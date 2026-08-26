import type { Model, ModelEdge, NodeType } from "@tensorium/model-ir";

/**
 * Pre-order traversal, children visited left-to-right in the order the
 * adapter's graph.ts actually created them — that order is what
 * layeredLayout uses to position siblings within a rank (see layout.ts), so
 * getting it backwards here doesn't just reorder an internal list, it makes
 * unrelated branches land on the wrong side of each other one or more ranks
 * down and cross visibly (confirmed against a real DeepSeek-V2 MLA block:
 * KV Down-projection/Q Projection swapped left-right relative to creation
 * order, crossing the very next rank's RoPE/RMSNorm edges). A stack-based
 * DFS visits children in *reverse* push order unless each node's children
 * are reversed before pushing — that's what the extra `.reverse()` calls
 * below correct for; dropping either one silently reintroduces the bug.
 */
export function getDescendants(model: Model, id: string): string[] {
  const out: string[] = [];
  const stack = [...model.nodes[id].children].reverse();
  while (stack.length) {
    const cur = stack.pop()!;
    out.push(cur);
    stack.push(...[...model.nodes[cur].children].reverse());
  }
  return out;
}

/**
 * Only the leaf (no-children) descendants — the actual computation steps.
 * Purely-organizational container nodes (e.g. "Attention" grouping Q/K/V/Out)
 * are skipped here; collapseEdges bridges over them automatically since it
 * walks each edge endpoint up to its nearest *visible* ancestor. Containers
 * stay fully browsable via the ModelTree, just not as their own graph box.
 */
export function getLeafDescendants(model: Model, id: string): string[] {
  return getDescendants(model, id).filter((d) => model.nodes[d].children.length === 0);
}

function findVisibleAncestor(model: Model, nodeId: string, visible: Set<string>): string | null {
  let cur: string | null = nodeId;
  while (cur) {
    if (visible.has(cur)) return cur;
    cur = model.nodes[cur]?.parentId ?? null;
  }
  return null;
}

/**
 * Given a set of "visible" node ids (some of which are containers standing
 * in for a hidden subtree), re-derives the edges between exactly those
 * nodes by walking each real edge's endpoints up to their nearest visible
 * ancestor. This is what makes level-1 (collapsed blocks) and level-2
 * (one block's internals) both renderable from the same underlying edge
 * list, with no per-architecture special-casing.
 *
 * `label` is carried through (e.g. adapters tag every residual connection
 * `"skip"`) so callers can style/route data-flow vs. residual edges
 * differently instead of every edge looking the same.
 */
export function collapseEdges(model: Model, visibleIds: string[]): ModelEdge[] {
  const visible = new Set(visibleIds);
  const seen = new Set<string>();
  const result: ModelEdge[] = [];
  for (const e of model.edges) {
    const s = findVisibleAncestor(model, e.source, visible);
    const t = findVisibleAncestor(model, e.target, visible);
    if (!s || !t || s === t) continue;
    const key = `${s}->${t}`;
    if (seen.has(key)) continue;
    seen.add(key);
    result.push({ id: key, source: s, target: t, label: e.label });
  }
  return result;
}

const ELLIPSIS = "__ellipsis__";

export interface Level1Graph {
  nodeIds: string[];
  edges: ModelEdge[];
  ellipsisCount: number;
}

/** The top-level architecture view: input, embeddings, every block (or an abbreviated run of them), head, output. */
export function buildLevel1Graph(model: Model, showAllBlocks: boolean, maxEdgeBlocks = 4): Level1Graph {
  const root = model.nodes[model.rootId];
  const blockGroup = root.children.find((id) => model.nodes[id].type === "block_group");
  const allBlockIds = blockGroup ? model.nodes[blockGroup].children : [];
  const nonBlockChildren = root.children.filter((id) => id !== blockGroup);

  const before = nonBlockChildren.filter((id) => {
    const rank = ["input", "embedding", "positional_embedding"].indexOf(model.nodes[id].type);
    return rank !== -1;
  });
  const after = nonBlockChildren.filter((id) => !before.includes(id));

  let blockIdsToShow = allBlockIds;
  let ellipsisCount = 0;
  if (!showAllBlocks && allBlockIds.length > maxEdgeBlocks * 2 + 1) {
    blockIdsToShow = [...allBlockIds.slice(0, maxEdgeBlocks), ...allBlockIds.slice(allBlockIds.length - maxEdgeBlocks)];
    ellipsisCount = allBlockIds.length - maxEdgeBlocks * 2;
  }

  const firstHalf = ellipsisCount > 0 ? blockIdsToShow.slice(0, maxEdgeBlocks) : blockIdsToShow;
  const secondHalf = ellipsisCount > 0 ? blockIdsToShow.slice(maxEdgeBlocks) : [];

  const nodeIds = [...before, ...firstHalf, ...(ellipsisCount > 0 ? [ELLIPSIS] : []), ...secondHalf, ...after];
  const edges = collapseEdges(model, [...before, ...blockIdsToShow, ...after]);

  if (ellipsisCount > 0) {
    edges.push({ id: "e1", source: firstHalf[firstHalf.length - 1], target: ELLIPSIS });
    edges.push({ id: "e2", source: ELLIPSIS, target: secondHalf[0] });
  }

  return { nodeIds, edges, ellipsisCount };
}

export { ELLIPSIS };

const BLOCK_INPUT = "__block_input__";

/**
 * The block-detail view: every computation step inside one transformer
 * block. Every adapter emits two edges sourced at the block's own id —
 * `edge(block, firstNorm)` (the main data-flow entry) and
 * `edge(block, residualAdd, "skip")` (the residual entry, carried around
 * attention/FFN) — representing "the block's input, from outside". These
 * need `blockId` added to the visible set to resolve at all (their source,
 * the block itself, is neither a leaf nor an ancestor of one), and get
 * remapped to a synthetic `BLOCK_INPUT` marker so they render as a real
 * "Block Input" box instead of reusing the block's own (misleadingly
 * "expandable") node.
 *
 * That can't be done by simply adding `blockId` to `collapseEdges`'s
 * visible set, though: some adapters also emit purely-organizational edges
 * sourced at a *hidden container* one level down (e.g. GPT-2's
 * `edge(ffnContainer, fc)`, kept only so the level-1 collapsed-block view
 * has a path through the container) — bubbled through `findVisibleAncestor`
 * once `blockId` is visible, a container with no other visible ancestor
 * resolves to `blockId` too, and gets misread as a genuine block-input edge.
 * The result: a bogus `Block Input -> Linear (expand)` edge that renders as
 * a plain (non-"skip") smoothstep line running nearly straight down through
 * every node in between, indistinguishable from a real connection. The
 * container's edge is redundant anyway — adapters already emit an explicit
 * leaf-to-leaf edge for the same case (`edge(ln2, fc)` alongside
 * `edge(ffn, fc)`) — so it's fine for it to simply be dropped instead.
 *
 * The fix: resolve the general edge set against the leaves only (never
 * `blockId`), and handle the two real block-input edges as a separate pass
 * that only looks at edges whose source is *exactly* `blockId` in the raw
 * model, before any ancestor-walking — so a hidden container can never be
 * mistaken for the block's own boundary.
 */
export function buildLevel2Graph(model: Model, blockId: string): { nodeIds: string[]; edges: ModelEdge[] } {
  const leafIds = getLeafDescendants(model, blockId);
  const leafSet = new Set(leafIds);
  const edges = collapseEdges(model, leafIds);

  let usesBlockInput = false;
  const seen = new Set(edges.map((e) => e.id));
  for (const e of model.edges) {
    if (e.source !== blockId) continue;
    const t = findVisibleAncestor(model, e.target, leafSet);
    if (!t) continue;
    const id = `${BLOCK_INPUT}->${t}`;
    if (seen.has(id)) continue;
    seen.add(id);
    usesBlockInput = true;
    edges.push({ id, source: BLOCK_INPUT, target: t, label: e.label });
  }

  const nodeIds = usesBlockInput ? [BLOCK_INPUT, ...leafIds] : leafIds;
  return { nodeIds, edges };
}

export { BLOCK_INPUT };

const STACK_PREFIX = "__stack__";

export interface StackGroup {
  /** The collapsed run's members, in sequence order. */
  memberIds: string[];
  /** Every member shares this NodeType, by construction. */
  type: NodeType;
}

/**
 * Collapses each maximal run of same-type nodes connected in a straight
 * chain (each one's sole outgoing edge feeds the next one's sole incoming
 * edge) into a single synthetic "stack" node — e.g. five sequential
 * Transformer Block nodes become one "Transformer Block × 5" node, with the
 * chain's boundary edges reattached to it and its internal edges dropped.
 * A run must be an actual straight line: a node with a branch (more than
 * one outgoing edge) or a merge (more than one incoming edge) breaks it,
 * same as a type change does.
 *
 * This only ever collapses *real* model nodes — synthetic markers like
 * `ELLIPSIS`/`BLOCK_INPUT` (absent from `model.nodes`) can't start or
 * extend a run, so an already-abbreviated block list naturally collapses
 * into two shorter runs on either side of the ellipsis rather than one.
 */
export function collapseRepeatedChains(
  model: Model,
  nodeIds: string[],
  edges: ModelEdge[]
): { nodeIds: string[]; edges: ModelEdge[]; stacks: Map<string, StackGroup> } {
  const idSet = new Set(nodeIds);
  const outBy = new Map<string, ModelEdge[]>();
  const inBy = new Map<string, ModelEdge[]>();
  for (const e of edges) {
    if (!idSet.has(e.source) || !idSet.has(e.target)) continue;
    if (!outBy.has(e.source)) outBy.set(e.source, []);
    outBy.get(e.source)!.push(e);
    if (!inBy.has(e.target)) inBy.set(e.target, []);
    inBy.get(e.target)!.push(e);
  }
  const isRealNode = (id: string) => !!model.nodes[id];

  function nextInChain(id: string): string | null {
    const outs = outBy.get(id);
    if (!outs || outs.length !== 1) return null;
    const target = outs[0].target;
    if (target === id || !isRealNode(target)) return null;
    if ((inBy.get(target)?.length ?? 0) !== 1) return null;
    return model.nodes[target].type === model.nodes[id].type ? target : null;
  }

  function hasChainPredecessor(id: string): boolean {
    const ins = inBy.get(id);
    if (!ins || ins.length !== 1) return false;
    return isRealNode(ins[0].source) && nextInChain(ins[0].source) === id;
  }

  const stacks = new Map<string, StackGroup>();
  const replaced = new Map<string, string>();
  for (const id of nodeIds) {
    if (!isRealNode(id) || hasChainPredecessor(id)) continue;
    const run = [id];
    for (let next = nextInChain(id); next; next = nextInChain(next)) run.push(next);
    if (run.length < 2) continue;
    const stackId = `${STACK_PREFIX}${id}`;
    stacks.set(stackId, { memberIds: run, type: model.nodes[id].type });
    for (const m of run) replaced.set(m, stackId);
  }
  if (stacks.size === 0) return { nodeIds, edges, stacks };

  const newNodeIds: string[] = [];
  const seenStack = new Set<string>();
  for (const id of nodeIds) {
    const stackId = replaced.get(id);
    if (!stackId) newNodeIds.push(id);
    else if (!seenStack.has(stackId)) {
      seenStack.add(stackId);
      newNodeIds.push(stackId);
    }
  }

  const newEdges: ModelEdge[] = [];
  const seenEdge = new Set<string>();
  for (const e of edges) {
    const s = replaced.get(e.source) ?? e.source;
    const t = replaced.get(e.target) ?? e.target;
    if (s === t) continue; // now-internal to a single stack
    const key = `${s}->${t}`;
    if (seenEdge.has(key)) continue;
    seenEdge.add(key);
    newEdges.push({ ...e, id: key, source: s, target: t });
  }

  return { nodeIds: newNodeIds, edges: newEdges, stacks };
}

export { STACK_PREFIX };
