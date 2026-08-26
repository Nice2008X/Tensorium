import type { ModelEdge } from "@tensorium/model-ir";

export interface LayoutPosition {
  x: number;
  y: number;
}

const CHAIN_COL_WIDTH = 220;
const BRANCH_COL_WIDTH = 240;
const CHAIN_ROW_HEIGHT = 120;
/** Extra vertical room wherever a rank fans out to (or in from) more than one node — gives the junction dot and offset ports space to read clearly instead of being cramped against the next rank. */
const BRANCH_ROW_HEIGHT = 170;

/** Barycenter sweeps to run before assigning x positions — enough for the ordering to settle on real graphs (a handful of ranks) without unbounded cost on a big one; each sweep is O(nodes), see reduceCrossings. */
const BARYCENTER_ITERATIONS = 4;

/**
 * Minimal layered ("Sugiyama-style") layout: rank nodes by longest path from
 * a source, then spread each rank horizontally. Works for a straight chain
 * (level 1) and for branching subgraphs like Attention's Q/K/V (level 2)
 * without any architecture-specific logic — it only reads the IR's edges.
 */
export function layeredLayout(nodeIds: string[], edges: ModelEdge[]): Map<string, LayoutPosition> {
  const idSet = new Set(nodeIds);
  const relevant = edges.filter((e) => idSet.has(e.source) && idSet.has(e.target));

  const preds = new Map<string, string[]>();
  const succs = new Map<string, string[]>();
  for (const id of nodeIds) {
    preds.set(id, []);
    succs.set(id, []);
  }
  for (const e of relevant) {
    preds.get(e.target)!.push(e.source);
    succs.get(e.source)!.push(e.target);
  }

  const rank = new Map<string, number>();
  const order = topologicalOrder(nodeIds, relevant);
  for (const id of order) {
    const p = preds.get(id) ?? [];
    const r = p.length === 0 ? 0 : Math.max(...p.map((x) => rank.get(x) ?? 0)) + 1;
    rank.set(id, r);
  }

  const byRank = new Map<number, string[]>();
  for (const id of nodeIds) {
    const r = rank.get(id) ?? 0;
    if (!byRank.has(r)) byRank.set(r, []);
    byRank.get(r)!.push(id);
  }
  const maxRank = Math.max(0, ...byRank.keys());

  // A node's insertion-order position within its rank isn't enough on its
  // own to avoid crossings: a node that both continues its own branch *and*
  // feeds into a shared merge point further down (DeepSeek-V2's MLA is a
  // real example — its KV Down-projection feeds both its own RMSNorm and,
  // directly, the RoPE node Q Projection also feeds) needs its column
  // chosen relative to where its neighbors actually landed, not just the
  // order it was created in. Standard fix: a few barycenter sweeps —
  // repeatedly reorder each rank by the average column of its neighbors in
  // the adjacent rank, alternating sweep direction so both predecessors and
  // successors get a say. Two kinds of edge are excluded, both for the same
  // reason — an excluded edge is always routed through its own dedicated
  // lane/detour regardless of x position (see ArchitectureGraph.tsx), so
  // letting it influence column choice would trade off ordering that
  // matters for edges that don't: "skip" (residual) edges, and any edge
  // that spans more than one rank (its target's column lives in a
  // differently-sized, incomparable rank — using it directly produced a
  // real bad reorder, caught by this fix's own regression sweep).
  reduceCrossings(
    byRank,
    maxRank,
    relevant.filter((e) => e.label !== "skip" && rank.get(e.target) === (rank.get(e.source) ?? -Infinity) + 1)
  );

  // Cumulative y per rank rather than a fixed `r * ROW_HEIGHT` — the gap
  // leading into or out of any rank with more than one node (a branch or a
  // merge) gets extra room; a plain chain segment stays compact.
  const rankY = new Map<number, number>();
  rankY.set(0, 0);
  for (let r = 1; r <= maxRank; r++) {
    const prevBranches = (byRank.get(r - 1)?.length ?? 1) > 1;
    const thisBranches = (byRank.get(r)?.length ?? 1) > 1;
    const gap = prevBranches || thisBranches ? BRANCH_ROW_HEIGHT : CHAIN_ROW_HEIGHT;
    rankY.set(r, (rankY.get(r - 1) ?? 0) + gap);
  }

  const positions = new Map<string, LayoutPosition>();
  for (const [r, ids] of byRank) {
    const n = ids.length;
    const colWidth = n > 1 ? BRANCH_COL_WIDTH : CHAIN_COL_WIDTH;
    ids.forEach((id, i) => {
      const x = (i - (n - 1) / 2) * colWidth;
      positions.set(id, { x, y: rankY.get(r) ?? r * CHAIN_ROW_HEIGHT });
    });
  }
  return positions;
}

/**
 * Reorders each rank's node list in place (mutating the arrays `byRank`
 * holds) by alternating downward sweeps (order each rank by the mean
 * column of its predecessors, freshly reordered one rank up) and upward
 * sweeps (mirror, using successors) — the standard barycenter heuristic for
 * reducing layered-graph edge crossings. A node with no positioned
 * neighbor on the sweep's side keeps its current slot (falls back to
 * insertion order) rather than collapsing to column 0.
 */
function reduceCrossings(byRank: Map<number, string[]>, maxRank: number, orderingEdges: ModelEdge[]) {
  const preds = new Map<string, string[]>();
  const succs = new Map<string, string[]>();
  for (const e of orderingEdges) {
    if (!preds.has(e.target)) preds.set(e.target, []);
    preds.get(e.target)!.push(e.source);
    if (!succs.has(e.source)) succs.set(e.source, []);
    succs.get(e.source)!.push(e.target);
  }

  const columnOf = new Map<string, number>();
  const reindex = () => {
    columnOf.clear();
    for (const ids of byRank.values()) ids.forEach((id, i) => columnOf.set(id, i));
  };
  reindex();

  const barycenterSort = (ids: string[], neighborsOf: Map<string, string[]>): string[] => {
    const withBary = ids.map((id, i) => {
      const cols = (neighborsOf.get(id) ?? []).map((n) => columnOf.get(n)).filter((c): c is number => c != null);
      const bary = cols.length > 0 ? cols.reduce((a, b) => a + b, 0) / cols.length : i;
      return { id, bary, i };
    });
    // Stable tie-break on the current index, so a node with no positioned
    // neighbor (bary falls back to `i`) never swaps past one that does.
    withBary.sort((a, b) => a.bary - b.bary || a.i - b.i);
    return withBary.map((x) => x.id);
  };

  for (let iter = 0; iter < BARYCENTER_ITERATIONS; iter++) {
    const downward = iter % 2 === 0;
    for (let r = downward ? 0 : maxRank; downward ? r <= maxRank : r >= 0; r += downward ? 1 : -1) {
      const ids = byRank.get(r);
      if (!ids || ids.length < 2) continue;
      byRank.set(r, barycenterSort(ids, downward ? preds : succs));
      reindex();
    }
  }

  // Barycenter alone can settle on a tie it resolves the wrong way: e.g. a
  // node with two children (each one column away) can end up with an
  // unrelated third node's single child sandwiched between them, which
  // barycenter's per-node averaging doesn't "see" as a problem even though
  // it forces a real crossing (caught by this fix's own regression sweep,
  // on a genuine two-child/one-child fan-out). A transpose pass — swap two
  // *adjacent* nodes whenever doing so strictly reduces how many of their
  // combined predecessor/successor edges cross — is the standard follow-up
  // in the Sugiyama method for exactly this. Bounded by `ids.length` passes
  // per rank, same as a worst-case bubble sort.
  for (const ids of byRank.values()) {
    if (ids.length < 2) continue;
    for (let pass = 0; pass < ids.length; pass++) {
      let improved = false;
      for (let i = 0; i < ids.length - 1; i++) {
        const a = ids[i];
        const b = ids[i + 1];
        const colsOf = (id: string) =>
          [...(preds.get(id) ?? []), ...(succs.get(id) ?? [])]
            .map((n) => columnOf.get(n))
            .filter((c): c is number => c != null);
        const colsA = colsOf(a);
        const colsB = colsOf(b);
        let before = 0;
        let after = 0;
        for (const ca of colsA) for (const cb of colsB) {
          if (ca > cb) before++;
          if (cb > ca) after++;
        }
        if (after < before) {
          ids[i] = b;
          ids[i + 1] = a;
          reindex();
          improved = true;
        }
      }
      if (!improved) break;
    }
  }
}

function topologicalOrder(nodeIds: string[], edges: ModelEdge[]): string[] {
  const inDegree = new Map<string, number>();
  const adj = new Map<string, string[]>();
  for (const id of nodeIds) {
    inDegree.set(id, 0);
    adj.set(id, []);
  }
  for (const e of edges) {
    adj.get(e.source)!.push(e.target);
    inDegree.set(e.target, (inDegree.get(e.target) ?? 0) + 1);
  }
  const queue = nodeIds.filter((id) => inDegree.get(id) === 0);
  const result: string[] = [];
  while (queue.length) {
    const id = queue.shift()!;
    result.push(id);
    for (const next of adj.get(id) ?? []) {
      inDegree.set(next, (inDegree.get(next) ?? 0) - 1);
      if (inDegree.get(next) === 0) queue.push(next);
    }
  }
  // any nodes left (cycle, shouldn't happen) — append in original order
  for (const id of nodeIds) if (!result.includes(id)) result.push(id);
  return result;
}
