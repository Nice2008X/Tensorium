import type { Model, ModelNode } from "@tensorium/model-ir";
import { componentRegistry } from "./registry.js";
import type { TranslationKey } from "./i18n.js";

type T = (key: TranslationKey) => string;

export interface InputSource {
  label: string;
  isBlockBoundary: boolean;
  /** A valid lookup key into ActivationCapture.activations for this source's real value. For a "block input" boundary source this is *not* the block container's own id (see resolveBoundarySources for why) — it's whichever node actually produced the value that crossed the boundary. */
  sourceId: string;
}

/** True if `ancestorId` is a strict ancestor of `node` (walking `parentId`, not including `node` itself). */
function isAncestor(model: Model, ancestorId: string, node: ModelNode): boolean {
  let cur = node.parentId;
  while (cur) {
    if (cur === ancestorId) return true;
    cur = model.nodes[cur]?.parentId ?? null;
  }
  return false;
}

/**
 * A block-boundary edge's `source` is the *container* (e.g. `block.3`), used
 * so the collapsed graph view stays connected — but every node's captured
 * activation is its own *output* (see every `record(id, ...)` call in each
 * adapter's `inference.ts`), containers included. So `activations[container]`
 * is that block's result, not what fed into it — using it directly as "this
 * boundary's input value" would silently show the wrong tensor (e.g. block
 * 3's own output, mislabeled as block 3's input). The real input value is
 * whatever produced the container itself, so resolve one hop further: the
 * container's own incoming edges. Those sources are genuine leaf/other-block
 * outputs, correctly captured under their own ids — e.g. block 0's boundary
 * resolves to Token Embedding + Positional Embedding (its two real inputs);
 * block N>0's resolves to block N-1 (its predecessor's real output).
 */
function resolveBoundarySources(model: Model, containerId: string, t: T): InputSource[] {
  const upstream = model.edges.filter((e) => e.target === containerId);
  if (upstream.length === 0) {
    // No recorded producer for this container (e.g. it's the graph root) —
    // fall back to its own id, still a valid (if less descriptive) lookup
    // key into ActivationCapture.activations.
    return [{ label: t("nodeInputs.blockInputFallback"), isBlockBoundary: true, sourceId: containerId }];
  }
  return upstream.map((e) => {
    const src = model.nodes[e.source];
    const parent = src?.parentId ? model.nodes[src.parentId] : null;
    // "Residual Add" / "LayerNorm 1" etc. are reused verbatim by every block
    // — for a source living inside some *other* transformer block (e.g. the
    // previous block's own output), name that block too so it doesn't read
    // as if it belongs to the block being inspected.
    const name = parent?.type === "transformer_block" ? `${parent.name} → ${src?.name ?? e.source}` : src?.name ?? e.source;
    return { label: t("nodeInputs.blockInputPrefix").replace("{name}", name), isBlockBoundary: true, sourceId: e.source };
  });
}

/**
 * Which upstream node(s) actually feed this one, and — where it's safe to
 * say so — how they combine. Shared by Inspector (the "Input construction"
 * formula + "This run" input rows) and TensorExplorer (the Input/Output
 * tab's input-source picker) so both agree on exactly the same sources,
 * rather than two independent implementations of this edge-resolution
 * logic drifting apart. Two things make this trickier than just reading
 * `model.edges`:
 *
 * - A container's own id is sometimes used as an edge source purely so the
 *   *collapsed* graph view stays connected once its children are hidden
 *   (e.g. GPT-2's `edge(ffnContainer, fc)` alongside the real `edge(ln2,
 *   fc)` — the same pattern `buildLevel2Graph` already has to work around).
 *   Those are dropped whenever a real sibling/leaf edge already covers the
 *   connection. When a container edge is the *only* incoming edge, though
 *   (e.g. `edge(block, firstNorm)`), it's genuine — that's the block's own
 *   boundary — and is kept, labeled as such rather than by the container's
 *   own name (matching the graph's "Block Input" convention).
 * - Once real edges are settled, the combining operator is only asserted
 *   when it's actually knowable in general: "addition" category or a
 *   "skip"-labeled edge means +, "elementwise" means ×. A "linear" node
 *   with several incoming edges (e.g. Output Projection reading Q/K/V) is
 *   doing real attention math, not a simple combine — its own formula
 *   (shown separately) already covers that, so no operator is guessed here.
 */
export function describeInputConstruction(model: Model, node: ModelNode, t: T): { sources: InputSource[]; operator: "+" | "×" | null } {
  const incoming = model.edges.filter((e) => e.target === node.id);
  const nonAncestor = incoming.filter((e) => e.label === "skip" || !isAncestor(model, e.source, node));
  // Container-sourced edges only survive if nothing more specific covers
  // the connection — otherwise they're the redundant "collapsed view" kind.
  const kept = nonAncestor.length > 0 ? nonAncestor : incoming;

  const sources: InputSource[] = kept.flatMap((e) => {
    // Ancestor-sourced edges get relabeled regardless of "skip" — a
    // residual's skip edge is very often exactly this pattern (the block's
    // own original input, carried around the sub-layer), and deserves the
    // same "Block Input" wording the graph itself uses rather than the
    // container's literal name. Resolved one hop further (see
    // resolveBoundarySources) since the container's *own* captured
    // activation is its output, not this boundary's input value.
    const boundary = isAncestor(model, e.source, node);
    return boundary ? resolveBoundarySources(model, e.source, t) : [{ label: model.nodes[e.source]?.name ?? e.source, isBlockBoundary: false, sourceId: e.source }];
  });

  if (sources.length < 2) return { sources, operator: null };

  const info = componentRegistry[node.type];
  if (info.category === "elementwise") return { sources, operator: "×" };
  if (info.category === "addition" || kept.some((e) => e.label === "skip")) return { sources, operator: "+" };
  if (info.category === "linear" || info.category === "other") return { sources, operator: null };
  // A structural node (e.g. a transformer block) combining two or more
  // untagged, non-container sources — every current instance of this
  // (token + positional embedding feeding the first block) is a plain sum.
  return { sources, operator: "+" };
}
