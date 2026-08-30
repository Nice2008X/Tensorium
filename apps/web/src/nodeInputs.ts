import type { Model, ModelNode } from "@tensorium/model-ir";
import { componentRegistry } from "./registry.js";

export interface InputSource {
  label: string;
  isBlockBoundary: boolean;
  /** The real edge source id — even for a "block input" boundary source, where it's the block container's own id. That container's captured activation *is* the tensor that actually flowed in at this boundary, so it's still a valid lookup key into ActivationCapture.activations, just not a very descriptive node name (hence the separate `label`). */
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
export function describeInputConstruction(model: Model, node: ModelNode): { sources: InputSource[]; operator: "+" | "×" | null } {
  const incoming = model.edges.filter((e) => e.target === node.id);
  const nonAncestor = incoming.filter((e) => e.label === "skip" || !isAncestor(model, e.source, node));
  // Container-sourced edges only survive if nothing more specific covers
  // the connection — otherwise they're the redundant "collapsed view" kind.
  const kept = nonAncestor.length > 0 ? nonAncestor : incoming;

  const sources: InputSource[] = kept.map((e) => {
    // Ancestor-sourced edges get relabeled regardless of "skip" — a
    // residual's skip edge is very often exactly this pattern (the block's
    // own original input, carried around the sub-layer), and deserves the
    // same "Block Input" wording the graph itself uses rather than the
    // container's literal name.
    const boundary = isAncestor(model, e.source, node);
    return { label: boundary ? "Block input (from outside this block)" : model.nodes[e.source]?.name ?? e.source, isBlockBoundary: boundary, sourceId: e.source };
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
