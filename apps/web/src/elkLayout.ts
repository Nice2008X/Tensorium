import type { ModelEdge } from "@tensorium/model-ir";
import type { ElkExtendedEdge, ElkNode } from "elkjs/lib/elk.bundled.js";

export interface NodeSize {
  width: number;
  height: number;
}

export interface LayoutPosition {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface EdgeRoute {
  points: { x: number; y: number }[];
}

export interface ElkLayoutResult {
  positions: Map<string, LayoutPosition>;
  /** Routed (non-residual) edges only, keyed by edge id — absolute flow-space points including bends, straight from ELK. Residual edges keep the existing dedicated side-lane renderer instead (see ResidualLaneEdge). */
  routes: Map<string, EdgeRoute>;
}

const NODE_NODE_SPACING = 50;
const LAYER_SPACING = 70;

/**
 * ELK orders FIXED_ORDER ports by ascending `port.index`, but which screen
 * direction that maps to depends on the side — confirmed empirically (see
 * .scratch smoke tests run before writing this file, not assumed from docs):
 * on SOUTH (a node's outgoing ports), ascending index runs RIGHT to LEFT, so
 * matching creation order (first edge reads leftmost, as every adapter
 * assumes) requires reversing the index; on NORTH (incoming ports),
 * ascending index already runs LEFT to RIGHT, so creation order maps
 * directly with no reversal. `flip` inverts this once more — used by the
 * crossing-retry loop below.
 */
function southIndex(i: number, count: number, flip: boolean): string {
  return String(flip ? i : count - 1 - i);
}
function northIndex(i: number, count: number, flip: boolean): string {
  return String(flip ? count - 1 - i : i);
}

function portId(edgeId: string, end: "source" | "target"): string {
  return `${edgeId}::${end}`;
}

export function segmentsIntersect(a1: { x: number; y: number }, a2: { x: number; y: number }, b1: { x: number; y: number }, b2: { x: number; y: number }): boolean {
  const d = (p: { x: number; y: number }, q: { x: number; y: number }, r: { x: number; y: number }) => (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x);
  const d1 = d(b1, b2, a1);
  const d2 = d(b1, b2, a2);
  const d3 = d(a1, a2, b1);
  const d4 = d(a1, a2, b2);
  return ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0));
}

/** Counts real geometric crossings among routed (non-residual) edges' own polylines — edges sharing an endpoint node are excluded, since they legitimately touch there. */
function countCrossings(routed: ModelEdge[], routes: Map<string, EdgeRoute>): number {
  const lines = routed.map((e) => ({ source: e.source, target: e.target, points: routes.get(e.id)?.points })).filter((l): l is { source: string; target: string; points: { x: number; y: number }[] } => !!l.points);
  let crossings = 0;
  for (let i = 0; i < lines.length; i++) {
    for (let j = i + 1; j < lines.length; j++) {
      const a = lines[i];
      const b = lines[j];
      if (a.source === b.source || a.source === b.target || a.target === b.source || a.target === b.target) continue;
      outer: for (let si = 0; si < a.points.length - 1; si++) {
        for (let sj = 0; sj < b.points.length - 1; sj++) {
          if (segmentsIntersect(a.points[si], a.points[si + 1], b.points[sj], b.points[sj + 1])) {
            crossings++;
            break outer;
          }
        }
      }
    }
  }
  return crossings;
}

interface RunResult {
  positions: Map<string, LayoutPosition>;
  routes: Map<string, EdgeRoute>;
}

async function runElk(
  nodeIds: string[],
  routed: ModelEdge[],
  outByNode: Map<string, ModelEdge[]>,
  inByNode: Map<string, ModelEdge[]>,
  sizes: Map<string, NodeSize>,
  fallbackSize: NodeSize,
  flippedNodes: ReadonlySet<string>
): Promise<RunResult> {
  const children: ElkNode[] = nodeIds.map((id) => {
    const size = sizes.get(id) ?? fallbackSize;
    const ports: NonNullable<ElkNode["ports"]> = [];
    const flip = flippedNodes.has(id);

    const outs = outByNode.get(id);
    if (outs && outs.length > 1) {
      outs.forEach((e, i) => {
        ports.push({ id: portId(e.id, "source"), width: 1, height: 1, layoutOptions: { "port.side": "SOUTH", "port.index": southIndex(i, outs.length, flip) } });
      });
    }
    const ins = inByNode.get(id);
    if (ins && ins.length > 1) {
      ins.forEach((e, i) => {
        ports.push({ id: portId(e.id, "target"), width: 1, height: 1, layoutOptions: { "port.side": "NORTH", "port.index": northIndex(i, ins.length, flip) } });
      });
    }

    return {
      id,
      width: size.width,
      height: size.height,
      ...(ports.length ? { ports, layoutOptions: { portConstraints: "FIXED_ORDER" } } : {}),
    };
  });

  const elkEdges: ElkExtendedEdge[] = routed.map((e) => {
    const outs = outByNode.get(e.source)!;
    const ins = inByNode.get(e.target)!;
    const sourceId = outs.length > 1 ? portId(e.id, "source") : e.source;
    const targetId = ins.length > 1 ? portId(e.id, "target") : e.target;
    return { id: e.id, sources: [sourceId], targets: [targetId] };
  });

  const graph: ElkNode = {
    id: "root",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": "DOWN",
      "elk.layered.spacing.nodeNodeBetweenLayers": String(LAYER_SPACING),
      "elk.spacing.nodeNode": String(NODE_NODE_SPACING),
      "elk.layered.layering.strategy": "LONGEST_PATH",
      "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
      "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
      "elk.layered.crossingMinimization.greedySwitch.type": "TWO_SIDED",
      "elk.layered.crossingMinimization.greedySwitch.activationThreshold": "0",
      "elk.layered.thoroughness": "100",
      "elk.layered.spacing.edgeNodeBetweenLayers": "30",
      "elk.layered.spacing.edgeEdgeBetweenLayers": "20",
      "elk.spacing.edgeNode": "20",
      "elk.spacing.edgeEdge": "15",
      "elk.edgeRouting": "ORTHOGONAL",
    },
    children,
    edges: elkEdges,
  };

  // Dynamically imported so Vite code-splits ELK's ~1.6MB bundle into its
  // own chunk instead of inflating the app's existing 553KB main bundle —
  // it only loads once a graph actually needs laying out.
  const { default: ELK } = await import("elkjs/lib/elk.bundled.js");
  const elk = new ELK();
  const result = await elk.layout(graph);

  const positions = new Map<string, LayoutPosition>();
  for (const c of result.children ?? []) {
    positions.set(c.id, { x: c.x ?? 0, y: c.y ?? 0, width: c.width ?? fallbackSize.width, height: c.height ?? fallbackSize.height });
  }

  const routes = new Map<string, EdgeRoute>();
  for (const e of result.edges ?? []) {
    const section = e.sections?.[0];
    if (!section) continue;
    routes.set(e.id, { points: [section.startPoint, ...(section.bendPoints ?? []), section.endPoint] });
  }

  return { positions, routes };
}

/**
 * Runs ELK's layered algorithm over one view's visible nodes/edges. Residual
 * ("kind: residual") edges are excluded from ELK's own graph entirely — they
 * keep the pre-existing dedicated side-lane treatment (ResidualLaneEdge),
 * which was never the source of any of the three real layout bugs fixed
 * earlier this session and doesn't need a general-purpose routing engine:
 * their source/target are always single-node ranks with nothing to dodge.
 * Every other edge is routed by ELK, including what used to need this app's
 * own hand-rolled "hub lane" special case (one source feeding many
 * far-apart ranks) — ELK's layered algorithm already inserts the dummy
 * nodes a long edge needs to route around intervening ranks, which is
 * exactly what that hand-rolled case was working around.
 *
 * Even well-tuned, ELK's crossing minimizer is a heuristic and can settle
 * on a local optimum with a real crossing it can't reach from where it
 * started (confirmed empirically against this app's own regression suite —
 * see apps/web/scripts/check-graph-layout.ts — on a real qwen3-5 topology
 * where a node with no port constraint of its own gets pulled toward its
 * upstream node's column at the cost of one crossing two ranks down). One
 * ELK.layout() call is cheap on graphs this size, so rather than build a
 * bespoke local-search optimizer, this retries with each individual
 * multi-port node's fixed order flipped, one at a time, and keeps whichever
 * attempt has the fewest real (geometric, not topological) crossings —
 * bounded to one extra ELK call per multi-port node, which is always a
 * small count for a single transformer block.
 */
export async function computeElkLayout(nodeIds: string[], edges: ModelEdge[], sizes: Map<string, NodeSize>, fallbackSize: NodeSize): Promise<ElkLayoutResult> {
  const idSet = new Set(nodeIds);
  const relevant = edges.filter((e) => idSet.has(e.source) && idSet.has(e.target) && e.source !== e.target);
  const routed = relevant.filter((e) => e.kind !== "residual");

  // Grouped in `nodeIds`/edge-array order, which is the adapter's own
  // creation order — the same order graphUtils.ts's getDescendants already
  // treats as authoritative for sibling layout (see its own doc comment on
  // the DFS-reversal bug fixed earlier this session).
  const outByNode = new Map<string, ModelEdge[]>();
  const inByNode = new Map<string, ModelEdge[]>();
  for (const e of routed) {
    if (!outByNode.has(e.source)) outByNode.set(e.source, []);
    outByNode.get(e.source)!.push(e);
    if (!inByNode.has(e.target)) inByNode.set(e.target, []);
    inByNode.get(e.target)!.push(e);
  }

  const multiPortNodes = nodeIds.filter((id) => (outByNode.get(id)?.length ?? 0) > 1 || (inByNode.get(id)?.length ?? 0) > 1);

  const baseline = await runElk(nodeIds, routed, outByNode, inByNode, sizes, fallbackSize, new Set());
  let best = baseline;
  let bestCrossings = countCrossings(routed, baseline.routes);

  for (const flipId of multiPortNodes) {
    if (bestCrossings === 0) break;
    const attempt = await runElk(nodeIds, routed, outByNode, inByNode, sizes, fallbackSize, new Set([flipId]));
    const crossings = countCrossings(routed, attempt.routes);
    if (crossings < bestCrossings) {
      best = attempt;
      bestCrossings = crossings;
    }
  }

  return best;
}
