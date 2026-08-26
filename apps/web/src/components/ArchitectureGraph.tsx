import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactFlow, {
  Background,
  BaseEdge,
  ControlButton,
  Controls,
  getNodesBounds,
  getViewportForBounds,
  Handle,
  MarkerType,
  MiniMap,
  Position,
  type Edge as RFEdge,
  type EdgeProps,
  type Node as RFNode,
  type ReactFlowInstance,
} from "reactflow";
import "reactflow/dist/style.css";
import { toPng } from "html-to-image";
import type { Model, ModelEdge, ModelNode } from "@tensorium/model-ir";
import { categoryGlyph, categoryLabel, componentRegistry } from "../registry.js";
import { layeredLayout } from "../layout.js";
import { BLOCK_INPUT, buildLevel1Graph, buildLevel2Graph, collapseRepeatedChains, ELLIPSIS, getLeafDescendants, STACK_PREFIX, type StackGroup } from "../graphUtils.js";
import { formatCount } from "../format.js";
import { useLocalStorageState } from "../useLocalStorageState.js";
import { useTranslation } from "./LanguageContext.js";

export type GraphView = { kind: "architecture" } | { kind: "block"; blockId: string };

interface Props {
  model: Model;
  view: GraphView;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  onEnterBlock: (blockId: string) => void;
  onExitBlock: () => void;
  /** Whether the surrounding panels (prediction, tree, inspector, bottom) are currently collapsed to give the graph maximum space. */
  isMaxFrame: boolean;
  onToggleMaxFrame: () => void;
}

interface IRNodeData {
  node?: ModelNode;
  label: string;
  sublabel: string;
  glyph?: string;
  dims?: string;
  color: string;
  selected: boolean;
  dimmed: boolean;
  expandable?: boolean;
  inputPorts: number;
  outputPorts: number;
  /** Set when this node stands in for a collapsed run of `stackCount` identical, sequentially-connected nodes — rendered as a small deck of cards instead of a single box. */
  stackCount?: number;
}

/**
 * A node with more than one sibling input/output gets one named Handle per
 * sibling instead of a single shared center point — otherwise every
 * incoming/outgoing edge visually converges on the exact same pixel, which
 * is the main reason a branching FFN/attention block reads as ambiguous
 * curves rather than a clear branch/merge.
 */
function PortHandles({ kind, position, count }: { kind: "target" | "source"; position: Position; count: number }) {
  if (count <= 1) return <Handle type={kind} position={position} style={{ opacity: 0 }} />;
  const prefix = kind === "target" ? "tgt" : "src";
  return (
    <>
      {Array.from({ length: count }, (_, i) => (
        <Handle key={i} type={kind} position={position} id={`${prefix}-${i}`} style={{ opacity: 0, left: `${((i + 1) / (count + 1)) * 100}%` }} />
      ))}
    </>
  );
}

function IRNodeComponent({ data }: { data: IRNodeData }) {
  const stacked = !!data.stackCount && data.stackCount > 1;
  // An adapter can shorten a node's displayed name to fit its column (see
  // .ir-node's max-width) while keeping the un-shortened name in
  // metadata.fullName — when it's actually longer/different from what's
  // shown, that's what the hover tooltip reveals, so nothing named "X (Y)"
  // actually loses the "(Y)" part, it just moves off the box and onto
  // hover. No tooltip at all when there's nothing extra to reveal — most
  // nodes' full name is identical to their label, and a tooltip that just
  // repeats the visible text on every single node is noise, not help.
  const fullName = data.node?.metadata.fullName as string | undefined;
  const tooltip = fullName && fullName !== data.label ? fullName : undefined;
  const card = (
    <div
      className={"ir-node nopan nodrag" + (data.selected ? " selected" : "") + (data.dimmed ? " dimmed" : "")}
      style={{ borderColor: data.color }}
      title={tooltip}
    >
      <PortHandles kind="target" position={Position.Top} count={data.inputPorts} />
      <div className="ir-node-label">
        {data.glyph && <span className="ir-node-glyph">{data.glyph}</span>}
        {data.label}
        {stacked && <span className="ir-node-stack-badge">× {data.stackCount}</span>}
      </div>
      <div className="ir-node-sub">{data.sublabel}</div>
      {data.dims && <div className="ir-node-dims">{data.dims}</div>}
      {data.expandable && <div className="ir-node-hint">double-click to expand</div>}
      <PortHandles kind="source" position={Position.Bottom} count={data.outputPorts} />
      {/* Dedicated right-side ports for residual/skip edges routed through the
          side lane (see ResidualEdge below) — kept separate from the
          top/bottom main-flow ports so a residual connection never competes
          with a sibling data-flow edge for the same numbered port. */}
      <Handle type="target" position={Position.Right} id="lane-in" style={{ opacity: 0 }} />
      <Handle type="source" position={Position.Right} id="lane-out" style={{ opacity: 0 }} />
    </div>
  );
  // A collapsed run gets two faint offset copies of its own border peeking
  // out behind the real card — a "deck of cards" cue that this one box
  // stands in for several, without needing extra DOM for the common
  // (non-stacked) case.
  if (!stacked) return card;
  return (
    <div className="ir-node-stack-outer">
      <div className="ir-node-stack-layer ir-node-stack-layer-2" style={{ borderColor: data.color }} />
      <div className="ir-node-stack-layer ir-node-stack-layer-1" style={{ borderColor: data.color }} />
      {card}
    </div>
  );
}

interface LaneEdgeData {
  laneX: number;
}

/**
 * Residual/skip edges: Block Input and Residual Add are always single-node
 * ranks with nothing else beside them, so exiting straight out their right
 * side and down a vertical lane never has anything to cross.
 */
function ResidualLaneEdge({ id, sourceX, sourceY, targetX, targetY, markerEnd, style, data }: EdgeProps<LaneEdgeData>) {
  const laneX = data?.laneX ?? Math.max(sourceX, targetX) + 60;
  const path = `M ${sourceX},${sourceY} L ${laneX},${sourceY} L ${laneX},${targetY} L ${targetX},${targetY}`;
  return <BaseEdge id={id} path={path} markerEnd={markerEnd} style={style} />;
}

/**
 * Any other edge that skips over an intermediate rank — e.g. Llama's
 * V Projection -> Output Projection, which bypasses the RoPE rank entirely
 * since RoPE only applies to Q/K. Unlike the residual case, V has siblings
 * (K, Q) sitting right beside it, so exiting straight out its side would cut
 * across them. This uses the *same* Top/Bottom ports an ordinary edge would
 * (still correctly offset among sibling edges), and only detours sideways
 * to `laneX` after already dropping clear of the entire source rank's row —
 * every node in a rank shares its row's height, so once the path is below
 * that row it can travel at any x without crossing a same-rank sibling,
 * then it rises back above the target's row the same way before arriving.
 * `laneX` itself is chosen (see the detour lane computation below) to additionally clear
 * whatever sits in the rank(s) actually being skipped.
 */
function DetourEdge({ id, sourceX, sourceY, targetX, targetY, markerEnd, style, data }: EdgeProps<LaneEdgeData>) {
  const laneX = data?.laneX ?? (sourceX + targetX) / 2;
  const clear = 20; // matches the drop React Flow's own smoothstep uses before its first bend
  const y1 = sourceY + clear;
  const y2 = targetY - clear;
  const path = `M ${sourceX},${sourceY} L ${sourceX},${y1} L ${laneX},${y1} L ${laneX},${y2} L ${targetX},${y2} L ${targetX},${targetY}`;
  return <BaseEdge id={id} path={path} markerEnd={markerEnd} style={style} />;
}

/** Purely decorative marker at a fan-out/fan-in point — not a real graph node, carries no data, has no handles/edges of its own. */
function JunctionDot() {
  return <div className="graph-junction-dot" />;
}

interface ScopeBoxData {
  label: string;
  color: string;
}

/**
 * A rounded, dashed frame drawn behind whichever leaf nodes belong to the
 * currently-selected *container* — e.g. selecting "Attention" in the model
 * tree while looking at a block's detail view has no node of its own to
 * highlight (only its leaf children — Q/K/V Projection, RoPE, Output
 * Projection — are actually rendered), so without this there was no visual
 * feedback for that selection at all. A light tinted fill plus a dashed
 * outline (rather than either alone) keeps the group legible whether
 * you're scanning the overall shape or looking closely at the boundary —
 * the same combination Figma/Miro use for named frames.
 */
function ScopeBox({ data }: { data: ScopeBoxData }) {
  return (
    <div className="graph-scope-box" style={{ borderColor: data.color, background: `${data.color}14` }}>
      <span className="graph-scope-label" style={{ color: data.color, borderColor: data.color }}>
        {data.label}
      </span>
    </div>
  );
}

/**
 * "[n]" bracket glyph for the tensor-shape toggle, drawn as vector paths
 * instead of literal "[" / "]" characters — those two glyphs don't share a
 * baseline in every monospace font, so at the control button's small size
 * they visibly sit at different heights. An SVG keeps both brackets exactly
 * level, and doubles as a plainer stand-in for "array dimensions" (bracket
 * pair around a value) if the toggle's meaning isn't obvious from the text
 * alone.
 */
function ShapeIcon({ active }: { active: boolean }) {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M4.5 2.5h-2v9h2" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M9.5 2.5h2v9h-2" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
      {active && (
        <text x="7" y="8.8" textAnchor="middle" fontSize="6.5" fontFamily="ui-monospace, monospace" fill="currentColor">
          n
        </text>
      )}
    </svg>
  );
}

/** Two overlapping rounded rectangles for the "stack repeated nodes" toggle — echoes the same offset-card motif used on an actual stacked node, so the button reads as a small preview of what it does. */
function StackIcon({ active }: { active: boolean }) {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="1.5" y="1.5" width="8" height="6" rx="1.3" stroke="currentColor" strokeWidth="1.3" opacity={active ? 0.55 : 0.9} />
      <rect x="4.5" y="6.5" width="8" height="6" rx="1.3" stroke="currentColor" strokeWidth="1.3" fill={active ? "currentColor" : "none"} fillOpacity={active ? 0.18 : 0} />
    </svg>
  );
}

/** Downward arrow into a tray — standard "export/download" glyph, spinning while an export is in flight. */
function ExportIcon({ busy }: { busy: boolean }) {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg" className={busy ? "control-icon-spin" : undefined}>
      {busy ? (
        <path d="M12 7A5 5 0 1 1 9.5 2.6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
      ) : (
        <>
          <path d="M7 1.5v7" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
          <path d="M4 5.8 7 8.8l3-3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M2 10v1.5a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V10" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
        </>
      )}
    </svg>
  );
}

const nodeTypes = { ir: IRNodeComponent, junction: JunctionDot, scope: ScopeBox };
const edgeTypes = { lane: ResidualLaneEdge, detour: DetourEdge };

const EDGE_COLOR = "#94a3b8";
const SKIP_EDGE_COLOR = "#64748b";
const HUB_EDGE_COLOR = "#38bdf8";
/** A source feeding this many or more rank-skipping targets (e.g. Gemma-4's Per-Layer Input Projection reaching every one of 35 blocks) gets bus-lane routing instead of the local per-edge detour: below this, the ordinary detour/port system already reads fine (an FFN's Gate+Up, an attention block's Q/K/V), but every edge past this count would otherwise get pushed one `LANE_SEPARATION` further out than the last as their spans pile up, fanning into an ever-widening staircase. */
const HUB_MIN_TARGETS = 3;
/** Rough node width used only to eyeball where a junction dot sits horizontally — nodes auto-size, so this is an approximation, not a measurement. */
const NOMINAL_NODE_WIDTH = 160;
/** Rough node height, same caveat as NOMINAL_NODE_WIDTH — used only to size the scope box around a selected container's leaf children. */
const NOMINAL_NODE_HEIGHT = 76;
/** Breathing room between a scope box's edge and the nodes it encloses. */
const SCOPE_PADDING = 26;
/** Clearance between the widest node a lane has to clear and the lane itself. */
const LANE_GAP = 90;
/** Minimum horizontal separation between two lanes whose vertical runs overlap — keeps concurrent detours (e.g. both block residuals, or a residual and an unrelated skip) from tracing the same line. */
const LANE_SEPARATION = 50;

export function ArchitectureGraph({ model, view, selectedId, onSelect, onEnterBlock, onExitBlock, isMaxFrame, onToggleMaxFrame }: Props) {
  const { t } = useTranslation();
  const containerRef = useRef<HTMLDivElement>(null);
  const rfInstanceRef = useRef<ReactFlowInstance | null>(null);
  const [hoveredEdgeId, setHoveredEdgeId] = useState<string | null>(null);
  // Off by default: a "[sequence_length, 16]" label on every touched edge
  // is genuinely useful when you're chasing shapes, but it's clutter for
  // just browsing the architecture, so it stays opt-in rather than
  // appearing automatically whenever a node gets selected/hovered.
  const [showTensorShapes, setShowTensorShapes] = useLocalStorageState("panel:graph-tensor-shapes", false);
  // On by default: collapsing a repeated run (e.g. 5 near-identical
  // Transformer Blocks) into one "× 5" node is the more readable starting
  // view for most models — the toggle still lets anyone switch back to
  // seeing every block individually.
  const [stackRepeats, setStackRepeats] = useLocalStorageState("panel:graph-stack-repeats", true);
  const [exportingImage, setExportingImage] = useState(false);

  const { nodeIds: rawNodeIds, edgeList: rawEdgeList } = useMemo(() => {
    if (view.kind === "architecture") {
      const g = buildLevel1Graph(model, false);
      return { nodeIds: g.nodeIds, edgeList: g.edges };
    }
    const g = buildLevel2Graph(model, view.blockId);
    return { nodeIds: g.nodeIds, edgeList: g.edges };
  }, [model, view]);

  // Collapsing happens once, right after the view's raw graph is built, so
  // every downstream step (layout, ports, lane routing, the scope box)
  // just sees a smaller graph and needs no awareness that stacking exists.
  const { nodeIds, edgeList, stackGroups } = useMemo(() => {
    if (!stackRepeats) return { nodeIds: rawNodeIds, edgeList: rawEdgeList, stackGroups: new Map<string, StackGroup>() };
    const collapsed = collapseRepeatedChains(model, rawNodeIds, rawEdgeList);
    return { nodeIds: collapsed.nodeIds, edgeList: collapsed.edges, stackGroups: collapsed.stacks };
  }, [stackRepeats, rawNodeIds, rawEdgeList, model]);

  const positions = useMemo(() => layeredLayout(nodeIds, edgeList), [nodeIds, edgeList]);

  // Any edge that skips over at least one intermediate rank (there's some
  // other node positioned strictly between its source and target) would
  // otherwise draw straight through whatever sits in between — that's the
  // overlap the lane fixes, whether or not the edge is a labeled residual.
  // Edges whose source/target are adjacent ranks don't need it; they never
  // had anything to collide with.
  //
  // Residual ("skip") edges all share one lane, cleared against every node
  // in the whole view: a block's two residuals should read as one
  // continuous line down the side, not jog inward/outward at the rank where
  // one hands off to the other. They exit straight out the source/target's
  // side, which is only safe because those nodes are always single-node
  // ranks with no siblings to cut across.
  //
  // Every other multi-rank edge (e.g. Llama's V Projection -> Output
  // Projection, which skips the RoPE rank since RoPE only applies to Q/K)
  // gets its own local detour instead, cleared only against the nodes it
  // actually spans — a short detour around one node shouldn't be pushed all
  // the way out past whatever's widest in the whole block. Unlike the
  // residual case, a source/target here typically *does* have siblings
  // (V's are K and Q), so the detour is routed via the ordinary Top/Bottom
  // ports rather than a side exit — see DetourEdge above for why that's
  // what keeps it from cutting across them.
  //
  // Each detour also picks whichever side — left or right of the obstacle —
  // costs less total sideways travel from its own source/target position,
  // rather than always going right: V sits at the *left* edge of its rank,
  // so detouring left around RoPE is both shorter and never has to cross
  // K or Q's column at all, whereas detouring right would (harmlessly, once
  // routed below the row — but there's no reason to prefer the longer path).
  // Detours are then greedily nudged further out, away from whichever side
  // they're already on, wherever two lanes' vertical runs would otherwise
  // overlap in y at the same x.
  const { skipLaneXByEdge, hubLaneXByEdge, detourByEdge, bypassEdgeIds, branchingHubEdgeIds } = useMemo(() => {
    type Span = { lo: number; hi: number };
    const spanOf = (e: ModelEdge): Span | null => {
      const sy = positions.get(e.source)?.y;
      const ty = positions.get(e.target)?.y;
      if (sy == null || ty == null || sy === ty) return null;
      const lo = Math.min(sy, ty);
      const hi = Math.max(sy, ty);
      const hasIntervening = nodeIds.some((id) => {
        if (id === e.source || id === e.target) return false;
        const y = positions.get(id)?.y;
        return y != null && y > lo && y < hi;
      });
      return hasIntervening ? { lo, hi } : null;
    };

    const skipLaneXByEdge = new Map<string, number>();
    const hubLaneXByEdge = new Map<string, number>();
    const placed: { lo: number; hi: number; x: number }[] = [];
    const hubSpans: Span[] = [];

    let viewMaxRight = 0;
    for (const id of nodeIds) {
      const p = positions.get(id);
      if (p) viewMaxRight = Math.max(viewMaxRight, p.x + NOMINAL_NODE_WIDTH / 2);
    }
    const skipLaneX = viewMaxRight + LANE_GAP;
    for (const e of edgeList) {
      if (e.label !== "skip") continue;
      const span = spanOf(e);
      if (!span) continue;
      skipLaneXByEdge.set(e.id, skipLaneX);
      placed.push({ ...span, x: skipLaneX });
    }

    // A single source reaching many rank-skipping targets (Gemma-4's
    // Per-Layer Input Projection feeding all 35 blocks is the case that
    // motivated this) gets the exact same treatment as a residual: one
    // shared lane, cleared against the whole view, exiting straight out the
    // side — safe for the same reason residuals are, since a hub source and
    // every one of its targets are always the sole node in their rank here.
    // Handled *before* the ordinary per-edge detours below so those can see
    // (and steer clear of) whatever side the bus already claimed.
    // Grouped by *raw* rank distance rather than spanOf's stricter
    // "something's actually in the way" test — once a source clears the hub
    // threshold, every one of its targets gets the same bus treatment for
    // consistency's sake, including any that happen to sit on the very next
    // rank with nothing between them (Gemma-4's Per-Layer Input Projection
    // reaching Block 0 itself, right below it, exactly as uniformly as it
    // reaches Block 34) — otherwise that one target would be the sole
    // holdout still merging onto the ordinary top port.
    const spanLoose = (e: ModelEdge): Span | null => {
      const sy = positions.get(e.source)?.y;
      const ty = positions.get(e.target)?.y;
      if (sy == null || ty == null || sy === ty) return null;
      return { lo: Math.min(sy, ty), hi: Math.max(sy, ty) };
    };
    const spanByCandidateEdge = new Map<string, Span>();
    const bySourceCandidate = new Map<string, ModelEdge[]>();
    for (const e of edgeList) {
      if (e.label === "skip") continue;
      const span = spanLoose(e);
      if (!span) continue;
      spanByCandidateEdge.set(e.id, span);
      if (!bySourceCandidate.has(e.source)) bySourceCandidate.set(e.source, []);
      bySourceCandidate.get(e.source)!.push(e);
    }
    const hubLaneX = skipLaneX + LANE_GAP;
    const hubIds = new Set<string>();
    for (const [source, edges] of bySourceCandidate) {
      // Counting *distinct ranks* reached, not raw edge count, is what
      // actually distinguishes a hub from an everyday parallel branch: an
      // attention block's RMSNorm splitting into Q/K/V has 3 edges too, but
      // all 3 land on the very same rank (ordinary siblings side by side,
      // exactly what the numbered-port system below already draws well) —
      // nothing like Per-Layer Input Projection's 35 edges, each landing on
      // its *own* rank down a long chain of blocks. Only the latter shape
      // benefits from bus routing; the former would just be sent on an
      // unnecessary detour to the far right for no visual gain.
      const distinctRanks = new Set(edges.map((e) => positions.get(e.target)?.y));
      if (distinctRanks.size < HUB_MIN_TARGETS) continue;
      hubIds.add(source);
      for (const e of edges) {
        const span = spanByCandidateEdge.get(e.id)!;
        hubLaneXByEdge.set(e.id, hubLaneX);
        placed.push({ ...span, x: hubLaneX });
        hubSpans.push(span);
      }
    }

    const localCandidates = edgeList
      .filter((e) => e.label !== "skip" && !hubLaneXByEdge.has(e.id))
      .map((e) => {
        const span = spanOf(e);
        if (!span) return null;
        const sourceX = positions.get(e.source)?.x ?? 0;
        const targetX = positions.get(e.target)?.x ?? 0;
        let obLeft = Infinity;
        let obRight = -Infinity;
        // Obstacle search includes the source's and target's *own* ranks
        // (>=/<=, not the strictly-between >/< that spanOf itself uses to
        // decide whether a detour is needed at all) because the jog's
        // horizontal run happens right at those ranks' boundary — only
        // `clear` (20px) below the source row and above the target row, not
        // fully outside them. A sibling sitting in that same boundary rank
        // (e.g. Qwen3.5's QKV Projection beside β/Decay Gate Projection,
        // which detour past it to reach the shared gated RMSNorm) still has
        // its own ordinary edge transiting straight down through that exact
        // band, and a real bug — the β/Decay detour's jog crossing right
        // through QKV Projection's own outgoing edge — showed this wasn't
        // being cleared before.
        for (const id of nodeIds) {
          if (id === e.source || id === e.target) continue;
          const p = positions.get(id);
          if (p && p.y >= span.lo && p.y <= span.hi) {
            obLeft = Math.min(obLeft, p.x - NOMINAL_NODE_WIDTH / 2);
            obRight = Math.max(obRight, p.x + NOMINAL_NODE_WIDTH / 2);
          }
        }
        const rightX = obRight + LANE_GAP;
        const leftX = obLeft - LANE_GAP;
        // A bus lane always sits on the right (see hubLaneX/skipLaneX
        // above); anything else whose span overlaps it should default to
        // the left instead of merely nudging away once too close, so it
        // never reads as "part of" the bus in the first place.
        const sharesHubSpan = hubSpans.some((h) => span.lo < h.hi && h.lo < span.hi);
        const costRight = Math.abs(rightX - sourceX) + Math.abs(rightX - targetX);
        const costLeft = Math.abs(leftX - sourceX) + Math.abs(leftX - targetX);
        const x = sharesHubSpan ? leftX : costLeft < costRight ? leftX : rightX;
        return { id: e.id, ...span, x };
      })
      .filter((c): c is { id: string; lo: number; hi: number; x: number } => !!c)
      // Widest span first: a short local detour should never have to dodge
      // a long-spanning one that just happens to be processed first.
      .sort((a, b) => b.hi - b.lo - (a.hi - a.lo));

    const detourByEdge = new Map<string, number>();
    for (const c of localCandidates) {
      let x = c.x;
      const side = x < 0 ? -1 : 1;
      let moved = true;
      while (moved) {
        moved = false;
        for (const p of placed) {
          const overlapsY = c.lo < p.hi && p.lo < c.hi;
          if (overlapsY && Math.abs(x - p.x) < LANE_SEPARATION) {
            x += side * LANE_SEPARATION;
            moved = true;
          }
        }
      }
      placed.push({ lo: c.lo, hi: c.hi, x });
      detourByEdge.set(c.id, x);
    }

    // A hub's own *direct* predecessors (an edge landing on the hub from an
    // adjacent rank, with nothing between them) get decluttered too: a
    // predecessor that *also* reaches past the hub to a later rank
    // (Gemma-4's Token Embedding starts the main residual stream at
    // Transformer Block 0 *and* feeds Per-Layer Input Projection) gets its
    // two edges pinned to opposite numbered ports instead of whatever order
    // they'd otherwise fall in: the bypass — which already detours out to
    // the far side via sharesHubSpan above — keeps that same side's port on
    // its own source, and the hub-bound edge takes the hub's port on the
    // *other* side, so the two stay visually paired with their own routing
    // rather than crossing back over each other right at the source/target.
    const bypassEdgeIds = new Set<string>();
    const branchingHubEdgeIds = new Set<string>();
    for (const hubId of hubIds) {
      const hubPos = positions.get(hubId);
      if (!hubPos) continue;
      const predecessorEdges = edgeList.filter((e) => e.target === hubId && e.source !== hubId && e.label !== "skip");
      for (const e of predecessorEdges) {
        const bypass = edgeList.find(
          (o) => o.source === e.source && o.id !== e.id && o.target !== hubId && o.label !== "skip" && (positions.get(o.target)?.y ?? -Infinity) > hubPos.y
        );
        if (bypass) {
          bypassEdgeIds.add(bypass.id);
          branchingHubEdgeIds.add(e.id);
        }
      }
    }

    return { skipLaneXByEdge, hubLaneXByEdge, detourByEdge, bypassEdgeIds, branchingHubEdgeIds };
  }, [nodeIds, edgeList, positions]);

  // Multiple edges sharing one source (a branch) or one target (a merge)
  // each get their own named Handle, ordered left-to-right to match their
  // sibling's actual x position — so a port on the left side of a node
  // connects to whichever sibling is laid out on the left, minimizing
  // crossings instead of assigning ports arbitrarily. Only residual/skip
  // edges are excluded here — they use their own dedicated right-side
  // handle instead of a numbered port. Detour edges (V Projection's, etc.)
  // stay in this grouping since they use ordinary Top/Bottom ports too.
  const { sourceHandleByEdge, targetHandleByEdge, outputPortsById, inputPortsById } = useMemo(() => {
    const bySource = new Map<string, ModelEdge[]>();
    const byTarget = new Map<string, ModelEdge[]>();
    for (const e of edgeList) {
      if (skipLaneXByEdge.has(e.id) || hubLaneXByEdge.has(e.id)) continue;
      if (!bySource.has(e.source)) bySource.set(e.source, []);
      bySource.get(e.source)!.push(e);
      if (!byTarget.has(e.target)) byTarget.set(e.target, []);
      byTarget.get(e.target)!.push(e);
    }

    const sourceHandleByEdge = new Map<string, string>();
    const outputPortsById = new Map<string, number>();
    for (const [id, edges] of bySource) {
      outputPortsById.set(id, edges.length);
      if (edges.length <= 1) continue;
      // A source with both a bypass edge and a hub-bound edge (Token
      // Embedding here) always gets the bypass on its leftmost port and the
      // hub-bound edge to its right, regardless of where their targets
      // happen to sit — both already detour to a specific side on their own
      // (the bypass via sharesHubSpan, the hub-bound edge by simply being
      // short), so the port order should agree with that instead of an
      // arbitrary x-position tie.
      const sourcePriority = (e: ModelEdge) => (bypassEdgeIds.has(e.id) ? -1 : branchingHubEdgeIds.has(e.id) ? 1 : 0);
      const sorted = [...edges].sort((a, b) => sourcePriority(a) - sourcePriority(b) || (positions.get(a.target)?.x ?? 0) - (positions.get(b.target)?.x ?? 0));
      sorted.forEach((e, i) => sourceHandleByEdge.set(e.id, `src-${i}`));
    }

    const targetHandleByEdge = new Map<string, string>();
    const inputPortsById = new Map<string, number>();
    for (const [id, edges] of byTarget) {
      inputPortsById.set(id, edges.length);
      if (edges.length <= 1) continue;
      // Mirrors the source-side bias just above: the incoming edge from a
      // branching predecessor (one that also bypasses this hub — Token
      // Embedding, here) always lands on the hub's leftmost port, so it
      // reads as a matched pair with that predecessor's own leftmost
      // (bypass) port rather than landing wherever raw x-position puts it.
      const targetPriority = (e: ModelEdge) => (branchingHubEdgeIds.has(e.id) ? -1 : 0);
      const sorted = [...edges].sort((a, b) => targetPriority(a) - targetPriority(b) || (positions.get(a.source)?.x ?? 0) - (positions.get(b.source)?.x ?? 0));
      sorted.forEach((e, i) => targetHandleByEdge.set(e.id, `tgt-${i}`));
    }

    return { sourceHandleByEdge, targetHandleByEdge, outputPortsById, inputPortsById };
  }, [edgeList, positions, skipLaneXByEdge, hubLaneXByEdge, bypassEdgeIds, branchingHubEdgeIds]);

  // Selecting a node highlights its whole computational neighborhood
  // (every ancestor and descendant reachable through this view's edges)
  // and fades everything else, instead of the selection ring being the
  // only visible feedback.
  const relatedIds = useMemo(() => {
    if (!selectedId || !nodeIds.includes(selectedId)) return null;
    const related = new Set<string>([selectedId]);
    let frontier = [selectedId];
    while (frontier.length) {
      const next: string[] = [];
      for (const cur of frontier) {
        for (const e of edgeList) {
          if (e.target === cur && !related.has(e.source)) {
            related.add(e.source);
            next.push(e.source);
          }
        }
      }
      frontier = next;
    }
    frontier = [selectedId];
    while (frontier.length) {
      const next: string[] = [];
      for (const cur of frontier) {
        for (const e of edgeList) {
          if (e.source === cur && !related.has(e.target)) {
            related.add(e.target);
            next.push(e.target);
          }
        }
      }
      frontier = next;
    }
    return related;
  }, [selectedId, nodeIds, edgeList]);

  const rfNodes: RFNode<IRNodeData>[] = useMemo(
    () =>
      nodeIds.map((id) => {
        const pos = positions.get(id) ?? { x: 0, y: 0 };
        const outputPorts = outputPortsById.get(id) ?? 1;
        const inputPorts = inputPortsById.get(id) ?? 1;
        const dimmed = relatedIds !== null && !relatedIds.has(id);

        if (id === ELLIPSIS) {
          return {
            id,
            type: "ir",
            position: pos,
            draggable: false,
            selectable: false,
            data: { label: "⋯", sublabel: "more blocks (collapsed)", color: "#94a3b8", selected: false, dimmed, inputPorts, outputPorts },
          };
        }
        if (id === BLOCK_INPUT) {
          return {
            id,
            type: "ir",
            position: pos,
            draggable: false,
            selectable: false,
            data: { label: "Block Input", sublabel: "from outside this block", color: "#94a3b8", selected: false, dimmed, inputPorts, outputPorts },
          };
        }

        const stackInfo = stackGroups.get(id);
        if (stackInfo) {
          const info = componentRegistry[stackInfo.type];
          const totalParams = stackInfo.memberIds.reduce((sum, mid) => {
            const n = model.nodes[mid];
            return sum + n.parameters.reduce((a, p) => a + p.logicalShape.reduce((x, y) => x * y, 1), 0);
          }, 0);
          const categoryText = categoryLabel[info.category] || info.label;
          const sublabel = totalParams > 0 ? `${categoryText} · ${formatCount(totalParams)} params total` : categoryText;
          const first = model.nodes[stackInfo.memberIds[0]];
          const last = model.nodes[stackInfo.memberIds[stackInfo.memberIds.length - 1]];
          let dims: string | undefined;
          if (info.category === "linear") {
            const inDims = first.inputs[0]?.dims;
            const outDims = last.outputs[0]?.dims;
            if (inDims?.length && outDims?.length) dims = `${inDims[inDims.length - 1]} → ${outDims[outDims.length - 1]}`;
          }
          return {
            id,
            type: "ir",
            position: pos,
            draggable: false,
            selectable: false,
            data: {
              label: info.label,
              sublabel,
              glyph: categoryGlyph[info.category] || undefined,
              dims,
              color: info.color,
              selected: false,
              dimmed,
              inputPorts,
              outputPorts,
              stackCount: stackInfo.memberIds.length,
            },
          };
        }

        const node = model.nodes[id];
        const info = componentRegistry[node.type];
        const paramCount = node.parameters.reduce((a, p) => a + p.logicalShape.reduce((x, y) => x * y, 1), 0);
        const categoryText = categoryLabel[info.category] || info.label;
        const sublabel = paramCount > 0 ? `${categoryText} · ${formatCount(paramCount)} params` : categoryText;

        // A "16 -> 32" shape summary is only meaningful where the last
        // dimension actually changes (projections, embeddings) — showing
        // it on every node (where it'd usually just read "32 -> 32") would
        // be noise, not signal.
        let dims: string | undefined;
        if (info.category === "linear") {
          const inDims = node.inputs[0]?.dims;
          const outDims = node.outputs[0]?.dims;
          if (inDims?.length && outDims?.length) dims = `${inDims[inDims.length - 1]} → ${outDims[outDims.length - 1]}`;
        }

        return {
          id,
          type: "ir",
          position: pos,
          draggable: false,
          data: {
            node,
            label: node.name,
            sublabel,
            glyph: categoryGlyph[info.category] || undefined,
            dims,
            color: info.color,
            selected: id === selectedId,
            dimmed,
            expandable: node.type === "transformer_block",
            inputPorts,
            outputPorts,
          },
        };
      }),
    [nodeIds, positions, model, selectedId, relatedIds, outputPortsById, inputPortsById, stackGroups]
  );

  const junctionNodes: RFNode[] = useMemo(() => {
    const junctions: RFNode[] = [];
    for (const [id, count] of outputPortsById) {
      if (count <= 1) continue;
      const pos = positions.get(id);
      if (!pos) continue;
      junctions.push({
        id: `junction-out-${id}`,
        type: "junction",
        position: { x: pos.x + NOMINAL_NODE_WIDTH / 2 - 4, y: pos.y + 96 },
        draggable: false,
        selectable: false,
        data: {},
      });
    }
    for (const [id, count] of inputPortsById) {
      if (count <= 1) continue;
      const pos = positions.get(id);
      if (!pos) continue;
      junctions.push({
        id: `junction-in-${id}`,
        type: "junction",
        position: { x: pos.x + NOMINAL_NODE_WIDTH / 2 - 4, y: pos.y - 24 },
        draggable: false,
        selectable: false,
        data: {},
      });
    }
    return junctions;
  }, [outputPortsById, inputPortsById, positions]);

  // A container (e.g. "Attention") selected via the model tree has no node
  // of its own in this view — only its leaf children do — so there'd
  // otherwise be no visual feedback for that selection at all. When the
  // selection is such a container, frame whichever of its leaf descendants
  // are actually visible here.
  const scopeBoxNode: RFNode<ScopeBoxData> | null = useMemo(() => {
    if (!selectedId || nodeIds.includes(selectedId)) return null;
    const container = model.nodes[selectedId];
    if (!container || container.children.length === 0) return null;
    const memberIds = getLeafDescendants(model, selectedId).filter((id) => nodeIds.includes(id));
    if (memberIds.length === 0) return null;
    const memberSet = new Set(memberIds);

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const id of memberIds) {
      const p = positions.get(id);
      if (!p) continue;
      // `p.x`/`p.y` are a node's top-left corner (React Flow's own
      // convention — confirmed against the actual rendered position, not
      // assumed), so the right/bottom edge is the position plus the full
      // nominal size, not half of it.
      minX = Math.min(minX, p.x);
      maxX = Math.max(maxX, p.x + NOMINAL_NODE_WIDTH);
      minY = Math.min(minY, p.y);
      maxY = Math.max(maxY, p.y + NOMINAL_NODE_HEIGHT);
    }
    if (minX === Infinity) return null;

    // An edge routed through a side lane (e.g. V Projection -> Output
    // Projection detouring around a Q/K-Norm rank) can bow out further than
    // either endpoint's own node — if both endpoints belong to this group,
    // that detour is purely internal to it and the frame should widen to
    // keep it inside rather than let it poke out past the border.
    for (const e of edgeList) {
      if (!memberSet.has(e.source) || !memberSet.has(e.target)) continue;
      const laneX = skipLaneXByEdge.get(e.id) ?? hubLaneXByEdge.get(e.id) ?? detourByEdge.get(e.id);
      if (laneX === undefined) continue;
      minX = Math.min(minX, laneX);
      maxX = Math.max(maxX, laneX);
    }

    return {
      id: `scope-${selectedId}`,
      type: "scope",
      position: { x: minX - SCOPE_PADDING, y: minY - SCOPE_PADDING },
      style: { width: maxX - minX + SCOPE_PADDING * 2, height: maxY - minY + SCOPE_PADDING * 2 },
      draggable: false,
      selectable: false,
      zIndex: -1,
      data: { label: container.name, color: componentRegistry[container.type]?.color ?? "#6b7280" },
    };
  }, [selectedId, nodeIds, model, positions, edgeList, skipLaneXByEdge, hubLaneXByEdge, detourByEdge]);

  const rfEdges: RFEdge[] = useMemo(
    () =>
      edgeList.map((e) => {
        const isSkip = e.label === "skip";
        const dimmed = relatedIds !== null && (!relatedIds.has(e.source) || !relatedIds.has(e.target));
        // Tensor shape is only worth showing when the user has opted in
        // AND is actually looking at this edge (hovered) or at one of its
        // endpoints (selected) — permanently labeling every edge with its
        // shape would be exactly the clutter the doc warns against.
        const showShape = showTensorShapes && (e.id === hoveredEdgeId || e.source === selectedId || e.target === selectedId);
        const shapeDims = showShape ? model.nodes[e.source]?.outputs[0]?.dims : undefined;
        const skipLaneX = skipLaneXByEdge.get(e.id);
        const hubLaneX = hubLaneXByEdge.get(e.id);
        const isHub = hubLaneX !== undefined;
        const detourX = detourByEdge.get(e.id);
        const laneX = skipLaneX ?? hubLaneX ?? detourX;
        return {
          id: e.id,
          source: e.source,
          target: e.target,
          sourceHandle: skipLaneX !== undefined || isHub ? "lane-out" : sourceHandleByEdge.get(e.id),
          targetHandle: skipLaneX !== undefined || isHub ? "lane-in" : targetHandleByEdge.get(e.id),
          type: skipLaneX !== undefined || isHub ? "lane" : detourX !== undefined ? "detour" : "smoothstep",
          data: laneX !== undefined ? { laneX } : undefined,
          label: shapeDims?.length ? `[${shapeDims.join(", ")}]` : undefined,
          markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14, color: isSkip ? SKIP_EDGE_COLOR : isHub ? HUB_EDGE_COLOR : EDGE_COLOR },
          style: {
            stroke: isSkip ? SKIP_EDGE_COLOR : isHub ? HUB_EDGE_COLOR : EDGE_COLOR,
            strokeWidth: isSkip ? 2 : 1.5,
            // Dash length is defined in flow-space units, which the
            // current zoom then scales down further — at the zoom level
            // "Maximize graph view" lands on for a tall block (~0.5x), a
            // "9 6" dash shrinks to a couple of screen pixels and
            // Chromium's rasterizer blurs it into what reads as a solid
            // line. A chunkier pattern stays legibly dashed even at that
            // zoom (verified down to 0.5x; non-scaling-stroke was tried
            // first but only cancels SVG-native transforms, not the CSS
            // scale React Flow applies to the whole canvas, so it had no
            // effect here).
            ...(isSkip ? { strokeDasharray: "16 10" } : {}),
          },
          className: "graph-edge" + (dimmed ? " graph-edge-dimmed" : ""),
        };
      }),
    [edgeList, hoveredEdgeId, selectedId, model, relatedIds, sourceHandleByEdge, targetHandleByEdge, showTensorShapes, skipLaneXByEdge, hubLaneXByEdge, detourByEdge]
  );

  const handleNodeClick = useCallback(
    (_: unknown, n: RFNode) => {
      if (n.id === ELLIPSIS || n.id === BLOCK_INPUT || n.type === "junction" || n.type === "scope" || n.id.startsWith(STACK_PREFIX)) return;
      onSelect(n.id);
    },
    [onSelect]
  );

  const handleNodeDoubleClick = useCallback(
    (_: unknown, n: RFNode) => {
      if (n.id.startsWith(STACK_PREFIX)) {
        setStackRepeats(false);
        return;
      }
      if (n.id === ELLIPSIS || n.id === BLOCK_INPUT || n.type === "junction" || n.type === "scope") return;
      const node = model.nodes[n.id];
      if (node?.type === "transformer_block") onEnterBlock(n.id);
    },
    [model, onEnterBlock, setStackRepeats]
  );

  const viewKey = view.kind === "architecture" ? "arch" : `block:${view.blockId}`;

  // A node selected elsewhere (most commonly the model tree, which can
  // force-expand and highlight a node the graph never scrolled to) can end
  // up entirely outside the current viewport — the highlight is real but
  // invisible. Pan (never zoom) to bring it fully into view whenever that's
  // the case; skip it when the node's already visible, so clicking a node
  // directly in the graph doesn't yank the camera out from under the
  // user's cursor for no reason.
  useEffect(() => {
    if (!selectedId || !nodeIds.includes(selectedId)) return;
    const instance = rfInstanceRef.current;
    const container = containerRef.current;
    if (!instance || !container) return;

    // Right after a view switch (entering/exiting a block) this node may
    // not be measured yet on the same tick it mounts — wait a frame.
    const raf = requestAnimationFrame(() => {
      const node = instance.getNode(selectedId);
      if (!node) return;
      const width = node.width ?? 220;
      const height = node.height ?? 70;
      const { x: vx, y: vy, zoom } = instance.getViewport();
      const rect = container.getBoundingClientRect();
      const screenX = node.position.x * zoom + vx;
      const screenY = node.position.y * zoom + vy;
      const margin = 24; // keep the node clear of pane edges/controls, not just barely on-screen
      const fullyVisible =
        screenX >= margin && screenY >= margin && screenX + width * zoom <= rect.width - margin && screenY + height * zoom <= rect.height - margin;
      if (fullyVisible) return;

      instance.setCenter(node.position.x + width / 2, node.position.y + height / 2, { zoom, duration: 400 });
    });
    return () => cancelAnimationFrame(raf);
  }, [selectedId, nodeIds]);

  // A couple of the doc's requested shortcuts: Esc backs out of a focused
  // selection (matching the same "Esc = back off" convention the settings
  // and load-model popovers already use), 0 re-fits the view without
  // reaching for the on-canvas button. Ignored while typing anywhere else
  // in the app so pressing "0" in a prompt or search box isn't hijacked.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) return;
      if (e.key === "Escape") onSelect(null);
      else if (e.key === "0") rfInstanceRef.current?.fitView({ padding: 0.2 });
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onSelect]);

  // Exports exactly what's currently drawn — every node's real position,
  // not just the on-screen viewport — by framing a fresh camera around all
  // node bounds (getNodesBounds + getViewportForBounds, the pattern React
  // Flow's own docs use for image export) and rendering only
  // `.react-flow__viewport` (nodes/edges), leaving out Controls/MiniMap.
  // html-to-image clones the DOM before applying that camera, so the live
  // graph on screen is never touched.
  const exportImage = useCallback(() => {
    const instance = rfInstanceRef.current;
    const viewportEl = containerRef.current?.querySelector<HTMLElement>(".react-flow__viewport");
    if (!instance || !viewportEl || exportingImage) return;

    const nodesBounds = getNodesBounds(instance.getNodes());
    const aspect = nodesBounds.width / Math.max(1, nodesBounds.height);
    const MAX_DIMENSION = 2400;
    const MIN_DIMENSION = 800;
    let imageWidth = Math.min(MAX_DIMENSION, Math.max(MIN_DIMENSION, Math.round(nodesBounds.width)));
    let imageHeight = Math.round(imageWidth / Math.max(aspect, 0.01));
    if (imageHeight > MAX_DIMENSION) {
      imageHeight = MAX_DIMENSION;
      imageWidth = Math.round(imageHeight * aspect);
    }
    const viewport = getViewportForBounds(nodesBounds, imageWidth, imageHeight, 0.1, 4, 0.1);
    const backgroundColor = getComputedStyle(document.documentElement).getPropertyValue("--bg-canvas").trim() || "#0b0d13";
    const viewLabel = view.kind === "block" ? (model.nodes[view.blockId]?.name ?? view.blockId) : "architecture";
    const filename = `${model.name}-${viewLabel}`.trim().replace(/[\\/:*?"<>|]+/g, "-").replace(/\s+/g, "_") + ".png";

    // A React re-render triggered here — even one that touches only the
    // sibling Controls button, nothing inside `.react-flow__viewport` —
    // reliably corrupts html-to-image's in-flight clone into a blank
    // image (root-caused empirically; html-to-image cloning races the
    // commit). Deferring the busy-state update past the current
    // microtask queue via setTimeout keeps the capture's DOM read clear
    // of any React commit.
    setTimeout(() => setExportingImage(true), 0);
    toPng(viewportEl, {
      backgroundColor,
      width: imageWidth,
      height: imageHeight,
      style: {
        width: `${imageWidth}px`,
        height: `${imageHeight}px`,
        transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.zoom})`,
      },
    })
      .then((dataUrl) => {
        const a = document.createElement("a");
        a.href = dataUrl;
        a.download = filename;
        a.click();
      })
      .catch((err) => {
        console.error("Failed to export graph image:", err);
      })
      .finally(() => setExportingImage(false));
  }, [model, view, exportingImage]);

  return (
    <div className="architecture-graph" ref={containerRef}>
      {view.kind === "block" && (
        <div className="graph-breadcrumb">
          <button onClick={onExitBlock}>← Architecture</button>
          <span>{model.nodes[view.blockId].name}</span>
        </div>
      )}
      <ReactFlow
        key={viewKey}
        nodes={[...(scopeBoxNode ? [scopeBoxNode] : []), ...rfNodes, ...junctionNodes] as RFNode[]}
        edges={rfEdges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodeClick={handleNodeClick}
        onNodeDoubleClick={handleNodeDoubleClick}
        onEdgeMouseEnter={(_, edge) => setHoveredEdgeId(edge.id)}
        onEdgeMouseLeave={() => setHoveredEdgeId(null)}
        onInit={(instance) => {
          rfInstanceRef.current = instance;
        }}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        proOptions={{ hideAttribution: true }}
        nodesConnectable={false}
        zoomOnDoubleClick={false}
        elementsSelectable
      >
        <Background />
        <MiniMap pannable zoomable nodeColor={(n) => (n.data as Partial<IRNodeData> | undefined)?.color ?? "#6b7280"} maskColor="rgba(15, 17, 23, 0.6)" />
        <Controls showInteractive={false}>
          <ControlButton onClick={onToggleMaxFrame} title={isMaxFrame ? t("graph.restorePanels") : t("graph.maximizeGraph")}>
            <span className="control-icon">{isMaxFrame ? "⤡" : "⤢"}</span>
          </ControlButton>
          <ControlButton
            className="control-button-gap"
            onClick={() => setShowTensorShapes((v) => !v)}
            title={showTensorShapes ? t("graph.hideTensorShapes") : t("graph.showTensorShapes")}
          >
            <span className={"control-icon" + (showTensorShapes ? " active" : "")}>
              <ShapeIcon active={showTensorShapes} />
            </span>
          </ControlButton>
          <ControlButton
            onClick={() => setStackRepeats((v) => !v)}
            title={stackRepeats ? t("graph.unstackRepeats") : t("graph.stackRepeats")}
          >
            <span className={"control-icon" + (stackRepeats ? " active" : "")}>
              <StackIcon active={stackRepeats} />
            </span>
          </ControlButton>
          <ControlButton className="control-button-gap" onClick={exportImage} disabled={exportingImage} title={t("graph.exportImage")}>
            <span className="control-icon">
              <ExportIcon busy={exportingImage} />
            </span>
          </ControlButton>
        </Controls>
      </ReactFlow>
    </div>
  );
}
