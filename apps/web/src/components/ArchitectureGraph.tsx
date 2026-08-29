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
import type { EdgeKind, Model, ModelNode } from "@tensorium/model-ir";
import { categoryGlyph, categoryLabel, componentRegistry } from "../registry.js";
import { computeElkLayout, type ElkLayoutResult, type NodeSize } from "../elkLayout.js";
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
  /** Mirrors the on-canvas zoom badge out to the caller — e.g. so the app-wide status footer can show the current zoom without duplicating React Flow's own viewport tracking. */
  onZoomChange?: (percent: number) => void;
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
  /** Set when this node stands in for a collapsed run of `stackCount` identical, sequentially-connected nodes — rendered as a small deck of cards instead of a single box. */
  stackCount?: number;
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
      {/* A single top/bottom handle per node, not one per sibling edge: ELK
          now computes each edge's real route (including where it leaves the
          node) directly, so the handle only needs to exist for React Flow's
          own bookkeeping — the rendered path comes from that route's own
          points (see ElkRoutedEdge), not from this handle's screen
          position. */}
      <Handle type="target" position={Position.Top} style={{ opacity: 0 }} />
      <div className="ir-node-label">
        {data.glyph && <span className="ir-node-glyph">{data.glyph}</span>}
        {data.label}
        {stacked && <span className="ir-node-stack-badge">× {data.stackCount}</span>}
      </div>
      <div className="ir-node-sub">{data.sublabel}</div>
      {data.dims && <div className="ir-node-dims">{data.dims}</div>}
      {data.expandable && <div className="ir-node-hint">double-click to expand</div>}
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0 }} />
      {/* Dedicated right-side ports for residual edges routed through the
          side lane (see ResidualLaneEdge below) — kept separate from the
          top/bottom main-flow handles so a residual connection never
          competes with a sibling data-flow edge for the same handle. */}
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
 * Residual edges: Block Input and Residual Add are always single-node ranks
 * with nothing else beside them, so exiting straight out their right side
 * and down a vertical lane never has anything to cross — this is the one
 * routing case that never needed a general-purpose layout engine, so it's
 * kept exactly as before rather than routed through ELK too.
 */
function ResidualLaneEdge({ id, sourceX, sourceY, targetX, targetY, markerEnd, style, data }: EdgeProps<LaneEdgeData>) {
  const laneX = data?.laneX ?? Math.max(sourceX, targetX) + 60;
  const path = `M ${sourceX},${sourceY} L ${laneX},${sourceY} L ${laneX},${targetY} L ${targetX},${targetY}`;
  return <BaseEdge id={id} path={path} markerEnd={markerEnd} style={style} />;
}

interface RouteEdgeData {
  points: { x: number; y: number }[];
}

/**
 * Every non-residual edge: drawn through ELK's own computed route (straight
 * line for an adjacent-rank edge, orthogonal bends for anything ELK had to
 * route around another node) instead of relying on React Flow's handle
 * positions at all — `data.points` is already in the same absolute
 * flow-space coordinates as node positions, straight from
 * `computeElkLayout`. This is what replaced this app's own hand-rolled
 * obstacle-avoidance detour/hub-lane system: ELK's layered algorithm
 * inserts the dummy nodes a long or crossing edge needs to route around
 * automatically, which is the general case that system was patching around
 * one bug at a time.
 */
function ElkRoutedEdge({ id, data, markerEnd, style }: EdgeProps<RouteEdgeData>) {
  const points = data?.points;
  if (!points || points.length < 2) return null;
  const path = "M " + points.map((p) => `${p.x},${p.y}`).join(" L ");
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
const edgeTypes = { lane: ResidualLaneEdge, routed: ElkRoutedEdge };

/** Style-by-kind, replacing the old isSkip/isHub ad hoc branching now that every edge carries a real `kind`. */
const EDGE_STYLE: Record<EdgeKind, { stroke: string; dash?: string }> = {
  data: { stroke: "#94a3b8" },
  branch: { stroke: "#94a3b8" },
  gate: { stroke: "#22d3ee" },
  residual: { stroke: "#64748b", dash: "16 10" },
};

/** Used only as the ELK/render fallback before a node's real DOM size has been measured (see the layout effect below) — never shown to the user, since the graph stays hidden until real ELK positions land. */
const FALLBACK_NODE_SIZE: NodeSize = { width: 160, height: 76 };
/** Breathing room between a scope box's edge and the nodes it encloses. */
const SCOPE_PADDING = 26;
/** Clearance between the widest thing in the view (a node or an ELK-routed edge's own bend) and the shared residual lane. */
const LANE_GAP = 90;

function resolveKind(e: { kind?: EdgeKind; label?: string }): EdgeKind {
  return e.kind ?? (e.label === "skip" ? "residual" : "data");
}

export function ArchitectureGraph({ model, view, selectedId, onSelect, onEnterBlock, onExitBlock, isMaxFrame, onToggleMaxFrame, onZoomChange }: Props) {
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
  // Starts at 100 (React Flow's own default zoom) rather than reading the
  // instance up front — it doesn't exist yet on first render — and is kept
  // in sync via onMove below, which React Flow fires for every pan/zoom
  // interaction (scroll, pinch, the Controls +/- buttons, and fitView's own
  // programmatic moves alike).
  const [zoomPercent, setZoomPercent] = useState(100);
  // Fires for every change regardless of which handler (onInit's first
  // read, or onMove's ongoing pan/zoom updates below) caused it, rather
  // than duplicating the callback at each call site.
  useEffect(() => onZoomChange?.(zoomPercent), [zoomPercent, onZoomChange]);

  const { nodeIds: rawNodeIds, edgeList: rawEdgeList } = useMemo(() => {
    if (view.kind === "architecture") {
      const g = buildLevel1Graph(model, false);
      return { nodeIds: g.nodeIds, edgeList: g.edges };
    }
    const g = buildLevel2Graph(model, view.blockId);
    return { nodeIds: g.nodeIds, edgeList: g.edges };
  }, [model, view]);

  // Collapsing happens once, right after the view's raw graph is built, so
  // every downstream step (layout, routing, the scope box) just sees a
  // smaller graph and needs no awareness that stacking exists.
  const { nodeIds, edgeList, stackGroups } = useMemo(() => {
    if (!stackRepeats) return { nodeIds: rawNodeIds, edgeList: rawEdgeList, stackGroups: new Map<string, StackGroup>() };
    const collapsed = collapseRepeatedChains(model, rawNodeIds, rawEdgeList);
    return { nodeIds: collapsed.nodeIds, edgeList: collapsed.edges, stackGroups: collapsed.stacks };
  }, [stackRepeats, rawNodeIds, rawEdgeList, model]);

  // ELK needs each node's *real* rendered size before it can lay anything
  // out (a size assumed up front and corrected later is exactly the "layout
  // first, then discover the DOM is 30px taller" trap) — so this runs a
  // two-pass cycle every time the visible graph changes: render once with
  // nodes parked at (0,0) so React Flow can measure their real DOM
  // width/height (already happening via its own ResizeObserver — the
  // existing "pan to selected node" effect below already reads
  // `instance.getNode(id).width` the same way), then hand ELK those real
  // sizes, then reveal the graph once positions land. The container stays
  // hidden for that first pass so it never flashes as a pile of
  // zero-positioned boxes.
  const [elkResult, setElkResult] = useState<ElkLayoutResult | null>(null);
  useEffect(() => {
    let cancelled = false;
    setElkResult(null);

    function tryLayout() {
      if (cancelled) return;
      const instance = rfInstanceRef.current;
      if (!instance) {
        requestAnimationFrame(tryLayout);
        return;
      }
      const rendered = instance.getNodes();
      const sizes = new Map<string, NodeSize>();
      for (const id of nodeIds) {
        const n = rendered.find((x) => x.id === id);
        if (!n || n.width == null || n.height == null) {
          requestAnimationFrame(tryLayout);
          return;
        }
        sizes.set(id, { width: n.width, height: n.height });
      }
      computeElkLayout(nodeIds, edgeList, sizes, FALLBACK_NODE_SIZE).then((result) => {
        if (!cancelled) setElkResult(result);
      });
    }
    const raf = requestAnimationFrame(tryLayout);
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
    };
  }, [nodeIds, edgeList]);

  // The default view is a real 100% zoom, not React Flow's own fit-to-view
  // scale (which shrinks to whatever fits the whole graph — often well
  // under 100% for a multi-block architecture) — centered horizontally on
  // the graph and aligned near its top edge, the natural place to start
  // reading a top-to-bottom diagram. This only runs once positions are
  // real (every node sits at (0,0) before that, so there's nothing
  // meaningful to center on yet); the on-canvas fit-to-screen button and
  // the "0" shortcut are still there for zooming out to see everything at
  // once.
  useEffect(() => {
    const instance = rfInstanceRef.current;
    const container = containerRef.current;
    if (!elkResult || !instance || !container) return;
    const bounds = getNodesBounds(instance.getNodes());
    const rect = container.getBoundingClientRect();
    const topMargin = 60;
    instance.setViewport({ x: rect.width / 2 - (bounds.x + bounds.width / 2), y: topMargin - bounds.y, zoom: 1 });
  }, [elkResult]);

  const positions = elkResult?.positions ?? new Map<string, { x: number; y: number; width: number; height: number }>();

  // The one routing decision left outside ELK: every residual edge in a
  // view shares a single lane, cleared against every node (and every
  // ELK-routed edge's own bends) in the whole view — a block's two
  // residuals should read as one continuous line down the side, not jog
  // inward/outward at the rank where one hands off to the other.
  const skipLaneX = useMemo(() => {
    const hasResidual = edgeList.some((e) => resolveKind(e) === "residual");
    if (!hasResidual || !elkResult) return null;
    let maxRight = 0;
    for (const id of nodeIds) {
      const p = positions.get(id);
      if (p) maxRight = Math.max(maxRight, p.x + p.width);
    }
    for (const route of elkResult.routes.values()) {
      for (const pt of route.points) maxRight = Math.max(maxRight, pt.x);
    }
    return maxRight + LANE_GAP;
  }, [edgeList, nodeIds, positions, elkResult]);

  // Junction dots only need to know *whether* a node fans out/in (not which
  // port order — ELK already decided that), counted straight from the
  // visible edges rather than any handle-assignment bookkeeping.
  const { outputCounts, inputCounts } = useMemo(() => {
    const outputCounts = new Map<string, number>();
    const inputCounts = new Map<string, number>();
    for (const e of edgeList) {
      if (resolveKind(e) === "residual") continue;
      outputCounts.set(e.source, (outputCounts.get(e.source) ?? 0) + 1);
      inputCounts.set(e.target, (inputCounts.get(e.target) ?? 0) + 1);
    }
    return { outputCounts, inputCounts };
  }, [edgeList]);

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
        const dimmed = relatedIds !== null && !relatedIds.has(id);

        if (id === ELLIPSIS) {
          return {
            id,
            type: "ir",
            position: { x: pos.x, y: pos.y },
            draggable: false,
            selectable: false,
            data: { label: "⋯", sublabel: "more blocks (collapsed)", color: "#94a3b8", selected: false, dimmed },
          };
        }
        if (id === BLOCK_INPUT) {
          return {
            id,
            type: "ir",
            position: { x: pos.x, y: pos.y },
            draggable: false,
            selectable: false,
            data: { label: "Block Input", sublabel: "from outside this block", color: "#94a3b8", selected: false, dimmed },
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
            position: { x: pos.x, y: pos.y },
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
          position: { x: pos.x, y: pos.y },
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
          },
        };
      }),
    [nodeIds, positions, model, selectedId, relatedIds, stackGroups]
  );

  const junctionNodes: RFNode[] = useMemo(() => {
    const junctions: RFNode[] = [];
    for (const [id, count] of outputCounts) {
      if (count <= 1) continue;
      const pos = positions.get(id);
      if (!pos) continue;
      junctions.push({
        id: `junction-out-${id}`,
        type: "junction",
        position: { x: pos.x + pos.width / 2 - 4, y: pos.y + pos.height + 20 },
        draggable: false,
        selectable: false,
        data: {},
      });
    }
    for (const [id, count] of inputCounts) {
      if (count <= 1) continue;
      const pos = positions.get(id);
      if (!pos) continue;
      junctions.push({
        id: `junction-in-${id}`,
        type: "junction",
        position: { x: pos.x + pos.width / 2 - 4, y: pos.y - 24 },
        draggable: false,
        selectable: false,
        data: {},
      });
    }
    return junctions;
  }, [outputCounts, inputCounts, positions]);

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
      minX = Math.min(minX, p.x);
      maxX = Math.max(maxX, p.x + p.width);
      minY = Math.min(minY, p.y);
      maxY = Math.max(maxY, p.y + p.height);
    }
    if (minX === Infinity) return null;

    // An ELK-routed edge can bow out further than either endpoint's own
    // node — if both endpoints belong to this group, that route is purely
    // internal to it and the frame should widen to keep it inside rather
    // than let it poke out past the border. Same for the shared residual
    // lane, when both its endpoints are inside the group too.
    if (elkResult) {
      for (const e of edgeList) {
        if (!memberSet.has(e.source) || !memberSet.has(e.target)) continue;
        if (resolveKind(e) === "residual") {
          if (skipLaneX != null) maxX = Math.max(maxX, skipLaneX);
          continue;
        }
        const route = elkResult.routes.get(e.id);
        if (!route) continue;
        for (const pt of route.points) {
          minX = Math.min(minX, pt.x);
          maxX = Math.max(maxX, pt.x);
        }
      }
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
  }, [selectedId, nodeIds, model, positions, edgeList, elkResult, skipLaneX]);

  // Memoized rather than a fresh `[...a, ...b, ...c]` spread inline in the
  // JSX below: React Flow is a controlled component that re-derives its own
  // internal node store from whatever array identity this prop has, and
  // this component already re-renders on every zoom/pan tick (zoomPercent
  // changes on every onMove below) — a brand-new array on every one of
  // those re-renders, even with identical contents, is unnecessary Flow-side
  // churn this avoids.
  const allNodes: RFNode[] = useMemo(
    () => [...(scopeBoxNode ? [scopeBoxNode] : []), ...rfNodes, ...junctionNodes] as RFNode[],
    [scopeBoxNode, rfNodes, junctionNodes]
  );

  const rfEdges: RFEdge[] = useMemo(
    () =>
      edgeList.map((e) => {
        const kind = resolveKind(e);
        const isResidual = kind === "residual";
        const dimmed = relatedIds !== null && (!relatedIds.has(e.source) || !relatedIds.has(e.target));
        // Tensor shape is only worth showing when the user has opted in
        // AND is actually looking at this edge (hovered) or at one of its
        // endpoints (selected) — permanently labeling every edge with its
        // shape would be exactly the clutter the doc warns against.
        const showShape = showTensorShapes && (e.id === hoveredEdgeId || e.source === selectedId || e.target === selectedId);
        const shapeDims = showShape ? model.nodes[e.source]?.outputs[0]?.dims : undefined;
        const edgeStyle = EDGE_STYLE[kind];
        return {
          id: e.id,
          source: e.source,
          target: e.target,
          sourceHandle: isResidual ? "lane-out" : undefined,
          targetHandle: isResidual ? "lane-in" : undefined,
          type: isResidual ? "lane" : "routed",
          data: isResidual ? { laneX: skipLaneX ?? undefined } : { points: elkResult?.routes.get(e.id)?.points ?? [] },
          label: shapeDims?.length ? `[${shapeDims.join(", ")}]` : undefined,
          markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14, color: edgeStyle.stroke },
          style: {
            stroke: edgeStyle.stroke,
            strokeWidth: isResidual ? 2 : 1.5,
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
            ...(edgeStyle.dash ? { strokeDasharray: edgeStyle.dash } : {}),
          },
          className: "graph-edge" + (dimmed ? " graph-edge-dimmed" : ""),
        };
      }),
    [edgeList, hoveredEdgeId, selectedId, model, relatedIds, showTensorShapes, skipLaneX, elkResult]
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
      {/* Hidden (not unmounted) until ELK's real layout lands — the nodes
          underneath still render and measure themselves at (0,0) during
          that first pass, which is what the layout effect above is
          waiting on; hiding it just keeps that pass from flashing on
          screen as a pile of stacked boxes. */}
      <div style={{ visibility: elkResult ? "visible" : "hidden", width: "100%", height: "100%" }}>
        <ReactFlow
          key={viewKey}
          nodes={allNodes}
          edges={rfEdges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          onNodeClick={handleNodeClick}
          onNodeDoubleClick={handleNodeDoubleClick}
          onEdgeMouseEnter={(_, edge) => setHoveredEdgeId(edge.id)}
          onEdgeMouseLeave={() => setHoveredEdgeId(null)}
          onInit={(instance) => {
            rfInstanceRef.current = instance;
            setZoomPercent(Math.round(instance.getViewport().zoom * 100));
          }}
          onMove={(_, viewport) => setZoomPercent(Math.round(viewport.zoom * 100))}
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
    </div>
  );
}
