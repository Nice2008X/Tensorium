/**
 * Permanent quality gate for the graph-layout pipeline (see
 * apps/web/src/elkLayout.ts and apps/web/src/components/ArchitectureGraph.tsx).
 * Formalizes the scratch verification scripts written by hand earlier in
 * this project's history (write one, eyeball it, delete it) into a
 * reusable, real geometric check: for every adapter's small fixture, every
 * view (architecture + every transformer block), and both the stacked and
 * unstacked rendering, it asserts zero node/node overlaps, zero
 * edge-crosses-edge intersections, and zero edge-cuts-through-an-unrelated-
 * node intersections, using ELK's own routed polylines — not a topological
 * approximation.
 *
 * Node sizes are a fixed nominal box (real DOM measurement isn't available
 * outside a browser) — the same simplification this app's very first
 * hand-rolled layout used, and sufficient for this checker's purpose, since
 * every bug found by hand so far was about arrangement, not exact pixel
 * dimensions.
 *
 * Run with: npx tsx apps/web/scripts/check-graph-layout.ts
 * (from the repo root, with Node >=22.12 active).
 */
import { LlamaAdapter } from "@tensorium/adapter-llama";
import { MistralAdapter } from "@tensorium/adapter-mistral";
import { GemmaAdapter } from "@tensorium/adapter-gemma";
import { QwenAdapter } from "@tensorium/adapter-qwen";
import { Qwen3Adapter } from "@tensorium/adapter-qwen3";
import { PhiAdapter } from "@tensorium/adapter-phi";
import { Glm4Adapter } from "@tensorium/adapter-glm4";
import { OlmoAdapter } from "@tensorium/adapter-olmo";
import { QwenMoeAdapter } from "@tensorium/adapter-qwen-moe";
import { Qwen3MoeAdapter } from "@tensorium/adapter-qwen3-moe";
import { DeepseekV2Adapter } from "@tensorium/adapter-deepseek-v2";
import { Qwen35Adapter } from "@tensorium/adapter-qwen3-5";
import { GPT2Adapter } from "@tensorium/adapter-gpt2";
import { Gemma4Adapter } from "@tensorium/adapter-gemma4";
import type { Model, ModelAdapter, ModelEdge } from "@tensorium/model-ir";
import { computeElkLayout, segmentsIntersect, type LayoutPosition, type NodeSize } from "../src/elkLayout.ts";
import { buildLevel1Graph, buildLevel2Graph, collapseRepeatedChains } from "../src/graphUtils.ts";

const NODE_SIZE: NodeSize = { width: 160, height: 76 };

function resolveKind(e: ModelEdge): string {
  return e.kind ?? (e.label === "skip" ? "residual" : "data");
}

function boxesOverlap(a: LayoutPosition, b: LayoutPosition): boolean {
  const margin = 1; // tolerate exact touching edges (adjacent-rank boxes sharing a boundary line)
  return a.x + margin < b.x + b.width && b.x + margin < a.x + a.width && a.y + margin < b.y + b.height && b.y + margin < a.y + a.height;
}

function segmentIntersectsBox(p1: { x: number; y: number }, p2: { x: number; y: number }, box: LayoutPosition): boolean {
  // Only care about a segment genuinely crossing through the box's
  // interior, not merely touching its boundary (every edge legitimately
  // touches its own source/target box boundary).
  const inset = 2;
  const bx0 = box.x + inset;
  const by0 = box.y + inset;
  const bx1 = box.x + box.width - inset;
  const by1 = box.y + box.height - inset;
  if (bx1 <= bx0 || by1 <= by0) return false;
  const corners = [
    { x: bx0, y: by0 },
    { x: bx1, y: by0 },
  ];
  const edges: [{ x: number; y: number }, { x: number; y: number }][] = [
    [{ x: bx0, y: by0 }, { x: bx1, y: by0 }],
    [{ x: bx1, y: by0 }, { x: bx1, y: by1 }],
    [{ x: bx1, y: by1 }, { x: bx0, y: by1 }],
    [{ x: bx0, y: by1 }, { x: bx0, y: by0 }],
  ];
  for (const [c1, c2] of edges) if (segmentsIntersect(p1, p2, c1, c2)) return true;
  void corners;
  // Fully inside (both segment endpoints inside the box, no boundary
  // crossing detected above) also counts.
  const inside = (p: { x: number; y: number }) => p.x > bx0 && p.x < bx1 && p.y > by0 && p.y < by1;
  return inside(p1) && inside(p2);
}

interface ViewResult {
  label: string;
  nodeOverlaps: number;
  edgeCrossings: number;
  edgeNodeIntersections: number;
}

async function checkView(label: string, nodeIds: string[], edgeList: ModelEdge[]): Promise<ViewResult> {
  const sizes = new Map<string, NodeSize>(nodeIds.map((id) => [id, NODE_SIZE]));
  const { positions, routes } = await computeElkLayout(nodeIds, edgeList, sizes, NODE_SIZE);

  let nodeOverlaps = 0;
  for (let i = 0; i < nodeIds.length; i++) {
    for (let j = i + 1; j < nodeIds.length; j++) {
      const a = positions.get(nodeIds[i]);
      const b = positions.get(nodeIds[j]);
      if (a && b && boxesOverlap(a, b)) {
        nodeOverlaps++;
        console.log(`  NODE OVERLAP in ${label}: ${nodeIds[i]} <-> ${nodeIds[j]}`);
      }
    }
  }

  // Build each edge's polyline: routed edges use ELK's own route; residual
  // edges use the same simple 3-point L-shape ResidualLaneEdge draws
  // (source -> laneX,sourceY -> laneX,targetY -> target), with laneX
  // computed the same way the component does (max right edge + gap).
  let maxRight = 0;
  for (const p of positions.values()) maxRight = Math.max(maxRight, p.x + p.width);
  for (const r of routes.values()) for (const pt of r.points) maxRight = Math.max(maxRight, pt.x);
  const laneX = maxRight + 90;

  const relevant = edgeList.filter((e) => nodeIds.includes(e.source) && nodeIds.includes(e.target) && e.source !== e.target);
  const polylines = relevant
    .map((e) => {
      const src = positions.get(e.source);
      const tgt = positions.get(e.target);
      if (!src || !tgt) return null;
      const kind = resolveKind(e);
      if (kind === "residual") {
        const sx = src.x + src.width / 2;
        const sy = src.y + src.height;
        const tx = tgt.x + tgt.width / 2;
        const ty = tgt.y;
        return { id: e.id, source: e.source, target: e.target, points: [{ x: sx, y: sy }, { x: laneX, y: sy }, { x: laneX, y: ty }, { x: tx, y: ty }] };
      }
      const route = routes.get(e.id);
      if (!route) return null;
      return { id: e.id, source: e.source, target: e.target, points: route.points };
    })
    .filter((x): x is { id: string; source: string; target: string; points: { x: number; y: number }[] } => !!x);

  let edgeCrossings = 0;
  for (let i = 0; i < polylines.length; i++) {
    for (let j = i + 1; j < polylines.length; j++) {
      const a = polylines[i];
      const b = polylines[j];
      // Two edges sharing an endpoint node legitimately touch there — not a crossing.
      if (a.source === b.source || a.source === b.target || a.target === b.source || a.target === b.target) continue;
      let crossed = false;
      for (let si = 0; si < a.points.length - 1 && !crossed; si++) {
        for (let sj = 0; sj < b.points.length - 1 && !crossed; sj++) {
          if (segmentsIntersect(a.points[si], a.points[si + 1], b.points[sj], b.points[sj + 1])) crossed = true;
        }
      }
      if (crossed) {
        edgeCrossings++;
        console.log(`  EDGE CROSSING in ${label}: ${a.id} x ${b.id}`);
      }
    }
  }

  let edgeNodeIntersections = 0;
  for (const line of polylines) {
    for (const id of nodeIds) {
      if (id === line.source || id === line.target) continue;
      const box = positions.get(id);
      if (!box) continue;
      let hit = false;
      for (let si = 0; si < line.points.length - 1 && !hit; si++) {
        if (segmentIntersectsBox(line.points[si], line.points[si + 1], box)) hit = true;
      }
      if (hit) {
        edgeNodeIntersections++;
        console.log(`  EDGE-THROUGH-NODE in ${label}: ${line.id} cuts through ${id}`);
      }
    }
  }

  return { label, nodeOverlaps, edgeCrossings, edgeNodeIntersections };
}

async function checkAdapter(name: string, adapter: ModelAdapter, repo: string): Promise<ViewResult[]> {
  const source = { kind: "hub" as const, repo };
  const metadata = await adapter.loadMetadata(source);
  const provider = await adapter.getWeightProvider(metadata, source);
  const model: Model = adapter.buildGraph(metadata, provider.id);

  const results: ViewResult[] = [];
  for (const stack of [false, true]) {
    const g1 = buildLevel1Graph(model, false);
    const view1 = stack ? collapseRepeatedChains(model, g1.nodeIds, g1.edges) : { nodeIds: g1.nodeIds, edges: g1.edges };
    results.push(await checkView(`${name}/architecture${stack ? " (stacked)" : ""}`, view1.nodeIds, view1.edges));

    for (const node of Object.values(model.nodes)) {
      if (node.type !== "transformer_block") continue;
      const g2 = buildLevel2Graph(model, node.id);
      const view2 = stack ? collapseRepeatedChains(model, g2.nodeIds, g2.edges) : { nodeIds: g2.nodeIds, edges: g2.edges };
      results.push(await checkView(`${name}/${node.id}${stack ? " (stacked)" : ""}`, view2.nodeIds, view2.edges));
    }
  }
  return results;
}

async function main() {
  const cases: [string, ModelAdapter, string][] = [
    ["llama", LlamaAdapter, "hf-internal-testing/tiny-random-LlamaForCausalLM"],
    ["mistral", MistralAdapter, "yujiepan/mistral-tiny-random"],
    ["gemma", GemmaAdapter, "fxmarty/tiny-random-GemmaForCausalLM"],
    ["qwen2", QwenAdapter, "yujiepan/qwen2-tiny-random"],
    ["qwen3", Qwen3Adapter, "tiny-random/qwen3"],
    ["phi3", PhiAdapter, "tiny-random/phi-4"],
    ["glm4", Glm4Adapter, "tiny-random/glm-4"],
    ["olmo", OlmoAdapter, "katuni4ka/tiny-random-olmo-hf"],
    ["qwen2_moe", QwenMoeAdapter, "katuni4ka/tiny-random-qwen1.5-moe"],
    ["qwen3_moe", Qwen3MoeAdapter, "tiny-random/qwen3-moe"],
    ["deepseek_v2", DeepseekV2Adapter, "yujiepan/deepseek-v2-0628-tiny-random"],
    ["qwen3-5", Qwen35Adapter, "tiny-random/qwen3.5"],
    ["gpt2", GPT2Adapter, "hf-internal-testing/tiny-random-gpt2"],
    ["gemma4", Gemma4Adapter, "google/gemma-4-E2B"],
  ];

  let totalNodeOverlaps = 0;
  let totalEdgeCrossings = 0;
  let totalEdgeNodeIntersections = 0;
  let viewCount = 0;

  for (const [name, adapter, repo] of cases) {
    try {
      const results = await checkAdapter(name, adapter, repo);
      const overlaps = results.reduce((a, r) => a + r.nodeOverlaps, 0);
      const crossings = results.reduce((a, r) => a + r.edgeCrossings, 0);
      const intersections = results.reduce((a, r) => a + r.edgeNodeIntersections, 0);
      totalNodeOverlaps += overlaps;
      totalEdgeCrossings += crossings;
      totalEdgeNodeIntersections += intersections;
      viewCount += results.length;
      console.log(`${name}: ${results.length} views, ${overlaps} overlaps, ${crossings} crossings, ${intersections} edge-through-node`);
    } catch (err) {
      console.error(`${name}: FAILED`, err);
      process.exitCode = 1;
    }
  }

  console.log(`\n${viewCount} total views checked.`);
  console.log(`node overlaps: ${totalNodeOverlaps}, edge crossings: ${totalEdgeCrossings}, edge-through-node: ${totalEdgeNodeIntersections}`);
  if (totalNodeOverlaps > 0 || totalEdgeCrossings > 0 || totalEdgeNodeIntersections > 0) {
    console.log("QUALITY GATE FAILED");
    process.exitCode = 1;
  } else {
    console.log("QUALITY GATE PASSED");
  }
}

main();
