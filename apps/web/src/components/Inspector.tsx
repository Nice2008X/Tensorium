import type { ReactNode } from "react";
import type { ActivationCapture, Model, ModelNode, Tensor } from "@tensorium/model-ir";
import { totalParameterCount } from "@tensorium/model-ir";
import type { Tokenizer } from "@tensorium/tokenizer";
import { componentRegistry } from "../registry.js";
import { topKFromLogits } from "../logits.js";
import { formatBytes, formatCount, formatPercent } from "../format.js";
import { Heatmap } from "./Heatmap.js";
import { useLocalStorageState } from "../useLocalStorageState.js";

function formatDims(dims: Array<number | string>): string {
  return `[${dims.join(", ")}]`;
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

interface InputSource {
  label: string;
  isBlockBoundary: boolean;
  /** The real edge source id — even for a "block input" boundary source, where it's the block container's own id. That container's captured activation *is* the tensor that actually flowed in at this boundary, so it's still a valid lookup key into ActivationCapture.activations, just not a very descriptive node name (hence the separate `label`). */
  sourceId: string;
}

/**
 * Which upstream node(s) actually feed this one, and — where it's safe to
 * say so — how they combine. Two things make this trickier than just
 * reading `model.edges`:
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
function describeInputConstruction(model: Model, node: ModelNode): { sources: InputSource[]; operator: "+" | "×" | null } {
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

interface Props {
  model: Model;
  node: ModelNode | null;
  /** The selected node's real captured activation shape/magnitude from the last run — undefined when no run has happened yet, or this node has no recorded activation (e.g. a purely organizational container). */
  activationShape?: number[];
  activationMagnitude?: number;
  onViewActivation?: () => void;
  onViewWeights?: () => void;
  /** The last completed forward pass (Prompt A), if any — feeds the "Last run" section of the no-selection overview below (and the root node's own view, which folds the same overview in — see `isRoot`). Not threaded any further than that: once a non-root node is selected, its own activation info comes from `activationShape`/`activationMagnitude` instead. */
  inferenceResult?: ActivationCapture;
  tokenizer?: Tokenizer;
  /** Real wall-clock duration of that forward pass, in ms — see useInference. Undefined for a result restored some other way (there isn't one today, but keeps this honestly optional rather than defaulting to 0). */
  elapsedMs?: number;
  /** Clears the selection, returning to the no-selection Model overview — surfaced as a back link once a node is selected, since Esc (the only other way back) isn't discoverable from the Inspector itself. */
  onDeselect?: () => void;
}

export function Inspector({ model, node, activationShape, activationMagnitude, onViewActivation, onViewWeights, inferenceResult, tokenizer, elapsedMs, onDeselect }: Props) {
  // Declared unconditionally, above the early return below, so the hook
  // count stays stable across a selection toggling node between a real
  // value and null (React's rules of hooks) — a per-user preference like
  // every other panel's collapse state, not reset per node.
  const [thisRunCollapsed, setThisRunCollapsed] = useLocalStorageState("panel:inspector-thisrun-collapsed", false);

  if (!node) {
    return (
      <div className="inspector">
        <ModelOverview model={model} inferenceResult={inferenceResult} tokenizer={tokenizer} elapsedMs={elapsedMs} />
      </div>
    );
  }

  const info = componentRegistry[node.type];
  const totalParams = node.parameters.reduce((a, p) => a + (p.slice ? p.logicalShape.reduce((x, y) => x * y, 1) : p.numElements), 0);
  const hasThisRun = activationShape !== undefined && activationMagnitude !== undefined;
  const { sources: inputSources, operator: inputOperator } = describeInputConstruction(model, node);
  // The root node's own per-node facts (no parameters, no shapes, no
  // incoming edges — every real adapter builds it as a bare container, see
  // e.g. gpt2/graph.ts's `node("model", "model", ...)`) are never
  // interesting on their own, so selecting it folds in the same Model/
  // Structure/Last-run summary the no-selection overview shows, instead of
  // leaving most of this view empty.
  const isRoot = node.id === model.rootId;

  return (
    <div className="inspector">
      {onDeselect && (
        <button type="button" className="inspector-back-link" onClick={onDeselect}>
          ‹ Model overview
        </button>
      )}
      <div className="inspector-title" style={{ borderColor: info.color }}>
        <span className="inspector-badge" style={{ background: info.color }}>
          {info.label}
        </span>
        <span className="inspector-name">{node.name}</span>
      </div>

      <Section title="What is it?">
        <p>{info.description}</p>
      </Section>

      {isRoot && <ModelSummarySections model={model} inferenceResult={inferenceResult} tokenizer={tokenizer} elapsedMs={elapsedMs} />}

      {hasThisRun && (
        <Section title="This run" collapsible collapsed={thisRunCollapsed} onToggleCollapsed={() => setThisRunCollapsed((v) => !v)}>
          {inputSources.length > 0 && (
            <div className="inspector-io-group">
              <div className="inspector-io-group-title">Input{inputSources.length > 1 ? "s" : ""}</div>
              {inputSources.map((s, i) => (
                <div key={i} className="inspector-io-item">
                  <div className="inspector-io-item-label" title={s.label}>
                    {s.label}
                  </div>
                  <TensorThumb tensor={inferenceResult?.activations[s.sourceId]} />
                </div>
              ))}
            </div>
          )}

          <div className="inspector-io-group">
            <div className="inspector-io-group-title">Output</div>
            <div className="inspector-io-item">
              <TensorThumb tensor={inferenceResult?.activations[node.id]} />
            </div>
          </div>

          <div className="io-row">
            <span className="io-label">activation shape</span>
            <span className="io-shape">{formatDims(activationShape!)}</span>
          </div>
          <div className="io-row">
            <span className="io-label">magnitude (L2 norm)</span>
            <span className="io-shape">{activationMagnitude!.toFixed(4)}</span>
          </div>
        </Section>
      )}

      {inputSources.length > 0 && (
        <Section title="Input construction">
          {inputOperator ? (
            <code className="formula">input = {inputSources.map((s) => s.label).join(` ${inputOperator} `)}</code>
          ) : inputSources.length === 1 ? (
            <code className="formula">input = {inputSources[0].label}</code>
          ) : (
            <>
              <p>Assembled from multiple sources:</p>
              <ul className="input-source-list">
                {inputSources.map((s, i) => (
                  <li key={i}>{s.label}</li>
                ))}
              </ul>
            </>
          )}
        </Section>
      )}

      {(hasThisRun || node.parameters.length > 0) && (
        <div className="inspector-actions">
          {hasThisRun && onViewActivation && (
            <button type="button" onClick={onViewActivation}>
              View activation
            </button>
          )}
          {node.parameters.length > 0 && onViewWeights && (
            <button type="button" onClick={onViewWeights}>
              View weights
            </button>
          )}
        </div>
      )}

      {info.formula && (
        <Section title="Show me the math">
          <code className="formula">{info.formula}</code>
        </Section>
      )}

      {(node.inputs.length > 0 || node.outputs.length > 0) && (
        <Section title="Shapes">
          {node.inputs.map((s, i) => (
            <div key={`in-${i}`} className="io-row">
              <span className="io-label">input</span>
              <span className="io-shape">{formatDims(s.dims)}</span>
            </div>
          ))}
          {node.outputs.map((s, i) => (
            <div key={`out-${i}`} className="io-row">
              <span className="io-label">output</span>
              <span className="io-shape">{formatDims(s.dims)}</span>
            </div>
          ))}
        </Section>
      )}

      {node.parameters.length > 0 && (
        <Section title={`Parameters (${totalParams.toLocaleString()})`}>
          {node.parameters.map((p, i) => (
            <div key={i} className="io-row">
              <span className="io-label">{p.slice ? `${p.name} (slice)` : p.name}</span>
              <span className="io-shape">
                {p.logicalShape.join(" × ")} · {p.dtype}
              </span>
            </div>
          ))}
        </Section>
      )}

      {Object.keys(node.metadata).length > 0 && (
        <Section title="Metadata">
          {Object.entries(node.metadata).map(([k, v]) => (
            <div key={k} className="io-row">
              <span className="io-label">{k}</span>
              <span className="io-shape">{typeof v === "object" ? JSON.stringify(v) : String(v)}</span>
            </div>
          ))}
        </Section>
      )}
    </div>
  );
}

/**
 * Rough per-top-level-group parameter share (Embedding / Transformer Blocks
 * / LM Head / ...) for the overview's structure bars. `seen` dedups a tensor
 * shared across groups (e.g. tied lm_head/embedding weights) so it's only
 * counted once, attributed to whichever group is walked first — good enough
 * for a rough share, not meant to be an exact accounting.
 */
function subtreeParamCount(model: Model, nodeId: string, seen: Set<string>): number {
  const node = model.nodes[nodeId];
  let count = 0;
  for (const p of node.parameters) {
    if (seen.has(p.name)) continue;
    seen.add(p.name);
    count += p.numElements;
  }
  for (const childId of node.children) count += subtreeParamCount(model, childId, seen);
  return count;
}

/**
 * Real byte size of everything this run actually captured (every
 * activation + attention-weight tensor, plus the logits) — each one is a
 * Float64Array regardless of the checkpoint's on-disk dtype (see model-ir's
 * Tensor doc comment), so `.byteLength` is exact, not estimated. This is
 * memory this specific result is holding onto, not total page/heap memory.
 */
function captureBytes(result: ActivationCapture): number {
  let bytes = result.logits.data.byteLength;
  for (const t of Object.values(result.activations)) bytes += t.data.byteLength;
  for (const t of Object.values(result.attentionWeights)) bytes += t.data.byteLength;
  return bytes;
}

/**
 * The Model / Structure / Last-run sections — shared by the no-selection
 * overview and the root node's own view (see `isRoot` above), since both
 * are "tell me about the whole model" moments and would otherwise duplicate
 * this exact content.
 */
function ModelSummarySections({ model, inferenceResult, tokenizer, elapsedMs }: { model: Model; inferenceResult?: ActivationCapture; tokenizer?: Tokenizer; elapsedMs?: number }) {
  const totalParams = totalParameterCount(model);

  const seen = new Set<string>();
  const groups = model.nodes[model.rootId].children
    .map((id) => ({ id, name: model.nodes[id].name, count: subtreeParamCount(model, id, seen) }))
    .filter((g) => g.count > 0)
    .sort((a, b) => b.count - a.count)
    .slice(0, 6);
  const maxGroupCount = groups[0]?.count ?? 1;

  const lastTokenIndex = inferenceResult ? inferenceResult.tokenIds.length - 1 : -1;
  const topPrediction = inferenceResult && tokenizer ? topKFromLogits(inferenceResult.logits, lastTokenIndex, 1)[0] : undefined;

  return (
    <>
      <Section title="Model">
        <div className="io-row">
          <span className="io-label">architecture</span>
          <span className="io-shape">{model.architecture}</span>
        </div>
        <div className="io-row">
          <span className="io-label">parameters</span>
          <span className="io-shape">{formatCount(totalParams)}</span>
        </div>
        <div className="io-row">
          <span className="io-label">layers</span>
          <span className="io-shape">{model.config.numLayers}</span>
        </div>
      </Section>

      {groups.length > 0 && (
        <Section title="Structure">
          {groups.map((g) => (
            <div key={g.id} className="overview-structure-row">
              <span className="overview-structure-label" title={g.name}>
                {g.name}
              </span>
              <div className="prediction-bar-track">
                <div className="prediction-bar-fill" style={{ width: `${(g.count / maxGroupCount) * 100}%` }} />
              </div>
              <span className="overview-structure-value">{formatCount(g.count)}</span>
            </div>
          ))}
        </Section>
      )}

      <Section title="Last run">
        {inferenceResult && tokenizer ? (
          <>
            <div className="io-row">
              <span className="io-label">prompt tokens</span>
              <span className="io-shape">{inferenceResult.tokenIds.length}</span>
            </div>
            <div className="io-row">
              <span className="io-label">decoded</span>
              <span className="io-shape overview-prompt-preview" title={inferenceResult.tokens.join("")}>
                {inferenceResult.tokens.join("")}
              </span>
            </div>
            {elapsedMs !== undefined && (
              <div className="io-row">
                <span className="io-label">forward pass</span>
                <span className="io-shape">{elapsedMs < 1 ? "<1 ms" : `${elapsedMs.toFixed(1)} ms`}</span>
              </div>
            )}
            <div className="io-row">
              <span className="io-label">memory (this run)</span>
              <span className="io-shape">{formatBytes(captureBytes(inferenceResult))}</span>
            </div>
            {topPrediction && (
              <div className="io-row">
                <span className="io-label">top prediction</span>
                <span className="io-shape">
                  {tokenizer.decodeToken(topPrediction.tokenId).trim() || `#${topPrediction.tokenId}`} · {formatPercent(topPrediction.prob)}
                </span>
              </div>
            )}
          </>
        ) : (
          <p>Run a forward pass to see prediction and activation info here.</p>
        )}
      </Section>
    </>
  );
}

/** The overview shown in place of "click a component" when nothing is selected — a model summary plus (once a prompt has been run) a snapshot of the last result, so the Inspector isn't dead space before the user picks a node. */
function ModelOverview({ model, inferenceResult, tokenizer, elapsedMs }: { model: Model; inferenceResult?: ActivationCapture; tokenizer?: Tokenizer; elapsedMs?: number }) {
  return (
    <div className="inspector inspector-overview">
      <div className="inspector-overview-title">Model overview</div>
      <ModelSummarySections model={model} inferenceResult={inferenceResult} tokenizer={tokenizer} elapsedMs={elapsedMs} />
      <p className="overview-hint">Click a component in the graph or tree to inspect it.</p>
    </div>
  );
}

/**
 * A small heatmap of one node's real captured tensor, plus its shape — the
 * compact "what did this actually compute" view for Input/Output rows.
 * `undefined` (no activation recorded for this node — e.g. a purely
 * organizational container) and >2D (e.g. something 3D-shaped, which
 * shouldn't occur for a plain `activations` entry, but this guards it
 * anyway) both fall back to a text hint instead of attempting a heatmap.
 */
function TensorThumb({ tensor }: { tensor?: Tensor }) {
  if (!tensor) return <p className="inspector-tensor-empty">Not captured for this run.</p>;
  if (tensor.shape.length > 2) return <p className="inspector-tensor-empty">Too many dimensions to preview here — see Tensor Explorer.</p>;
  return (
    <div className="inspector-tensor-thumb">
      {tensor.shape.length === 2 ? (
        <Heatmap data={tensor.data} rows={tensor.shape[0]} cols={tensor.shape[1]} cellSize={3} />
      ) : (
        <Heatmap data={tensor.data} rows={1} cols={tensor.shape[0]} cellSize={4} />
      )}
      <span className="inspector-tensor-thumb-shape">{formatDims(tensor.shape)}</span>
    </div>
  );
}

function Section({
  title,
  children,
  collapsible,
  collapsed,
  onToggleCollapsed,
}: {
  title: string;
  children: ReactNode;
  collapsible?: boolean;
  collapsed?: boolean;
  onToggleCollapsed?: () => void;
}) {
  return (
    <div className="inspector-section">
      <div className="inspector-section-header">
        <div className="inspector-section-title">{title}</div>
        {collapsible && (
          <button type="button" className="inspector-section-collapse-btn" onClick={onToggleCollapsed} title={collapsed ? "Expand" : "Collapse"}>
            {collapsed ? "▸" : "▾"}
          </button>
        )}
      </div>
      {!collapsed && children}
    </div>
  );
}
