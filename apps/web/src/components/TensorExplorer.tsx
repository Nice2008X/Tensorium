import { useEffect, useMemo, useState } from "react";
import type { Model, ModelNode, ParameterRef, WeightProvider, Tensor } from "@tensorium/model-ir";
import { numElements } from "@tensorium/model-ir";
import { computeStats, type TensorStats } from "@tensorium/tensor-core";
import { Heatmap } from "./Heatmap.js";
import { Histogram } from "./Histogram.js";
import { AttentionView } from "./AttentionView.js";
import { composeSlice, defaultWindow, parameterKey } from "../tensor.js";
import type { InferenceState } from "../useInference.js";

interface Props {
  model: Model;
  weightProvider: WeightProvider;
  selectedNode: ModelNode | null;
  inference?: InferenceState;
  selectedTokenIndex: number | null;
  promptBInference?: InferenceState;
  /** A one-shot request to switch the Weights/Activations source tab — e.g. from the Inspector's "View activation"/"View weights" quick actions. Bump `nonce` on every request so a repeat click of the same source still re-applies (a user may have since clicked to a different tab themselves). */
  sourceRequest?: { value: "weights" | "activations"; nonce: number } | null;
  /** True when weightProvider is a SyntheticWeightProvider — every value shown below (weights, and any activation, since those come from a forward pass over the same fabricated weights) is randomly generated, not real. */
  structureOnly?: boolean;
}

interface ParamEntry {
  ref: ParameterRef;
  ownerName: string;
  ownerId: string;
}

// Tied weights (e.g. Gemma's lm_head reusing the embedding tensor) mean two
// different owner nodes can produce the exact same parameterKey(ref) — key
// list entries and selection state by (owner, ref) together so those stay
// distinguishable instead of colliding.
function entryKey(p: ParamEntry): string {
  return `${p.ownerId}:${parameterKey(p.ref)}`;
}

type ViewMode = "heatmap" | "matrix" | "histogram" | "tokens";
type Source = "weights" | "activations" | "compare";

const MAX_MATRIX_CELLS = 128 * 128; // beyond this, the Matrix tab is disabled rather than freezing the tab

function formatBytes(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`;
  return `${n} B`;
}

export function TensorExplorer({ model, weightProvider, selectedNode, inference, selectedTokenIndex, promptBInference, sourceRequest, structureOnly }: Props) {
  const allParams = useMemo<ParamEntry[]>(() => {
    const list: ParamEntry[] = [];
    for (const node of Object.values(model.nodes)) {
      for (const ref of node.parameters) list.push({ ref, ownerName: node.name, ownerId: node.id });
    }
    return list;
  }, [model]);

  const [search, setSearch] = useState("");
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [tensor, setTensor] = useState<Tensor | null>(null);
  const [stats, setStats] = useState<TensorStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [windowRanges, setWindowRanges] = useState<{ start: number; end: number }[] | null>(null);
  const [view, setView] = useState<ViewMode>("heatmap");
  const [source, setSource] = useState<Source>("weights");
  // Which prompt's own activation the "Activations" tab shows — independent
  // of "Compare", which always shows both at once (plus their diff) rather
  // than one at a time at full detail (Matrix/Histogram, full stats). Reset
  // to A whenever B's result goes away (Compare disabled, or B re-run and
  // not yet ready again) so this doesn't silently keep pointing at a stale
  // or now-absent result.
  const [activationSource, setActivationSource] = useState<"A" | "B">("A");

  // a freshly-finished run is almost always what the user wants to look at next
  useEffect(() => {
    if (inference?.status === "ready") setSource("activations");
  }, [inference?.result]);

  // Explicit request from outside (Inspector's quick actions) — keyed on
  // `nonce` rather than `value` so clicking the same source again (after the
  // user has since switched tabs themselves) still re-applies it.
  useEffect(() => {
    if (sourceRequest) setSource(sourceRequest.value);
  }, [sourceRequest?.nonce]);

  useEffect(() => {
    if (selectedNode && selectedNode.parameters.length > 0) {
      setSelectedKey(`${selectedNode.id}:${parameterKey(selectedNode.parameters[0])}`);
      setWindowRanges(null);
    }
  }, [selectedNode]);

  const selectedEntry = useMemo(
    () => allParams.find((p) => entryKey(p) === selectedKey) ?? null,
    [allParams, selectedKey]
  );

  useEffect(() => {
    if (source !== "weights" || !selectedEntry) {
      setTensor(null);
      setStats(null);
      return;
    }
    const ref = selectedEntry.ref;
    const win = windowRanges ? { ranges: windowRanges } : defaultWindow(ref.logicalShape);
    let cancelled = false;
    setLoading(true);
    weightProvider
      .loadTensor(ref.name, composeSlice(ref, win))
      .then((t) => {
        if (cancelled) return;
        setTensor(t);
        setStats(computeStats(t.data));
        setLoading(false);
      })
      .catch(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [source, selectedEntry, windowRanges, weightProvider]);

  const hasInferenceResult = inference?.status === "ready" && !!inference.result;
  const hasPromptB = promptBInference?.status === "ready" && !!promptBInference.result;

  // Falls back to A automatically once B's result is gone — otherwise this
  // tab would keep pointing at a source that no longer has anything to show.
  useEffect(() => {
    if (activationSource === "B" && !hasPromptB) setActivationSource("A");
  }, [activationSource, hasPromptB]);

  const activeInference = activationSource === "B" ? promptBInference : inference;
  const activationTensor = source === "activations" && selectedNode ? activeInference?.result?.activations[selectedNode.id] ?? null : null;
  const activationStats = useMemo(() => (activationTensor ? computeStats(activationTensor.data) : null), [activationTensor]);
  const attentionTensor = source === "activations" && selectedNode ? activeInference?.result?.attentionWeights[selectedNode.id] ?? null : null;

  const filtered = useMemo(() => {
    const q = search.toLowerCase();
    return allParams.filter((p) => p.ref.name.toLowerCase().includes(q) || p.ownerName.toLowerCase().includes(q));
  }, [allParams, search]);

  const ref = selectedEntry?.ref;
  const loadedElements = tensor ? numElements(tensor.shape) : 0;
  const loadedBytes = ref ? loadedElements * (ref.bytes / ref.numElements) : 0;

  const displayTensor = source === "weights" ? tensor : activationTensor;
  const displayStats = source === "weights" ? stats : activationStats;
  const canShowMatrix = !!displayTensor && displayTensor.data.length <= MAX_MATRIX_CELLS;
  // Only meaningful for an activation whose rows are literally "one per
  // token" — a weight tensor's rows aren't tokens, and a 1D tensor (e.g. a
  // LayerNorm bias) has no per-token axis at all.
  const canShowTokens =
    source === "activations" && !!displayTensor && displayTensor.shape.length === 2 && !!activeInference?.displayTokens && displayTensor.shape[0] === activeInference.displayTokens.length;

  // keep the active tab valid as the selection changes (e.g. a 1-value bias has no useful histogram)
  useEffect(() => {
    if (view === "matrix" && !canShowMatrix) setView("heatmap");
    if (view === "tokens" && !canShowTokens) setView("heatmap");
  }, [view, canShowMatrix, canShowTokens]);

  const compare = useMemo(() => {
    if (source !== "compare" || !selectedNode || !inference?.result || !promptBInference?.result) return null;
    const a = inference.result.activations[selectedNode.id];
    const b = promptBInference.result.activations[selectedNode.id];
    if (!a || !b || a.shape.length !== 2 || b.shape.length !== 2) return null;
    const rows = Math.min(a.shape[0], b.shape[0]);
    const cols = Math.min(a.shape[1], b.shape[1]);
    const truncated = a.shape[0] !== b.shape[0] || a.shape[1] !== b.shape[1];
    const diffData = new Float64Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        diffData[r * cols + c] = a.data[r * a.shape[1] + c] - b.data[r * b.shape[1] + c];
      }
    }
    const diff: Tensor = { shape: [rows, cols], dtype: "F32", data: diffData };
    return { a, b, diff, truncated, statsA: computeStats(a.data), statsB: computeStats(b.data), statsDiff: computeStats(diff.data) };
  }, [source, selectedNode, inference?.result, promptBInference?.result]);

  return (
    <div className="tensor-explorer">
      {hasInferenceResult && (
        <div className="source-tabs">
          <button className={source === "weights" ? "active" : ""} onClick={() => setSource("weights")}>
            Weights
          </button>
          <button className={source === "activations" ? "active" : ""} onClick={() => setSource("activations")}>
            Activations (last run)
          </button>
          <button className={source === "compare" ? "active" : ""} disabled={!hasPromptB} onClick={() => setSource("compare")} title={!hasPromptB ? "Run Prompt B first" : undefined}>
            Compare (A vs B)
          </button>
        </div>
      )}

      <div className="tensor-explorer-body">
        {/* Lives here — inside the Weights tab's own content, below the tab
            switcher above — rather than as a permanent left rail, since
            searching/browsing raw parameters only makes sense for Weights;
            Activations/Compare are always scoped to whatever's selected in
            the graph/tree instead. */}
        {source === "weights" && (
          <div className="tensor-explorer-list">
            <input
              className="search-input"
              placeholder="Search parameters (e.g. attn, ln_1, wte)…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            <div className="param-list-count">
              {filtered.length} of {allParams.length} parameter tensors
            </div>
            <div className="param-list">
              {filtered.map((p) => {
                const key = entryKey(p);
                return (
                  <button
                    key={key}
                    className={"param-item" + (key === selectedKey ? " selected" : "")}
                    onClick={() => {
                      setSelectedKey(key);
                      setWindowRanges(null);
                    }}
                  >
                    <div className="param-name">
                      {p.ref.name}
                      {p.ref.slice ? <span className="param-slice"> [sliced]</span> : null}
                    </div>
                    <div className="param-shape">
                      {p.ref.logicalShape.join(" × ")} · {p.ref.dtype}
                    </div>
                  </button>
                );
              })}
              {filtered.length === 0 && <div className="empty-hint">No parameters match.</div>}
            </div>
          </div>
        )}

        <div className="tensor-explorer-detail">
        {/* Compare (above) always shows A, B, and their diff together — this
            is the opposite: one prompt's own activation at a time, at full
            detail (Matrix/Histogram, full stats), same as picking any other
            single tensor. */}
        {source === "activations" && hasPromptB && (
          <div className="source-tabs activation-source-tabs">
            <button className={activationSource === "A" ? "active" : ""} onClick={() => setActivationSource("A")}>
              Prompt A
            </button>
            <button className={activationSource === "B" ? "active" : ""} onClick={() => setActivationSource("B")}>
              Prompt B
            </button>
          </div>
        )}

        {source === "weights" && !ref && <div className="empty-hint">Select a component with weights to inspect its tensor.</div>}
        {source === "activations" && !selectedNode && <div className="empty-hint">Select a component to inspect its activation from the last forward pass.</div>}
        {source === "activations" && selectedNode && !activationTensor && (
          <div className="empty-hint">No activation was captured for "{selectedNode.name}" — try a leaf computation node (LayerNorm, a projection, the activation function, …).</div>
        )}

        {source === "weights" && ref && (
          <div className="tensor-header">
            <div className="tensor-title">
              {ref.name}
              {ref.slice ? " (sliced)" : ""}
            </div>
            <div className="tensor-meta">
              <span>Shape {ref.logicalShape.join(" × ")}</span>
              <span>dtype {ref.dtype}</span>
              <span>{ref.numElements.toLocaleString()} params (full tensor)</span>
              <span>{formatBytes(ref.bytes)} (full tensor)</span>
              {structureOnly && (
                <span className="synthetic-badge" title="This checkpoint's real weights were never downloaded (too large or sharded) — these values are randomly generated, not real.">
                  Synthetic
                </span>
              )}
              {tensor && (
                <span className="loaded-badge" title="Bytes actually pulled into the browser for the window currently shown">
                  {formatBytes(loadedBytes)} loaded ({((loadedBytes / ref.bytes) * 100 || 0).toFixed(1)}% of tensor)
                </span>
              )}
            </div>
          </div>
        )}
        {source === "activations" && selectedNode && activationTensor && (
          <div className="tensor-header">
            <div className="tensor-title">
              {selectedNode.name} — activation
              {hasPromptB && <span className="tensor-title-prompt-tag">{activationSource === "A" ? "Prompt A" : "Prompt B"}</span>}
            </div>
            <div className="tensor-meta">
              <span>Shape {activationTensor.shape.join(" × ")}</span>
              <span>dtype {activationTensor.dtype}</span>
              <span>from prompt: "{activeInference?.displayTokens?.join("")}"</span>
              {structureOnly && (
                <span className="synthetic-badge" title="Computed from randomly generated weights (this checkpoint's real weights were never downloaded) — not a real forward pass.">
                  Synthetic
                </span>
              )}
            </div>
          </div>
        )}

        {source === "weights" && ref && ref.logicalShape.reduce((a, b) => a * b, 1) > 64 * 64 && (
          <WindowControls
            shape={ref.logicalShape}
            ranges={windowRanges ?? defaultWindow(ref.logicalShape).ranges!}
            onChange={setWindowRanges}
          />
        )}

        {source === "weights" && loading && <div className="empty-hint">Loading tensor…</div>}

        {!loading && displayTensor && displayStats && (
          <div className="tensor-body">
            <div className="tensor-visual">
              <div className="view-tabs">
                <button className={view === "heatmap" ? "active" : ""} onClick={() => setView("heatmap")}>
                  Heatmap
                </button>
                <button
                  className={view === "matrix" ? "active" : ""}
                  disabled={!canShowMatrix}
                  onClick={() => setView("matrix")}
                  title={!canShowMatrix ? "Too many values to render as a table — narrow the window first" : undefined}
                >
                  Matrix
                </button>
                <button className={view === "histogram" ? "active" : ""} onClick={() => setView("histogram")}>
                  Histogram
                </button>
                {canShowTokens && (
                  <button className={view === "tokens" ? "active" : ""} onClick={() => setView("tokens")}>
                    Per Token
                  </button>
                )}
              </div>

              {view === "heatmap" && displayTensor.shape.length === 2 && <Heatmap data={displayTensor.data} rows={displayTensor.shape[0]} cols={displayTensor.shape[1]} />}
              {view === "heatmap" && displayTensor.shape.length === 1 && <Heatmap data={displayTensor.data} rows={1} cols={displayTensor.shape[0]} />}
              {view === "matrix" && canShowMatrix && <RawGrid tensor={displayTensor} />}
              {view === "histogram" && <Histogram stats={displayStats} />}
              {view === "tokens" && canShowTokens && <PerTokenVectors tensor={displayTensor} tokens={activeInference!.displayTokens!} />}
            </div>

            <div className="tensor-stats">
              <StatRow label="Showing" value={`${displayTensor.shape.join(" × ")} (${displayTensor.data.length.toLocaleString()} values)`} />
              <StatRow label="Min" value={displayStats.min.toFixed(4)} />
              <StatRow label="Max" value={displayStats.max.toFixed(4)} />
              <StatRow label="Mean" value={displayStats.mean.toFixed(4)} />
              <StatRow label="Std" value={displayStats.std.toFixed(4)} />
              <StatRow label="Sparsity" value={`${(displayStats.sparsity * 100).toFixed(1)}% (${displayStats.zeros.toLocaleString()} zeros)`} />
              <div className="stat-divider">Percentiles</div>
              <StatRow label="p1" value={displayStats.percentiles.p1.toFixed(4)} />
              <StatRow label="p25" value={displayStats.percentiles.p25.toFixed(4)} />
              <StatRow label="p50 (median)" value={displayStats.percentiles.p50.toFixed(4)} />
              <StatRow label="p75" value={displayStats.percentiles.p75.toFixed(4)} />
              <StatRow label="p99" value={displayStats.percentiles.p99.toFixed(4)} />
            </div>
          </div>
        )}

        {source === "activations" && attentionTensor && activeInference?.displayTokens && (
          <AttentionView attentionWeights={attentionTensor} tokens={activeInference.displayTokens} queryTokenIndex={selectedTokenIndex ?? activeInference.displayTokens.length - 1} />
        )}

        {source === "compare" && !selectedNode && <div className="empty-hint">Select a component to compare its activation across Prompt A and Prompt B.</div>}
        {source === "compare" && selectedNode && !compare && (
          <div className="empty-hint">
            No 2D activation was captured for "{selectedNode.name}" in both runs — try a leaf computation node (LayerNorm, a projection, the activation function, …).
          </div>
        )}
        {source === "compare" && compare && (
          <div className="compare-view">
            {compare.truncated && (
              <div className="compare-note">Prompt A and B have different token counts here — comparing only the overlapping {compare.diff.shape[0]}×{compare.diff.shape[1]} region.</div>
            )}
            <div className="compare-columns">
              <CompareColumn title="Prompt A" tensor={compare.a} stats={compare.statsA} />
              <CompareColumn title="Prompt B" tensor={compare.b} stats={compare.statsB} />
              <CompareColumn title="A − B" tensor={compare.diff} stats={compare.statsDiff} diverging />
            </div>
          </div>
        )}
        </div>
      </div>
    </div>
  );
}

function CompareColumn({ title, tensor, stats, diverging }: { title: string; tensor: Tensor; stats: TensorStats; diverging?: boolean }) {
  return (
    <div className="compare-column">
      <div className="compare-column-title">{title}</div>
      <Heatmap data={tensor.data} rows={tensor.shape[0]} cols={tensor.shape[1]} />
      <div className="compare-stats">
        <StatRow label="Mean" value={stats.mean.toFixed(4)} />
        <StatRow label="Std" value={stats.std.toFixed(4)} />
        <StatRow label={diverging ? "Max |Δ|" : "Max"} value={diverging ? Math.max(Math.abs(stats.min), Math.abs(stats.max)).toFixed(4) : stats.max.toFixed(4)} />
      </div>
    </div>
  );
}

function StatRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat-row">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
    </div>
  );
}

// How many of a (typically 32- to several-thousand-dimensional) vector's
// real values to print per row before truncating with "+N more" — enough to
// get a feel for the numbers without the row wrapping across the panel.
const PER_TOKEN_PREVIEW_DIMS = 8;

/**
 * One row per input token, each showing a prefix of that token's own real
 * activation vector — e.g. `"cat" → [0.91, 0.12, ...] (+29 more, 32 dims)`.
 * The plain-numbers alternative to the heatmap above: same underlying
 * values, no color scale or stats to read, just "here's what this token's
 * vector actually contains".
 */
function PerTokenVectors({ tensor, tokens }: { tensor: Tensor; tokens: string[] }) {
  const [seqLen, hidden] = tensor.shape;
  const previewCount = Math.min(PER_TOKEN_PREVIEW_DIMS, hidden);
  return (
    <div className="per-token-vectors">
      {Array.from({ length: seqLen }, (_, i) => {
        const rowStart = i * hidden;
        const values: string[] = [];
        for (let d = 0; d < previewCount; d++) values.push(tensor.data[rowStart + d].toFixed(4));
        return (
          <div key={i} className="per-token-row">
            <span className="per-token-label">"{tokens[i] || "·"}"</span>
            <span className="per-token-arrow">→</span>
            <span className="per-token-vector">
              [{values.join(", ")}
              {hidden > previewCount ? `, …` : ""}]
            </span>
            {hidden > previewCount && (
              <span className="per-token-dims">
                +{hidden - previewCount} more · {hidden} dims
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

function RawGrid({ tensor }: { tensor: Tensor }) {
  const cols = tensor.shape.length === 2 ? tensor.shape[1] : tensor.shape[0];
  return (
    <div className="raw-grid-scroll">
      <table className="raw-grid">
        <tbody>
          {Array.from({ length: Math.ceil(tensor.data.length / cols) }, (_, r) => (
            <tr key={r}>
              {Array.from({ length: cols }, (_, c) => {
                const v = tensor.data[r * cols + c];
                return <td key={c}>{v.toFixed(3)}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function WindowControls({
  shape,
  ranges,
  onChange,
}: {
  shape: number[];
  ranges: { start: number; end: number }[];
  onChange: (ranges: { start: number; end: number }[]) => void;
}) {
  const clamp = (v: number, dim: number) => Math.max(0, Math.min(dim, Math.round(Number.isFinite(v) ? v : 0)));

  return (
    <div className="window-controls">
      {shape.map((dim, i) => (
        <label key={i} className="window-dim">
          dim {i} (0–{dim})
          <input
            type="number"
            value={ranges[i]?.start ?? 0}
            min={0}
            max={dim}
            onChange={(e) => {
              const next = ranges.slice();
              const start = clamp(Number(e.target.value), dim);
              const end = Math.max(start, next[i]?.end ?? dim);
              next[i] = { start, end: clamp(end, dim) };
              onChange(next);
            }}
          />
          <span className="window-dash">–</span>
          <input
            type="number"
            value={ranges[i]?.end ?? dim}
            min={0}
            max={dim}
            onChange={(e) => {
              const next = ranges.slice();
              const start = next[i]?.start ?? 0;
              const end = Math.max(start, clamp(Number(e.target.value), dim));
              next[i] = { start, end };
              onChange(next);
            }}
          />
        </label>
      ))}
    </div>
  );
}
