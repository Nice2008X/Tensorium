import type { ReactNode } from "react";
import type { ActivationCapture, Model, ModelNode } from "@tensorium/model-ir";
import { totalParameterCount } from "@tensorium/model-ir";
import type { Tokenizer } from "@tensorium/tokenizer";
import { componentRegistry } from "../registry.js";
import { topKFromLogits } from "../logits.js";
import { formatBytes, formatCount, formatPercent } from "../format.js";
import { useLocalStorageState } from "../useLocalStorageState.js";
import { describeInputConstruction } from "../nodeInputs.js";
import { useTranslation } from "./LanguageContext.js";

function formatDims(dims: Array<number | string>): string {
  return `[${dims.join(", ")}]`;
}

interface Props {
  model: Model;
  node: ModelNode | null;
  /** The selected node's real captured activation shape/magnitude from the last run — undefined when no run has happened yet, or this node has no recorded activation (e.g. a purely organizational container). */
  activationShape?: number[];
  activationMagnitude?: number;
  onViewActivation?: () => void;
  onViewWeights?: () => void;
  /** Deep-links into Tensor Explorer's Input/Output tab, pre-selected to this specific input source (by id) or to the node's own output — the "This run" section's replacement for showing a heatmap inline. */
  onViewInput?: (sourceId: string) => void;
  onViewOutput?: () => void;
  /** The last completed forward pass (Prompt A), if any — feeds the "Last run" section of the no-selection overview below (and the root node's own view, which folds the same overview in — see `isRoot`). Not threaded any further than that: once a non-root node is selected, its own activation info comes from `activationShape`/`activationMagnitude` instead. */
  inferenceResult?: ActivationCapture;
  tokenizer?: Tokenizer;
  /** Real wall-clock duration of that forward pass, in ms — see useInference. Undefined for a result restored some other way (there isn't one today, but keeps this honestly optional rather than defaulting to 0). */
  elapsedMs?: number;
  /** Clears the selection, returning to the no-selection Model overview — surfaced as a back link once a node is selected, since Esc (the only other way back) isn't discoverable from the Inspector itself. */
  onDeselect?: () => void;
}

export function Inspector({
  model,
  node,
  activationShape,
  activationMagnitude,
  onViewActivation,
  onViewWeights,
  onViewInput,
  onViewOutput,
  inferenceResult,
  tokenizer,
  elapsedMs,
  onDeselect,
}: Props) {
  const { t } = useTranslation();
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
  const { sources: inputSources, operator: inputOperator } = describeInputConstruction(model, node, t);
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
          {t("inspector.backToOverview")}
        </button>
      )}
      <div className="inspector-title" style={{ borderColor: info.color }}>
        <span className="inspector-badge" style={{ background: info.color }}>
          {info.label}
        </span>
        <span className="inspector-name">{node.name}</span>
      </div>

      <Section title={t("inspector.whatIsIt")}>
        <p>{info.description}</p>
      </Section>

      {isRoot && <ModelSummarySections model={model} inferenceResult={inferenceResult} tokenizer={tokenizer} elapsedMs={elapsedMs} />}

      {hasThisRun && (
        <Section title={t("inspector.thisRun")} collapsible collapsed={thisRunCollapsed} onToggleCollapsed={() => setThisRunCollapsed((v) => !v)}>
          {inputSources.length > 0 && (
            <div className="inspector-io-group">
              <div className="inspector-io-group-title">{inputSources.length > 1 ? t("inspector.inputs") : t("inspector.input")}</div>
              {inputSources.map((s, i) => (
                <div key={i} className="inspector-io-item">
                  <div className="inspector-io-item-label" title={s.label}>
                    {s.label}
                  </div>
                  {/* Detail (heatmap/matrix/histogram/stats) lives one click
                      away in Tensor Explorer's Input/Output tab rather than
                      inline here — this row is just "does this run actually
                      have a captured value" plus a way in. */}
                  {inferenceResult?.activations[s.sourceId] ? (
                    onViewInput && (
                      <button type="button" className="inspector-io-link" onClick={() => onViewInput(s.sourceId)}>
                        {t("inspector.viewMatrix")}
                      </button>
                    )
                  ) : (
                    <span className="inspector-tensor-empty">{t("inspector.notCaptured")}</span>
                  )}
                </div>
              ))}
            </div>
          )}

          <div className="inspector-io-group">
            <div className="inspector-io-group-title">{t("inspector.output")}</div>
            <div className="inspector-io-item">
              {inferenceResult?.activations[node.id] ? (
                onViewOutput && (
                  <button type="button" className="inspector-io-link" onClick={onViewOutput}>
                    {t("inspector.viewMatrix")}
                  </button>
                )
              ) : (
                <span className="inspector-tensor-empty">{t("inspector.notCaptured")}</span>
              )}
            </div>
          </div>

          <div className="io-row">
            <span className="io-label">{t("inspector.activationShape")}</span>
            <span className="io-shape">{formatDims(activationShape!)}</span>
          </div>
          <div className="io-row">
            <span className="io-label">{t("inspector.magnitudeL2")}</span>
            <span className="io-shape">{activationMagnitude!.toFixed(4)}</span>
          </div>
        </Section>
      )}

      {inputSources.length > 0 && (
        <Section title={t("inspector.inputConstruction")}>
          {inputOperator ? (
            <code className="formula">
              {t("inspector.shapesInput")} = {inputSources.map((s) => s.label).join(` ${inputOperator} `)}
            </code>
          ) : inputSources.length === 1 ? (
            <code className="formula">
              {t("inspector.shapesInput")} = {inputSources[0].label}
            </code>
          ) : (
            <>
              <p>{t("inspector.assembledFromMultiple")}</p>
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
              {t("inspector.viewActivation")}
            </button>
          )}
          {node.parameters.length > 0 && onViewWeights && (
            <button type="button" onClick={onViewWeights}>
              {t("inspector.viewWeights")}
            </button>
          )}
        </div>
      )}

      {info.formula && (
        <Section title={t("inspector.showMeTheMath")}>
          <code className="formula">{info.formula}</code>
        </Section>
      )}

      {(node.inputs.length > 0 || node.outputs.length > 0) && (
        <Section title={t("inspector.shapes")}>
          {node.inputs.map((s, i) => (
            <div key={`in-${i}`} className="io-row">
              <span className="io-label">{t("inspector.shapesInput")}</span>
              <span className="io-shape">{formatDims(s.dims)}</span>
            </div>
          ))}
          {node.outputs.map((s, i) => (
            <div key={`out-${i}`} className="io-row">
              <span className="io-label">{t("inspector.shapesOutput")}</span>
              <span className="io-shape">{formatDims(s.dims)}</span>
            </div>
          ))}
        </Section>
      )}

      {node.parameters.length > 0 && (
        <Section title={t("inspector.parametersCount").replace("{n}", totalParams.toLocaleString())}>
          {node.parameters.map((p, i) => (
            <div key={i} className="io-row">
              <span className="io-label">{p.slice ? t("inspector.paramSlice").replace("{name}", p.name) : p.name}</span>
              <span className="io-shape">
                {p.logicalShape.join(" × ")} · {p.dtype}
              </span>
            </div>
          ))}
        </Section>
      )}

      {Object.keys(node.metadata).length > 0 && (
        <Section title={t("inspector.metadata")}>
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
  const { t } = useTranslation();
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
      <Section title={t("inspector.model")}>
        <div className="io-row">
          <span className="io-label">{t("inspector.architecture")}</span>
          <span className="io-shape">{model.architecture}</span>
        </div>
        <div className="io-row">
          <span className="io-label">{t("inspector.parameters")}</span>
          <span className="io-shape">{formatCount(totalParams)}</span>
        </div>
        <div className="io-row">
          <span className="io-label">{t("inspector.layers")}</span>
          <span className="io-shape">{model.config.numLayers}</span>
        </div>
      </Section>

      {groups.length > 0 && (
        <Section title={t("inspector.structure")}>
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

      <Section title={t("inspector.lastRun")}>
        {inferenceResult && tokenizer ? (
          <>
            <div className="io-row">
              <span className="io-label">{t("inspector.promptTokens")}</span>
              <span className="io-shape">{inferenceResult.tokenIds.length}</span>
            </div>
            <div className="io-row">
              <span className="io-label">{t("inspector.decoded")}</span>
              <span className="io-shape overview-prompt-preview" title={inferenceResult.tokens.join("")}>
                {inferenceResult.tokens.join("")}
              </span>
            </div>
            {elapsedMs !== undefined && (
              <div className="io-row">
                <span className="io-label">{t("inspector.forwardPass")}</span>
                <span className="io-shape">{elapsedMs < 1 ? t("inspector.lessThan1Ms") : t("inspector.msValue").replace("{ms}", elapsedMs.toFixed(1))}</span>
              </div>
            )}
            <div className="io-row">
              <span className="io-label">{t("inspector.memoryThisRun")}</span>
              <span className="io-shape">{formatBytes(captureBytes(inferenceResult))}</span>
            </div>
            {topPrediction && (
              <div className="io-row">
                <span className="io-label">{t("inspector.topPrediction")}</span>
                <span className="io-shape">
                  {tokenizer.decodeToken(topPrediction.tokenId).trim() || `#${topPrediction.tokenId}`} · {formatPercent(topPrediction.prob)}
                </span>
              </div>
            )}
          </>
        ) : (
          <p>{t("inspector.runForwardHint")}</p>
        )}
      </Section>
    </>
  );
}

/** The overview shown in place of "click a component" when nothing is selected — a model summary plus (once a prompt has been run) a snapshot of the last result, so the Inspector isn't dead space before the user picks a node. */
function ModelOverview({ model, inferenceResult, tokenizer, elapsedMs }: { model: Model; inferenceResult?: ActivationCapture; tokenizer?: Tokenizer; elapsedMs?: number }) {
  const { t } = useTranslation();
  return (
    <div className="inspector inspector-overview">
      <div className="inspector-overview-title">{t("inspector.modelOverview")}</div>
      <ModelSummarySections model={model} inferenceResult={inferenceResult} tokenizer={tokenizer} elapsedMs={elapsedMs} />
      <p className="overview-hint">{t("inspector.clickToInspectHint")}</p>
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
  const { t } = useTranslation();
  return (
    <div className="inspector-section">
      <div className="inspector-section-header">
        <div className="inspector-section-title">{title}</div>
        {collapsible && (
          <button type="button" className="inspector-section-collapse-btn" onClick={onToggleCollapsed} title={collapsed ? t("inspector.expand") : t("inspector.collapse")}>
            {collapsed ? "▸" : "▾"}
          </button>
        )}
      </div>
      {!collapsed && children}
    </div>
  );
}
