import { useEffect, useState } from "react";
import type { Model, ModelAdapter, WeightProvider } from "@tensorium/model-ir";
import { computeTokenAttribution, computeHeadAttribution, type TokenAttributionResult, type HeadAttributionResult } from "@tensorium/interpretability";
import type { Tokenizer } from "@tensorium/tokenizer";
import { useTranslation } from "./LanguageContext.js";

interface Props {
  model: Model;
  weightProvider: WeightProvider;
  adapter: ModelAdapter;
  tokenIds: number[];
  /** When Prompt B has also been run, lets this panel switch between the two instead of only ever attributing Prompt A's prediction. */
  promptBTokenIds?: number[];
  tokenizer: Tokenizer;
  /** Which position's next-token prediction to attribute toward — shared with the Prediction panel/prompt chips, defaults to the last position. */
  selectedTokenIndex: number | null;
  /** Which prompt's tokens to attribute — controlled by App so the Prediction panel's "Why?" link (Prompt A's or Prompt B's) can land on the matching side instead of always defaulting back to A. */
  source: "A" | "B";
  onSourceChange: (source: "A" | "B") => void;
  /** Clicking an influential-head row jumps the graph/tree/inspector to that block's Attention node. */
  onSelectNode: (nodeId: string) => void;
  /** Reported whenever this panel's own background computation starts/stops — lets the app show a busy cursor while it runs. */
  onBusyChange?: (busy: boolean) => void;
}

export function TokenAttributionPanel({ model, weightProvider, adapter, tokenIds, promptBTokenIds, tokenizer, selectedTokenIndex, source, onSourceChange, onSelectNode, onBusyChange }: Props) {
  const { t } = useTranslation();
  const activeTokenIds = source === "B" && promptBTokenIds ? promptBTokenIds : tokenIds;

  const [result, setResult] = useState<TokenAttributionResult | null>(null);
  const [headResult, setHeadResult] = useState<HeadAttributionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // A stale selectedTokenIndex from a longer previous prompt (App only
  // resets it on model change, not on every re-run) would otherwise index
  // past this prompt's logits.
  const predictIndex = Math.min(selectedTokenIndex ?? activeTokenIds.length - 1, activeTokenIds.length - 1);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    onBusyChange?.(true);
    Promise.all([
      computeTokenAttribution(model, weightProvider, adapter, activeTokenIds, { predictIndex }),
      computeHeadAttribution(model, weightProvider, adapter, activeTokenIds, { predictIndex }),
    ])
      .then(([tokenResult, heads]) => {
        if (!cancelled) {
          setResult(tokenResult);
          setHeadResult(heads);
          setLoading(false);
          onBusyChange?.(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
          setLoading(false);
          onBusyChange?.(false);
        }
      });
    return () => {
      cancelled = true;
      onBusyChange?.(false);
    };
  }, [model, weightProvider, adapter, activeTokenIds, predictIndex]);

  const sourceToggle = promptBTokenIds && (
    <div className="source-tabs">
      <button className={source === "A" ? "active" : ""} onClick={() => onSourceChange("A")}>
        {t("inference.promptA")}
      </button>
      <button className={source === "B" ? "active" : ""} onClick={() => onSourceChange("B")}>
        {t("inference.promptB")}
      </button>
    </div>
  );

  if (loading)
    return (
      <div className="token-attribution">
        {sourceToggle}
        <div className="loading-hint">
          <span className="spinner" />
          Occluding each token and attention head in turn and re-running…
        </div>
      </div>
    );
  if (error)
    return (
      <div className="token-attribution">
        {sourceToggle}
        <div className="inference-error">{error}</div>
      </div>
    );
  if (!result || !headResult) return sourceToggle ? <div className="token-attribution">{sourceToggle}</div> : null;

  const maxAbs = Math.max(...result.entries.map((e) => Math.abs(e.logitDrop)), 1e-9);
  const targetDisplay = tokenizer.decodeToken(result.targetTokenId) || `#${result.targetTokenId}`;
  const positionNote = predictIndex !== activeTokenIds.length - 1 ? ` at position ${predictIndex}` : "";

  const topHeads = [...headResult.entries].sort((a, b) => Math.abs(b.logitDrop) - Math.abs(a.logitDrop)).slice(0, 8);
  const maxHeadAbs = Math.max(...topHeads.map((e) => Math.abs(e.logitDrop)), 1e-9);

  return (
    <div className="token-attribution">
      {sourceToggle}
      <div className="attribution-intro">
        Occlusion attribution toward predicting <strong>"{targetDisplay}"</strong>{positionNote} (the model's actual top prediction, logit {result.baselineLogit.toFixed(4)}).
        Each bar shows how much removing that token or head <em>hurt</em> (blue, right) or <em>helped</em> (red, left) the prediction.
      </div>
      <div className="attribution-section-title">Most influential tokens</div>
      <div className="attribution-bars">
        {result.entries.map((e) => {
          const displayToken = tokenizer.decodeToken(activeTokenIds[e.tokenIndex]) || `#${activeTokenIds[e.tokenIndex]}`;
          const widthPct = (Math.abs(e.logitDrop) / maxAbs) * 50;
          return (
            <div key={e.tokenIndex} className="attribution-row">
              <span className="attribution-token">{displayToken.trim() || "·"}</span>
              <div className="attribution-track">
                <div className="attribution-center" />
                {e.logitDrop >= 0 ? (
                  <div className="attribution-fill attribution-positive" style={{ width: `${widthPct}%` }} />
                ) : (
                  <div className="attribution-fill attribution-negative" style={{ width: `${widthPct}%` }} />
                )}
              </div>
              <span className="attribution-value">{e.logitDrop >= 0 ? "+" : ""}{e.logitDrop.toFixed(4)}</span>
            </div>
          );
        })}
      </div>

      <div className="attribution-section-title">
        Most influential heads
        {headResult.truncated && <span className="attribution-truncated-note"> (showing the first 64 of more combinations in this model)</span>}
      </div>
      <div className="attribution-bars">
        {topHeads.map((e) => {
          const widthPct = (Math.abs(e.logitDrop) / maxHeadAbs) * 50;
          return (
            <button key={`${e.nodeId}:${e.headIndex}`} className="attribution-row attribution-row-clickable" onClick={() => onSelectNode(e.nodeId)}>
              <span className="attribution-token">{e.blockLabel} · Head {e.headIndex}</span>
              <div className="attribution-track">
                <div className="attribution-center" />
                {e.logitDrop >= 0 ? (
                  <div className="attribution-fill attribution-positive" style={{ width: `${widthPct}%` }} />
                ) : (
                  <div className="attribution-fill attribution-negative" style={{ width: `${widthPct}%` }} />
                )}
              </div>
              <span className="attribution-value">{e.logitDrop >= 0 ? "+" : ""}{e.logitDrop.toFixed(4)}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
