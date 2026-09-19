import { useEffect, useState } from "react";
import type { Model, WeightProvider, ActivationCapture } from "@tensorium/model-ir";
import { computeLogitLens, type LogitLensEntry } from "@tensorium/interpretability";
import type { Tokenizer } from "@tensorium/tokenizer";
import { formatPercent } from "../format.js";
import { outputTokenLabel } from "../logits.js";
import { useTranslation } from "./LanguageContext.js";

interface Props {
  model: Model;
  weightProvider: WeightProvider;
  capture: ActivationCapture;
  /** When Prompt B has also been run, lets this panel switch between the two instead of only ever showing Prompt A's. */
  promptBCapture?: ActivationCapture;
  tokenizer: Tokenizer;
  /** Shared with the rest of the app (prompt token chips, Prediction panel, Token Attribution) — clicking a token anywhere moves this same position, instead of each panel tracking its own. */
  selectedTokenIndex: number | null;
  onSelectToken: (i: number) => void;
  /** Reported whenever this panel's own background computation starts/stops — lets the app show a busy cursor while it runs. */
  onBusyChange?: (busy: boolean) => void;
}

export function LogitLensPanel({ model, weightProvider, capture, promptBCapture, tokenizer, selectedTokenIndex, onSelectToken, onBusyChange }: Props) {
  const { t } = useTranslation();
  const [source, setSource] = useState<"A" | "B">("A");
  const activeCapture = source === "B" && promptBCapture ? promptBCapture : capture;

  // A stale selectedTokenIndex from a longer previous prompt (App only
  // resets it on model change, not on every re-run) would otherwise index
  // past this capture's logits.
  const tokenIndex = Math.min(selectedTokenIndex ?? activeCapture.tokenIds.length - 1, activeCapture.tokenIds.length - 1);
  const [layers, setLayers] = useState<LogitLensEntry[] | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    onBusyChange?.(true);
    computeLogitLens(model, weightProvider, activeCapture, { tokenIndex, topK: 5 }).then((result) => {
      if (!cancelled) {
        setLayers(result);
        setLoading(false);
        onBusyChange?.(false);
      }
    });
    return () => {
      cancelled = true;
      onBusyChange?.(false);
    };
  }, [model, weightProvider, activeCapture, tokenIndex]);

  const displayTokens = activeCapture.tokenIds.map((id) => tokenizer.decodeToken(id));

  return (
    <div className="logit-lens">
      {promptBCapture && (
        <div className="source-tabs">
          <button className={source === "A" ? "active" : ""} onClick={() => setSource("A")}>
            {t("inference.promptA")}
          </button>
          <button className={source === "B" ? "active" : ""} onClick={() => setSource("B")}>
            {t("inference.promptB")}
          </button>
        </div>
      )}
      <div className="logit-lens-header">
        <span>
          Predicting the token <em>after</em>:
        </span>
        <div className="token-chips">
          {displayTokens.map((t, i) => (
            <button key={i} className={"token-chip" + (i === tokenIndex ? " selected" : "")} onClick={() => onSelectToken(i)}>
              {t.trim() === "" ? "·".repeat(Math.max(1, t.length)) : t}
            </button>
          ))}
        </div>
      </div>

      {loading && (
        <div className="loading-hint">
          <span className="spinner" />
          Projecting each layer through the LM head…
        </div>
      )}

      {!loading && layers && (
        <div className="logit-lens-layers">
          {layers.map((layer) => (
            <div key={layer.nodeId} className="logit-lens-row">
              <div className="logit-lens-label">{layer.label}</div>
              <div className="logit-lens-bars">
                {layer.topTokens.map((t, i) => (
                  <div key={i} className="logit-lens-token" style={{ opacity: 0.4 + 0.6 * t.prob }} title={formatPercent(t.prob)}>
                    <span className="logit-lens-token-text">{outputTokenLabel(tokenizer, model.config.classLabels, t.tokenId) || `#${t.tokenId}`}</span>
                    <span className="logit-lens-token-prob">{formatPercent(t.prob)}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
