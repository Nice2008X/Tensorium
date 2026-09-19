import type { ActivationCapture } from "@tensorium/model-ir";
import type { Tokenizer } from "@tensorium/tokenizer";
import { outputTokenLabel, topKFromLogits } from "../logits.js";
import { formatPercent } from "../format.js";
import { useTranslation } from "./LanguageContext.js";

interface Props {
  result: ActivationCapture;
  tokenizer: Tokenizer;
  /** Set for a sequence-classification model — output ids are then class indices, shown by label instead of decoded as tokens. */
  classLabels?: string[];
  selectedTokenIndex: number | null;
  onViewWhy: () => void;
  /** Lifted to App so the "maximize graph" control can collapse/expand this panel together with the tree/inspector/bottom panels, not just this panel's own toggle. */
  collapsed: boolean;
  onToggleCollapsed: () => void;
  /** Set when this panel is one of a side-by-side pair (Prompt A / Prompt B) — shown as a small tag next to the title so the two are distinguishable. Omitted in the single-prompt case, where there's nothing to disambiguate. */
  promptLabel?: string;
}

/**
 * The "what did the model predict" answer, front and center right after a
 * run — reuses topKFromLogits (already existed, only ever wired into
 * ExperimentPanel's before/after comparison) so this needs no new
 * computation, just a place to show it without digging into a bottom tab.
 */
export function PredictionPanel({ result, tokenizer, classLabels, selectedTokenIndex, onViewWhy, collapsed, onToggleCollapsed, promptLabel }: Props) {
  const { t } = useTranslation();
  // A stale selectedTokenIndex from a longer previous prompt (App only
  // resets it on model change, not on every re-run) would otherwise index
  // past this result's logits.
  const tokenIndex = Math.min(selectedTokenIndex ?? result.tokenIds.length - 1, result.tokenIds.length - 1);
  const ranked = topKFromLogits(result.logits, tokenIndex, 5);
  const maxProb = ranked[0]?.prob ?? 1;

  return (
    <div className="prediction-panel">
      <div className="prediction-header">
        <button
          type="button"
          className="prediction-collapse-btn"
          onClick={onToggleCollapsed}
          title={collapsed ? t("app.expandPanel") : t("app.collapsePanel")}
        >
          {collapsed ? "▸" : "▾"}
        </button>
        <span className="prediction-title">{t("prediction.title")}</span>
        {promptLabel && <span className="prediction-prompt-tag">{promptLabel}</span>}
        <span className="prediction-position">{t("prediction.position").replace("{n}", String(tokenIndex))}</span>
        <button type="button" className="prediction-why-link" onClick={onViewWhy}>
          {t("prediction.why")}
        </button>
      </div>
      {!collapsed && (
        <div className="prediction-rows">
          {ranked.map((r) => {
            const display = outputTokenLabel(tokenizer, classLabels, r.tokenId);
            return (
              <div key={r.tokenId} className="prediction-row">
                <span className="prediction-token">{display.trim() || `#${r.tokenId}`}</span>
                <div className="prediction-bar-track">
                  <div className="prediction-bar-fill" style={{ width: `${maxProb > 0 ? (r.prob / maxProb) * 100 : 0}%` }} />
                </div>
                <span className="prediction-pct">{formatPercent(r.prob)}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
