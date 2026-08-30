import { useState } from "react";
import type { Tensor } from "@tensorium/model-ir";
import { useTranslation } from "./LanguageContext.js";

interface Props {
  attentionWeights: Tensor; // [numHeads, seqLen, seqLen]
  tokens: string[];
  queryTokenIndex: number;
}

export function AttentionView({ attentionWeights, tokens, queryTokenIndex }: Props) {
  const { t } = useTranslation();
  const [head, setHead] = useState(0);
  const [numHeads, seqLen] = attentionWeights.shape;
  const h = Math.min(head, numHeads - 1);
  const q = Math.min(queryTokenIndex, seqLen - 1);

  const weights: number[] = [];
  for (let j = 0; j < seqLen; j++) {
    weights.push(attentionWeights.data[h * seqLen * seqLen + q * seqLen + j]);
  }
  const max = Math.max(...weights, 1e-9);

  // "{token}" splits the translated sentence around the <strong>-wrapped
  // token so the emphasis survives translation regardless of where the
  // token falls in that language's word order.
  const [headingBefore, headingAfter] = t("attention.heading").replace("{q}", String(q)).split("{token}");

  return (
    <div className="attention-view">
      <div className="attention-header">
        <span>
          {headingBefore}
          <strong>{tokens[q] || `#${q}`}</strong>
          {headingAfter}
        </span>
        {numHeads > 1 && (
          <div className="head-tabs">
            {Array.from({ length: numHeads }, (_, i) => (
              <button key={i} className={i === h ? "active" : ""} onClick={() => setHead(i)}>
                {t("attention.head").replace("{i}", String(i))}
              </button>
            ))}
          </div>
        )}
      </div>
      <div className="attention-bars">
        {weights.map((w, j) => (
          <div key={j} className="attention-bar-row">
            <span className="attention-bar-token" title={tokens[j]}>
              {tokens[j]?.trim() || "·"}
            </span>
            <div className="attention-bar-track">
              <div className="attention-bar-fill" style={{ width: `${(w / max) * 100}%` }} />
            </div>
            <span className="attention-bar-value">{w.toFixed(4)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
