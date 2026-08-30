import { useState } from "react";
import type { Tensor } from "@tensorium/model-ir";

interface Props {
  attentionWeights: Tensor; // [numHeads, seqLen, seqLen]
  tokens: string[];
  queryTokenIndex: number;
}

export function AttentionView({ attentionWeights, tokens, queryTokenIndex }: Props) {
  const [head, setHead] = useState(0);
  const [numHeads, seqLen] = attentionWeights.shape;
  const h = Math.min(head, numHeads - 1);
  const q = Math.min(queryTokenIndex, seqLen - 1);

  const weights: number[] = [];
  for (let j = 0; j < seqLen; j++) {
    weights.push(attentionWeights.data[h * seqLen * seqLen + q * seqLen + j]);
  }
  const max = Math.max(...weights, 1e-9);

  return (
    <div className="attention-view">
      <div className="attention-header">
        <span>
          Attention from token <strong>{tokens[q] || `#${q}`}</strong> (position {q}) to each earlier token
        </span>
        {numHeads > 1 && (
          <div className="head-tabs">
            {Array.from({ length: numHeads }, (_, i) => (
              <button key={i} className={i === h ? "active" : ""} onClick={() => setHead(i)}>
                head {i}
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
