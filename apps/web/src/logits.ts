import type { Tensor } from "@tensorium/model-ir";

/** Display text for an output id: the class label for a sequence-classification model, otherwise the decoded vocabulary token. */
export function outputTokenLabel(tokenizer: { decodeToken(id: number): string }, classLabels: string[] | undefined, id: number): string {
  return classLabels ? (classLabels[id] ?? `#${id}`) : tokenizer.decodeToken(id);
}

export interface RankedToken {
  tokenId: number;
  prob: number;
}

/** Softmaxes one sequence position of a [sequence, vocab] logits tensor and returns the top-k tokens. */
export function topKFromLogits(logits: Tensor, tokenIndex: number, k = 5): RankedToken[] {
  const vocab = logits.shape[1];
  const row = Array.from(logits.data.slice(tokenIndex * vocab, (tokenIndex + 1) * vocab));
  // Not `Math.max(...row)`: spreading into a function call passes every
  // element as an individual argument, and V8's argument-count ceiling can
  // throw "Maximum call stack size exceeded" for a large-vocab model's
  // logits row (real GLM-4/Qwen-class vocabularies run 150K+ tokens) — see
  // nn-ops' softmaxRow, which hit the exact same bug.
  let max = -Infinity;
  for (const v of row) if (v > max) max = v;
  const exps = row.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return row
    .map((_, tokenId) => ({ tokenId, prob: exps[tokenId] / sum }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, k);
}
