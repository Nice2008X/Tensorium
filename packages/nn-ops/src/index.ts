// Small, dependency-free numeric primitives for running an actual forward
// pass over an already-loaded checkpoint. Works on plain `number[][]`
// ([sequence][dim]) rather than typed arrays — these are tiny debug models,
// so readability/correctness matters far more than raw throughput here.
//
// Every formula below is written to match the specific HF `transformers`
// implementation it stands in for (not just "a" GELU, "a" RMSNorm, etc.),
// since this is meant to reproduce real model behavior for debugging, not
// just something GELU-shaped.

import type { Tensor } from "@tensorium/model-ir";

export type Matrix = number[][]; // [rows][cols]

export function tensorToMatrix(t: Tensor): Matrix {
  if (t.shape.length !== 2) throw new Error(`tensorToMatrix expects a 2D tensor, got shape [${t.shape.join(",")}]`);
  const [rows, cols] = t.shape;
  const out: Matrix = [];
  for (let r = 0; r < rows; r++) {
    const row = new Array(cols);
    for (let c = 0; c < cols; c++) row[c] = t.data[r * cols + c];
    out.push(row);
  }
  return out;
}

export function tensorToVector(t: Tensor): number[] {
  if (t.shape.length !== 1) throw new Error(`tensorToVector expects a 1D tensor, got shape [${t.shape.join(",")}]`);
  return Array.from(t.data);
}

export function matrixToTensor(m: Matrix, dtype = "F32"): Tensor {
  const rows = m.length;
  const cols = rows > 0 ? m[0].length : 0;
  const data = new Float64Array(rows * cols);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) data[r * cols + c] = m[r][c];
  return { shape: [rows, cols], dtype, data };
}

export function zeros(rows: number, cols: number): Matrix {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

export function addMatrices(a: Matrix, b: Matrix): Matrix {
  return a.map((row, r) => row.map((v, c) => v + b[r][c]));
}

export function mulMatricesElementwise(a: Matrix, b: Matrix): Matrix {
  return a.map((row, r) => row.map((v, c) => v * b[r][c]));
}

/**
 * x: [S, in]. `layout` reflects how the weight tensor is actually stored:
 *  - "out_in": standard nn.Linear, W is [out, in], computes x @ W^T (+ bias)
 *  - "in_out": GPT-2's Conv1D, W is [in, out], computes x @ W (+ bias)
 */
export function linear(x: Matrix, W: Matrix, bias: number[] | null, layout: "out_in" | "in_out"): Matrix {
  const S = x.length;
  const inDim = x[0]?.length ?? 0;
  const outDim = layout === "out_in" ? W.length : W[0].length;
  const out = zeros(S, outDim);
  for (let s = 0; s < S; s++) {
    for (let o = 0; o < outDim; o++) {
      let acc = bias ? bias[o] : 0;
      if (layout === "out_in") {
        const wRow = W[o]; // [in]
        for (let i = 0; i < inDim; i++) acc += x[s][i] * wRow[i];
      } else {
        for (let i = 0; i < inDim; i++) acc += x[s][i] * W[i][o];
      }
      out[s][o] = acc;
    }
  }
  return out;
}

/** PyTorch-style LayerNorm: biased variance, learned scale + shift. */
export function layerNorm(x: Matrix, gamma: number[], beta: number[], eps: number): Matrix {
  return x.map((row) => {
    const n = row.length;
    const mean = row.reduce((a, b) => a + b, 0) / n;
    const variance = row.reduce((a, b) => a + (b - mean) * (b - mean), 0) / n;
    const denom = Math.sqrt(variance + eps);
    return row.map((v, i) => ((v - mean) / denom) * gamma[i] + beta[i]);
  });
}

/** LlamaRMSNorm: no re-centering, no bias. */
export function rmsNorm(x: Matrix, gamma: number[], eps: number): Matrix {
  return x.map((row) => {
    const n = row.length;
    const meanSq = row.reduce((a, b) => a + b * b, 0) / n;
    const scale = 1 / Math.sqrt(meanSq + eps);
    return row.map((v, i) => v * scale * gamma[i]);
  });
}

/**
 * GemmaRMSNorm: same normalization as rmsNorm, but scales by (1 + weight)
 * instead of weight directly — Gemma's weight is zero-initialized, so
 * "no-op" is weight=0, not weight=1 like every other RMSNorm variant.
 * Using the standard `rmsNorm` here would silently zero out the signal.
 */
export function gemmaRmsNorm(x: Matrix, gamma: number[], eps: number): Matrix {
  return x.map((row) => {
    const n = row.length;
    const meanSq = row.reduce((a, b) => a + b * b, 0) / n;
    const scale = 1 / Math.sqrt(meanSq + eps);
    return row.map((v, i) => v * scale * (1 + gamma[i]));
  });
}

/** Multiplies every row by a constant — Gemma scales token embeddings by sqrt(hidden_size) right after lookup. */
export function scaleMatrix(x: Matrix, factor: number): Matrix {
  return x.map((row) => row.map((v) => v * factor));
}

/** Element-wise clamp to [min, max] — OLMo's optional clip_qkv clips Q/K/V projections right after they're computed, before RoPE. */
export function clamp(x: Matrix, min: number, max: number): Matrix {
  return x.map((row) => row.map((v) => Math.min(max, Math.max(min, v))));
}

/** HF's "gelu_new" / gelu_pytorch_tanh approximation — what GPT-2 actually uses. */
export function geluNew(x: number): number {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
}

export function gelu(x: number): number {
  // exact erf-based GELU
  return 0.5 * x * (1 + erf(x / Math.SQRT2));
}

function erf(x: number): number {
  // Abramowitz & Stegun 7.1.26 approximation, accurate to ~1.5e-7
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
  return sign * y;
}

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function silu(x: number): number {
  return x / (1 + Math.exp(-x));
}

export function relu(x: number): number {
  return Math.max(0, x);
}

export function applyActivation(x: Matrix, kind: string): Matrix {
  const fn = kind.includes("new") || kind === "gelu_pytorch_tanh" ? geluNew : kind === "silu" || kind === "swish" ? silu : kind === "relu" ? relu : gelu;
  return x.map((row) => row.map(fn));
}

export function softmaxRow(row: number[]): number[] {
  // Not `Math.max(...row)`: spreading into a function call passes every
  // element as an individual argument, and V8's argument-count ceiling
  // (not a fixed constant — it depends on how much call stack is already in
  // use) can throw "Maximum call stack size exceeded" for a large-vocab
  // model's logits row (real GLM-4/Qwen-class vocabularies run 150K+
  // tokens). A plain loop has no such limit.
  let max = -Infinity;
  for (const v of row) if (v > max) max = v;
  const exps = row.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

/** Indices of the k largest values, sorted descending — a Mixture-of-Experts router's "pick the top-k experts" step. */
export function topKIndices(values: number[], k: number): number[] {
  return values
    .map((v, i) => ({ v, i }))
    .sort((a, b) => b.v - a.v)
    .slice(0, k)
    .map(({ i }) => i);
}

export function embed(tokenIds: number[], table: Matrix): Matrix {
  return tokenIds.map((id) => table[id].slice());
}

/** RoPE cos/sin tables, HF's "rotate_half" convention: freqs duplicated across both halves of head_dim. */
export function ropeCosSin(seqLen: number, headDim: number, theta: number): { cos: Matrix; sin: Matrix } {
  const half = headDim / 2;
  const invFreq = Array.from({ length: half }, (_, i) => 1 / theta ** ((2 * i) / headDim));
  const cos: Matrix = [];
  const sin: Matrix = [];
  for (let pos = 0; pos < seqLen; pos++) {
    const freqs = invFreq.map((f) => pos * f);
    const full = [...freqs, ...freqs];
    cos.push(full.map(Math.cos));
    sin.push(full.map(Math.sin));
  }
  return { cos, sin };
}

function rotateHalf(row: number[]): number[] {
  const half = row.length / 2;
  const x1 = row.slice(0, half);
  const x2 = row.slice(half);
  return [...x2.map((v) => -v), ...x1];
}

/** Applies RoPE to one head's [S, headDim] slice in place semantics (returns a new matrix). */
export function applyRopeToHead(x: Matrix, cos: Matrix, sin: Matrix): Matrix {
  return x.map((row, pos) => {
    const rotated = rotateHalf(row);
    return row.map((v, i) => v * cos[pos][i] + rotated[i] * sin[pos][i]);
  });
}

function sliceHead(x: Matrix, headIdx: number, headDim: number): Matrix {
  return x.map((row) => row.slice(headIdx * headDim, (headIdx + 1) * headDim));
}

function matmul(a: Matrix, bT: Matrix): Matrix {
  // a: [m, k], bT: [n, k] (i.e. b transposed) -> [m, n]
  const m = a.length;
  const n = bT.length;
  const k = a[0]?.length ?? 0;
  const out = zeros(m, n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let acc = 0;
      for (let p = 0; p < k; p++) acc += a[i][p] * bT[j][p];
      out[i][j] = acc;
    }
  }
  return out;
}

export interface AttentionResult {
  output: Matrix; // [S, numHeads * headDim]
  /** attentionWeights[head][query_pos][key_pos], softmax already applied. */
  attentionWeights: number[][][];
}

/**
 * Scaled dot-product self-attention with a causal mask, GQA-aware (K/V
 * heads are repeated to match the number of query heads when
 * numKeyValueHeads < numHeads).
 *
 * `scale` defaults to the usual `1/sqrt(headDim)` — pass an explicit value
 * for a model that deliberately departs from it (Gemma-4's text decoder
 * fixes it at 1.0, relying on its Q/K RMSNorm instead). `slidingWindow`,
 * when given, additionally masks out any key more than that many positions
 * behind the query — on top of, not instead of, the causal mask — for a
 * local/global hybrid attention pattern (also Gemma-4).
 */
export function causalSelfAttention(
  q: Matrix,
  k: Matrix,
  v: Matrix,
  numHeads: number,
  numKeyValueHeads: number,
  headDim: number,
  options?: { scale?: number; slidingWindow?: number }
): AttentionResult {
  const S = q.length;
  const groupSize = numHeads / numKeyValueHeads;
  const scale = options?.scale ?? 1 / Math.sqrt(headDim);
  const slidingWindow = options?.slidingWindow;
  const perHeadOutputs: Matrix[] = [];
  const attentionWeights: number[][][] = [];

  for (let h = 0; h < numHeads; h++) {
    const kvHead = Math.floor(h / groupSize);
    const qh = sliceHead(q, h, headDim);
    const kh = sliceHead(k, kvHead, headDim);
    const vh = sliceHead(v, kvHead, headDim);

    const scores = matmul(qh, kh).map((row) => row.map((v2) => v2 * scale));
    for (let i = 0; i < S; i++) {
      for (let j = 0; j < S; j++) {
        const outOfWindow = slidingWindow != null && i - j >= slidingWindow;
        if (j > i || outOfWindow) scores[i][j] = -Infinity; // causal mask, plus the sliding window's lower bound when set
      }
    }
    const weights = scores.map(softmaxRow);
    attentionWeights.push(weights);

    const headOut = zeros(S, headDim);
    for (let i = 0; i < S; i++) {
      for (let j = 0; j <= i; j++) {
        const w = weights[i][j];
        for (let d = 0; d < headDim; d++) headOut[i][d] += w * vh[j][d];
      }
    }
    perHeadOutputs.push(headOut);
  }

  const output = zeros(S, numHeads * headDim);
  for (let s = 0; s < S; s++) {
    for (let h = 0; h < numHeads; h++) {
      for (let d = 0; d < headDim; d++) output[s][h * headDim + d] = perHeadOutputs[h][s][d];
    }
  }

  return { output, attentionWeights };
}

export * from "./intervene.js";
