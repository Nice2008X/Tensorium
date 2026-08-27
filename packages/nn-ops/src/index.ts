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

/** Numerically stable softplus: log(1+e^x), computed directly from x for large x to avoid overflowing e^x first. Used by Gated DeltaNet-style decay gates (Qwen3-Next/Qwen3.5). */
export function softplus(x: number): number {
  return x > 20 ? x : Math.log1p(Math.exp(x));
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

/**
 * Causal depthwise 1D convolution over the sequence dimension, each
 * channel convolved independently against its own kernel, then SiLU —
 * the short-conv preprocessing a Gated DeltaNet layer (Qwen3-Next /
 * Qwen3.5-style hybrid attention) applies to its fused Q/K/V projection
 * before the recurrence. Matches `causal_conv1d_fn(..., activation="silu")`
 * in the real `transformers` implementation: left-padded with zeros (so
 * output[t] only ever depends on input[t-kernelWidth+1 .. t]), and
 * `kernel[c]` holds channel c's own taps in PyTorch conv1d's
 * cross-correlation order (`kernel[c][0]` weights the earliest input in
 * the window, `kernel[c][kernelWidth-1]` weights input[t] itself).
 */
export function causalConv1dSilu(x: Matrix, kernel: Matrix, bias: number[] | null): Matrix {
  const S = x.length;
  const C = x[0]?.length ?? 0;
  const width = kernel[0]?.length ?? 0;
  const out = zeros(S, C);
  for (let t = 0; t < S; t++) {
    for (let c = 0; c < C; c++) {
      let acc = bias ? bias[c] : 0;
      for (let w = 0; w < width; w++) {
        const srcT = t - (width - 1) + w;
        if (srcT >= 0) acc += x[srcT][c] * kernel[c][w];
      }
      out[t][c] = silu(acc);
    }
  }
  return out;
}

export interface GatedDeltaRuleResult {
  output: Matrix; // [S, numValueHeads * valueHeadDim]
}

/**
 * The sequential (non-chunked) form of the gated delta rule, ported
 * directly from `torch_recurrent_gated_delta_rule` in the real
 * `transformers` Qwen3-Next implementation (not derived independently —
 * confirmed against the actual source, statement order included). For
 * each head, maintains a `[keyHeadDim, valueHeadDim]` recurrent state
 * matrix and, at every timestep: decays the state by that step's `decay`
 * factor, computes `kv_mem = stateᵀk_t` (what the decayed state already
 * predicts for this key), corrects the state by the outer product of
 * `k_t` and `beta_t * (v_t - kv_mem)`, then reads the output out as
 * `stateᵀq_t`.
 *
 * `numKeyHeads` divides `numValueHeads`: each key/query/decay/beta head is
 * shared, GQA-style, across `numValueHeads / numKeyHeads` value heads —
 * matches Qwen3-Next/Qwen3.5's real head-count asymmetry (e.g. 16 key
 * heads, 48 value heads). `q`/`k` are laid out with `numKeyHeads` heads of
 * `keyHeadDim` each (query shares key's head layout in this architecture,
 * there's no separate query head_dim); `v`/`decay`/`beta` are laid out
 * with `numValueHeads` heads.
 *
 * `decay` must already be the per-step multiplicative factor (i.e.
 * `exp(g_t)` in the real source, where `g_t = -exp(A_log) *
 * softplus(a_t + dt_bias)`) — computing that from the model's own
 * `A_log`/`dt_bias` parameters is left to the caller, since those are
 * architecture-specific learned parameters, not something this generic
 * primitive should know the name of.
 *
 * `q`/`k` are L2-normalized per head (over `keyHeadDim`) before use —
 * the real source always passes `use_qk_l2norm_in_kernel=True`, it's not
 * behind a config flag, so this primitive always does it too rather than
 * exposing a toggle nothing in this codebase would ever set to false.
 */
export function gatedDeltaRule(
  q: Matrix,
  k: Matrix,
  v: Matrix,
  decay: Matrix, // [S, numValueHeads], already exp(g_t)
  beta: Matrix, // [S, numValueHeads], already sigmoid(b_t)
  numKeyHeads: number,
  numValueHeads: number,
  keyHeadDim: number,
  valueHeadDim: number
): GatedDeltaRuleResult {
  const S = q.length;
  const groupSize = numValueHeads / numKeyHeads;
  const output = zeros(S, numValueHeads * valueHeadDim);
  const l2norm = (row: number[], off: number, len: number): number[] => {
    let sumSq = 0;
    for (let i = 0; i < len; i++) sumSq += row[off + i] * row[off + i];
    const denom = Math.sqrt(sumSq) + 1e-6;
    return Array.from({ length: len }, (_, i) => row[off + i] / denom);
  };

  for (let h = 0; h < numValueHeads; h++) {
    const kh = Math.floor(h / groupSize);
    const qOff = kh * keyHeadDim;
    const kOff = kh * keyHeadDim;
    const vOff = h * valueHeadDim;
    const state = zeros(keyHeadDim, valueHeadDim);

    for (let t = 0; t < S; t++) {
      const gt = decay[t][h];
      const betaT = beta[t][h];
      const qt = l2norm(q[t], qOff, keyHeadDim);
      const kt = l2norm(k[t], kOff, keyHeadDim);

      for (let i = 0; i < keyHeadDim; i++) for (let j = 0; j < valueHeadDim; j++) state[i][j] *= gt;

      const kvMem = new Array(valueHeadDim).fill(0);
      for (let i = 0; i < keyHeadDim; i++) {
        const ki = kt[i];
        for (let j = 0; j < valueHeadDim; j++) kvMem[j] += state[i][j] * ki;
      }

      const delta = new Array(valueHeadDim);
      for (let j = 0; j < valueHeadDim; j++) delta[j] = (v[t][vOff + j] - kvMem[j]) * betaT;

      for (let i = 0; i < keyHeadDim; i++) {
        const ki = kt[i];
        for (let j = 0; j < valueHeadDim; j++) state[i][j] += ki * delta[j];
      }

      for (let j = 0; j < valueHeadDim; j++) {
        let acc = 0;
        for (let i = 0; i < keyHeadDim; i++) acc += state[i][j] * qt[i];
        output[t][vOff + j] = acc;
      }
    }
  }

  return { output };
}

export * from "./intervene.js";
