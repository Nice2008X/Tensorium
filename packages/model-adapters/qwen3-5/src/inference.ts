import type { ActivationCapture, InferenceProgress, Intervention, Model, Tensor, WeightProvider } from "@tensorium/model-ir";
import {
  addMatrices,
  applyActivation,
  applyHeadIntervention,
  applyInterventions,
  applyRopeToHead,
  causalConv1dSilu,
  causalSelfAttention,
  embed,
  gatedDeltaRule,
  linear,
  matrixToTensor,
  mulMatricesElementwise,
  ropeCosSin,
  rmsNorm,
  sigmoid,
  silu,
  softplus,
  tensorToMatrix,
  tensorToVector,
  type Matrix,
} from "@tensorium/nn-ops";

/** Qwen3.5/3.8's QK-Norm: the same [head_dim] RMSNorm weight applied independently to each head's slice — same convention every other QK-Norm model in this app uses (Qwen3, DeepSeek-V2, Gemma-4). */
function applyNormPerHead(x: Matrix, numHeads: number, headDim: number, gamma: number[], eps: number): Matrix {
  const S = x.length;
  const out: Matrix = Array.from({ length: S }, () => new Array(numHeads * headDim));
  for (let h = 0; h < numHeads; h++) {
    const headSlice = x.map((row) => row.slice(h * headDim, (h + 1) * headDim));
    const normed = rmsNorm(headSlice, gamma, eps);
    for (let s = 0; s < S; s++) {
      for (let d = 0; d < headDim; d++) out[s][h * headDim + d] = normed[s][d];
    }
  }
  return out;
}

/** Rotates only the first `rotaryDim` of each head's `headDim` dims (partial_rotary_factor < 1) — the rest pass through unrotated. Same "slice, rotate, concat" convention as GLM-4's real apply_rotary_pos_emb, confirmed against Qwen3.5's own source too (not Gemma-4's proportional/zero-frequency variant). */
function applyRopePerHead(x: Matrix, numHeads: number, headDim: number, rotaryDim: number, cos: Matrix, sin: Matrix): Matrix {
  const S = x.length;
  const out: Matrix = Array.from({ length: S }, () => new Array(numHeads * headDim));
  for (let h = 0; h < numHeads; h++) {
    const headSlice = x.map((row) => row.slice(h * headDim, (h + 1) * headDim));
    const rotaryPart = headSlice.map((row) => row.slice(0, rotaryDim));
    const roped = rotaryDim > 0 ? applyRopeToHead(rotaryPart, cos, sin) : rotaryPart;
    for (let s = 0; s < S; s++) {
      for (let d = 0; d < rotaryDim; d++) out[s][h * headDim + d] = roped[s][d];
      for (let d = rotaryDim; d < headDim; d++) out[s][h * headDim + d] = headSlice[s][d];
    }
  }
  return out;
}

/**
 * Splits Q Projection's doubled output into the real query and the output
 * gate — per-head interleaved (`view(..., numHeads, headDim*2).chunk(2,
 * dim=-1)` in the real source), NOT a global first-half/second-half split:
 * head h's query lives at columns `[h*2*headDim, h*2*headDim+headDim)`,
 * its gate right after at `[h*2*headDim+headDim, (h+1)*2*headDim)`.
 */
function splitInterleavedGate(x: Matrix, numHeads: number, headDim: number): { query: Matrix; gate: Matrix } {
  const S = x.length;
  const query: Matrix = Array.from({ length: S }, () => new Array(numHeads * headDim));
  const gate: Matrix = Array.from({ length: S }, () => new Array(numHeads * headDim));
  for (let h = 0; h < numHeads; h++) {
    for (let s = 0; s < S; s++) {
      for (let d = 0; d < headDim; d++) {
        query[s][h * headDim + d] = x[s][h * headDim * 2 + d];
        gate[s][h * headDim + d] = x[s][h * headDim * 2 + headDim + d];
      }
    }
  }
  return { query, gate };
}

function sliceColumns(m: Matrix, start: number, end: number): Matrix {
  return m.map((row) => row.slice(start, end));
}

function applyElementwise(x: Matrix, fn: (v: number) => number): Matrix {
  return x.map((row) => row.map(fn));
}

function headsToTensor(headWeights: number[][][]): Tensor {
  const numHeads = headWeights.length;
  const S = headWeights[0]?.length ?? 0;
  const data = new Float64Array(numHeads * S * S);
  let idx = 0;
  for (let h = 0; h < numHeads; h++) {
    for (let i = 0; i < S; i++) {
      for (let j = 0; j < S; j++) {
        const w = headWeights[h][i][j];
        data[idx++] = Number.isFinite(w) ? w : 0;
      }
    }
  }
  return { shape: [numHeads, S, S], dtype: "F32", data };
}

const LP = "model.language_model";

export async function runInference(
  model: Model,
  weightProvider: WeightProvider,
  tokenIds: number[],
  interventions?: Intervention[],
  onProgress?: (progress: InferenceProgress) => void
): Promise<ActivationCapture> {
  const cfg = model.config;
  const S = tokenIds.length;
  // One step for the embedding lookup, one per transformer block, one for
  // the final norm + LM head — real steps of the loop below, not a guess.
  const totalSteps = cfg.numLayers + 2;
  const numHeads = cfg.numHeads;
  const numKVHeads = Number(cfg.extra.numKeyValueHeads);
  const headDim = Number(cfg.extra.headDim);
  const eps = Number(cfg.extra.rmsNormEps ?? 1e-6);
  const activationKind = String(cfg.extra.activationFunction ?? "silu");
  const hasAttnBias = cfg.extra.attentionBias === true;
  const attnOutputGate = cfg.extra.attnOutputGate === true;
  const layerTypes = cfg.extra.layerTypes as string[];
  const linearConvKernelDim = Number(cfg.extra.linearConvKernelDim);
  const linearKeyHeadDim = Number(cfg.extra.linearKeyHeadDim);
  const linearNumKeyHeads = Number(cfg.extra.linearNumKeyHeads);
  const linearValueHeadDim = Number(cfg.extra.linearValueHeadDim);
  const linearNumValueHeads = Number(cfg.extra.linearNumValueHeads);
  const keyDim = linearNumKeyHeads * linearKeyHeadDim;
  const valueDim = linearNumValueHeads * linearValueHeadDim;
  const ropeTheta = Number(cfg.extra.ropeTheta ?? 10000000);
  const rotaryDim = Math.round(headDim * Number(cfg.extra.partialRotaryFactor ?? 1));

  const H = cfg.hiddenSize;

  const activations: ActivationCapture["activations"] = {};
  const attentionWeights: ActivationCapture["attentionWeights"] = {};
  // Token IDs are real per-run data too (the tokenizer's output for this
  // exact prompt) — capturing them under the root "input" node id lets the
  // Embedding node's "input" view show real values instead of reporting
  // nothing captured, since "input" otherwise never appears on the left
  // side of a `record()` call below.
  activations["input"] = matrixToTensor(
    tokenIds.map((id) => [id]),
    "I32"
  );

  const loadMatrix = async (name: string): Promise<Matrix> => tensorToMatrix(await weightProvider.loadTensor(name));
  const loadVector = async (name: string): Promise<number[]> => tensorToVector(await weightProvider.loadTensor(name));
  // conv1d.weight is real-shaped [channels, 1, kernelWidth] (depthwise: the
  // middle "in_channels_per_group" dim is always 1) — reshape directly off
  // the raw tensor rather than routing through tensorToMatrix, which only
  // accepts 2D.
  const loadConvKernel = async (name: string): Promise<Matrix> => {
    const t = await weightProvider.loadTensor(name);
    const [channels, , width] = t.shape;
    return Array.from({ length: channels }, (_, c) => Array.from({ length: width }, (_, w) => t.data[c * width + w]));
  };
  const record = (nodeId: string, m: Matrix): Matrix => {
    const patched = applyInterventions(nodeId, m, interventions);
    activations[nodeId] = matrixToTensor(patched);
    return patched;
  };

  const embedTokensW = await loadMatrix(`${LP}.embed_tokens.weight`);
  let x = record("embed", embed(tokenIds, embedTokensW));
  onProgress?.({ completed: 1, total: totalSteps });

  for (let i = 0; i < cfg.numLayers; i++) {
    const b = `block.${i}`;
    const L = `${LP}.layers.${i}`;
    const blockInput = x;
    const isLinear = layerTypes[i] === "linear_attention";
    const attn = `${b}.attn`;

    const rms1g = await loadVector(`${L}.input_layernorm.weight`);
    const rms1Out = record(`${b}.rms1`, rmsNorm(x, rms1g, eps));

    let attnOut: Matrix;

    if (isLinear) {
      const qkvW = await loadMatrix(`${L}.linear_attn.in_proj_qkv.weight`);
      const mixedQkv = record(`${attn}.qkv`, linear(rms1Out, qkvW, null, "out_in"));

      const convKernel = await loadConvKernel(`${L}.linear_attn.conv1d.weight`);
      const conv = record(`${attn}.conv`, causalConv1dSilu(mixedQkv, convKernel, null));

      const q = sliceColumns(conv, 0, keyDim);
      const k = sliceColumns(conv, keyDim, 2 * keyDim);
      const v = sliceColumns(conv, 2 * keyDim, 2 * keyDim + valueDim);

      const betaW = await loadMatrix(`${L}.linear_attn.in_proj_b.weight`);
      const beta = record(`${attn}.beta`, applyElementwise(linear(rms1Out, betaW, null, "out_in"), sigmoid));

      const decayW = await loadMatrix(`${L}.linear_attn.in_proj_a.weight`);
      const aLog = await loadVector(`${L}.linear_attn.A_log`);
      const dtBias = await loadVector(`${L}.linear_attn.dt_bias`);
      const aRaw = linear(rms1Out, decayW, null, "out_in");
      // g = -exp(A_log) * softplus(a + dt_bias); the recurrence wants the
      // already-exponentiated per-step multiplicative factor, exp(g).
      const decay = record(
        `${attn}.decay`,
        aRaw.map((row) => row.map((a, hIdx) => Math.exp(-Math.exp(aLog[hIdx]) * softplus(a + dtBias[hIdx]))))
      );

      const zW = await loadMatrix(`${L}.linear_attn.in_proj_z.weight`);
      const zRaw = record(`${attn}.gate`, linear(rms1Out, zW, null, "out_in"));

      const { output: coreOut } = gatedDeltaRule(q, k, v, decay, beta, linearNumKeyHeads, linearNumValueHeads, linearKeyHeadDim, linearValueHeadDim);
      const corePatched = applyHeadIntervention(attn, coreOut, interventions, linearValueHeadDim);

      const gatedNormG = await loadVector(`${L}.linear_attn.norm.weight`);
      const gatedNormOut = record(`${attn}.gated_norm`, applyNormPerHead(corePatched, linearNumValueHeads, linearValueHeadDim, gatedNormG, eps));

      const gateAct = record(`${attn}.gate_act`, applyElementwise(zRaw, silu));
      const gatedMul = record(`${attn}.gated_mul`, mulMatricesElementwise(gatedNormOut, gateAct));

      const outW = await loadMatrix(`${L}.linear_attn.out_proj.weight`);
      attnOut = record(`${attn}.out`, linear(gatedMul, outW, null, "out_in"));
    } else {
      const qW = await loadMatrix(`${L}.self_attn.q_proj.weight`);
      const qB = hasAttnBias ? await loadVector(`${L}.self_attn.q_proj.bias`) : null;
      const qWidth = numHeads * headDim * (attnOutputGate ? 2 : 1);
      const qRaw = record(`${attn}.q`, linear(rms1Out, qW, qB, "out_in"));

      const { query: qOnly, gate: gateRaw } = attnOutputGate ? splitInterleavedGate(qRaw, numHeads, headDim) : { query: qRaw, gate: null as Matrix | null };

      const qNormG = await loadVector(`${L}.self_attn.q_norm.weight`);
      let q = record(`${attn}.q_norm`, applyNormPerHead(qOnly, numHeads, headDim, qNormG, eps));

      const kW = await loadMatrix(`${L}.self_attn.k_proj.weight`);
      const kB = hasAttnBias ? await loadVector(`${L}.self_attn.k_proj.bias`) : null;
      const kRaw = record(`${attn}.k`, linear(rms1Out, kW, kB, "out_in"));
      const kNormG = await loadVector(`${L}.self_attn.k_norm.weight`);
      let k = record(`${attn}.k_norm`, applyNormPerHead(kRaw, numKVHeads, headDim, kNormG, eps));

      const vW = await loadMatrix(`${L}.self_attn.v_proj.weight`);
      const vB = hasAttnBias ? await loadVector(`${L}.self_attn.v_proj.bias`) : null;
      const v = record(`${attn}.v`, linear(rms1Out, vW, vB, "out_in"));

      const rope = ropeCosSin(S, rotaryDim, ropeTheta);
      q = applyRopePerHead(q, numHeads, headDim, rotaryDim, rope.cos, rope.sin);
      k = applyRopePerHead(k, numKVHeads, headDim, rotaryDim, rope.cos, rope.sin);
      q = record(`${attn}.rope`, q);

      const { output: attnHeadsRaw, attentionWeights: headWeights } = causalSelfAttention(q, k, v, numHeads, numKVHeads, headDim);
      let attnResult = applyHeadIntervention(attn, attnHeadsRaw, interventions, headDim);
      attentionWeights[attn] = headsToTensor(headWeights);

      if (attnOutputGate && gateRaw) {
        const gateRecorded = record(`${attn}.gate`, gateRaw);
        const gateAct = record(`${attn}.gate_act`, applyElementwise(gateRecorded, sigmoid));
        attnResult = record(`${attn}.gated_mul`, mulMatricesElementwise(attnResult, gateAct));
      }

      const oW = await loadMatrix(`${L}.self_attn.o_proj.weight`);
      const oB = hasAttnBias ? await loadVector(`${L}.self_attn.o_proj.bias`) : null;
      attnOut = record(`${attn}.out`, linear(attnResult, oW, oB, "out_in"));
    }

    record(attn, attnOut);

    const res1 = record(`${b}.res1`, addMatrices(attnOut, blockInput));

    const rms2g = await loadVector(`${L}.post_attention_layernorm.weight`);
    const rms2Out = record(`${b}.rms2`, rmsNorm(res1, rms2g, eps));

    const gateW = await loadMatrix(`${L}.mlp.gate_proj.weight`);
    const upW = await loadMatrix(`${L}.mlp.up_proj.weight`);
    const gateOut = record(`${b}.ffn.gate`, linear(rms2Out, gateW, null, "out_in"));
    const gateAct = record(`${b}.ffn.gate_act`, applyActivation(gateOut, activationKind));
    const upOut = record(`${b}.ffn.up`, linear(rms2Out, upW, null, "out_in"));
    const mulOut = record(`${b}.ffn.mul`, mulMatricesElementwise(gateAct, upOut));

    const downW = await loadMatrix(`${L}.mlp.down_proj.weight`);
    const ffnOut = record(`${b}.ffn.down`, linear(mulOut, downW, null, "out_in"));

    const res2 = record(`${b}.res2`, addMatrices(ffnOut, res1));
    x = record(b, res2);
    onProgress?.({ completed: 2 + i, total: totalSteps });
  }

  const normG = await loadVector(`${LP}.norm.weight`);
  const normOut = record("norm", rmsNorm(x, normG, eps));

  const lmHeadRef = model.nodes["lm_head"].parameters[0];
  const lmHeadW = await loadMatrix(lmHeadRef.name);
  const logits = record("lm_head", linear(normOut, lmHeadW, null, "out_in"));
  onProgress?.({ completed: totalSteps, total: totalSteps });

  return {
    tokenIds,
    tokens: tokenIds.map((id) => `#${id}`),
    activations,
    attentionWeights,
    logits: matrixToTensor(logits),
  };
}
