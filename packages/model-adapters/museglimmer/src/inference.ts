import type { ActivationCapture, InferenceProgress, Intervention, Model, Tensor, WeightProvider } from "@tensorium/model-ir";
import {
  addMatrices,
  applyActivation,
  applyHeadIntervention,
  applyInterventions,
  applyRopeToHead,
  causalSelfAttention,
  embed,
  linear,
  matrixToTensor,
  mulMatricesElementwise,
  ropeCosSin,
  rmsNorm,
  sigmoid,
  tensorToMatrix,
  tensorToVector,
  type Matrix,
} from "@tensorium/nn-ops";

/** MuseGlimmerTextCenteredRMSNorm (Gemma2RMSNorm): scales by (1 + weight) instead of weight — same shape as gemmaRmsNorm in nn-ops, kept local since it's the only caller here that needs a per-eps variant mixed with the plain-weight kind below. */
function centeredRmsNorm(x: Matrix, gamma: number[], eps: number): Matrix {
  return x.map((row) => {
    const n = row.length;
    const meanSq = row.reduce((a, b) => a + b * b, 0) / n;
    const scale = 1 / Math.sqrt(meanSq + eps);
    return row.map((v, i) => v * scale * (1 + gamma[i]));
  });
}

/** Same per-head RMSNorm-with-a-single-shared-weight pattern every QK-norm model in this app uses, generalized to an optional (weightless) gamma — MuseGlimmer's qk_norm has with_scale=False, so gamma is an implicit all-ones vector. */
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

function applyRopePerHead(x: Matrix, numHeads: number, headDim: number, cos: Matrix, sin: Matrix): Matrix {
  const S = x.length;
  const out: Matrix = Array.from({ length: S }, () => new Array(numHeads * headDim));
  for (let h = 0; h < numHeads; h++) {
    const headSlice = x.map((row) => row.slice(h * headDim, (h + 1) * headDim));
    const roped = applyRopeToHead(headSlice, cos, sin);
    for (let s = 0; s < S; s++) {
      for (let d = 0; d < headDim; d++) out[s][h * headDim + d] = roped[s][d];
    }
  }
  return out;
}

function scaleMatrix(x: Matrix, factor: number): Matrix {
  return x.map((row) => row.map((v) => v * factor));
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
  const totalSteps = cfg.numLayers + 2;
  const numHeads = cfg.numHeads;
  const numKVHeads = Number(cfg.extra.numKeyValueHeads);
  const headDim = Number(cfg.extra.headDim);
  const rmsNormEps = Number(cfg.extra.rmsNormEps);
  const postNormEps = Number(cfg.extra.postNormEps);
  const activationKind = String(cfg.extra.activationFunction ?? "silu");
  const hasAttnBias = cfg.extra.attentionBias === true;
  const layerTypes = cfg.extra.layerTypes as string[];
  const layerHasRope = cfg.extra.layerHasRope as boolean[];
  const slidingWindow = Number(cfg.extra.slidingWindow ?? 4096);
  const ropeTheta = Number(cfg.extra.ropeTheta ?? 500000);
  const qkScaleFactor = Number(cfg.extra.qkScaleFactor ?? 1);
  const outputMultiplier = Number(cfg.extra.outputMultiplier ?? 1);
  const finalLogitSoftcapping = cfg.extra.finalLogitSoftcapping != null ? Number(cfg.extra.finalLogitSoftcapping) : null;
  const qOnes = new Array(numHeads * headDim).fill(1);
  const kOnes = new Array(numKVHeads * headDim).fill(1);

  const activations: ActivationCapture["activations"] = {};
  const attentionWeights: ActivationCapture["attentionWeights"] = {};
  activations["input"] = matrixToTensor(
    tokenIds.map((id) => [id]),
    "I32"
  );

  const loadMatrix = async (name: string): Promise<Matrix> => tensorToMatrix(await weightProvider.loadTensor(name));
  const loadVector = async (name: string): Promise<number[]> => tensorToVector(await weightProvider.loadTensor(name));
  const record = (nodeId: string, m: Matrix): Matrix => {
    const patched = applyInterventions(nodeId, m, interventions);
    activations[nodeId] = matrixToTensor(patched);
    return patched;
  };

  // --- Embeddings ---------------------------------------------------------
  const embedTokensW = await loadMatrix(`${LP}.embed_tokens.weight`);
  let x = embed(tokenIds, embedTokensW);
  x = record("embed", x);
  // Weightless RMSNorm right on top of the raw lookup (MuseGlimmerTextNormedEmbedding) — no √hidden_size scaling here, unlike Gemma's classic embedding convention.
  x = record("embed_norm", rmsNorm(x, new Array(cfg.hiddenSize).fill(1), rmsNormEps));
  onProgress?.({ completed: 1, total: totalSteps });

  const rope = ropeCosSin(S, headDim, ropeTheta);

  for (let i = 0; i < cfg.numLayers; i++) {
    const b = `block.${i}`;
    const L = `${LP}.layers.${i}`;
    const blockInput = x;
    const layerType = layerTypes[i];
    const hasRope = layerHasRope[i];

    const rms1g = await loadVector(`${L}.input_layernorm.weight`);
    const rms1Out = record(`${b}.rms1`, centeredRmsNorm(x, rms1g, rmsNormEps));

    const qW = await loadMatrix(`${L}.self_attn.q_proj.weight`);
    const qB = hasAttnBias ? await loadVector(`${L}.self_attn.q_proj.bias`) : null;
    let q = record(`${b}.attn.q`, linear(rms1Out, qW, qB, "out_in"));

    const kW = await loadMatrix(`${L}.self_attn.k_proj.weight`);
    const kB = hasAttnBias ? await loadVector(`${L}.self_attn.k_proj.bias`) : null;
    let k = record(`${b}.attn.k`, linear(rms1Out, kW, kB, "out_in"));

    const vW = await loadMatrix(`${L}.self_attn.v_proj.weight`);
    const vB = hasAttnBias ? await loadVector(`${L}.self_attn.v_proj.bias`) : null;
    const v = record(`${b}.attn.v`, linear(rms1Out, vW, vB, "out_in"));

    // Same weightless RMSNorm module normalizes Q and K per head; Q alone gets the extra qk_scale_factor multiply.
    q = record(`${b}.attn.q_norm`, applyNormPerHead(q, numHeads, headDim, qOnes, rmsNormEps));
    q = record(`${b}.attn.q_scale`, scaleMatrix(q, qkScaleFactor));
    k = record(`${b}.attn.k_norm`, applyNormPerHead(k, numKVHeads, headDim, kOnes, rmsNormEps));

    if (hasRope) {
      q = applyRopePerHead(q, numHeads, headDim, rope.cos, rope.sin);
      k = applyRopePerHead(k, numKVHeads, headDim, rope.cos, rope.sin);
      record(`${b}.attn.rope`, q);
    }

    const { output: attnHeadsRaw, attentionWeights: headWeights } = causalSelfAttention(q, k, v, numHeads, numKVHeads, headDim, {
      slidingWindow: layerType === "sliding_attention" ? slidingWindow : undefined,
    });
    const attnRaw = applyHeadIntervention(`${b}.attn`, attnHeadsRaw, interventions, headDim);
    attentionWeights[`${b}.attn`] = headsToTensor(headWeights);

    const gateW = await loadMatrix(`${L}.self_attn.gate_proj.weight`);
    const gateOut = record(`${b}.attn.gate`, linear(rms1Out, gateW, null, "out_in"));
    const gateAct = record(`${b}.attn.gate_act`, gateOut.map((row) => row.map(sigmoid)));
    const gatedMul = record(`${b}.attn.gated_mul`, mulMatricesElementwise(attnRaw, gateAct));

    const oW = await loadMatrix(`${L}.self_attn.o_proj.weight`);
    const oB = hasAttnBias ? await loadVector(`${L}.self_attn.o_proj.bias`) : null;
    const attnOutRaw = linear(gatedMul, oW, oB, "out_in");
    const attnProjected = record(`${b}.attn.out`, attnOutRaw);
    record(`${b}.attn`, attnProjected);

    const postAttnG = await loadVector(`${L}.post_attention_layernorm.weight`);
    const postAttnNorm = record(`${b}.post_attn_norm`, centeredRmsNorm(attnProjected, postAttnG, postNormEps));

    const res1 = record(`${b}.res1`, addMatrices(postAttnNorm, blockInput));

    const preFfnG = await loadVector(`${L}.pre_feedforward_layernorm.weight`);
    const preFfnNorm = record(`${b}.pre_ffn_norm`, centeredRmsNorm(res1, preFfnG, rmsNormEps));

    const gateWFfn = await loadMatrix(`${L}.mlp.gate_proj.weight`);
    const upW = await loadMatrix(`${L}.mlp.up_proj.weight`);
    const gateFfnOut = record(`${b}.ffn.gate`, linear(preFfnNorm, gateWFfn, null, "out_in"));
    const gateFfnAct = record(`${b}.ffn.gate_act`, applyActivation(gateFfnOut, activationKind));
    const upOut = record(`${b}.ffn.up`, linear(preFfnNorm, upW, null, "out_in"));
    const mulOut = record(`${b}.ffn.mul`, mulMatricesElementwise(gateFfnAct, upOut));

    const downW = await loadMatrix(`${L}.mlp.down_proj.weight`);
    const ffnProjected = record(`${b}.ffn.down`, linear(mulOut, downW, null, "out_in"));

    const postFfnG = await loadVector(`${L}.post_feedforward_layernorm.weight`);
    const postFfnNorm = record(`${b}.post_ffn_norm`, centeredRmsNorm(ffnProjected, postFfnG, postNormEps));

    const res2 = record(`${b}.res2`, addMatrices(postFfnNorm, res1));
    x = record(b, res2);
    onProgress?.({ completed: 2 + i, total: totalSteps });
  }

  const normG = await loadVector(`${LP}.norm.weight`);
  const normOut = record("norm", rmsNorm(x, normG, rmsNormEps));

  const lmHeadRef = model.nodes["lm_head"].parameters[0];
  const lmHeadW = await loadMatrix(lmHeadRef.name);
  let logits = linear(normOut, lmHeadW, null, "out_in");
  logits = record("lm_head", logits);

  if (outputMultiplier !== 1) {
    logits = scaleMatrix(logits, outputMultiplier);
    logits = record("output_multiplier", logits);
  }
  if (finalLogitSoftcapping != null) {
    logits = logits.map((row) => row.map((v) => Math.tanh(v / finalLogitSoftcapping) * finalLogitSoftcapping));
    logits = record("logit_softcap", logits);
  }
  onProgress?.({ completed: totalSteps, total: totalSteps });

  return {
    tokenIds,
    tokens: tokenIds.map((id) => `#${id}`),
    activations,
    attentionWeights,
    logits: matrixToTensor(logits),
  };
}
