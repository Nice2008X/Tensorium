import type { ActivationCapture, Intervention, Model, Tensor, WeightProvider } from "@tensorium/model-ir";
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
  scaleMatrix,
  tensorToMatrix,
  tensorToVector,
  type Matrix,
} from "@tensorium/nn-ops";
import { donorLayerIndex } from "./graph.js";

/**
 * The "proportional" RoPE variant full-attention layers use (sliding
 * layers use the ordinary full-head rotation, i.e. nn-ops' ropeCosSin
 * directly). Ported from `_compute_proportional_rope_parameters` in the
 * real `transformers` package (v5.15.1) — the key difference from a
 * GLM-4-style partial rotary slice is that the frequency exponent is
 * normalized against the *full* head_dim, not the rotated sub-range, so
 * reusing GLM-4's existing partial-rotary math here would silently produce
 * the wrong frequencies. The dims past `ropeAngles` get a zero frequency
 * (cos=1, sin=0) rather than being sliced off, which is mathematically a
 * no-op rotation — so applyRopeToHead can still run over the *full*
 * head_dim uniformly for both layer types, nothing needs to be sliced out
 * and reattached by hand.
 */
function proportionalRopeCosSin(seqLen: number, headDim: number, theta: number, partialRotaryFactor: number, factor: number): { cos: Matrix; sin: Matrix } {
  const half = headDim / 2;
  const ropeAngles = Math.floor((partialRotaryFactor * headDim) / 2);
  const invFreq = new Array(half).fill(0);
  for (let j = 0; j < ropeAngles; j++) invFreq[j] = 1 / theta ** ((2 * j) / headDim) / factor;

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

/** Same per-head RMSNorm-with-a-single-shared-weight pattern every QK-norm model in this app uses (Qwen3, DeepSeek-V2, ...) — repeated here rather than imported since it's a small, self-contained helper and gemma4 doesn't otherwise depend on adapter-llama-family. */
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

function sliceColumns(m: Matrix, start: number, end: number): Matrix {
  return m.map((row) => row.slice(start, end));
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

export async function runInference(model: Model, weightProvider: WeightProvider, tokenIds: number[], interventions?: Intervention[]): Promise<ActivationCapture> {
  const cfg = model.config;
  const S = tokenIds.length;
  const numHeads = cfg.numHeads;
  const numKVHeads = Number(cfg.extra.numKeyValueHeads);
  const eps = Number(cfg.extra.rmsNormEps ?? 1e-6);
  const activationKind = String(cfg.extra.activationFunction ?? "gelu_pytorch_tanh");
  const hasAttnBias = cfg.extra.attentionBias === true;
  const layerTypes = cfg.extra.layerTypes as string[];
  const headDimSliding = Number(cfg.extra.headDim);
  const headDimGlobal = Number(cfg.extra.globalHeadDim);
  const firstKvSharedLayerIdx = Number(cfg.extra.firstKvSharedLayerIdx);
  const hiddenSizePerLayerInput = Number(cfg.extra.hiddenSizePerLayerInput ?? 0);
  const hasPerLayerInput = hiddenSizePerLayerInput > 0;
  const ropeTheta = Number(cfg.extra.ropeTheta ?? 10000);
  const ropeThetaGlobal = Number(cfg.extra.ropeThetaGlobal ?? 1000000);
  const partialRotaryFactorGlobal = Number(cfg.extra.partialRotaryFactorGlobal ?? 0.25);
  const slidingWindow = Number(cfg.extra.slidingWindow ?? 512);
  const finalLogitSoftcapping = cfg.extra.finalLogitSoftcapping != null ? Number(cfg.extra.finalLogitSoftcapping) : null;
  const headDimForLayer = (i: number) => (layerTypes[i] === "full_attention" ? headDimGlobal : headDimSliding);
  const isKvSharedLayer = (i: number) => i >= firstKvSharedLayerIdx;

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
  const record = (nodeId: string, m: Matrix): Matrix => {
    const patched = applyInterventions(nodeId, m, interventions);
    activations[nodeId] = matrixToTensor(patched);
    return patched;
  };

  // --- Embeddings ---------------------------------------------------------
  const embedTokensW = await loadMatrix(`${LP}.embed_tokens.weight`);
  let x = embed(tokenIds, embedTokensW);
  x = scaleMatrix(x, Math.sqrt(H));
  x = record("embed", x);

  // Per-Layer Embeddings — computed once, sliced per layer below. See
  // graph.ts's "Per-Layer Input Projection" node for the same math with
  // more narrative explanation.
  let perLayerInputs: Matrix[] | null = null;
  if (hasPerLayerInput) {
    const embedPerLayerW = await loadMatrix(`${LP}.embed_tokens_per_layer.weight`);
    let perLayerEmbedFull = embed(tokenIds, embedPerLayerW);
    perLayerEmbedFull = scaleMatrix(perLayerEmbedFull, Math.sqrt(hiddenSizePerLayerInput));

    const perLayerProjW = await loadMatrix(`${LP}.per_layer_model_projection.weight`);
    let perLayerProjFull = linear(x, perLayerProjW, null, "out_in");
    perLayerProjFull = scaleMatrix(perLayerProjFull, H ** -0.5);

    const perLayerProjNormG = await loadVector(`${LP}.per_layer_projection_norm.weight`);

    perLayerInputs = [];
    for (let i = 0; i < cfg.numLayers; i++) {
      const start = i * hiddenSizePerLayerInput;
      const end = start + hiddenSizePerLayerInput;
      const embedSlice = sliceColumns(perLayerEmbedFull, start, end);
      const projSlice = rmsNorm(sliceColumns(perLayerProjFull, start, end), perLayerProjNormG, eps);
      const combined = scaleMatrix(addMatrices(projSlice, embedSlice), 2 ** -0.5);
      perLayerInputs.push(combined);
    }
    record("per_layer_input", perLayerInputs[0]); // representative slice for display — every layer's is a same-shape different slice
  }

  // K/V computed by a non-shared layer, keyed by that layer's own index —
  // any later KV-shared layer looks up its donor's entry here instead of
  // running its own (unused, checkpoint-only) k_proj/v_proj.
  const computedKV = new Map<number, { k: Matrix; v: Matrix }>();

  for (let i = 0; i < cfg.numLayers; i++) {
    const b = `block.${i}`;
    const L = `${LP}.layers.${i}`;
    const blockInput = x;
    const layerType = layerTypes[i];
    const headDim = headDimForLayer(i);
    const sharedLayer = isKvSharedLayer(i);

    const rms1g = await loadVector(`${L}.input_layernorm.weight`);
    const rms1Out = record(`${b}.rms1`, rmsNorm(x, rms1g, eps));

    const qW = await loadMatrix(`${L}.self_attn.q_proj.weight`);
    const qB = hasAttnBias ? await loadVector(`${L}.self_attn.q_proj.bias`) : null;
    let q = record(`${b}.attn.q`, linear(rms1Out, qW, qB, "out_in"));

    const qNormG = await loadVector(`${L}.self_attn.q_norm.weight`);
    q = record(`${b}.attn.q_norm`, applyNormPerHead(q, numHeads, headDim, qNormG, eps));

    const rope =
      layerType === "full_attention"
        ? proportionalRopeCosSin(S, headDim, ropeThetaGlobal, partialRotaryFactorGlobal, 1)
        : ropeCosSin(S, headDim, ropeTheta);
    q = applyRopePerHead(q, numHeads, headDim, rope.cos, rope.sin);
    q = record(`${b}.attn.rope`, q);

    let k: Matrix;
    let v: Matrix;
    if (!sharedLayer) {
      const kW = await loadMatrix(`${L}.self_attn.k_proj.weight`);
      const kB = hasAttnBias ? await loadVector(`${L}.self_attn.k_proj.bias`) : null;
      k = record(`${b}.attn.k`, linear(rms1Out, kW, kB, "out_in"));
      const kNormG = await loadVector(`${L}.self_attn.k_norm.weight`);
      k = record(`${b}.attn.k_norm`, applyNormPerHead(k, numKVHeads, headDim, kNormG, eps));
      k = applyRopePerHead(k, numKVHeads, headDim, rope.cos, rope.sin);

      const vW = await loadMatrix(`${L}.self_attn.v_proj.weight`);
      const vB = hasAttnBias ? await loadVector(`${L}.self_attn.v_proj.bias`) : null;
      const vRaw = record(`${b}.attn.v`, linear(rms1Out, vW, vB, "out_in"));
      // v_norm has no learnable weight (with_scale=False in the real model) — RMS-normalize with an implicit all-ones gamma.
      v = record(`${b}.attn.v_norm`, rmsNorm(vRaw, new Array(numKVHeads * headDim).fill(1), eps));

      computedKV.set(i, { k, v });
    } else {
      const donor = donorLayerIndex(layerTypes, firstKvSharedLayerIdx, i);
      const donorKV = computedKV.get(donor);
      if (!donorKV) throw new Error(`Block ${i} expected to share K/V from block ${donor}, but that block's K/V was never computed.`);
      k = donorKV.k;
      v = donorKV.v;
    }

    const { output: attnHeadsRaw, attentionWeights: headWeights } = causalSelfAttention(q, k, v, numHeads, numKVHeads, headDim, {
      scale: 1.0,
      slidingWindow: layerType === "sliding_attention" ? slidingWindow : undefined,
    });
    const attnRaw = applyHeadIntervention(`${b}.attn`, attnHeadsRaw, interventions, headDim);
    attentionWeights[`${b}.attn`] = headsToTensor(headWeights);

    const oW = await loadMatrix(`${L}.self_attn.o_proj.weight`);
    const oB = hasAttnBias ? await loadVector(`${L}.self_attn.o_proj.bias`) : null;
    const attnOutRaw = linear(attnRaw, oW, oB, "out_in");
    const attnProjected = record(`${b}.attn.out`, attnOutRaw);
    record(`${b}.attn`, attnProjected);

    const postAttnG = await loadVector(`${L}.post_attention_layernorm.weight`);
    const postAttnNorm = record(`${b}.post_attn_norm`, rmsNorm(attnProjected, postAttnG, eps));

    const res1 = record(`${b}.res1`, addMatrices(postAttnNorm, blockInput));

    const preFfnG = await loadVector(`${L}.pre_feedforward_layernorm.weight`);
    const preFfnNorm = record(`${b}.pre_ffn_norm`, rmsNorm(res1, preFfnG, eps));

    const gateW = await loadMatrix(`${L}.mlp.gate_proj.weight`);
    const upW = await loadMatrix(`${L}.mlp.up_proj.weight`);
    const gateOut = record(`${b}.ffn.gate`, linear(preFfnNorm, gateW, null, "out_in"));
    const gateAct = record(`${b}.ffn.gate_act`, applyActivation(gateOut, activationKind));
    const upOut = record(`${b}.ffn.up`, linear(preFfnNorm, upW, null, "out_in"));
    const mulOut = record(`${b}.ffn.mul`, mulMatricesElementwise(gateAct, upOut));

    const downW = await loadMatrix(`${L}.mlp.down_proj.weight`);
    const ffnProjected = record(`${b}.ffn.down`, linear(mulOut, downW, null, "out_in"));

    const postFfnG = await loadVector(`${L}.post_feedforward_layernorm.weight`);
    const postFfnNorm = record(`${b}.post_ffn_norm`, rmsNorm(ffnProjected, postFfnG, eps));

    const res2 = record(`${b}.res2`, addMatrices(postFfnNorm, res1));

    let blockPreScale = res2;
    if (hasPerLayerInput && perLayerInputs) {
      const plGateW = await loadMatrix(`${L}.per_layer_input_gate.weight`);
      const plGateOut = record(`${b}.pl_gate`, linear(res2, plGateW, null, "out_in"));
      const plAct = record(`${b}.pl_act`, applyActivation(plGateOut, activationKind));
      const plMul = record(`${b}.pl_mul`, mulMatricesElementwise(plAct, perLayerInputs[i]));

      const plProjW = await loadMatrix(`${L}.per_layer_projection.weight`);
      const plProj = record(`${b}.pl_proj`, linear(plMul, plProjW, null, "out_in"));

      const plNormG = await loadVector(`${L}.post_per_layer_input_norm.weight`);
      const plNorm = record(`${b}.pl_norm`, rmsNorm(plProj, plNormG, eps));

      blockPreScale = record(`${b}.res3`, addMatrices(plNorm, res2));
    }

    const layerScalarVec = await loadVector(`${L}.layer_scalar`);
    const layerScalar = layerScalarVec[0] ?? 1;
    const scaled = record(`${b}.scale`, scaleMatrix(blockPreScale, layerScalar));
    x = record(b, scaled);
  }

  const normG = await loadVector(`${LP}.norm.weight`);
  const normOut = record("norm", rmsNorm(x, normG, eps));

  const lmHeadRef = model.nodes["lm_head"].parameters[0];
  const lmHeadW = await loadMatrix(lmHeadRef.name);
  let logits = linear(normOut, lmHeadW, null, "out_in");
  logits = record("lm_head", logits);

  if (finalLogitSoftcapping != null) {
    logits = logits.map((row) => row.map((v) => Math.tanh(v / finalLogitSoftcapping) * finalLogitSoftcapping));
    logits = record("logit_softcap", logits);
  }

  return {
    tokenIds,
    tokens: tokenIds.map((id) => `#${id}`),
    activations,
    attentionWeights,
    logits: matrixToTensor(logits),
  };
}
