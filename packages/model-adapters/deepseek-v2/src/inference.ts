import type { ActivationCapture, Intervention, Model, Tensor, WeightProvider } from "@tensorium/model-ir";
import {
  addMatrices,
  applyActivation,
  applyHeadIntervention,
  applyInterventions,
  embed,
  linear,
  matrixToTensor,
  mulMatricesElementwise,
  rmsNorm,
  softmaxRow,
  tensorToMatrix,
  tensorToVector,
  topKIndices,
  zeros,
  type Matrix,
} from "@tensorium/nn-ops";

interface RopeScaling {
  type: string;
  factor: number;
  original_max_position_embeddings?: number;
  beta_fast?: number;
  beta_slow?: number;
  mscale?: number;
  mscale_all_dim?: number;
}

/**
 * YARN's per-dimension frequency interpolation: dimensions inside
 * [low, high] (found via yarnFindCorrectionRange, derived from beta_fast /
 * beta_slow) use the scaled-down "interpolated" frequency; dimensions
 * outside it keep the original "extrapolated" one; a linear ramp blends
 * across the boundary instead of a hard cutoff. Ported 1:1 from
 * DeepseekV2YarnRotaryEmbedding._set_cos_sin_cache in DeepSeek-V2's
 * modeling_deepseek.py (the formulas, not just "a" NTK-aware scaling) since
 * this exists to reproduce real model behavior.
 */
function yarnFindCorrectionDim(numRotations: number, dim: number, base: number, maxPositionEmbeddings: number): number {
  return (dim * Math.log(maxPositionEmbeddings / (numRotations * 2 * Math.PI))) / (2 * Math.log(base));
}

function yarnFindCorrectionRange(lowRot: number, highRot: number, dim: number, base: number, maxPositionEmbeddings: number): [number, number] {
  const low = Math.floor(yarnFindCorrectionDim(lowRot, dim, base, maxPositionEmbeddings));
  const high = Math.ceil(yarnFindCorrectionDim(highRot, dim, base, maxPositionEmbeddings));
  return [Math.max(low, 0), Math.min(high, dim - 1)];
}

function yarnGetMscale(scale: number, mscale: number): number {
  if (scale <= 1) return 1.0;
  return 0.1 * mscale * Math.log(scale) + 1.0;
}

function yarnLinearRampMask(min: number, max: number, dim: number): number[] {
  if (min === max) max += 0.001; // avoid a divide-by-zero singularity
  return Array.from({ length: dim }, (_, i) => Math.min(1, Math.max(0, (i - min) / (max - min))));
}

/**
 * cos/sin tables for the rope-carrying slice of each head (qk_rope_head_dim
 * wide), YARN-scaled when this checkpoint's config sets rope_scaling, plain
 * otherwise. Returns the extra softmax_scale multiplier (mscale²) that
 * DeepSeek-V2 folds into attention scoring alongside the usual 1/sqrt(d) —
 * 1 when there's no YARN scaling to apply.
 */
function ropeCosSinForDeepseek(seqLen: number, dim: number, base: number, scaling: RopeScaling | null): { cos: Matrix; sin: Matrix; softmaxScaleMul: number } {
  const half = dim / 2;
  if (!scaling) {
    const invFreq = Array.from({ length: half }, (_, i) => 1 / base ** ((2 * i) / dim));
    const cos: Matrix = [];
    const sin: Matrix = [];
    for (let pos = 0; pos < seqLen; pos++) {
      const freqs = invFreq.map((f) => pos * f);
      const full = [...freqs, ...freqs];
      cos.push(full.map(Math.cos));
      sin.push(full.map(Math.sin));
    }
    return { cos, sin, softmaxScaleMul: 1 };
  }

  const scalingFactor = scaling.factor;
  const originalMaxPositionEmbeddings = scaling.original_max_position_embeddings ?? 4096;
  const betaFast = scaling.beta_fast ?? 32;
  const betaSlow = scaling.beta_slow ?? 1;
  const mscaleParam = scaling.mscale ?? 1;
  const mscaleAllDimParam = scaling.mscale_all_dim ?? 0;

  const freqExtra = Array.from({ length: half }, (_, i) => 1 / base ** ((2 * i) / dim));
  const freqInter = Array.from({ length: half }, (_, i) => 1 / (scalingFactor * base ** ((2 * i) / dim)));
  const [low, high] = yarnFindCorrectionRange(betaFast, betaSlow, dim, base, originalMaxPositionEmbeddings);
  const ramp = yarnLinearRampMask(low, high, half); // 1 - inv_freq_mask, in the source's terms
  const invFreq = freqInter.map((fi, i) => fi * ramp[i] + freqExtra[i] * (1 - ramp[i]));

  const cosSinScale = yarnGetMscale(scalingFactor, mscaleParam) / yarnGetMscale(scalingFactor, mscaleAllDimParam || 1);
  const cos: Matrix = [];
  const sin: Matrix = [];
  for (let pos = 0; pos < seqLen; pos++) {
    const freqs = invFreq.map((f) => pos * f);
    const full = [...freqs, ...freqs];
    cos.push(full.map((v) => Math.cos(v) * cosSinScale));
    sin.push(full.map((v) => Math.sin(v) * cosSinScale));
  }

  // The attention scale multiplier only kicks in once mscale_all_dim is
  // actually set (matches DeepseekV2Attention.__init__ exactly — it checks
  // truthiness of mscale_all_dim, not just "rope_scaling is not None").
  const softmaxScaleMul = mscaleAllDimParam ? yarnGetMscale(scalingFactor, mscaleAllDimParam) ** 2 : 1;
  return { cos, sin, softmaxScaleMul };
}

function rotateHalf(row: number[]): number[] {
  const half = row.length / 2;
  return [...row.slice(half).map((v) => -v), ...row.slice(0, half)];
}

/**
 * DeepSeek-V2's apply_rotary_pos_emb does one extra step before the usual
 * cos/sin * rotate_half: it de-interleaves each head's rope slice — pairs
 * (x0,x1),(x2,x3),... become (x0,x2,x4,...) followed by (x1,x3,x5,...) —
 * because these checkpoints store Q/K in the original interleaved layout
 * rather than the pre-permuted layout Llama-derived conversion scripts
 * produce (which is what lets every other adapter's RoPE skip this step).
 */
function deinterleave(row: number[]): number[] {
  const half = row.length / 2;
  const out = new Array(row.length);
  for (let i = 0; i < half; i++) {
    out[i] = row[2 * i];
    out[half + i] = row[2 * i + 1];
  }
  return out;
}

function applyDeepseekRope(x: Matrix, cos: Matrix, sin: Matrix): Matrix {
  return x.map((row, pos) => {
    const deinterleaved = deinterleave(row);
    const rotated = rotateHalf(deinterleaved);
    return deinterleaved.map((v, i) => v * cos[pos][i] + rotated[i] * sin[pos][i]);
  });
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

export async function runInference(model: Model, weightProvider: WeightProvider, tokenIds: number[], interventions?: Intervention[]): Promise<ActivationCapture> {
  const cfg = model.config;
  const S = tokenIds.length;
  const numHeads = cfg.numHeads;
  const eps = Number(cfg.extra.rmsNormEps ?? 1e-6);
  const activationKind = String(cfg.extra.activationFunction ?? "silu");
  const hasAttnBias = cfg.extra.attentionBias === true;
  const qLoraRank = cfg.extra.qLoraRank as number | null;
  const kvLoraRank = Number(cfg.extra.kvLoraRank);
  const qkNopeHeadDim = Number(cfg.extra.qkNopeHeadDim);
  const qkRopeHeadDim = Number(cfg.extra.qkRopeHeadDim);
  const vHeadDim = Number(cfg.extra.vHeadDim);
  const qHeadDim = Number(cfg.extra.qHeadDim);
  const ropeTheta = Number(cfg.extra.ropeTheta ?? 10000);
  const ropeScaling = (cfg.extra.ropeScaling as RopeScaling | null) ?? null;
  const isMoE = cfg.extra.isMoE === true;
  const numRoutedExperts = Number(cfg.extra.numRoutedExperts ?? 0);
  const numSharedExperts = Number(cfg.extra.numSharedExperts ?? 0);
  const numExpertsPerTok = Number(cfg.extra.numExpertsPerTok ?? 0);
  const normTopkProb = cfg.extra.normTopkProb === true;
  const routedScalingFactor = Number(cfg.extra.routedScalingFactor ?? 1);
  const topkMethod = String(cfg.extra.topkMethod ?? "greedy");
  const numGroups = Number(cfg.extra.numGroups ?? 1);
  const topkGroup = Number(cfg.extra.topkGroup ?? 1);
  const firstKDenseReplace = Number(cfg.extra.firstKDenseReplace ?? 0);
  const moeLayerFreq = Number(cfg.extra.moeLayerFreq ?? 1);
  const isSparseLayer = (i: number) => isMoE && i >= firstKDenseReplace && i % moeLayerFreq === 0;

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

  const { cos: ropeCos, sin: ropeSin, softmaxScaleMul } = ropeCosSinForDeepseek(S, qkRopeHeadDim, ropeTheta, ropeScaling);
  const softmaxScale = qHeadDim ** -0.5 * softmaxScaleMul;

  /** DeepSeekMoE's gate: softmax over every routed expert, then either a plain top-k or a group-limited top-k (only experts inside the best-scoring groups are eligible). */
  const selectExperts = (scores: number[]): { expertIdx: number; weight: number }[] => {
    let candidateScores = scores;
    if (topkMethod === "group_limited_greedy" && numGroups > 1) {
      const groupSize = numRoutedExperts / numGroups;
      const groupScores = Array.from({ length: numGroups }, (_, g) => {
        let max = -Infinity;
        for (let j = 0; j < groupSize; j++) max = Math.max(max, scores[g * groupSize + j]);
        return max;
      });
      const topGroups = new Set(topKIndices(groupScores, topkGroup));
      candidateScores = scores.map((v, i) => (topGroups.has(Math.floor(i / groupSize)) ? v : 0));
    }
    const idx = topKIndices(candidateScores, numExpertsPerTok);
    let weights = idx.map((i) => candidateScores[i]);
    if (numExpertsPerTok > 1 && normTopkProb) {
      const sum = weights.reduce((a, v) => a + v, 0) + 1e-20;
      weights = weights.map((w) => w / sum);
    } else {
      weights = weights.map((w) => w * routedScalingFactor);
    }
    return idx.map((expertIdx, slot) => ({ expertIdx, weight: weights[slot] }));
  };

  const runMoEFfn = async (b: string, L: string, x: Matrix): Promise<Matrix> => {
    const routerW = await loadMatrix(`${L}.mlp.gate.weight`); // [numRoutedExperts, hidden], out_in
    const routerLogits = linear(x, routerW, null, "out_in");
    const routerProbs = record(`${b}.ffn.router`, routerLogits.map((row) => softmaxRow(row)));

    const H = x[0]?.length ?? 0;
    const selection = routerProbs.map(selectExperts);

    const needed = new Set<number>();
    for (const sel of selection) for (const { expertIdx } of sel) needed.add(expertIdx);

    const expertWeights = new Map<number, { gateW: Matrix; upW: Matrix; downW: Matrix }>();
    await Promise.all(
      [...needed].map(async (e) => {
        const [gateW, upW, downW] = await Promise.all([
          loadMatrix(`${L}.mlp.experts.${e}.gate_proj.weight`),
          loadMatrix(`${L}.mlp.experts.${e}.up_proj.weight`),
          loadMatrix(`${L}.mlp.experts.${e}.down_proj.weight`),
        ]);
        expertWeights.set(e, { gateW, upW, downW });
      })
    );

    const combined: Matrix = zeros(S, H);
    for (let s = 0; s < S; s++) {
      for (const { expertIdx, weight } of selection[s]) {
        const { gateW, upW, downW } = expertWeights.get(expertIdx)!;
        const xRow: Matrix = [x[s]];
        const act = applyActivation(linear(xRow, gateW, null, "out_in"), activationKind);
        const upOut = linear(xRow, upW, null, "out_in");
        const downOut = linear(mulMatricesElementwise(act, upOut), downW, null, "out_in");
        for (let d = 0; d < H; d++) combined[s][d] += weight * downOut[0][d];
      }
    }

    const expertsOut = record(`${b}.ffn.experts`, combined);
    return expertsOut;
  };

  const embedTokens = await loadMatrix("model.embed_tokens.weight");
  let x = embed(tokenIds, embedTokens);
  x = record("embed", x);

  for (let i = 0; i < cfg.numLayers; i++) {
    const b = `block.${i}`;
    const L = `model.layers.${i}`;
    const blockInput = x;

    const rms1g = await loadVector(`${L}.input_layernorm.weight`);
    const rms1Out = record(`${b}.rms1`, rmsNorm(x, rms1g, eps));

    // --- Q branch ---------------------------------------------------------
    let q: Matrix;
    if (qLoraRank != null) {
      const qDownW = await loadMatrix(`${L}.self_attn.q_a_proj.weight`);
      const qDownB = hasAttnBias ? await loadVector(`${L}.self_attn.q_a_proj.bias`) : null;
      const qDown = record(`${b}.attn.q_down`, linear(rms1Out, qDownW, qDownB, "out_in"));

      const qNormG = await loadVector(`${L}.self_attn.q_a_layernorm.weight`);
      const qDownNorm = record(`${b}.attn.q_down_norm`, rmsNorm(qDown, qNormG, eps));

      const qUpW = await loadMatrix(`${L}.self_attn.q_b_proj.weight`);
      q = record(`${b}.attn.q_up`, linear(qDownNorm, qUpW, null, "out_in"));
    } else {
      const qW = await loadMatrix(`${L}.self_attn.q_proj.weight`);
      q = record(`${b}.attn.q`, linear(rms1Out, qW, null, "out_in"));
    }

    // --- KV branch ----------------------------------------------------------
    const kvDownW = await loadMatrix(`${L}.self_attn.kv_a_proj_with_mqa.weight`);
    const kvDownB = hasAttnBias ? await loadVector(`${L}.self_attn.kv_a_proj_with_mqa.bias`) : null;
    const kvDown = record(`${b}.attn.kv_down`, linear(rms1Out, kvDownW, kvDownB, "out_in")); // [S, kvLoraRank + qkRopeHeadDim]

    const compressedKv = kvDown.map((row) => row.slice(0, kvLoraRank));
    const kPeShared = kvDown.map((row) => row.slice(kvLoraRank));

    const kvNormG = await loadVector(`${L}.self_attn.kv_a_layernorm.weight`);
    const kvDownNorm = record(`${b}.attn.kv_down_norm`, rmsNorm(compressedKv, kvNormG, eps));

    const kvUpW = await loadMatrix(`${L}.self_attn.kv_b_proj.weight`);
    const kv = record(`${b}.attn.kv_up`, linear(kvDownNorm, kvUpW, null, "out_in")); // [S, numHeads*(qkNopeHeadDim+vHeadDim)]

    // Rotate just the rope-carrying slices: every Q head's own trailing
    // qkRopeHeadDim dims, and the one k_pe slice shared by every head.
    const kPeRoped = applyDeepseekRope(kPeShared, ropeCos, ropeSin);

    const qHeadsFull: Matrix[] = [];
    const kHeadsFull: Matrix[] = [];
    const vHeads: Matrix[] = [];
    for (let h = 0; h < numHeads; h++) {
      const qNope = q.map((row) => row.slice(h * qHeadDim, h * qHeadDim + qkNopeHeadDim));
      const qPe = q.map((row) => row.slice(h * qHeadDim + qkNopeHeadDim, h * qHeadDim + qHeadDim));
      const qPeRoped = applyDeepseekRope(qPe, ropeCos, ropeSin);
      qHeadsFull.push(qNope.map((row, s) => [...row, ...qPeRoped[s]]));

      const kvOff = h * (qkNopeHeadDim + vHeadDim);
      const kNope = kv.map((row) => row.slice(kvOff, kvOff + qkNopeHeadDim));
      vHeads.push(kv.map((row) => row.slice(kvOff + qkNopeHeadDim, kvOff + qkNopeHeadDim + vHeadDim)));
      kHeadsFull.push(kNope.map((row, s) => [...row, ...kPeRoped[s]]));
    }

    // Record the reconstructed (post-rope) Q as this node's activation —
    // same convention the llama-family engine uses for its "rope" node:
    // one node conceptually rotates both Q and K, but a recorded Tensor can
    // only hold one, so the (more informative, per-head-varying) Q side wins.
    const qRopedFlat: Matrix = Array.from({ length: S }, (_, s) => qHeadsFull.reduce<number[]>((acc, head) => [...acc, ...head[s]], []));
    record(`${b}.attn.rope`, qRopedFlat);

    const perHeadOutputs: Matrix[] = [];
    const headWeights: number[][][] = [];
    for (let h = 0; h < numHeads; h++) {
      const qh = qHeadsFull[h];
      const kh = kHeadsFull[h];
      const vh = vHeads[h];
      const scores = zeros(S, S);
      for (let si = 0; si < S; si++) {
        for (let sj = 0; sj < S; sj++) {
          if (sj > si) {
            scores[si][sj] = -Infinity;
            continue;
          }
          let acc = 0;
          for (let d = 0; d < qHeadDim; d++) acc += qh[si][d] * kh[sj][d];
          scores[si][sj] = acc * softmaxScale;
        }
      }
      const weights = scores.map(softmaxRow);
      headWeights.push(weights);
      const headOut = zeros(S, vHeadDim);
      for (let si = 0; si < S; si++) {
        for (let sj = 0; sj <= si; sj++) {
          const w = weights[si][sj];
          for (let d = 0; d < vHeadDim; d++) headOut[si][d] += w * vh[sj][d];
        }
      }
      perHeadOutputs.push(headOut);
    }
    attentionWeights[`${b}.attn`] = headsToTensor(headWeights);

    const attnConcat = zeros(S, numHeads * vHeadDim);
    for (let s = 0; s < S; s++) {
      for (let h = 0; h < numHeads; h++) {
        for (let d = 0; d < vHeadDim; d++) attnConcat[s][h * vHeadDim + d] = perHeadOutputs[h][s][d];
      }
    }
    const attnConcatPatched = applyHeadIntervention(`${b}.attn`, attnConcat, interventions, vHeadDim);

    const oW = await loadMatrix(`${L}.self_attn.o_proj.weight`);
    const oB = hasAttnBias ? await loadVector(`${L}.self_attn.o_proj.bias`) : null;
    const attnOutRaw = linear(attnConcatPatched, oW, oB, "out_in");
    const attnProjected = record(`${b}.attn.out`, attnOutRaw);
    const attnOut = record(`${b}.attn`, attnProjected);

    const res1 = record(`${b}.res1`, addMatrices(attnOut, blockInput));

    const rms2g = await loadVector(`${L}.post_attention_layernorm.weight`);
    const rms2Out = record(`${b}.rms2`, rmsNorm(res1, rms2g, eps));

    let ffnOut: Matrix;
    if (isSparseLayer(i)) {
      const routedOut = await runMoEFfn(b, L, rms2Out);
      let combined = routedOut;
      // DeepSeek's shared expert(s) — merged into one wider MLP at
      // conversion time — run on every token unconditionally, no gate at
      // all (unlike Qwen2-MoE's sigmoid-gated single shared expert).
      if (numSharedExperts > 0) {
        const [sharedGateW, sharedUpW, sharedDownW] = await Promise.all([
          loadMatrix(`${L}.mlp.shared_experts.gate_proj.weight`),
          loadMatrix(`${L}.mlp.shared_experts.up_proj.weight`),
          loadMatrix(`${L}.mlp.shared_experts.down_proj.weight`),
        ]);
        const sharedAct = applyActivation(linear(rms2Out, sharedGateW, null, "out_in"), activationKind);
        const sharedUpOut = linear(rms2Out, sharedUpW, null, "out_in");
        const sharedDownOut = linear(mulMatricesElementwise(sharedAct, sharedUpOut), sharedDownW, null, "out_in");
        combined = addMatrices(routedOut, sharedDownOut);
      }
      ffnOut = record(`${b}.ffn`, combined);
    } else {
      const gateW = await loadMatrix(`${L}.mlp.gate_proj.weight`);
      const upW = await loadMatrix(`${L}.mlp.up_proj.weight`);
      const gateOut = record(`${b}.ffn.gate`, linear(rms2Out, gateW, null, "out_in"));
      const gateAct = record(`${b}.ffn.gate_act`, applyActivation(gateOut, activationKind));
      const upOut = record(`${b}.ffn.up`, linear(rms2Out, upW, null, "out_in"));
      const mulOut = record(`${b}.ffn.mul`, mulMatricesElementwise(gateAct, upOut));
      const downW = await loadMatrix(`${L}.mlp.down_proj.weight`);
      const ffnProjected = record(`${b}.ffn.down`, linear(mulOut, downW, null, "out_in"));
      ffnOut = record(`${b}.ffn`, ffnProjected);
    }

    const res2 = record(`${b}.res2`, addMatrices(ffnOut, res1));
    x = record(b, res2);
  }

  const normg = await loadVector("model.norm.weight");
  const normOut = record("norm", rmsNorm(x, normg, eps));

  const lmHeadRef = model.nodes["lm_head"].parameters[0];
  const lmHeadW = await loadMatrix(lmHeadRef.name);
  const logits = record("lm_head", linear(normOut, lmHeadW, null, "out_in"));

  return {
    tokenIds,
    tokens: tokenIds.map((id) => `#${id}`),
    activations,
    attentionWeights,
    logits: matrixToTensor(logits),
  };
}
