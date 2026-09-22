import type { EdgeKind, Model, ModelConfig, ModelMetadata, ModelNode, NodeType, ParameterRef, TensorSlice } from "@tensorium/model-ir";
import { numElements, dtypeSize, modelSourceLabel } from "@tensorium/model-ir";

/**
 * `meta-models/Muse-Glimmer-30B` ships a genuinely multimodal checkpoint —
 * a text decoder plus a separate ViT-style vision tower and a small
 * vision-to-text adapter, under `model.language_model.*` /
 * `model.vision_tower.*` / `model.vision_adapter.*` +
 * `model.vision_projection.*` weight prefixes respectively. Like Gemma 4
 * and Qwen3.5/Qwen3.8, this adapter deliberately covers the text decoder
 * only (`text_config` below, weights under `model.language_model.*`); the
 * vision tower isn't a transformer block in the sense this app's
 * graph/Inspector/forward-pass machinery understands (patch embedding,
 * windowed non-causal attention, pixel-shuffle merging).
 *
 * Every field below was confirmed against the real `transformers` package
 * source for `model_type: "muse_glimmer"` (`modeling_muse_glimmer.py` /
 * `modular_muse_glimmer.py` / `configuration_muse_glimmer.py`, fetched
 * directly from the `meta-models/Muse-Glimmer-30B` checkpoint's own repo
 * reference and from the model's real `transformers` module on GitHub) and
 * the checkpoint's real `model.safetensors.index.json`, not guessed.
 */
export interface MuseGlimmerRawConfig {
  model_type?: string;
  architectures?: string[];
  text_config: MuseGlimmerTextRawConfig;
}

export interface MuseGlimmerTextRawConfig {
  model_type?: string;
  hidden_size: number;
  intermediate_size: number;
  num_hidden_layers: number;
  num_attention_heads: number;
  num_key_value_heads?: number;
  /** Explicit, like Gemma — not derived from hidden_size / num_attention_heads (real checkpoint: 6656/32 != 128). */
  head_dim?: number;
  vocab_size: number;
  max_position_embeddings?: number;
  /** Epsilon for the sandwich norms' *pre* pair (input_layernorm, pre_feedforward_layernorm). */
  rms_norm_eps?: number;
  /** Epsilon for the sandwich norms' *post* pair (post_attention_layernorm, post_feedforward_layernorm) — genuinely different from rms_norm_eps in the real checkpoint (1e-8 vs 1e-5), not a typo. */
  post_norm_eps?: number;
  hidden_activation?: string;
  tie_word_embeddings?: boolean;
  attention_bias?: boolean;
  /** Gemma-2/3-style final-logits softcap, applied once after the LM head — but see output_multiplier below, which this checkpoint applies *before* the softcap divide, on top of the usual mechanism. */
  final_logit_softcapping?: number | null;
  /** Multiplier applied to Q right after the (weightless) QK-norm, on top of the standard 1/sqrt(head_dim) attention scaling baked into the attention op itself — i.e. the real effective score scale is qk_scale_factor / sqrt(head_dim), not one or the other. */
  qk_scale_factor?: number;
  /** Scales logits before the final tanh softcap divide: `softcap * tanh(logits * output_multiplier / softcap)`. */
  output_multiplier?: number;
  sliding_window?: number | null;
  /** One entry per layer: "full_attention" or "sliding_attention" — every 4th layer counted backward from the last is full_attention (and NoPE, see layer_rope_theta), the rest are sliding_attention (window = sliding_window, with RoPE). */
  layer_types: string[];
  /** One entry per layer: 0 means that layer gets no positional encoding at all (NoPE — position_embeddings=None reaches attention); non-zero (always the same global theta in every real checkpoint seen) means the layer gets ordinary RoPE. Despite the per-layer-value shape, there's only ever one theta in practice — rope_parameters.rope_theta below — this field is really just a per-layer NoPE on/off switch. */
  layer_rope_theta?: (number | null)[] | null;
  rope_parameters?: { rope_theta?: number; rope_type?: string } | null;
}

export function buildModelConfig(raw: MuseGlimmerRawConfig): ModelConfig {
  const t = raw.text_config;
  const numLayers = t.num_hidden_layers;

  const layerTypes =
    t.layer_types ?? Array.from({ length: numLayers }, (_, i) => ((numLayers - 1 - i) % 4 === 0 ? "full_attention" : "sliding_attention"));
  if (layerTypes.length !== numLayers) {
    throw new Error(`text_config.layer_types has ${layerTypes.length} entries but num_hidden_layers is ${numLayers} — can't tell each layer's attention pattern.`);
  }

  const ropeTheta = t.rope_parameters?.rope_theta ?? 500000;
  const layerRopeTheta = t.layer_rope_theta ?? layerTypes.map((type) => (type === "full_attention" ? 0 : ropeTheta));
  if (layerRopeTheta.length !== numLayers) {
    throw new Error(`text_config.layer_rope_theta has ${layerRopeTheta.length} entries but num_hidden_layers is ${numLayers}.`);
  }

  return {
    modelType: raw.model_type ?? "muse_glimmer",
    numLayers,
    numHeads: t.num_attention_heads,
    hiddenSize: t.hidden_size,
    intermediateSize: t.intermediate_size,
    vocabSize: t.vocab_size,
    contextLength: t.max_position_embeddings ?? 131072,
    extra: {
      rmsNormEps: t.rms_norm_eps ?? 1e-5,
      postNormEps: t.post_norm_eps ?? 1e-8,
      activationFunction: t.hidden_activation ?? "silu",
      tiedEmbeddings: t.tie_word_embeddings ?? false,
      attentionBias: t.attention_bias ?? false,
      numKeyValueHeads: t.num_key_value_heads ?? t.num_attention_heads,
      headDim: t.head_dim ?? Math.floor(t.hidden_size / t.num_attention_heads),
      layerTypes,
      layerHasRope: layerTypes.map((_, i) => Boolean(layerRopeTheta[i])),
      slidingWindow: t.sliding_window ?? 4096,
      ropeTheta,
      qkScaleFactor: t.qk_scale_factor ?? 1,
      outputMultiplier: t.output_multiplier ?? 1,
      finalLogitSoftcapping: t.final_logit_softcapping ?? null,
    },
  };
}

type WeightIndex = Record<string, { shape: number[]; dtype: string }>;

function param(name: string, weightIndex: WeightIndex, providerId: string, slice?: TensorSlice): ParameterRef {
  const entry = weightIndex[name];
  if (!entry) throw new Error(`Missing weight in checkpoint: ${name}`);
  const full = entry.shape;
  const logicalShape = slice?.ranges ? full.map((dim, i) => (slice.ranges![i] ? slice.ranges![i].end - slice.ranges![i].start : dim)) : full;
  return {
    name,
    shape: full,
    dtype: entry.dtype,
    numElements: numElements(full),
    bytes: numElements(full) * dtypeSize(entry.dtype),
    providerId,
    slice,
    logicalShape,
  };
}

/** model.language_model.* — the text decoder's own weight prefix, distinct from the (unloaded) vision_tower/vision_adapter/vision_projection prefixes in the same file. */
const LP = "model.language_model";

export function buildGraph(metadata: ModelMetadata, providerId: string): Model {
  const cfg = metadata.config;
  const wi = metadata.weightIndex;
  const nodes: Record<string, ModelNode> = {};
  const edges: Model["edges"] = [];

  const numHeads = cfg.numHeads;
  const numKVHeads = Number(cfg.extra.numKeyValueHeads);
  const headDim = Number(cfg.extra.headDim);
  const qDim = numHeads * headDim;
  const kvDim = numKVHeads * headDim;
  const layerTypes = cfg.extra.layerTypes as string[];
  const layerHasRope = cfg.extra.layerHasRope as boolean[];
  const hasAttnBias = cfg.extra.attentionBias === true;
  const rmsNormEps = Number(cfg.extra.rmsNormEps);
  const postNormEps = Number(cfg.extra.postNormEps);
  const qkScaleFactor = Number(cfg.extra.qkScaleFactor ?? 1);
  const outputMultiplier = Number(cfg.extra.outputMultiplier ?? 1);
  const finalLogitSoftcapping = cfg.extra.finalLogitSoftcapping != null ? Number(cfg.extra.finalLogitSoftcapping) : null;

  function node(id: string, type: NodeType, name: string, parentId: string | null, opts: Partial<ModelNode> = {}): ModelNode {
    const n: ModelNode = {
      id,
      type,
      name,
      inputs: opts.inputs ?? [],
      outputs: opts.outputs ?? [],
      parameters: opts.parameters ?? [],
      children: [],
      parentId,
      metadata: opts.metadata ?? {},
    };
    nodes[id] = n;
    if (parentId) nodes[parentId].children.push(id);
    return n;
  }

  function edge(source: string, target: string, label?: string, kind?: EdgeKind) {
    edges.push({ id: `${source}->${target}`, source, target, label, kind: kind ?? (label === "skip" ? "residual" : "data") });
  }

  /** MuseGlimmerTextCenteredRMSNorm (Gemma2RMSNorm): scales by (1 + weight), weight zero-initialized — the four sandwich norms every block has. */
  function centeredNormNode(id: string, label: string, parentId: string, weightName: string, eps: number) {
    return node(id, "rms_norm", label, parentId, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: seqH }],
      parameters: [param(weightName, wi, providerId)],
      metadata: { note: "Gemma-2-style (1 + weight) RMSNorm, weight zero-initialized.", eps },
    });
  }

  /** MuseGlimmerRMSNorm (plain-weight, Gemma4RMSNorm-derived): used for the embedding norm (weightless), Q/K norm (weightless), and the final norm (weighted, real `norm.weight` tensor). */
  function plainNormNode(id: string, label: string, parentId: string, dim: number, weightName: string | null, eps: number) {
    return node(id, "rms_norm", label, parentId, {
      inputs: [{ dims: ["sequence_length", dim] }],
      outputs: [{ dims: ["sequence_length", dim] }],
      parameters: weightName ? [param(weightName, wi, providerId)] : [],
      metadata: weightName ? { eps } : { note: "Parameter-free (with_scale=False in the real model) — normalizes by RMS only, no learned scale.", eps },
    });
  }

  const H = cfg.hiddenSize;
  const seqH: Array<number | string> = ["sequence_length", H];

  // --- root -------------------------------------------------------------
  node("model", "model", metadata.architecture, null);

  node("input", "input", "Input tokens", "model", {
    outputs: [{ dims: ["sequence_length"] }],
    metadata: { description: "Token IDs produced by the tokenizer." },
  });

  node("embed", "embedding", "Token Embedding", "model", {
    inputs: [{ dims: ["sequence_length"] }],
    outputs: [{ dims: seqH }],
    parameters: [param(`${LP}.embed_tokens.weight`, wi, providerId)],
    metadata: { description: "No learned positional embedding table — position is injected later, inside attention, via RoPE (only on sliding-attention layers, see below)." },
  });
  edge("input", "embed");

  const embedNorm = "embed_norm";
  plainNormNode(embedNorm, "RMSNorm (post-embedding)", "model", H, null, rmsNormEps);
  edge("embed", embedNorm);

  node("blocks", "block_group", `Transformer Blocks × ${cfg.numLayers}`, "model", {
    metadata: { count: cfg.numLayers },
  });

  let prevOut = embedNorm;
  for (let i = 0; i < cfg.numLayers; i++) {
    const b = `block.${i}`;
    const L = `${LP}.layers.${i}`;
    const layerType = layerTypes[i];
    const hasRope = layerHasRope[i];

    node(b, "transformer_block", `Transformer Block ${i}`, "blocks", {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: seqH }],
      metadata: { layerType, hasRope },
    });
    edge(prevOut, b);

    const rms1 = `${b}.rms1`;
    centeredNormNode(rms1, "RMSNorm (pre-attention)", b, `${L}.input_layernorm.weight`, rmsNormEps);
    edge(b, rms1);

    const attn = `${b}.attn`;
    node(attn, "attention", `Attention (${layerType === "full_attention" ? "full" : "sliding"}${hasRope ? "" : ", NoPE"})`, b, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: seqH }],
      metadata: {
        numHeads,
        numKeyValueHeads: numKVHeads,
        headDim,
        groupedQueryAttention: numKVHeads !== numHeads,
        slidingWindow: layerType === "sliding_attention" ? cfg.extra.slidingWindow : undefined,
        effectiveScale: `qk_scale_factor(${qkScaleFactor}) / sqrt(head_dim)`,
        description: hasRope
          ? `Causal attention restricted to the last ${cfg.extra.slidingWindow} positions (a local window), head_dim=${headDim}, with ordinary RoPE (θ=${cfg.extra.ropeTheta}).`
          : `Ordinary causal attention over the whole sequence, head_dim=${headDim} — one of the periodic NoPE layers: no positional encoding of any kind reaches this layer's Q/K, unlike the RoPE'd sliding-window layers around it.`,
      },
    });

    const q = `${attn}.q`;
    const k = `${attn}.k`;
    const v = `${attn}.v`;
    const qkvBiasParams = (name: string) => (hasAttnBias ? [param(`${L}.self_attn.${name}.weight`, wi, providerId), param(`${L}.self_attn.${name}.bias`, wi, providerId)] : [param(`${L}.self_attn.${name}.weight`, wi, providerId)]);
    node(q, "q_projection", "Q Projection", attn, { inputs: [{ dims: seqH }], outputs: [{ dims: ["sequence_length", qDim] }], parameters: qkvBiasParams("q_proj") });
    node(k, "k_projection", "K Projection", attn, { inputs: [{ dims: seqH }], outputs: [{ dims: ["sequence_length", kvDim] }], parameters: qkvBiasParams("k_proj") });
    node(v, "v_projection", "V Projection", attn, { inputs: [{ dims: seqH }], outputs: [{ dims: ["sequence_length", kvDim] }], parameters: qkvBiasParams("v_proj") });
    edge(rms1, q);
    edge(rms1, k);
    edge(rms1, v);

    // A single shared, weightless RMSNorm module normalizes Q and K
    // per-head (over head_dim) — the same nn.Module instance in the real
    // model, just called twice; Q additionally gets multiplied by the
    // constant qk_scale_factor right after, K doesn't.
    const qNorm = `${attn}.q_norm`;
    plainNormNode(qNorm, "Q Norm", attn, qDim, null, rmsNormEps);
    edge(q, qNorm);
    const qScale = `${attn}.q_scale`;
    node(qScale, "activation", `× qk_scale_factor (${qkScaleFactor})`, attn, {
      inputs: [{ dims: ["sequence_length", qDim] }],
      outputs: [{ dims: ["sequence_length", qDim] }],
      metadata: { note: "Constant multiply from config, applied to Q only (not K) — on top of, not instead of, attention's own 1/sqrt(head_dim) scaling." },
    });
    edge(qNorm, qScale);

    const kNorm = `${attn}.k_norm`;
    plainNormNode(kNorm, "K Norm", attn, kvDim, null, rmsNormEps);
    edge(k, kNorm);

    // No separate node represents the softmax/weighted-value computation
    // itself (same convention every other adapter here uses) — Q's and
    // K's final pre-attention node just feed straight into whatever's
    // next (the output gate below), alongside V.
    let qkIntoNext: string[];
    if (hasRope) {
      const rope = `${attn}.rope`;
      node(rope, "rope", "RoPE", attn, {
        inputs: [{ dims: ["sequence_length", qDim] }, { dims: ["sequence_length", kvDim] }],
        outputs: [{ dims: ["sequence_length", qDim] }, { dims: ["sequence_length", kvDim] }],
        metadata: { description: "Rotates Q and K by an angle proportional to sequence position — no learned parameters.", ropeTheta: cfg.extra.ropeTheta },
      });
      edge(qScale, rope);
      edge(kNorm, rope);
      qkIntoNext = [rope];
    } else {
      qkIntoNext = [qScale, kNorm];
    }

    // Every layer (not just sliding ones) gates its attention output: a
    // separate gate_proj reads the same pre-attention-normed input as
    // Q/K/V, and its sigmoid scales the attention heads' concatenated
    // output elementwise, right before the output projection.
    const gateId = `${attn}.gate`;
    node(gateId, "linear", "Attention Output Gate", attn, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: ["sequence_length", qDim] }],
      parameters: [param(`${L}.self_attn.gate_proj.weight`, wi, providerId)],
    });
    edge(rms1, gateId);

    const gateAct = `${attn}.gate_act`;
    node(gateAct, "activation", "sigmoid", attn, {
      inputs: [{ dims: ["sequence_length", qDim] }],
      outputs: [{ dims: ["sequence_length", qDim] }],
    });
    edge(gateId, gateAct);

    const gatedMul = `${attn}.gated_mul`;
    node(gatedMul, "elementwise_mul", "× Output Gate", attn, {
      inputs: [{ dims: ["sequence_length", qDim] }, { dims: ["sequence_length", qDim] }],
      outputs: [{ dims: ["sequence_length", qDim] }],
    });
    for (const src of qkIntoNext) edge(src, gatedMul);
    edge(v, gatedMul);
    edge(gateAct, gatedMul);

    const outp = `${attn}.out`;
    node(outp, "output_projection", "Output Projection", attn, {
      inputs: [{ dims: ["sequence_length", qDim] }],
      outputs: [{ dims: seqH }],
      parameters: hasAttnBias ? [param(`${L}.self_attn.o_proj.weight`, wi, providerId), param(`${L}.self_attn.o_proj.bias`, wi, providerId)] : [param(`${L}.self_attn.o_proj.weight`, wi, providerId)],
    });
    edge(gatedMul, outp);

    const postAttnNorm = `${b}.post_attn_norm`;
    centeredNormNode(postAttnNorm, "RMSNorm (post-attention, sandwich)", b, `${L}.post_attention_layernorm.weight`, postNormEps);
    edge(outp, postAttnNorm);

    const res1 = `${b}.res1`;
    node(res1, "residual", "Residual Add", b, {
      inputs: [{ dims: seqH }, { dims: seqH }],
      outputs: [{ dims: seqH }],
      metadata: { description: "Adds the block's input back in around Attention (after Attention's own post-norm)." },
    });
    edge(postAttnNorm, res1);
    edge(b, res1, "skip");

    const preFfnNorm = `${b}.pre_ffn_norm`;
    centeredNormNode(preFfnNorm, "RMSNorm (pre-FFN, sandwich)", b, `${L}.pre_feedforward_layernorm.weight`, rmsNormEps);
    edge(res1, preFfnNorm);

    const ffn = `${b}.ffn`;
    node(ffn, "ffn", "Feed Forward (gated)", b, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: seqH }],
    });

    const gate = `${ffn}.gate`;
    const up = `${ffn}.up`;
    node(gate, "linear", "Gate Projection", ffn, { inputs: [{ dims: seqH }], outputs: [{ dims: ["sequence_length", cfg.intermediateSize] }], parameters: [param(`${L}.mlp.gate_proj.weight`, wi, providerId)] });
    node(up, "linear", "Up Projection", ffn, { inputs: [{ dims: seqH }], outputs: [{ dims: ["sequence_length", cfg.intermediateSize] }], parameters: [param(`${L}.mlp.up_proj.weight`, wi, providerId)] });
    edge(preFfnNorm, gate);
    edge(preFfnNorm, up);

    const gateAct2 = `${ffn}.gate_act`;
    node(gateAct2, "activation", String(cfg.extra.activationFunction ?? "silu"), ffn, {
      inputs: [{ dims: ["sequence_length", cfg.intermediateSize] }],
      outputs: [{ dims: ["sequence_length", cfg.intermediateSize] }],
    });
    edge(gate, gateAct2);

    const mul = `${ffn}.mul`;
    node(mul, "elementwise_mul", "Gate × Up", ffn, {
      inputs: [{ dims: ["sequence_length", cfg.intermediateSize] }, { dims: ["sequence_length", cfg.intermediateSize] }],
      outputs: [{ dims: ["sequence_length", cfg.intermediateSize] }],
    });
    edge(gateAct2, mul);
    edge(up, mul);

    const down = `${ffn}.down`;
    node(down, "linear", "Down Projection", ffn, {
      inputs: [{ dims: ["sequence_length", cfg.intermediateSize] }],
      outputs: [{ dims: seqH }],
      parameters: [param(`${L}.mlp.down_proj.weight`, wi, providerId)],
    });
    edge(mul, down);

    const postFfnNorm = `${b}.post_ffn_norm`;
    centeredNormNode(postFfnNorm, "RMSNorm (post-FFN, sandwich)", b, `${L}.post_feedforward_layernorm.weight`, postNormEps);
    edge(down, postFfnNorm);

    const res2 = `${b}.res2`;
    node(res2, "residual", "Residual Add", b, {
      inputs: [{ dims: seqH }, { dims: seqH }],
      outputs: [{ dims: seqH }],
      metadata: { description: "Adds the pre-FFN state back in around the Feed Forward block (after the FFN's own post-norm)." },
    });
    edge(postFfnNorm, res2);
    edge(res1, res2, "skip");

    prevOut = res2;
  }

  const finalNorm = "norm";
  plainNormNode(finalNorm, "Final RMSNorm", "model", H, `${LP}.norm.weight`, Number(cfg.extra.rmsNormEps));
  edge(prevOut, finalNorm);

  // Unlike every layer's weights, lm_head.weight (when untied) lives at the
  // very top of the checkpoint, not under model.language_model.* — real
  // model puts `self.lm_head = nn.Linear(...)` directly on
  // MuseGlimmerForConditionalGeneration, not inside MuseGlimmerModel.
  const tied = !wi["lm_head.weight"];
  node("lm_head", "lm_head", "LM Head", "model", {
    inputs: [{ dims: seqH }],
    outputs: [{ dims: ["sequence_length", cfg.vocabSize] }],
    parameters: [param(tied ? `${LP}.embed_tokens.weight` : "lm_head.weight", wi, providerId)],
    metadata: { tied, description: tied ? "Tied to the token embedding weight (transposed)." : undefined },
  });
  edge(finalNorm, "lm_head");

  let lastId = "lm_head";
  if (outputMultiplier !== 1) {
    const scaleId = "output_multiplier";
    node(scaleId, "activation", `× Output Multiplier (${outputMultiplier})`, "model", {
      inputs: [{ dims: ["sequence_length", cfg.vocabSize] }],
      outputs: [{ dims: ["sequence_length", cfg.vocabSize] }],
      metadata: { description: `Constant multiply from config, applied before the softcap divide below: logits *= ${outputMultiplier}.` },
    });
    edge(lastId, scaleId);
    lastId = scaleId;
  }

  if (finalLogitSoftcapping != null) {
    const softcapId = "logit_softcap";
    node(softcapId, "activation", "Logit Softcap (tanh)", "model", {
      inputs: [{ dims: ["sequence_length", cfg.vocabSize] }],
      outputs: [{ dims: ["sequence_length", cfg.vocabSize] }],
      metadata: { description: `Gemma-style softcap: logits = tanh(logits / ${finalLogitSoftcapping}) * ${finalLogitSoftcapping} — keeps any single logit from growing unboundedly large.` },
    });
    edge(lastId, softcapId);
    lastId = softcapId;
  }

  node("output", "output", "Logits", "model", {
    inputs: [{ dims: ["sequence_length", cfg.vocabSize] }],
  });
  edge(lastId, "output");

  return {
    id: modelSourceLabel(metadata.source),
    name: modelSourceLabel(metadata.source),
    architecture: metadata.architecture,
    config: cfg,
    inputs: nodes["input"].outputs,
    outputs: nodes["output"].inputs,
    nodes,
    edges,
    rootId: "model",
  };
}
