import type { Model, ModelConfig, ModelMetadata, ModelNode, NodeType, ParameterRef, TensorSlice } from "@tensorium/model-ir";
import { numElements, dtypeSize, modelSourceLabel } from "@tensorium/model-ir";

/**
 * `google/gemma-4-*` ships a genuinely multimodal checkpoint — a text
 * decoder plus separate vision and audio towers, each under its own
 * `model.{language_model,vision_tower,audio_tower}.*` weight prefix. This
 * adapter deliberately covers the text decoder only (`text_config` below,
 * weights under `model.language_model.*`); the vision/audio towers aren't
 * transformer blocks in the sense this app's graph/Inspector/forward-pass
 * machinery understands at all, and every other model here is text-only.
 *
 * Every field below was confirmed against the real `transformers` package
 * source (v5.15.1's `models/gemma4/{modeling,configuration}_gemma4.py`),
 * not guessed — this architecture is new enough that a plausible-looking
 * guess had a real chance of being subtly wrong (see e.g. the RoPE
 * comments below, and DecoderLayer's residual/norm order in graph.ts).
 */
export interface Gemma4RawConfig {
  model_type?: string;
  architectures?: string[];
  text_config: Gemma4TextRawConfig;
}

export interface Gemma4TextRawConfig {
  model_type?: string;
  hidden_size: number;
  intermediate_size: number;
  num_hidden_layers: number;
  num_attention_heads: number;
  /** The one shared KV head every layer uses (extreme MQA) — real config.json values seen so far are always 1. */
  num_key_value_heads?: number;
  /** Sliding-attention layers' head_dim. Full-attention layers use `global_head_dim` instead (see per_layer_config in the real config — confirmed against real weight shapes: q_proj is [num_heads*global_head_dim, hidden] on full-attention layers, [num_heads*head_dim, hidden] on sliding ones). */
  head_dim?: number;
  global_head_dim?: number;
  num_global_key_value_heads?: number | null;
  /** Gates whether num_global_key_value_heads actually overrides anything — false in every real checkpoint seen so far, meaning full-attention layers keep the same (1) KV head as sliding ones. */
  attention_k_eq_v?: boolean;
  vocab_size: number;
  max_position_embeddings?: number;
  rms_norm_eps?: number;
  /** Real value seen: "gelu_pytorch_tanh", not "silu" — read directly, never assumed. */
  hidden_activation?: string;
  tie_word_embeddings?: boolean;
  attention_bias?: boolean;
  /** Gemma-2/3-style final-logits softcap: logits = tanh(logits/cap) * cap, applied once, only after the LM head — never inside per-layer attention (confirmed: no `softcap=` kwarg reaches eager_attention_forward from this model's attention layer). */
  final_logit_softcapping?: number | null;
  sliding_window?: number;
  /** One entry per layer: "full_attention" or "sliding_attention" — determines that layer's head_dim, RoPE config, and attention mask. */
  layer_types: string[];
  /** The last N layers (N = this value) don't compute their own K/V at all — they reuse a frozen K/V computed once by the *last* layer of the same type among the first `num_hidden_layers - num_kv_shared_layers` layers. Confirmed both via source (Gemma4TextAttention.__init__/forward) and by checking real checkpoints: those later layers' own k_proj/k_norm/v_proj/v_norm weights are present in the file but genuinely unused (never constructed as submodules by the real model, so never loaded) — an intentional bit of extra baggage in the checkpoint format, not a bug in this reading. */
  num_kv_shared_layers?: number;
  /** > 0 enables Per-Layer Embeddings: a second embedding table, looked up once per token and injected additively into every layer (see buildGraph's "Per-Layer Input Projection" node and each layer's own injection step). 0/absent disables the whole mechanism. */
  hidden_size_per_layer_input?: number;
  vocab_size_per_layer_input?: number;
  /** Doubles intermediate_size, but only on the KV-shared layers (same layer_idx >= first_kv_shared_layer_idx boundary) — confirmed against real weight shapes (mlp.gate_proj: 6144-wide on layers 0-14, 12288-wide on layers 15-34 for the E2B checkpoint). */
  use_double_wide_mlp?: boolean;
  rope_parameters?: {
    full_attention?: { rope_theta?: number; partial_rotary_factor?: number; rope_type?: string; factor?: number };
    sliding_attention?: { rope_theta?: number; rope_type?: string; factor?: number };
  } | null;
}

export function buildModelConfig(raw: Gemma4RawConfig): ModelConfig {
  const t = raw.text_config;
  const numLayers = t.num_hidden_layers;
  const numKvSharedLayers = t.num_kv_shared_layers ?? 0;
  const firstKvSharedLayerIdx = numLayers - numKvSharedLayers;
  const layerTypes = t.layer_types;

  if (layerTypes.length !== numLayers) {
    throw new Error(`text_config.layer_types has ${layerTypes.length} entries but num_hidden_layers is ${numLayers} — can't tell each layer's attention pattern.`);
  }

  return {
    modelType: raw.model_type ?? "gemma4",
    numLayers,
    numHeads: t.num_attention_heads,
    hiddenSize: t.hidden_size,
    intermediateSize: t.intermediate_size,
    vocabSize: t.vocab_size,
    contextLength: t.max_position_embeddings ?? 131072,
    extra: {
      rmsNormEps: t.rms_norm_eps ?? 1e-6,
      activationFunction: t.hidden_activation ?? "gelu_pytorch_tanh",
      tiedEmbeddings: t.tie_word_embeddings ?? true,
      attentionBias: t.attention_bias ?? false,
      finalLogitSoftcapping: t.final_logit_softcapping ?? null,
      slidingWindow: t.sliding_window ?? 512,
      layerTypes,
      numKeyValueHeads: t.num_key_value_heads ?? 1,
      headDim: t.head_dim ?? Math.floor(t.hidden_size / t.num_attention_heads),
      globalHeadDim: t.global_head_dim ?? 512,
      numKvSharedLayers,
      firstKvSharedLayerIdx,
      hiddenSizePerLayerInput: t.hidden_size_per_layer_input ?? 0,
      vocabSizePerLayerInput: t.vocab_size_per_layer_input ?? t.vocab_size,
      useDoubleWideMlp: t.use_double_wide_mlp ?? false,
      ropeTheta: t.rope_parameters?.sliding_attention?.rope_theta ?? 10000,
      ropeThetaGlobal: t.rope_parameters?.full_attention?.rope_theta ?? 1000000,
      partialRotaryFactorGlobal: t.rope_parameters?.full_attention?.partial_rotary_factor ?? 0.25,
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

/** model.language_model.* — the text decoder's own weight prefix, distinct from the (unloaded) vision_tower/audio_tower prefixes in the same file. */
const LP = "model.language_model";

/** The layer index (< firstKvSharedLayerIdx) whose K/V a KV-shared layer reuses: the *last* occurrence of the same layer_type among the non-shared layers — mirrors DeepseekV2Attention.__init__'s store_full_length_kv logic exactly, just computed the straightforward way instead of via reversed-list index arithmetic. */
export function donorLayerIndex(layerTypes: string[], firstKvSharedLayerIdx: number, layerIdx: number): number {
  const type = layerTypes[layerIdx];
  for (let j = firstKvSharedLayerIdx - 1; j >= 0; j--) {
    if (layerTypes[j] === type) return j;
  }
  return -1;
}

export function buildGraph(metadata: ModelMetadata, providerId: string): Model {
  const cfg = metadata.config;
  const wi = metadata.weightIndex;
  const nodes: Record<string, ModelNode> = {};
  const edges: Model["edges"] = [];

  const layerTypes = cfg.extra.layerTypes as string[];
  const numHeads = cfg.numHeads;
  const numKVHeads = Number(cfg.extra.numKeyValueHeads);
  const headDimSliding = Number(cfg.extra.headDim);
  const headDimGlobal = Number(cfg.extra.globalHeadDim);
  const firstKvSharedLayerIdx = Number(cfg.extra.firstKvSharedLayerIdx);
  const useDoubleWideMlp = cfg.extra.useDoubleWideMlp === true;
  const hasAttnBias = cfg.extra.attentionBias === true;
  const hiddenSizePerLayerInput = Number(cfg.extra.hiddenSizePerLayerInput ?? 0);
  const hasPerLayerInput = hiddenSizePerLayerInput > 0;

  const headDimForLayer = (i: number) => (layerTypes[i] === "full_attention" ? headDimGlobal : headDimSliding);
  const isKvSharedLayer = (i: number) => i >= firstKvSharedLayerIdx;

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

  function edge(source: string, target: string, label?: string) {
    edges.push({ id: `${source}->${target}`, source, target, label });
  }

  function normNode(id: string, label: string, parentId: string, weightName: string, dim: number, opts: { withScale?: boolean } = {}) {
    const withScale = opts.withScale ?? true;
    return node(id, "rms_norm", label, parentId, {
      inputs: [{ dims: ["sequence_length", dim] }],
      outputs: [{ dims: ["sequence_length", dim] }],
      parameters: withScale ? [param(weightName, wi, providerId)] : [],
      metadata: withScale ? {} : { note: "Parameter-free (with_scale=False in the real model) — normalizes by RMS only, no learned scale, so there's no weight tensor for this one." },
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
    metadata: { description: "Scaled by √hidden_size right after lookup (standard Gemma convention) — no learned positional embedding table; position is injected inside attention via RoPE." },
  });
  edge("input", "embed");

  // Per-Layer Embeddings: a *second*, separate embedding table (real weight:
  // embed_tokens_per_layer, [vocab_size, hidden_size_per_layer_input *
  // num_layers]) is looked up once per token, combined with a global
  // projection of the *main* embedding, and the result gets sliced per
  // layer and injected additively into every single transformer block —
  // computed once here rather than duplicated per layer.
  let perLayerInputId: string | null = null;
  if (hasPerLayerInput) {
    perLayerInputId = "per_layer_input";
    node(perLayerInputId, "linear", "Per-Layer Input Projection", "model", {
      inputs: [{ dims: ["sequence_length"] }, { dims: seqH }],
      outputs: [{ dims: ["sequence_length", cfg.numLayers, hiddenSizePerLayerInput] }],
      parameters: [
        param(`${LP}.embed_tokens_per_layer.weight`, wi, providerId),
        param(`${LP}.per_layer_model_projection.weight`, wi, providerId),
        param(`${LP}.per_layer_projection_norm.weight`, wi, providerId),
      ],
      metadata: {
        description:
          "Per-Layer Embeddings: looks up a second embedding table by token id (scaled by √hidden_size_per_layer_input), separately projects the main token embedding down to the same width (scaled by 1/√hidden_size) and RMS-normalizes it, sums the two, then scales by 1/√2 — producing one hidden_size_per_layer_input-wide vector per layer, per token, sliced out and injected into each block below.",
      },
    });
    edge("input", perLayerInputId);
    edge("embed", perLayerInputId);
  }

  node("blocks", "block_group", `Transformer Blocks × ${cfg.numLayers}`, "model", {
    metadata: { count: cfg.numLayers },
  });

  let prevOut = "embed";
  for (let i = 0; i < cfg.numLayers; i++) {
    const b = `block.${i}`;
    const L = `${LP}.layers.${i}`;
    const layerType = layerTypes[i];
    const headDim = headDimForLayer(i);
    const qDim = numHeads * headDim;
    const kvDim = numKVHeads * headDim;
    const sharedLayer = isKvSharedLayer(i);
    const doubledMlp = useDoubleWideMlp && sharedLayer;
    const intermediateSize = cfg.intermediateSize * (doubledMlp ? 2 : 1);

    node(b, "transformer_block", `Transformer Block ${i}`, "blocks", {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: seqH }],
      metadata: { layerType, headDim, kvShared: sharedLayer },
    });
    edge(prevOut, b);

    const rms1 = `${b}.rms1`;
    normNode(rms1, "RMSNorm (pre-attention)", b, `${L}.input_layernorm.weight`, H);
    edge(b, rms1);

    const attn = `${b}.attn`;
    node(attn, "attention", `Attention (${layerType === "full_attention" ? "global" : "sliding"})`, b, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: seqH }],
      metadata: {
        numHeads,
        numKeyValueHeads: numKVHeads,
        headDim,
        layerType,
        slidingWindow: layerType === "sliding_attention" ? cfg.extra.slidingWindow : undefined,
        fixedAttentionScale: 1.0,
        description:
          layerType === "full_attention"
            ? `Ordinary causal attention over the whole sequence, head_dim=${headDim} — this is one of the periodic "full_attention" layers that give every token eventual access to the whole context, unlike the sliding-window layers around it.`
            : `Causal attention restricted to the last ${cfg.extra.slidingWindow} positions (a local window), head_dim=${headDim} — most layers in this model are this cheaper local form; only every 5th layer is full-context.`,
      },
    });

    const q = `${attn}.q`;
    node(q, "q_projection", "Q Projection", attn, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: ["sequence_length", qDim] }],
      parameters: hasAttnBias
        ? [param(`${L}.self_attn.q_proj.weight`, wi, providerId), param(`${L}.self_attn.q_proj.bias`, wi, providerId)]
        : [param(`${L}.self_attn.q_proj.weight`, wi, providerId)],
    });
    edge(rms1, q);

    const qNorm = `${attn}.q_norm`;
    normNode(qNorm, "Q Norm", attn, `${L}.self_attn.q_norm.weight`, qDim);
    edge(q, qNorm);

    let kSource: string;
    let vSource: string;
    if (!sharedLayer) {
      const k = `${attn}.k`;
      node(k, "k_projection", "K Projection", attn, {
        inputs: [{ dims: seqH }],
        outputs: [{ dims: ["sequence_length", kvDim] }],
        parameters: hasAttnBias
          ? [param(`${L}.self_attn.k_proj.weight`, wi, providerId), param(`${L}.self_attn.k_proj.bias`, wi, providerId)]
          : [param(`${L}.self_attn.k_proj.weight`, wi, providerId)],
      });
      edge(rms1, k);
      const kNorm = `${attn}.k_norm`;
      normNode(kNorm, "K Norm", attn, `${L}.self_attn.k_norm.weight`, kvDim);
      edge(k, kNorm);
      kSource = kNorm;

      const v = `${attn}.v`;
      node(v, "v_projection", "V Projection", attn, {
        inputs: [{ dims: seqH }],
        outputs: [{ dims: ["sequence_length", kvDim] }],
        parameters: hasAttnBias
          ? [param(`${L}.self_attn.v_proj.weight`, wi, providerId), param(`${L}.self_attn.v_proj.bias`, wi, providerId)]
          : [param(`${L}.self_attn.v_proj.weight`, wi, providerId)],
      });
      edge(rms1, v);
      const vNorm = `${attn}.v_norm`;
      normNode(vNorm, "V Norm", attn, `${L}.self_attn.v_norm.weight`, kvDim, { withScale: false });
      edge(v, vNorm);
      vSource = vNorm;
    } else {
      const donor = donorLayerIndex(layerTypes, firstKvSharedLayerIdx, i);
      // No k_proj/k_norm/v_proj/v_norm nodes at all for this layer — the
      // real model never constructs those submodules once past the
      // sharing boundary, so there's nothing here to run, even though the
      // checkpoint still carries (unused) weight tensors under this
      // layer's name. K/V instead come from whichever earlier layer of the
      // same attention type is the last one before the sharing boundary —
      // referencing that donor's K/V-norm nodes here, the closest real
      // per-K/per-V nodes this graph has (the donor's own RoPE node
      // conflates Q and K, so it isn't a clean K-only reference).
      kSource = `block.${donor}.attn.k_norm`;
      vSource = `block.${donor}.attn.v_norm`;
      node(`${attn}.shared_kv_note`, "linear", `K/V reused from Block ${donor}`, attn, {
        inputs: [],
        outputs: [{ dims: ["sequence_length", kvDim] }],
        parameters: [],
        metadata: {
          description: `This layer computes no K/V of its own — from layer ${firstKvSharedLayerIdx} onward, every layer reuses the frozen K/V that Block ${donor} (the last ${layerType === "full_attention" ? "full-attention" : "sliding-window"} layer before the sharing boundary) computed once, already rotated by that layer's own RoPE step. Its checkpoint still ships k_proj/k_norm/v_proj/v_norm weights under this layer's name, but the real model never loads or runs them.`,
        },
      });
      edge(kSource, `${attn}.shared_kv_note`, "skip");
    }

    const rope = `${attn}.rope`;
    node(rope, "rope", "RoPE (decoupled by layer type)", attn, {
      inputs: [{ dims: ["sequence_length", qDim] }],
      outputs: [{ dims: ["sequence_length", qDim] }],
      metadata: {
        description:
          layerType === "full_attention"
            ? `A "proportional" RoPE variant: only the first ${Math.floor((Number(cfg.extra.partialRotaryFactorGlobal) * headDim) / 2) * 2} of this head's ${headDim} dims actually rotate (θ=${cfg.extra.ropeThetaGlobal}) — the rest pass through unrotated, encoded here as zero rotation frequency rather than a separate slice.`
            : `Standard full-head RoPE over all ${headDim} dims, θ=${cfg.extra.ropeTheta}.`,
        ropeTheta: layerType === "full_attention" ? cfg.extra.ropeThetaGlobal : cfg.extra.ropeTheta,
      },
    });
    edge(qNorm, rope);
    if (!sharedLayer) edge(kSource, rope);

    const outp = `${attn}.out`;
    node(outp, "output_projection", "Output Projection", attn, {
      inputs: [{ dims: ["sequence_length", numHeads * headDim] }],
      outputs: [{ dims: seqH }],
      parameters: hasAttnBias
        ? [param(`${L}.self_attn.o_proj.weight`, wi, providerId), param(`${L}.self_attn.o_proj.bias`, wi, providerId)]
        : [param(`${L}.self_attn.o_proj.weight`, wi, providerId)],
    });
    edge(rope, outp);
    edge(vSource, outp);

    const postAttnNorm = `${b}.post_attn_norm`;
    normNode(postAttnNorm, "RMSNorm (post-attention)", b, `${L}.post_attention_layernorm.weight`, H);
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
    normNode(preFfnNorm, "RMSNorm (pre-FFN)", b, `${L}.pre_feedforward_layernorm.weight`, H);
    edge(res1, preFfnNorm);

    const ffn = `${b}.ffn`;
    node(ffn, "ffn", doubledMlp ? "Feed Forward (gated, double-wide)" : "Feed Forward (gated)", b, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: seqH }],
    });

    const gate = `${ffn}.gate`;
    const up = `${ffn}.up`;
    node(gate, "linear", "Gate Projection", ffn, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: ["sequence_length", intermediateSize] }],
      parameters: [param(`${L}.mlp.gate_proj.weight`, wi, providerId)],
    });
    node(up, "linear", "Up Projection", ffn, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: ["sequence_length", intermediateSize] }],
      parameters: [param(`${L}.mlp.up_proj.weight`, wi, providerId)],
    });
    edge(preFfnNorm, gate);
    edge(preFfnNorm, up);

    const gateAct = `${ffn}.gate_act`;
    node(gateAct, "activation", String(cfg.extra.activationFunction ?? "gelu_pytorch_tanh"), ffn, {
      inputs: [{ dims: ["sequence_length", intermediateSize] }],
      outputs: [{ dims: ["sequence_length", intermediateSize] }],
    });
    edge(gate, gateAct);

    const mul = `${ffn}.mul`;
    node(mul, "elementwise_mul", "Gate × Up", ffn, {
      inputs: [{ dims: ["sequence_length", intermediateSize] }, { dims: ["sequence_length", intermediateSize] }],
      outputs: [{ dims: ["sequence_length", intermediateSize] }],
    });
    edge(gateAct, mul);
    edge(up, mul);

    const down = `${ffn}.down`;
    node(down, "linear", "Down Projection", ffn, {
      inputs: [{ dims: ["sequence_length", intermediateSize] }],
      outputs: [{ dims: seqH }],
      parameters: [param(`${L}.mlp.down_proj.weight`, wi, providerId)],
    });
    edge(mul, down);

    const postFfnNorm = `${b}.post_ffn_norm`;
    normNode(postFfnNorm, "RMSNorm (post-FFN)", b, `${L}.post_feedforward_layernorm.weight`, H);
    edge(down, postFfnNorm);

    const res2 = `${b}.res2`;
    node(res2, "residual", "Residual Add", b, {
      inputs: [{ dims: seqH }, { dims: seqH }],
      outputs: [{ dims: seqH }],
      metadata: { description: "Adds the pre-FFN state back in around the Feed Forward block (after the FFN's own post-norm)." },
    });
    edge(postFfnNorm, res2);
    edge(res1, res2, "skip");

    let blockPreScale = res2;
    if (hasPerLayerInput && perLayerInputId) {
      const plGate = `${b}.pl_gate`;
      node(plGate, "linear", "Per-Layer Input Gate", b, {
        inputs: [{ dims: seqH }],
        outputs: [{ dims: ["sequence_length", hiddenSizePerLayerInput] }],
        parameters: [param(`${L}.per_layer_input_gate.weight`, wi, providerId)],
      });
      edge(res2, plGate);

      const plAct = `${b}.pl_act`;
      node(plAct, "activation", String(cfg.extra.activationFunction ?? "gelu_pytorch_tanh"), b, {
        inputs: [{ dims: ["sequence_length", hiddenSizePerLayerInput] }],
        outputs: [{ dims: ["sequence_length", hiddenSizePerLayerInput] }],
      });
      edge(plGate, plAct);

      const plMul = `${b}.pl_mul`;
      node(plMul, "elementwise_mul", "× Per-Layer Input", b, {
        inputs: [{ dims: ["sequence_length", hiddenSizePerLayerInput] }, { dims: ["sequence_length", hiddenSizePerLayerInput] }],
        outputs: [{ dims: ["sequence_length", hiddenSizePerLayerInput] }],
        metadata: { note: `Elementwise-multiplies against this layer's own slice (index ${i} of ${cfg.numLayers}) of the Per-Layer Input Projection computed once at the top of the model.` },
      });
      edge(plAct, plMul);
      edge(perLayerInputId, plMul);

      const plProj = `${b}.pl_proj`;
      node(plProj, "linear", "Per-Layer Projection", b, {
        inputs: [{ dims: ["sequence_length", hiddenSizePerLayerInput] }],
        outputs: [{ dims: seqH }],
        parameters: [param(`${L}.per_layer_projection.weight`, wi, providerId)],
      });
      edge(plMul, plProj);

      const plNorm = `${b}.pl_norm`;
      normNode(plNorm, "RMSNorm (post-per-layer-input)", b, `${L}.post_per_layer_input_norm.weight`, H);
      edge(plProj, plNorm);

      const res3 = `${b}.res3`;
      node(res3, "residual", "Residual Add", b, {
        inputs: [{ dims: seqH }, { dims: seqH }],
        outputs: [{ dims: seqH }],
        metadata: { description: "Adds the Per-Layer Embedding contribution back in — a third residual branch this model has on top of the usual attention/FFN pair." },
      });
      edge(plNorm, res3);
      edge(res2, res3, "skip");
      blockPreScale = res3;
    }

    const scaleNode = `${b}.scale`;
    node(scaleNode, "linear", "× Layer Scalar", b, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: seqH }],
      parameters: [param(`${L}.layer_scalar`, wi, providerId)],
      metadata: { description: "Multiplies this block's entire output by one learned scalar before it's handed to the next block — a real, stored per-layer weight, not a fixed constant." },
    });
    edge(blockPreScale, scaleNode);

    prevOut = scaleNode;
  }

  const finalNorm = "norm";
  normNode(finalNorm, "Final RMSNorm", "model", `${LP}.norm.weight`, H);
  edge(prevOut, finalNorm);

  const tied = !wi[`${LP}.lm_head.weight`];
  node("lm_head", "lm_head", "LM Head", "model", {
    inputs: [{ dims: seqH }],
    outputs: [{ dims: ["sequence_length", cfg.vocabSize] }],
    parameters: [param(tied ? `${LP}.embed_tokens.weight` : `${LP}.lm_head.weight`, wi, providerId)],
    metadata: { tied, description: tied ? "Tied to the token embedding weight (transposed)." : undefined },
  });
  edge(finalNorm, "lm_head");

  const softcap = cfg.extra.finalLogitSoftcapping;
  let lastId = "lm_head";
  if (softcap != null) {
    const softcapId = "logit_softcap";
    node(softcapId, "activation", "Logit Softcap (tanh)", "model", {
      inputs: [{ dims: ["sequence_length", cfg.vocabSize] }],
      outputs: [{ dims: ["sequence_length", cfg.vocabSize] }],
      metadata: { description: `Gemma-style softcap: logits = tanh(logits / ${softcap}) * ${softcap} — keeps any single logit from growing unboundedly large, applied once, only here.` },
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
