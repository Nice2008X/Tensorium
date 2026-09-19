import type { EdgeKind, Model, ModelConfig, ModelMetadata, ModelNode, NodeType, ParameterRef, TensorSlice } from "@tensorium/model-ir";
import { numElements, dtypeSize, modelSourceLabel } from "@tensorium/model-ir";

/**
 * The shared config surface across Llama, Mistral, and Gemma's config.json —
 * field names happen to line up almost exactly (Mistral's are identical to
 * Llama's; Gemma adds `head_dim`, which the others omit and derive instead).
 */
export interface LlamaFamilyRawConfig {
  model_type?: string;
  architectures?: string[];
  hidden_size: number;
  intermediate_size: number;
  num_hidden_layers: number;
  num_attention_heads: number;
  num_key_value_heads?: number;
  /** Gemma specifies this explicitly (and it need not equal hidden_size / num_attention_heads); Llama/Mistral don't and it's derived. */
  head_dim?: number;
  vocab_size: number;
  max_position_embeddings?: number;
  rms_norm_eps?: number;
  hidden_act?: string;
  rope_theta?: number;
  tie_word_embeddings?: boolean;
  /** GLM-4 rotates only this fraction of each head's dimensions (the rest pass through RoPE unrotated) — a real config.json field, unlike the other family-defining traits below, which read as adapter options instead since nothing in config.json states them directly. */
  partial_rotary_factor?: number;
  /** OLMo: clamps Q/K/V projections to [-clip_qkv, clip_qkv] right after projection, before RoPE. null (the common case) disables it. */
  clip_qkv?: number | null;
  /** MoE config.json fields — present only on sparse checkpoints (e.g. Qwen2-MoE, Qwen3-MoE). How many experts exist, how many run per token, each expert's FFN width, whether the router's top-k weights get renormalized to sum to 1, and (Qwen2-MoE only) an always-on shared expert's FFN width. */
  num_experts?: number;
  num_experts_per_tok?: number;
  moe_intermediate_size?: number;
  norm_topk_prob?: boolean;
  shared_expert_intermediate_size?: number;
  /** MoE checkpoints don't have to route every layer: a layer is sparse only if it isn't listed in mlp_only_layers AND (layer_idx + 1) % decoder_sparse_step === 0 — everything else runs the plain dense gated FFN instead. Qwen2-MoE's step is 1 (every layer sparse); Qwen3-MoE's is 2 (every other layer). Defaults (1, []) make every layer sparse when omitted, matching the common case. */
  decoder_sparse_step?: number;
  mlp_only_layers?: number[];
  /** ZGCM-style hybrid attention (real config.json fields): a window size, and — per layer — whether it uses that window ("sliding_attention") or sees the whole sequence ("full_attention"). Only read when the adapter opts in via `perLayerSlidingWindow`. Missing `layer_types` falls back to the model's own rule: every `window_attn_skip_freq`-th layer (default 6) is full attention. */
  sliding_window?: number | null;
  layer_types?: string[];
  window_attn_skip_freq?: number;
  /** ZGCM: which layers carry a `g_proj` output gate. Missing means "every sliding-attention layer" (the model's own default). Only read when the adapter opts in via `attentionOutputGate`. */
  attention_gate_layers?: boolean[];
}

export interface LlamaFamilyOptions {
  /** Falls back to this when config.json omits model_type. */
  defaultModelType: string;
  /** Gemma's RMSNorm scales by (1 + weight), not weight — see nn-ops' gemmaRmsNorm doc comment. */
  rmsNormVariant?: "standard" | "gemma";
  /** Gemma multiplies token embeddings by sqrt(hidden_size) right after lookup. */
  embeddingScale?: "none" | "sqrt_hidden";
  /** Used only when config.json omits tie_word_embeddings outright. */
  tiedByDefault?: boolean;
  /** Qwen2 keeps a bias on q_proj/k_proj/v_proj (but not o_proj) — everything else in this family has no attention/MLP biases at all. */
  qkvBias?: boolean;
  /** Qwen3 applies a per-head RMSNorm (shared weight across heads) to Q and K right after projection, before RoPE — replaces Qwen2's qkvBias as its stabilization mechanism. */
  qkNorm?: boolean;
  /** Phi3/Phi4 fuse Q/K/V into one `qkv_proj` weight (like GPT-2's c_attn, but GQA-sized: Q/K/V get unequal row-slices instead of equal thirds). */
  fusedQkv?: boolean;
  /** Phi3/Phi4 also fuse the gated-FFN's gate and up projections into one `gate_up_proj` weight, split into equal halves. */
  fusedGateUp?: boolean;
  /** GLM-4's "sandwich" norm: an extra RMSNorm applied to each sub-layer's output (post_self_attn_layernorm after attention, post_mlp_layernorm after the FFN) right before it's added back into the residual stream — on top of, not instead of, the usual pre-norms. */
  sandwichNorm?: boolean;
  /** OLMo (v1) replaces RMSNorm with a LayerNorm that has no learnable weight or bias at all (fixed gamma=1, beta=0) — every other model in this family uses "rmsnorm". */
  normType?: "rmsnorm" | "layernorm_no_affine";
  /** Qwen2-MoE/Qwen3-MoE: this checkpoint's FFN is a sparse Mixture-of-Experts (a router picks num_experts_per_tok of num_experts SwiGLU experts per token, weighted-summed) instead of one dense gated FFN. Not derivable from config.json alone (a dense model simply omits the MoE fields above), so — like qkvBias/qkNorm — the adapter states it explicitly. */
  moe?: boolean;
  /** ZGCM: attention uses a per-layer window (config.json's `layer_types` + `sliding_window`) — sliding layers only see the last `sliding_window` tokens. Opt-in because other families (e.g. Mistral) list a `sliding_window` in config.json that this app has never applied. */
  perLayerSlidingWindow?: boolean;
  /** ZGCM: layers listed in `attention_gate_layers` have a separate `self_attn.g_proj` whose sigmoid multiplies the attention heads' output right before o_proj. */
  attentionOutputGate?: boolean;
  /** Weight-name suffixes (under `model.layers.N.`) of the norm run before attention and the one run before the FFN. Standard Llama-family names are `input_layernorm` / `post_attention_layernorm`; ZGCM calls them `post_attention_layernorm` / `post_feedforward_layernorm` even though both are ordinary pre-norms. */
  normWeightNames?: { preAttention: string; preFfn: string };
  /** How partial_rotary_factor turns into a rotary width. "round" (default) is what every existing adapter here uses; ZGCM's own code does `int(head_dim * factor)` then drops to an even number, which is not the same when the product isn't an integer (128 × 0.334 = 42.75 → 42, not 43). */
  rotaryDimRounding?: "round" | "floor_even";
}

export function buildModelConfig(raw: LlamaFamilyRawConfig, options: LlamaFamilyOptions): ModelConfig {
  const numHeads = raw.num_attention_heads;
  const headDim = raw.head_dim ?? raw.hidden_size / numHeads;
  const partialRotaryFactor = raw.partial_rotary_factor ?? 1;
  // GLM-4 only rotates a leading slice of each head (the rest of the
  // dimensions pass through unrotated); every other adapter in this family
  // implicitly has partial_rotary_factor 1, so rotaryDim === headDim there.
  const rotaryDim =
    options.rotaryDimRounding === "floor_even"
      ? Math.floor(headDim * partialRotaryFactor) - (Math.floor(headDim * partialRotaryFactor) % 2)
      : Math.round(headDim * partialRotaryFactor);

  // RoPE rotates its dimensions in (x, y) pairs (see nn-ops'
  // ropeCosSin/rotateHalf), so it's only defined for an even rotary
  // dimension — real checkpoints always land here (head_dim, and any
  // partial_rotary_factor slice of it, is a power of two in practice), but
  // a "tiny-random" test fixture can shrink hidden_size/num_attention_heads
  // down to a degenerate odd dimension that no RoPE implementation (this
  // one or PyTorch's) can rotate. Failing fast here beats silently
  // propagating NaN through every downstream layer and surfacing as a
  // confusing "NaN%" in the Logit Lens.
  if (rotaryDim % 2 !== 0) {
    throw new Error(
      `This checkpoint's rotary dimension is ${rotaryDim} (head_dim ${headDim}${partialRotaryFactor !== 1 ? ` × partial_rotary_factor ${partialRotaryFactor}` : ""}), which is odd. RoPE rotates dimensions in pairs and requires an even value, so this model can't run a forward pass here.`
    );
  }

  const normType = options.normType ?? "rmsnorm";

  // Per-layer hybrid attention (ZGCM). Same defaults as the model's own
  // config class: every skip_freq-th layer is full attention, and only the
  // sliding layers carry an output gate unless attention_gate_layers says otherwise.
  const layerTypes = Array.from({ length: raw.num_hidden_layers }, (_, i) =>
    raw.layer_types?.[i] ?? ((i + 1) % (raw.window_attn_skip_freq ?? 6) === 0 ? "full_attention" : "sliding_attention")
  );
  const slidingWindowSizes: (number | null)[] = layerTypes.map((t) =>
    options.perLayerSlidingWindow && t === "sliding_attention" && raw.sliding_window != null ? raw.sliding_window : null
  );
  const attentionGateLayers: boolean[] = layerTypes.map((t, i) =>
    options.attentionOutputGate ? Boolean(raw.attention_gate_layers?.[i] ?? t === "sliding_attention") : false
  );
  // OLMo's LayerNorm has no config.json field for epsilon — HF hardcodes
  // 1e-5 for it. rms_norm_eps (default 1e-6) is a different family's knob;
  // only fall back to it here if a checkpoint's config explicitly sets it.
  const normEps = normType === "layernorm_no_affine" ? raw.rms_norm_eps ?? 1e-5 : raw.rms_norm_eps ?? 1e-6;

  return {
    modelType: raw.model_type ?? options.defaultModelType,
    numLayers: raw.num_hidden_layers,
    numHeads,
    hiddenSize: raw.hidden_size,
    intermediateSize: raw.intermediate_size,
    vocabSize: raw.vocab_size,
    contextLength: raw.max_position_embeddings ?? 4096,
    extra: {
      numKeyValueHeads: raw.num_key_value_heads ?? numHeads,
      headDim,
      partialRotaryFactor,
      rotaryDim,
      slidingWindowSizes,
      attentionGateLayers,
      preAttentionNormName: options.normWeightNames?.preAttention ?? "input_layernorm",
      preFfnNormName: options.normWeightNames?.preFfn ?? "post_attention_layernorm",
      perLayerSlidingWindow: options.perLayerSlidingWindow ?? false,
      rmsNormEps: normEps,
      activationFunction: raw.hidden_act ?? "silu",
      ropeTheta: raw.rope_theta ?? 10000,
      tiedEmbeddings: raw.tie_word_embeddings ?? options.tiedByDefault ?? false,
      rmsNormVariant: options.rmsNormVariant ?? "standard",
      embeddingScale: options.embeddingScale ?? "none",
      qkvBias: options.qkvBias ?? false,
      qkNorm: options.qkNorm ?? false,
      fusedQkv: options.fusedQkv ?? false,
      fusedGateUp: options.fusedGateUp ?? false,
      sandwichNorm: options.sandwichNorm ?? false,
      normType,
      clipQkv: raw.clip_qkv ?? null,
      isMoE: options.moe ?? false,
      numExperts: raw.num_experts,
      numExpertsPerTok: raw.num_experts_per_tok,
      moeIntermediateSize: raw.moe_intermediate_size,
      normTopkProb: raw.norm_topk_prob ?? false,
      hasSharedExpert: raw.shared_expert_intermediate_size != null,
      decoderSparseStep: raw.decoder_sparse_step ?? 1,
      mlpOnlyLayers: raw.mlp_only_layers ?? [],
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

export function buildGraph(metadata: ModelMetadata, providerId: string): Model {
  const cfg = metadata.config;
  const wi = metadata.weightIndex;
  const nodes: Record<string, ModelNode> = {};
  const edges: Model["edges"] = [];
  const isGemmaNorm = cfg.extra.rmsNormVariant === "gemma";
  const scalesEmbedding = cfg.extra.embeddingScale === "sqrt_hidden";
  const hasQkvBias = cfg.extra.qkvBias === true;
  const hasQkNorm = cfg.extra.qkNorm === true;
  const hasFusedQkv = cfg.extra.fusedQkv === true;
  const hasFusedGateUp = cfg.extra.fusedGateUp === true;
  const hasSandwichNorm = cfg.extra.sandwichNorm === true;
  const isLayerNormNoAffine = cfg.extra.normType === "layernorm_no_affine";
  const clipQkv = cfg.extra.clipQkv != null ? Number(cfg.extra.clipQkv) : null;
  const isMoE = cfg.extra.isMoE === true;
  const numExperts = Number(cfg.extra.numExperts ?? 0);
  const numExpertsPerTok = Number(cfg.extra.numExpertsPerTok ?? numExperts);
  const hasSharedExpert = cfg.extra.hasSharedExpert === true;
  const decoderSparseStep = Number(cfg.extra.decoderSparseStep ?? 1);
  const mlpOnlyLayers = (cfg.extra.mlpOnlyLayers as number[] | undefined) ?? [];
  const preAttentionNormName = String(cfg.extra.preAttentionNormName ?? "input_layernorm");
  const preFfnNormName = String(cfg.extra.preFfnNormName ?? "post_attention_layernorm");
  const perLayerSlidingWindow = cfg.extra.perLayerSlidingWindow === true;
  const slidingWindowSizes = (cfg.extra.slidingWindowSizes as (number | null)[] | undefined) ?? [];
  const attentionGateLayers = (cfg.extra.attentionGateLayers as boolean[] | undefined) ?? [];
  // Not every layer of an MoE checkpoint has to be sparse — Qwen2-MoE routes
  // every layer (step 1), Qwen3-MoE only every other one (step 2), and
  // either family can also name specific always-dense layers explicitly.
  // Everything else falls back to the plain dense gated FFN below.
  const isSparseLayer = (i: number) => isMoE && numExperts > 0 && !mlpOnlyLayers.includes(i) && (i + 1) % decoderSparseStep === 0;
  const normLabel = (suffix: string) => (isLayerNormNoAffine ? `LayerNorm (${suffix})` : `RMSNorm (${suffix})`);

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

  function normNode(id: string, label: string, parentId: string, weightName: string) {
    if (isLayerNormNoAffine) {
      // OLMo: true LayerNorm (re-centers by mean, not just RMS-scaled) with
      // no learnable weight or bias at all — fixed gamma=1, beta=0.
      return node(id, "layer_norm", label, parentId, {
        inputs: [{ dims: seqH }],
        outputs: [{ dims: seqH }],
        metadata: { note: "OLMo variant: non-parametric — no learnable weight or bias (fixed gamma=1, beta=0)." },
      });
    }
    return node(id, "rms_norm", label, parentId, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: seqH }],
      parameters: [param(weightName, wi, providerId)],
      metadata: isGemmaNorm
        ? { note: "Gemma variant: scales by (1 + weight), and weight is zero-initialized — not the standard (x_normalized * weight)." }
        : {},
    });
  }

  const H = cfg.hiddenSize;
  const seqH: Array<number | string> = ["sequence_length", H];
  const numKVHeads = Number(cfg.extra.numKeyValueHeads ?? cfg.numHeads);
  const headDim = Number(cfg.extra.headDim ?? H / cfg.numHeads);
  const qDim = cfg.numHeads * headDim;
  const kvDim = numKVHeads * headDim;

  // --- root -------------------------------------------------------------
  node("model", "model", metadata.architecture, null);

  node("input", "input", "Input tokens", "model", {
    outputs: [{ dims: ["sequence_length"] }],
    metadata: { description: "Token IDs produced by the tokenizer." },
  });

  node("embed", "embedding", "Token Embedding", "model", {
    inputs: [{ dims: ["sequence_length"] }],
    outputs: [{ dims: seqH }],
    parameters: [param("model.embed_tokens.weight", wi, providerId)],
    metadata: {
      description: scalesEmbedding
        ? "No learned positional embedding table — position is injected inside attention via RoPE. Additionally scaled by √hidden_size right after lookup."
        : "No learned positional embedding table — position is injected later, inside attention, via RoPE.",
    },
  });
  edge("input", "embed");

  node("blocks", "block_group", `Transformer Blocks × ${cfg.numLayers}`, "model", {
    metadata: { count: cfg.numLayers },
  });

  let prevOut = "embed";
  for (let i = 0; i < cfg.numLayers; i++) {
    const b = `block.${i}`;
    const L = `model.layers.${i}`;
    node(b, "transformer_block", `Transformer Block ${i}`, "blocks", {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: seqH }],
    });
    edge(prevOut, b);

    const rms1 = `${b}.rms1`;
    normNode(rms1, normLabel("pre-attention"), b, `${L}.${preAttentionNormName}.weight`);
    edge(b, rms1);

    const attn = `${b}.attn`;
    const layerWindow = slidingWindowSizes[i] ?? null;
    const layerHasGate = attentionGateLayers[i] === true;
    node(attn, "attention", perLayerSlidingWindow ? `Attention (${layerWindow != null ? "sliding" : "full"})` : "Attention", b, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: seqH }],
      metadata: {
        numHeads: cfg.numHeads,
        numKeyValueHeads: numKVHeads,
        headDim,
        groupedQueryAttention: numKVHeads !== cfg.numHeads,
        ...(perLayerSlidingWindow
          ? {
              slidingWindow: layerWindow ?? undefined,
              description:
                layerWindow != null
                  ? `Causal attention restricted to the last ${layerWindow} positions (a local window) — most layers in this model are this cheaper local form.`
                  : "Ordinary causal attention over the whole sequence — one of the periodic full-attention layers that give every token eventual access to the whole context.",
            }
          : {}),
      },
    });

    const q = `${attn}.q`;
    const k = `${attn}.k`;
    const v = `${attn}.v`;
    if (hasFusedQkv) {
      // Phi3/Phi4: one qkv_proj weight, [out_features, in_features] layout
      // (unlike GPT-2's Conv1D [in,out]) — Q/K/V are unequal row-slices
      // (GQA-sized), the same ParameterRef.slice mechanism GPT-2 uses for
      // its fused c_attn, just sliced along the other dimension.
      const qkvName = `${L}.self_attn.qkv_proj.weight`;
      node(q, "q_projection", "Q Projection", attn, {
        inputs: [{ dims: seqH }],
        outputs: [{ dims: ["sequence_length", qDim] }],
        parameters: [param(qkvName, wi, providerId, { ranges: [{ start: 0, end: qDim }] })],
        metadata: { note: "Sliced out of the fused qkv_proj weight." },
      });
      node(k, "k_projection", "K Projection", attn, {
        inputs: [{ dims: seqH }],
        outputs: [{ dims: ["sequence_length", kvDim] }],
        parameters: [param(qkvName, wi, providerId, { ranges: [{ start: qDim, end: qDim + kvDim }] })],
        metadata: { note: "Sliced out of the fused qkv_proj weight." },
      });
      node(v, "v_projection", "V Projection", attn, {
        inputs: [{ dims: seqH }],
        outputs: [{ dims: ["sequence_length", kvDim] }],
        parameters: [param(qkvName, wi, providerId, { ranges: [{ start: qDim + kvDim, end: qDim + 2 * kvDim }] })],
        metadata: { note: "Sliced out of the fused qkv_proj weight." },
      });
    } else {
      const clipNote = clipQkv != null ? { note: `Clamped to [-${clipQkv}, ${clipQkv}] right after projection (this checkpoint's clip_qkv).` } : {};
      node(q, "q_projection", "Q Projection", attn, {
        inputs: [{ dims: seqH }],
        outputs: [{ dims: ["sequence_length", qDim] }],
        parameters: hasQkvBias
          ? [param(`${L}.self_attn.q_proj.weight`, wi, providerId), param(`${L}.self_attn.q_proj.bias`, wi, providerId)]
          : [param(`${L}.self_attn.q_proj.weight`, wi, providerId)],
        metadata: clipNote,
      });
      node(k, "k_projection", "K Projection", attn, {
        inputs: [{ dims: seqH }],
        outputs: [{ dims: ["sequence_length", kvDim] }],
        parameters: hasQkvBias
          ? [param(`${L}.self_attn.k_proj.weight`, wi, providerId), param(`${L}.self_attn.k_proj.bias`, wi, providerId)]
          : [param(`${L}.self_attn.k_proj.weight`, wi, providerId)],
        metadata: clipNote,
      });
      node(v, "v_projection", "V Projection", attn, {
        inputs: [{ dims: seqH }],
        outputs: [{ dims: ["sequence_length", kvDim] }],
        parameters: hasQkvBias
          ? [param(`${L}.self_attn.v_proj.weight`, wi, providerId), param(`${L}.self_attn.v_proj.bias`, wi, providerId)]
          : [param(`${L}.self_attn.v_proj.weight`, wi, providerId)],
        metadata: clipNote,
      });
    }
    edge(rms1, q);
    edge(rms1, k);
    edge(rms1, v);

    // Qwen3 normalizes each head's Q/K vector (one shared [head_dim] weight
    // reused across every head) right after projection, before RoPE rotates
    // it — a stabilization mechanism that replaces Qwen2's qkvBias.
    let qIntoRope = q;
    let kIntoRope = k;
    if (hasQkNorm) {
      const qNorm = `${attn}.q_norm`;
      node(qNorm, "rms_norm", "Q Norm", attn, {
        inputs: [{ dims: ["sequence_length", qDim] }],
        outputs: [{ dims: ["sequence_length", qDim] }],
        parameters: [param(`${L}.self_attn.q_norm.weight`, wi, providerId)],
        metadata: { note: "Applied per-head: the same head_dim-sized weight normalizes every head's Q slice independently." },
      });
      edge(q, qNorm);
      qIntoRope = qNorm;

      const kNorm = `${attn}.k_norm`;
      node(kNorm, "rms_norm", "K Norm", attn, {
        inputs: [{ dims: ["sequence_length", kvDim] }],
        outputs: [{ dims: ["sequence_length", kvDim] }],
        parameters: [param(`${L}.self_attn.k_norm.weight`, wi, providerId)],
        metadata: { note: "Applied per-head: the same head_dim-sized weight normalizes every head's K slice independently." },
      });
      edge(k, kNorm);
      kIntoRope = kNorm;
    }

    const rope = `${attn}.rope`;
    node(rope, "rope", "RoPE", attn, {
      inputs: [{ dims: ["sequence_length", qDim] }, { dims: ["sequence_length", kvDim] }],
      outputs: [{ dims: ["sequence_length", qDim] }, { dims: ["sequence_length", kvDim] }],
      metadata: { description: "Rotates Q and K by an angle proportional to sequence position — no learned parameters.", ropeTheta: cfg.extra.ropeTheta },
    });
    edge(qIntoRope, rope);
    edge(kIntoRope, rope);

    // ZGCM's attention output gate: a separate g_proj reads the same
    // pre-attention-normed input as Q/K/V, and its sigmoid scales the
    // attention heads' output elementwise right before the output projection.
    let attnResultSource = rope;
    let gateActId: string | null = null;
    if (layerHasGate) {
      const gateId = `${attn}.gate`;
      node(gateId, "linear", "Attention Output Gate", attn, {
        inputs: [{ dims: seqH }],
        outputs: [{ dims: ["sequence_length", qDim] }],
        parameters: [param(`${L}.self_attn.g_proj.weight`, wi, providerId)],
      });
      edge(rms1, gateId);

      gateActId = `${attn}.gate_act`;
      node(gateActId, "activation", "sigmoid", attn, {
        inputs: [{ dims: ["sequence_length", qDim] }],
        outputs: [{ dims: ["sequence_length", qDim] }],
      });
      edge(gateId, gateActId);

      const gatedMul = `${attn}.gated_mul`;
      node(gatedMul, "elementwise_mul", "× Output Gate", attn, {
        inputs: [{ dims: ["sequence_length", qDim] }, { dims: ["sequence_length", qDim] }],
        outputs: [{ dims: ["sequence_length", qDim] }],
      });
      edge(rope, gatedMul);
      edge(v, gatedMul);
      edge(gateActId, gatedMul);
      attnResultSource = gatedMul;
    }

    const outp = `${attn}.out`;
    node(outp, "output_projection", "Output Projection", attn, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: seqH }],
      parameters: [param(`${L}.self_attn.o_proj.weight`, wi, providerId)],
    });
    edge(attnResultSource, outp);
    if (!gateActId) edge(v, outp);

    // GLM-4's sandwich norm: an extra RMSNorm on the attention sub-layer's
    // *output*, on top of (not instead of) rms1's pre-attention norm —
    // applied right before this gets added back into the residual stream.
    let attnIntoResidual = outp;
    if (hasSandwichNorm) {
      const postAttnNorm = `${attn}.post_norm`;
      normNode(postAttnNorm, "RMSNorm (post-attention, sandwich)", attn, `${L}.post_self_attn_layernorm.weight`);
      edge(outp, postAttnNorm);
      attnIntoResidual = postAttnNorm;
    }

    const res1 = `${b}.res1`;
    node(res1, "residual", "Residual Add", b, {
      inputs: [{ dims: seqH }, { dims: seqH }],
      outputs: [{ dims: seqH }],
      metadata: { description: "Adds the block's input back in around Attention." },
    });
    edge(attnIntoResidual, res1);
    edge(b, res1, "skip");

    const rms2 = `${b}.rms2`;
    normNode(rms2, normLabel("pre-FFN"), b, `${L}.${preFfnNormName}.weight`);
    edge(res1, rms2);

    const layerIsSparse = isSparseLayer(i);
    const ffn = `${b}.ffn`;
    node(ffn, "ffn", layerIsSparse ? "Feed Forward (Mixture of Experts)" : "Feed Forward (gated)", b, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: seqH }],
    });

    let ffnOut: string;
    if (layerIsSparse) {
      const router = `${ffn}.router`;
      node(router, "router", "Router", ffn, {
        inputs: [{ dims: seqH }],
        outputs: [{ dims: ["sequence_length", numExperts] }],
        parameters: [param(`${L}.mlp.gate.weight`, wi, providerId)],
        metadata: { numExperts, numExpertsPerTok, normTopkProb: cfg.extra.normTopkProb === true },
      });
      edge(rms2, router);

      // Every expert's weights are attached here (not split into per-expert
      // graph nodes) so the graph stays readable regardless of expert count
      // — the Tensor Explorer's search already lets you find any single
      // expert's weight by name (e.g. "experts.3.gate_proj") from this one
      // box, and the Inspector's parameter list shows all of them.
      const expertParams: ParameterRef[] = [];
      for (let e = 0; e < numExperts; e++) {
        expertParams.push(
          param(`${L}.mlp.experts.${e}.gate_proj.weight`, wi, providerId),
          param(`${L}.mlp.experts.${e}.up_proj.weight`, wi, providerId),
          param(`${L}.mlp.experts.${e}.down_proj.weight`, wi, providerId)
        );
      }
      if (hasSharedExpert) {
        expertParams.push(
          param(`${L}.mlp.shared_expert.gate_proj.weight`, wi, providerId),
          param(`${L}.mlp.shared_expert.up_proj.weight`, wi, providerId),
          param(`${L}.mlp.shared_expert.down_proj.weight`, wi, providerId),
          param(`${L}.mlp.shared_expert_gate.weight`, wi, providerId)
        );
      }

      const experts = `${ffn}.experts`;
      node(experts, "moe_experts", `Experts (top ${numExpertsPerTok} of ${numExperts}${hasSharedExpert ? " + shared" : ""})`, ffn, {
        inputs: [{ dims: seqH }, { dims: ["sequence_length", numExperts] }],
        outputs: [{ dims: seqH }],
        parameters: expertParams,
        metadata: { numExperts, numExpertsPerTok, hasSharedExpert, moeIntermediateSize: cfg.extra.moeIntermediateSize },
      });
      edge(rms2, experts);
      edge(router, experts);
      ffnOut = experts;
    } else {
      const gate = `${ffn}.gate`;
      const up = `${ffn}.up`;
      if (hasFusedGateUp) {
        // Phi3/Phi4: one gate_up_proj weight, split into two equal row-halves.
        const gateUpName = `${L}.mlp.gate_up_proj.weight`;
        node(gate, "linear", "Gate Projection", ffn, {
          inputs: [{ dims: seqH }],
          outputs: [{ dims: ["sequence_length", cfg.intermediateSize] }],
          parameters: [param(gateUpName, wi, providerId, { ranges: [{ start: 0, end: cfg.intermediateSize }] })],
          metadata: { note: "Sliced out of the fused gate_up_proj weight (first half)." },
        });
        node(up, "linear", "Up Projection", ffn, {
          inputs: [{ dims: seqH }],
          outputs: [{ dims: ["sequence_length", cfg.intermediateSize] }],
          parameters: [param(gateUpName, wi, providerId, { ranges: [{ start: cfg.intermediateSize, end: 2 * cfg.intermediateSize }] })],
          metadata: { note: "Sliced out of the fused gate_up_proj weight (second half)." },
        });
      } else {
        node(gate, "linear", "Gate Projection", ffn, {
          inputs: [{ dims: seqH }],
          outputs: [{ dims: ["sequence_length", cfg.intermediateSize] }],
          parameters: [param(`${L}.mlp.gate_proj.weight`, wi, providerId)],
        });
        node(up, "linear", "Up Projection", ffn, {
          inputs: [{ dims: seqH }],
          outputs: [{ dims: ["sequence_length", cfg.intermediateSize] }],
          parameters: [param(`${L}.mlp.up_proj.weight`, wi, providerId)],
        });
      }
      edge(rms2, gate);
      edge(rms2, up);

      const gateAct = `${ffn}.gate_act`;
      node(gateAct, "activation", String(cfg.extra.activationFunction ?? "silu"), ffn, {
        inputs: [{ dims: ["sequence_length", cfg.intermediateSize] }],
        outputs: [{ dims: ["sequence_length", cfg.intermediateSize] }],
      });
      edge(gate, gateAct);

      const mul = `${ffn}.mul`;
      node(mul, "elementwise_mul", "Gate × Up", ffn, {
        inputs: [{ dims: ["sequence_length", cfg.intermediateSize] }, { dims: ["sequence_length", cfg.intermediateSize] }],
        outputs: [{ dims: ["sequence_length", cfg.intermediateSize] }],
      });
      edge(gateAct, mul);
      edge(up, mul);

      const down = `${ffn}.down`;
      node(down, "linear", "Down Projection", ffn, {
        inputs: [{ dims: ["sequence_length", cfg.intermediateSize] }],
        outputs: [{ dims: seqH }],
        parameters: [param(`${L}.mlp.down_proj.weight`, wi, providerId)],
      });
      edge(mul, down);
      ffnOut = down;
    }

    let ffnIntoResidual = ffnOut;
    if (hasSandwichNorm) {
      const postMlpNorm = `${ffn}.post_norm`;
      normNode(postMlpNorm, "RMSNorm (post-FFN, sandwich)", ffn, `${L}.post_mlp_layernorm.weight`);
      edge(ffnOut, postMlpNorm);
      ffnIntoResidual = postMlpNorm;
    }

    const res2 = `${b}.res2`;
    node(res2, "residual", "Residual Add", b, {
      inputs: [{ dims: seqH }, { dims: seqH }],
      outputs: [{ dims: seqH }],
      metadata: { description: "Adds the pre-FFN state back in around the Feed Forward block." },
    });
    edge(ffnIntoResidual, res2);
    edge(res1, res2, "skip");

    prevOut = res2;
  }

  const finalNorm = "norm";
  normNode(finalNorm, isLayerNormNoAffine ? "Final LayerNorm" : "Final RMSNorm", "model", "model.norm.weight");
  edge(prevOut, finalNorm);

  const tied = !wi["lm_head.weight"];
  node("lm_head", "lm_head", "LM Head", "model", {
    inputs: [{ dims: seqH }],
    outputs: [{ dims: ["sequence_length", cfg.vocabSize] }],
    parameters: [param(tied ? "model.embed_tokens.weight" : "lm_head.weight", wi, providerId)],
    metadata: { tied, description: tied ? "Tied to the token embedding weight (transposed)." : undefined },
  });
  edge(finalNorm, "lm_head");

  node("output", "output", "Logits", "model", {
    inputs: [{ dims: ["sequence_length", cfg.vocabSize] }],
  });
  edge("lm_head", "output");

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
