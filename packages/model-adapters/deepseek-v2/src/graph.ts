import type { EdgeKind, Model, ModelConfig, ModelMetadata, ModelNode, NodeType, ParameterRef, TensorSlice } from "@tensorium/model-ir";
import { numElements, dtypeSize, modelSourceLabel } from "@tensorium/model-ir";

/**
 * DeepSeek-V2 (and V2-Lite)'s config.json. Two things make this family
 * structurally different from every other adapter here, not just a
 * parameter variant of Llama's shape:
 *
 * 1. Multi-head Latent Attention (MLA): instead of separate Q/K/V
 *    projections, K and V are both reconstructed from one shared
 *    low-rank "latent" (kv_a_proj_with_mqa -> kv_lora_rank, expanded back
 *    up by kv_b_proj) — this is what lets a 236B/16B-parameter model keep
 *    a KV-cache the size of a much smaller GQA model's. Only a slice of
 *    each head (qk_rope_head_dim) actually gets rotated by RoPE; the rest
 *    (qk_nope_head_dim) carries content with no positional signal at all.
 *    Q optionally goes through the same kind of low-rank compression
 *    (q_lora_rank) — DeepSeek-V2-Lite skips it (q_lora_rank: null) and
 *    projects Q directly, but other checkpoints in the family (including
 *    this adapter's own tiny-random test fixture) set it.
 * 2. DeepSeekMoE: a fine-grained Mixture-of-Experts (many small experts
 *    instead of a few large ones) with one or more "shared" experts that
 *    run on every token unconditionally (no gate at all, unlike Qwen2-MoE's
 *    sigmoid-gated shared expert) on top of the router's top-k selection.
 *    Routing can also be group-limited: experts are split into n_group
 *    buckets, only the top topk_group buckets are eligible, and top-k is
 *    then taken from inside just those — DeepSeek-V2-Lite's own config
 *    uses the simpler ungrouped "greedy" method (n_group: 1), but larger
 *    DeepSeek-V2 checkpoints (and this adapter's tiny-random fixture) use
 *    "group_limited_greedy", so both are implemented.
 */
export interface DeepseekV2RawConfig {
  model_type?: string;
  architectures?: string[];
  hidden_size: number;
  intermediate_size: number;
  num_hidden_layers: number;
  num_attention_heads: number;
  vocab_size: number;
  max_position_embeddings?: number;
  rms_norm_eps?: number;
  hidden_act?: string;
  rope_theta?: number;
  tie_word_embeddings?: boolean;
  attention_bias?: boolean;
  /** MLA sizing. q_lora_rank is nullable — omitting the Q-side low-rank compression entirely (DeepSeek-V2-Lite does this; K/V compression is never optional). */
  q_lora_rank?: number | null;
  kv_lora_rank: number;
  qk_nope_head_dim: number;
  qk_rope_head_dim: number;
  v_head_dim: number;
  rope_scaling?: {
    type: string;
    factor: number;
    original_max_position_embeddings?: number;
    beta_fast?: number;
    beta_slow?: number;
    mscale?: number;
    mscale_all_dim?: number;
  } | null;
  /** DeepSeekMoE fields. A layer is sparse once its index reaches first_k_dense_replace (and, on checkpoints that set it, only every moe_layer_freq'th layer after that) — everything before that runs the plain dense gated FFN below, using the family's regular intermediate_size rather than the (usually much smaller) moe_intermediate_size. */
  n_routed_experts?: number;
  n_shared_experts?: number | null;
  num_experts_per_tok?: number;
  moe_intermediate_size?: number;
  first_k_dense_replace?: number;
  moe_layer_freq?: number;
  norm_topk_prob?: boolean;
  routed_scaling_factor?: number;
  scoring_func?: string;
  topk_method?: "greedy" | "group_limited_greedy";
  n_group?: number;
  topk_group?: number;
}

export function buildModelConfig(raw: DeepseekV2RawConfig): ModelConfig {
  const qkRopeHeadDim = raw.qk_rope_head_dim;
  const qkNopeHeadDim = raw.qk_nope_head_dim;
  const qHeadDim = qkNopeHeadDim + qkRopeHeadDim;

  // Same reasoning as the llama-family engine's rotaryDim check: RoPE
  // rotates dimensions in (x, y) pairs, so the rope-carrying slice of each
  // head must be even. Real checkpoints always satisfy this; only a
  // degenerately shrunk "tiny-random" fixture could violate it.
  if (qkRopeHeadDim % 2 !== 0) {
    throw new Error(`This checkpoint's qk_rope_head_dim is ${qkRopeHeadDim}, which is odd. RoPE rotates dimensions in pairs and requires an even value.`);
  }

  const numRoutedExperts = raw.n_routed_experts;
  const isMoE = numRoutedExperts != null && numRoutedExperts > 0;

  return {
    modelType: raw.model_type ?? "deepseek_v2",
    numLayers: raw.num_hidden_layers,
    numHeads: raw.num_attention_heads,
    hiddenSize: raw.hidden_size,
    intermediateSize: raw.intermediate_size,
    vocabSize: raw.vocab_size,
    contextLength: raw.max_position_embeddings ?? 4096,
    extra: {
      rmsNormEps: raw.rms_norm_eps ?? 1e-6,
      activationFunction: raw.hidden_act ?? "silu",
      ropeTheta: raw.rope_theta ?? 10000,
      tiedEmbeddings: raw.tie_word_embeddings ?? false,
      attentionBias: raw.attention_bias ?? false,
      qLoraRank: raw.q_lora_rank ?? null,
      kvLoraRank: raw.kv_lora_rank,
      qkNopeHeadDim,
      qkRopeHeadDim,
      vHeadDim: raw.v_head_dim,
      qHeadDim,
      ropeScaling: raw.rope_scaling ?? null,
      isMoE,
      numRoutedExperts: numRoutedExperts ?? 0,
      numSharedExperts: raw.n_shared_experts ?? 0,
      numExpertsPerTok: raw.num_experts_per_tok ?? 0,
      moeIntermediateSize: raw.moe_intermediate_size ?? raw.intermediate_size,
      firstKDenseReplace: raw.first_k_dense_replace ?? 0,
      moeLayerFreq: raw.moe_layer_freq ?? 1,
      normTopkProb: raw.norm_topk_prob ?? false,
      routedScalingFactor: raw.routed_scaling_factor ?? 1,
      scoringFunc: raw.scoring_func ?? "softmax",
      topkMethod: raw.topk_method ?? "greedy",
      numGroups: raw.n_group ?? 1,
      topkGroup: raw.topk_group ?? 1,
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

  const hasAttnBias = cfg.extra.attentionBias === true;
  const qLoraRank = cfg.extra.qLoraRank as number | null;
  const kvLoraRank = Number(cfg.extra.kvLoraRank);
  const qkNopeHeadDim = Number(cfg.extra.qkNopeHeadDim);
  const qkRopeHeadDim = Number(cfg.extra.qkRopeHeadDim);
  const vHeadDim = Number(cfg.extra.vHeadDim);
  const qHeadDim = Number(cfg.extra.qHeadDim);
  const isMoE = cfg.extra.isMoE === true;
  const numRoutedExperts = Number(cfg.extra.numRoutedExperts ?? 0);
  const numSharedExperts = Number(cfg.extra.numSharedExperts ?? 0);
  const numExpertsPerTok = Number(cfg.extra.numExpertsPerTok ?? 0);
  const firstKDenseReplace = Number(cfg.extra.firstKDenseReplace ?? 0);
  const moeLayerFreq = Number(cfg.extra.moeLayerFreq ?? 1);
  const isSparseLayer = (i: number) => isMoE && i >= firstKDenseReplace && i % moeLayerFreq === 0;

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

  function normNode(id: string, label: string, parentId: string, weightName: string, dim: number) {
    return node(id, "rms_norm", label, parentId, {
      inputs: [{ dims: ["sequence_length", dim] }],
      outputs: [{ dims: ["sequence_length", dim] }],
      parameters: [param(weightName, wi, providerId)],
    });
  }

  const H = cfg.hiddenSize;
  const seqH: Array<number | string> = ["sequence_length", H];
  const qOutDim = cfg.numHeads * qHeadDim;
  const kvLatentDim = kvLoraRank + qkRopeHeadDim;
  const kvExpandedDim = cfg.numHeads * (qkNopeHeadDim + vHeadDim);
  const attnConcatDim = cfg.numHeads * vHeadDim;

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
    metadata: { description: "No learned positional embedding table — position is injected inside attention, via RoPE on just the rope-carrying slice of each head." },
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
    normNode(rms1, "RMSNorm (pre-attention)", b, `${L}.input_layernorm.weight`, H);
    edge(b, rms1);

    const attn = `${b}.attn`;
    node(attn, "attention", "Attention (MLA)", b, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: seqH }],
      metadata: {
        numHeads: cfg.numHeads,
        qkNopeHeadDim,
        qkRopeHeadDim,
        vHeadDim,
        kvLoraRank,
        qLoraRank,
        description:
          "Multi-head Latent Attention: K and V are both reconstructed from one shared low-rank latent instead of separate projections, keeping the KV-cache small. Only a qk_rope_head_dim-wide slice of each head is rotated by RoPE — the rest carries content with no positional signal.",
      },
    });

    // --- Q branch: either a direct projection, or (when q_lora_rank is
    // set) the same down-project -> RMSNorm -> up-project compression V/K
    // gets, just applied to Q instead. Either way it ends at a single node
    // producing [seq, numHeads * (qk_nope_head_dim + qk_rope_head_dim)] —
    // each head's nope/rope slices are described in metadata rather than
    // split into their own nodes, the same way GLM-4's partial-rotary slice
    // is handled, to keep the graph readable.
    let qOut: string;
    if (qLoraRank != null) {
      const qDown = `${attn}.q_down`;
      node(qDown, "linear", "Q Down-projection", attn, {
        inputs: [{ dims: seqH }],
        outputs: [{ dims: ["sequence_length", qLoraRank] }],
        parameters: hasAttnBias
          ? [param(`${L}.self_attn.q_a_proj.weight`, wi, providerId), param(`${L}.self_attn.q_a_proj.bias`, wi, providerId)]
          : [param(`${L}.self_attn.q_a_proj.weight`, wi, providerId)],
        metadata: {
          // The box shows the short name (long labels overlap their sibling
          // column — see layout.ts's fixed column spacing); the full name
          // survives here so the graph can still show it as a hover tooltip.
          fullName: "Q Down-projection (LoRA)",
          note: "Compresses Q into a small low-rank latent (LoRA-style), same idea as the K/V compression below — applied to Q only when this checkpoint sets q_lora_rank.",
        },
      });
      edge(rms1, qDown);

      const qDownNorm = `${attn}.q_down_norm`;
      normNode(qDownNorm, "RMSNorm (Q latent)", attn, `${L}.self_attn.q_a_layernorm.weight`, qLoraRank);
      edge(qDown, qDownNorm);

      const qUp = `${attn}.q_up`;
      node(qUp, "q_projection", "Q Up-projection", attn, {
        inputs: [{ dims: ["sequence_length", qLoraRank] }],
        outputs: [{ dims: ["sequence_length", qOutDim] }],
        parameters: [param(`${L}.self_attn.q_b_proj.weight`, wi, providerId)],
        metadata: { note: `Expands the Q latent back up to ${cfg.numHeads} heads × (qk_nope_head_dim + qk_rope_head_dim). Each head's first ${qkNopeHeadDim} dims are content-only (nope); the last ${qkRopeHeadDim} get rotated by RoPE below.` },
      });
      edge(qDownNorm, qUp);
      qOut = qUp;
    } else {
      const qProj = `${attn}.q`;
      node(qProj, "q_projection", "Q Projection", attn, {
        inputs: [{ dims: seqH }],
        outputs: [{ dims: ["sequence_length", qOutDim] }],
        parameters: [param(`${L}.self_attn.q_proj.weight`, wi, providerId)],
        metadata: { note: `Projects directly to ${cfg.numHeads} heads × (qk_nope_head_dim + qk_rope_head_dim) — this checkpoint has no q_lora_rank set, so Q skips the low-rank compression K/V always go through.` },
      });
      edge(rms1, qProj);
      qOut = qProj;
    }

    // --- KV branch: one shared down-projection produces both the KV latent
    // (compressed_kv, which gets RMSNorm'd and expanded back up into
    // per-head K-content + V) and a single shared rope-key slice (k_pe) —
    // that slice is *not* per-head (only one copy exists, effectively
    // multi-query for the positional part alone), broadcast into every
    // head's key vector after rotation.
    const kvDown = `${attn}.kv_down`;
    node(kvDown, "linear", "KV Down-projection", attn, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: ["sequence_length", kvLatentDim] }],
      parameters: hasAttnBias
        ? [param(`${L}.self_attn.kv_a_proj_with_mqa.weight`, wi, providerId), param(`${L}.self_attn.kv_a_proj_with_mqa.bias`, wi, providerId)]
        : [param(`${L}.self_attn.kv_a_proj_with_mqa.weight`, wi, providerId)],
      metadata: {
        fullName: "KV Down-projection (latent + shared rope key)",
        note: `Produces two pieces concatenated together: a ${kvLoraRank}-wide KV latent (expanded below into per-head K-content + V) and a ${qkRopeHeadDim}-wide rope key shared by every head — this second piece is why the weight is named "_with_mqa".`,
      },
    });
    edge(rms1, kvDown);

    const kvDownNorm = `${attn}.kv_down_norm`;
    normNode(kvDownNorm, "RMSNorm (KV latent)", attn, `${L}.self_attn.kv_a_layernorm.weight`, kvLoraRank);
    edge(kvDown, kvDownNorm);

    const kvUp = `${attn}.kv_up`;
    node(kvUp, "linear", "KV Up-projection", attn, {
      inputs: [{ dims: ["sequence_length", kvLoraRank] }],
      outputs: [{ dims: ["sequence_length", kvExpandedDim] }],
      parameters: [param(`${L}.self_attn.kv_b_proj.weight`, wi, providerId)],
      metadata: {
        fullName: "KV Up-projection (K-content + V)",
        note: `Expands the KV latent back up to ${cfg.numHeads} heads × (qk_nope_head_dim + v_head_dim) — each head's first ${qkNopeHeadDim} dims are K-content (nope), the last ${vHeadDim} are V.`,
      },
    });
    edge(kvDownNorm, kvUp);

    const rope = `${attn}.rope`;
    node(rope, "rope", "RoPE (decoupled)", attn, {
      inputs: [{ dims: ["sequence_length", qOutDim] }, { dims: ["sequence_length", kvLatentDim] }],
      outputs: [{ dims: ["sequence_length", qOutDim] }, { dims: ["sequence_length", kvLatentDim] }],
      metadata: {
        description: `Rotates only the trailing ${qkRopeHeadDim}-wide slice of each Q head, plus the one shared rope-key slice from the KV down-projection — the nope slices (Q's leading ${qkNopeHeadDim} dims, and K's content slice from the KV up-projection) pass through untouched.`,
        ropeTheta: cfg.extra.ropeTheta,
        ropeScaling: cfg.extra.ropeScaling,
      },
    });
    edge(qOut, rope);
    edge(kvDown, rope);

    const outp = `${attn}.out`;
    node(outp, "output_projection", "Output Projection", attn, {
      inputs: [{ dims: ["sequence_length", attnConcatDim] }],
      outputs: [{ dims: seqH }],
      parameters: hasAttnBias
        ? [param(`${L}.self_attn.o_proj.weight`, wi, providerId), param(`${L}.self_attn.o_proj.bias`, wi, providerId)]
        : [param(`${L}.self_attn.o_proj.weight`, wi, providerId)],
    });
    edge(rope, outp);
    edge(kvUp, outp);

    const res1 = `${b}.res1`;
    node(res1, "residual", "Residual Add", b, {
      inputs: [{ dims: seqH }, { dims: seqH }],
      outputs: [{ dims: seqH }],
      metadata: { description: "Adds the block's input back in around Attention." },
    });
    edge(outp, res1);
    edge(b, res1, "skip");

    const rms2 = `${b}.rms2`;
    normNode(rms2, "RMSNorm (pre-FFN)", b, `${L}.post_attention_layernorm.weight`, H);
    edge(res1, rms2);

    const layerIsSparse = isSparseLayer(i);
    const ffn = `${b}.ffn`;
    node(ffn, "ffn", layerIsSparse ? "Feed Forward (DeepSeekMoE)" : "Feed Forward (gated)", b, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: seqH }],
    });

    let ffnOut: string;
    if (layerIsSparse) {
      const router = `${ffn}.router`;
      node(router, "router", "Router", ffn, {
        inputs: [{ dims: seqH }],
        outputs: [{ dims: ["sequence_length", numRoutedExperts] }],
        parameters: [param(`${L}.mlp.gate.weight`, wi, providerId)],
        metadata: {
          numExperts: numRoutedExperts,
          numExpertsPerTok,
          scoringFunc: cfg.extra.scoringFunc,
          topkMethod: cfg.extra.topkMethod,
          numGroups: cfg.extra.numGroups,
          topkGroup: cfg.extra.topkGroup,
          normTopkProb: cfg.extra.normTopkProb,
          routedScalingFactor: cfg.extra.routedScalingFactor,
          description:
            cfg.extra.topkMethod === "group_limited_greedy"
              ? `Experts are split into ${cfg.extra.numGroups} groups; only the top ${cfg.extra.topkGroup} group(s) (by their best-scoring expert) are eligible, then the top ${numExpertsPerTok} experts overall are picked from inside just those groups.`
              : `Scores every expert with softmax, keeps the top ${numExpertsPerTok}.`,
        },
      });
      edge(rms2, router);

      const expertParams: ParameterRef[] = [];
      for (let e = 0; e < numRoutedExperts; e++) {
        expertParams.push(
          param(`${L}.mlp.experts.${e}.gate_proj.weight`, wi, providerId),
          param(`${L}.mlp.experts.${e}.up_proj.weight`, wi, providerId),
          param(`${L}.mlp.experts.${e}.down_proj.weight`, wi, providerId)
        );
      }
      if (numSharedExperts > 0) {
        expertParams.push(
          param(`${L}.mlp.shared_experts.gate_proj.weight`, wi, providerId),
          param(`${L}.mlp.shared_experts.up_proj.weight`, wi, providerId),
          param(`${L}.mlp.shared_experts.down_proj.weight`, wi, providerId)
        );
      }

      const experts = `${ffn}.experts`;
      node(experts, "moe_experts", `Experts (top ${numExpertsPerTok} of ${numRoutedExperts}${numSharedExperts > 0 ? ` + ${numSharedExperts} shared` : ""})`, ffn, {
        inputs: [{ dims: seqH }, { dims: ["sequence_length", numRoutedExperts] }],
        outputs: [{ dims: seqH }],
        parameters: expertParams,
        metadata: {
          numExperts: numRoutedExperts,
          numExpertsPerTok,
          numSharedExperts,
          moeIntermediateSize: cfg.extra.moeIntermediateSize,
          description:
            numSharedExperts > 0
              ? `Unlike Qwen2-MoE's sigmoid-gated shared expert, DeepSeek's ${numSharedExperts} shared expert(s) (merged into one wide MLP) run on every token unconditionally — no gate at all — on top of the router's top-${numExpertsPerTok} weighted sum.`
              : undefined,
        },
      });
      edge(rms2, experts);
      edge(router, experts);
      ffnOut = experts;
    } else {
      const gate = `${ffn}.gate`;
      const up = `${ffn}.up`;
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

    const res2 = `${b}.res2`;
    node(res2, "residual", "Residual Add", b, {
      inputs: [{ dims: seqH }, { dims: seqH }],
      outputs: [{ dims: seqH }],
      metadata: { description: "Adds the pre-FFN state back in around the Feed Forward block." },
    });
    edge(ffnOut, res2);
    edge(res1, res2, "skip");

    prevOut = res2;
  }

  const finalNorm = "norm";
  normNode(finalNorm, "Final RMSNorm", "model", "model.norm.weight", H);
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
