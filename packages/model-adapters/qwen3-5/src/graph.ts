import type { Model, ModelConfig, ModelMetadata, ModelNode, NodeType, ParameterRef, TensorSlice } from "@tensorium/model-ir";
import { numElements, dtypeSize, modelSourceLabel } from "@tensorium/model-ir";

/**
 * `Qwen/Qwen3.8-27B` and its sibling `Qwen/Qwen3.5-27B` (`model_type:
 * "qwen3_5"`, `architectures: ["Qwen3_5ForConditionalGeneration"]`) ship a
 * genuinely multimodal checkpoint — a hybrid text decoder plus a separate
 * vision tower (`model.visual.*`) and an optional multi-token-prediction
 * head (`mtp.*`). This adapter deliberately covers the text decoder only
 * (`text_config` below, weights under `model.language_model.*` and the
 * top-level `lm_head.weight`) — the same choice `adapter-gemma4` makes for
 * its own vision/audio towers, for the same reason: neither the vision
 * tower nor the MTP head are transformer blocks this app's graph/Inspector/
 * forward-pass machinery understands.
 *
 * Every field and formula below was confirmed against the real
 * `transformers` `models/qwen3_5/modeling_qwen3_5.py` source and the real
 * `tiny-random/qwen3.5` checkpoint's own safetensors header (exact weight
 * names/shapes), not guessed — this architecture is brand new (released
 * 2026-08), so a plausible-looking guess had a real chance of being subtly
 * wrong. See inference.ts's doc comments for the attention/linear-attention
 * math itself.
 */
export interface Qwen35RawConfig {
  model_type?: string;
  architectures?: string[];
  text_config: Qwen35TextRawConfig;
}

export interface Qwen35TextRawConfig {
  model_type?: string;
  hidden_size: number;
  intermediate_size: number;
  num_hidden_layers: number;
  num_attention_heads: number;
  num_key_value_heads?: number;
  head_dim: number;
  vocab_size: number;
  max_position_embeddings?: number;
  rms_norm_eps?: number;
  /** Real value seen: "silu" — read directly, never assumed. */
  hidden_act?: string;
  tie_word_embeddings?: boolean;
  attention_bias?: boolean;
  /** Doubles q_proj's output width to also emit a per-head sigmoid output gate on ordinary attention layers — real checkpoints seen so far always have this true; kept as a real flag (not hardcoded) in case a future checkpoint disables it. */
  attn_output_gate?: boolean;
  /** One entry per layer: "full_attention" (ordinary causal GQA) or "linear_attention" (Gated DeltaNet). */
  layer_types: string[];
  /** Every Nth layer is full_attention, the rest linear_attention — informational only, buildGraph reads layer_types directly rather than recomputing this. */
  full_attention_interval?: number;
  linear_conv_kernel_dim: number;
  linear_key_head_dim: number;
  linear_num_key_heads: number;
  linear_value_head_dim: number;
  linear_num_value_heads: number;
  /** Fraction of head_dim that's actually rotated by RoPE — "slice the first N dims, rotate those, pass the rest through unchanged" (GLM-4's convention, confirmed against the real apply_rotary_pos_emb source: NOT Gemma-4's proportional/zero-frequency variant). */
  partial_rotary_factor?: number;
  rope_parameters?: { rope_theta?: number; rope_type?: string } | null;
}

export function buildModelConfig(raw: Qwen35RawConfig): ModelConfig {
  const t = raw.text_config;
  const numLayers = t.num_hidden_layers;
  const layerTypes = t.layer_types;

  if (layerTypes.length !== numLayers) {
    throw new Error(`text_config.layer_types has ${layerTypes.length} entries but num_hidden_layers is ${numLayers} — can't tell each layer's attention kind.`);
  }

  return {
    modelType: raw.model_type ?? "qwen3_5",
    numLayers,
    numHeads: t.num_attention_heads,
    hiddenSize: t.hidden_size,
    intermediateSize: t.intermediate_size,
    vocabSize: t.vocab_size,
    contextLength: t.max_position_embeddings ?? 262144,
    extra: {
      rmsNormEps: t.rms_norm_eps ?? 1e-6,
      activationFunction: t.hidden_act ?? "silu",
      tiedEmbeddings: t.tie_word_embeddings ?? false,
      attentionBias: t.attention_bias ?? false,
      attnOutputGate: t.attn_output_gate ?? true,
      layerTypes,
      numKeyValueHeads: t.num_key_value_heads ?? t.num_attention_heads,
      headDim: t.head_dim,
      linearConvKernelDim: t.linear_conv_kernel_dim,
      linearKeyHeadDim: t.linear_key_head_dim,
      linearNumKeyHeads: t.linear_num_key_heads,
      linearValueHeadDim: t.linear_value_head_dim,
      linearNumValueHeads: t.linear_num_value_heads,
      partialRotaryFactor: t.partial_rotary_factor ?? 1,
      ropeTheta: t.rope_parameters?.rope_theta ?? 10000000,
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

/** model.language_model.* — the text decoder's own weight prefix; the vision tower and mtp prefixes stay present in weightIndex but are never referenced by any node this adapter creates. Note lm_head.weight itself lives at the top level, not under this prefix — confirmed against the real checkpoint. */
const LP = "model.language_model";

export function buildGraph(metadata: ModelMetadata, providerId: string): Model {
  const cfg = metadata.config;
  const wi = metadata.weightIndex;
  const nodes: Record<string, ModelNode> = {};
  const edges: Model["edges"] = [];

  const layerTypes = cfg.extra.layerTypes as string[];
  const numHeads = cfg.numHeads;
  const numKVHeads = Number(cfg.extra.numKeyValueHeads);
  const headDim = Number(cfg.extra.headDim);
  const attnOutputGate = cfg.extra.attnOutputGate === true;
  const hasAttnBias = cfg.extra.attentionBias === true;
  const linearConvKernelDim = Number(cfg.extra.linearConvKernelDim);
  const linearKeyHeadDim = Number(cfg.extra.linearKeyHeadDim);
  const linearNumKeyHeads = Number(cfg.extra.linearNumKeyHeads);
  const linearValueHeadDim = Number(cfg.extra.linearValueHeadDim);
  const linearNumValueHeads = Number(cfg.extra.linearNumValueHeads);
  const keyDim = linearNumKeyHeads * linearKeyHeadDim;
  const valueDim = linearNumValueHeads * linearValueHeadDim;
  const convDim = keyDim * 2 + valueDim;

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

  function normNode(id: string, label: string, parentId: string, weightName: string, dim: number, opts: { note?: string } = {}) {
    return node(id, "rms_norm", label, parentId, {
      inputs: [{ dims: ["sequence_length", dim] }],
      outputs: [{ dims: ["sequence_length", dim] }],
      parameters: [param(weightName, wi, providerId)],
      metadata: opts.note ? { note: opts.note } : {},
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
  });
  edge("input", "embed");

  node("blocks", "block_group", `Transformer Blocks × ${cfg.numLayers}`, "model", {
    metadata: { count: cfg.numLayers },
  });

  let prevOut = "embed";
  for (let i = 0; i < cfg.numLayers; i++) {
    const b = `block.${i}`;
    const L = `${LP}.layers.${i}`;
    const layerType = layerTypes[i];
    const isLinear = layerType === "linear_attention";

    node(b, "transformer_block", `Transformer Block ${i}`, "blocks", {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: seqH }],
      metadata: { layerType },
    });
    edge(prevOut, b);

    const rms1 = `${b}.rms1`;
    normNode(rms1, "RMSNorm (input)", b, `${L}.input_layernorm.weight`, H);
    edge(b, rms1);

    const attn = `${b}.attn`;
    let attnOutSource: string;

    if (isLinear) {
      node(attn, "linear_attention", "Gated DeltaNet", b, {
        inputs: [{ dims: seqH }],
        outputs: [{ dims: seqH }],
        metadata: {
          linearNumKeyHeads,
          linearNumValueHeads,
          linearKeyHeadDim,
          linearValueHeadDim,
          description: `A linear/recurrent attention layer (Gated DeltaNet), not ordinary softmax attention: a short causal convolution over a fused Q/K/V projection, then a per-head recurrent state (${linearKeyHeadDim}×${linearValueHeadDim}) updated one token at a time by the "delta rule" (decay, then correct toward the new value, then read out against the query) — O(sequence_length) instead of attention's O(sequence_length²). ${linearNumKeyHeads} key/decay/beta heads are each shared, GQA-style, across ${linearNumValueHeads / linearNumKeyHeads} of the ${linearNumValueHeads} value heads.`,
        },
      });

      const qkv = `${attn}.qkv`;
      node(qkv, "qkv_projection", "QKV Projection", attn, {
        inputs: [{ dims: seqH }],
        outputs: [{ dims: ["sequence_length", convDim] }],
        parameters: [param(`${L}.linear_attn.in_proj_qkv.weight`, wi, providerId)],
        metadata: { note: `Fused: the first ${keyDim} columns are Q, the next ${keyDim} are K, the last ${valueDim} are V.` },
      });
      edge(rms1, qkv);

      const conv = `${attn}.conv`;
      node(conv, "linear", "Causal Conv1D (SiLU)", attn, {
        inputs: [{ dims: ["sequence_length", convDim] }],
        outputs: [{ dims: ["sequence_length", convDim] }],
        parameters: [param(`${L}.linear_attn.conv1d.weight`, wi, providerId)],
        metadata: { description: `Depthwise (per-channel) causal convolution, kernel width ${linearConvKernelDim}, followed by SiLU — short-range mixing applied independently to each of Q/K/V's own channels before the recurrence below.` },
      });
      edge(qkv, conv);

      const beta = `${attn}.beta`;
      node(beta, "linear", "β Gate Projection", attn, {
        inputs: [{ dims: seqH }],
        outputs: [{ dims: ["sequence_length", linearNumValueHeads] }],
        parameters: [param(`${L}.linear_attn.in_proj_b.weight`, wi, providerId)],
        metadata: { description: "Sigmoid-activated per-head correction strength (β) for the delta rule below." },
      });
      edge(rms1, beta);

      const decay = `${attn}.decay`;
      node(decay, "linear", "Decay Gate Projection", attn, {
        inputs: [{ dims: seqH }],
        outputs: [{ dims: ["sequence_length", linearNumValueHeads] }],
        parameters: [param(`${L}.linear_attn.in_proj_a.weight`, wi, providerId), param(`${L}.linear_attn.A_log`, wi, providerId), param(`${L}.linear_attn.dt_bias`, wi, providerId)],
        metadata: { description: "Per-head recurrent-state decay: g = -exp(A_log) · softplus(a + dt_bias), exponentiated to a (0,1] multiplicative factor before each timestep's state update." },
      });
      edge(rms1, decay);

      const gate = `${attn}.gate`;
      node(gate, "linear", "Output Gate Projection", attn, {
        inputs: [{ dims: seqH }],
        outputs: [{ dims: ["sequence_length", valueDim] }],
        parameters: [param(`${L}.linear_attn.in_proj_z.weight`, wi, providerId)],
      });
      edge(rms1, gate);

      const gatedNorm = `${attn}.gated_norm`;
      normNode(gatedNorm, "RMSNorm (per-head, gated)", attn, `${L}.linear_attn.norm.weight`, valueDim, {
        note: `One [${linearValueHeadDim}]-wide learned weight, reused identically across all ${linearNumValueHeads} value heads — normalizes the recurrence's output (from Conv1D/β/Decay above) ahead of the output gate.`,
      });
      edge(conv, gatedNorm);
      edge(beta, gatedNorm);
      edge(decay, gatedNorm);

      const gateAct = `${attn}.gate_act`;
      node(gateAct, "activation", "silu", attn, {
        inputs: [{ dims: ["sequence_length", valueDim] }],
        outputs: [{ dims: ["sequence_length", valueDim] }],
      });
      edge(gate, gateAct);

      const gatedMul = `${attn}.gated_mul`;
      node(gatedMul, "elementwise_mul", "× Output Gate", attn, {
        inputs: [{ dims: ["sequence_length", valueDim] }, { dims: ["sequence_length", valueDim] }],
        outputs: [{ dims: ["sequence_length", valueDim] }],
      });
      edge(gatedNorm, gatedMul);
      edge(gateAct, gatedMul);

      const outp = `${attn}.out`;
      node(outp, "output_projection", "Output Projection", attn, {
        inputs: [{ dims: ["sequence_length", valueDim] }],
        outputs: [{ dims: seqH }],
        parameters: [param(`${L}.linear_attn.out_proj.weight`, wi, providerId)],
      });
      edge(gatedMul, outp);
      attnOutSource = outp;
    } else {
      const qWidth = numHeads * headDim * (attnOutputGate ? 2 : 1);
      node(attn, "attention", "Attention (full)", b, {
        inputs: [{ dims: seqH }],
        outputs: [{ dims: seqH }],
        metadata: {
          numHeads,
          numKeyValueHeads: numKVHeads,
          headDim,
          description: "Ordinary causal GQA attention over the whole sequence — one of this model's periodic full-context layers, interleaved with the Gated DeltaNet layers around it.",
        },
      });

      const q = `${attn}.q`;
      node(q, "q_projection", "Q Projection", attn, {
        inputs: [{ dims: seqH }],
        outputs: [{ dims: ["sequence_length", qWidth] }],
        parameters: hasAttnBias
          ? [param(`${L}.self_attn.q_proj.weight`, wi, providerId), param(`${L}.self_attn.q_proj.bias`, wi, providerId)]
          : [param(`${L}.self_attn.q_proj.weight`, wi, providerId)],
        metadata: attnOutputGate
          ? { note: `Doubled width: each head's ${headDim * 2}-wide slice packs the real ${headDim}-wide query first, then a ${headDim}-wide output gate (used below, after attention, never rotated or normalized).` }
          : {},
      });
      edge(rms1, q);

      const qNorm = `${attn}.q_norm`;
      normNode(qNorm, "Q Norm", attn, `${L}.self_attn.q_norm.weight`, numHeads * headDim, {
        note: attnOutputGate ? "Normalizes only the query half of Q Projection's doubled output — the gate half passes through untouched." : undefined,
      });
      edge(q, qNorm);

      // Created here (right after Q Norm, before K/V) purely so this
      // shares Q Projection's column in the graph's layered layout — its
      // only real dependency is Q, but creation order also doubles as the
      // layout's left-right tie-break within a rank (see layout.ts), and
      // creating it later (after K/V/RoPE) left it sandwiched apart from
      // Q Norm with K Norm's column in between, forcing a real edge
      // crossing between K->K Norm and Q->Gate.
      let gateActId: string | null = null;
      let gateId: string | null = null;
      if (attnOutputGate) {
        gateId = `${attn}.gate`;
        node(gateId, "linear", "Attention Output Gate", attn, {
          inputs: [{ dims: ["sequence_length", qWidth] }],
          outputs: [{ dims: ["sequence_length", numHeads * headDim] }],
          metadata: { description: "The second half of Q Projection's doubled output — not a separate weight — sigmoid-activated and multiplied onto the attention output below." },
        });
        edge(q, gateId);

        gateActId = `${attn}.gate_act`;
        node(gateActId, "activation", "sigmoid", attn, {
          inputs: [{ dims: ["sequence_length", numHeads * headDim] }],
          outputs: [{ dims: ["sequence_length", numHeads * headDim] }],
        });
        edge(gateId, gateActId);
      }

      const k = `${attn}.k`;
      node(k, "k_projection", "K Projection", attn, {
        inputs: [{ dims: seqH }],
        outputs: [{ dims: ["sequence_length", numKVHeads * headDim] }],
        parameters: hasAttnBias
          ? [param(`${L}.self_attn.k_proj.weight`, wi, providerId), param(`${L}.self_attn.k_proj.bias`, wi, providerId)]
          : [param(`${L}.self_attn.k_proj.weight`, wi, providerId)],
      });
      edge(rms1, k);
      const kNorm = `${attn}.k_norm`;
      normNode(kNorm, "K Norm", attn, `${L}.self_attn.k_norm.weight`, numKVHeads * headDim);
      edge(k, kNorm);

      const v = `${attn}.v`;
      node(v, "v_projection", "V Projection", attn, {
        inputs: [{ dims: seqH }],
        outputs: [{ dims: ["sequence_length", numKVHeads * headDim] }],
        parameters: hasAttnBias
          ? [param(`${L}.self_attn.v_proj.weight`, wi, providerId), param(`${L}.self_attn.v_proj.bias`, wi, providerId)]
          : [param(`${L}.self_attn.v_proj.weight`, wi, providerId)],
      });
      edge(rms1, v);

      const rope = `${attn}.rope`;
      const rotaryDim = Math.round(headDim * Number(cfg.extra.partialRotaryFactor ?? 1));
      node(rope, "rope", "RoPE (partial)", attn, {
        inputs: [{ dims: ["sequence_length", numHeads * headDim] }],
        outputs: [{ dims: ["sequence_length", numHeads * headDim] }],
        metadata: {
          ropeTheta: cfg.extra.ropeTheta,
          description: `Rotates only the first ${rotaryDim} of each head's ${headDim} dims (θ=${cfg.extra.ropeTheta}) — the remaining ${headDim - rotaryDim} dims pass through unrotated.`,
        },
      });
      edge(qNorm, rope);
      edge(kNorm, rope);

      let attnResultSource = rope;
      if (gateActId) {
        const gatedMul = `${attn}.gated_mul`;
        node(gatedMul, "elementwise_mul", "× Output Gate", attn, {
          inputs: [{ dims: ["sequence_length", numHeads * headDim] }, { dims: ["sequence_length", numHeads * headDim] }],
          outputs: [{ dims: ["sequence_length", numHeads * headDim] }],
        });
        edge(rope, gatedMul);
        edge(v, gatedMul);
        edge(gateActId, gatedMul);
        attnResultSource = gatedMul;
      }

      const outp = `${attn}.out`;
      node(outp, "output_projection", "Output Projection", attn, {
        inputs: [{ dims: ["sequence_length", numHeads * headDim] }],
        outputs: [{ dims: seqH }],
        parameters: hasAttnBias
          ? [param(`${L}.self_attn.o_proj.weight`, wi, providerId), param(`${L}.self_attn.o_proj.bias`, wi, providerId)]
          : [param(`${L}.self_attn.o_proj.weight`, wi, providerId)],
      });
      edge(attnResultSource, outp);
      if (!gateActId) edge(v, outp);
      attnOutSource = outp;
    }

    const res1 = `${b}.res1`;
    node(res1, "residual", "Residual Add", b, {
      inputs: [{ dims: seqH }, { dims: seqH }],
      outputs: [{ dims: seqH }],
      metadata: { description: `Adds the block's input back in around ${isLinear ? "Gated DeltaNet" : "Attention"}.` },
    });
    edge(attnOutSource, res1);
    edge(b, res1, "skip");

    const rms2 = `${b}.rms2`;
    normNode(rms2, "RMSNorm (post-attention)", b, `${L}.post_attention_layernorm.weight`, H);
    edge(res1, rms2);

    const ffn = `${b}.ffn`;
    node(ffn, "ffn", "Feed Forward (gated)", b, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: seqH }],
    });

    const gateProj = `${ffn}.gate`;
    const up = `${ffn}.up`;
    node(gateProj, "linear", "Gate Projection", ffn, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: ["sequence_length", cfg.intermediateSize] }],
      parameters: [param(`${L}.mlp.gate_proj.weight`, wi, providerId)],
    });
    node(up, "linear", "Up Projection", ffn, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: ["sequence_length", cfg.intermediateSize] }],
      parameters: [param(`${L}.mlp.up_proj.weight`, wi, providerId)],
    });
    edge(rms2, gateProj);
    edge(rms2, up);

    const gateAct = `${ffn}.gate_act`;
    node(gateAct, "activation", String(cfg.extra.activationFunction ?? "silu"), ffn, {
      inputs: [{ dims: ["sequence_length", cfg.intermediateSize] }],
      outputs: [{ dims: ["sequence_length", cfg.intermediateSize] }],
    });
    edge(gateProj, gateAct);

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

    const res2 = `${b}.res2`;
    node(res2, "residual", "Residual Add", b, {
      inputs: [{ dims: seqH }, { dims: seqH }],
      outputs: [{ dims: seqH }],
      metadata: { description: "Adds the pre-FFN state back in around the Feed Forward block." },
    });
    edge(down, res2);
    edge(res1, res2, "skip");

    prevOut = res2;
  }

  const finalNorm = "norm";
  normNode(finalNorm, "Final RMSNorm", "model", `${LP}.norm.weight`, H);
  edge(prevOut, finalNorm);

  const tied = !wi["lm_head.weight"];
  node("lm_head", "lm_head", "LM Head", "model", {
    inputs: [{ dims: seqH }],
    outputs: [{ dims: ["sequence_length", cfg.vocabSize] }],
    parameters: [param(tied ? `${LP}.embed_tokens.weight` : "lm_head.weight", wi, providerId)],
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
