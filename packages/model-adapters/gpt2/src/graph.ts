import type { EdgeKind, Model, ModelConfig, ModelMetadata, ModelNode, NodeType, ParameterRef, TensorSlice } from "@tensorium/model-ir";
import { numElements, dtypeSize, modelSourceLabel } from "@tensorium/model-ir";

export interface GPT2RawConfig {
  model_type?: string;
  architectures?: string[];
  n_layer: number;
  n_head: number;
  n_embd: number;
  vocab_size: number;
  n_positions?: number;
  n_ctx?: number;
  n_inner?: number | null;
  activation_function?: string;
  layer_norm_epsilon?: number;
  tie_word_embeddings?: boolean;
}

export function buildModelConfig(raw: GPT2RawConfig, weightIndex: Record<string, { shape: number[]; dtype: string }>): ModelConfig {
  const fcWeight = weightIndex["transformer.h.0.mlp.c_fc.weight"];
  const intermediateSize = fcWeight ? fcWeight.shape[1] : raw.n_inner ?? raw.n_embd * 4;

  return {
    modelType: raw.model_type ?? "gpt2",
    numLayers: raw.n_layer,
    numHeads: raw.n_head,
    hiddenSize: raw.n_embd,
    intermediateSize,
    vocabSize: raw.vocab_size,
    contextLength: raw.n_positions ?? raw.n_ctx ?? 1024,
    extra: {
      activationFunction: raw.activation_function ?? "gelu_new",
      layerNormEpsilon: raw.layer_norm_epsilon ?? 1e-5,
      tiedEmbeddings: raw.tie_word_embeddings ?? true,
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

  const H = cfg.hiddenSize;
  const seqH: Array<number | string> = ["sequence_length", H];

  // --- root -------------------------------------------------------------
  node("model", "model", metadata.architecture, null);

  node("input", "input", "Input tokens", "model", {
    outputs: [{ dims: ["sequence_length"] }],
    metadata: { description: "Token IDs produced by the tokenizer." },
  });

  node("wte", "embedding", "Token Embedding", "model", {
    inputs: [{ dims: ["sequence_length"] }],
    outputs: [{ dims: ["sequence_length", H] }],
    parameters: [param("transformer.wte.weight", wi, providerId)],
  });
  node("wpe", "positional_embedding", "Positional Embedding", "model", {
    inputs: [{ dims: ["sequence_length"] }],
    outputs: [{ dims: ["sequence_length", H] }],
    parameters: [param("transformer.wpe.weight", wi, providerId)],
  });
  edge("input", "wte");
  edge("input", "wpe");

  node("blocks", "block_group", `Transformer Blocks × ${cfg.numLayers}`, "model", {
    metadata: { count: cfg.numLayers },
  });

  let prevOut = "wte"; // wpe merges in via edge, but wte is the "main" carrier for layout purposes
  for (let i = 0; i < cfg.numLayers; i++) {
    const b = `block.${i}`;
    node(b, "transformer_block", `Transformer Block ${i}`, "blocks", {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: seqH }],
    });
    edge(prevOut, b);
    if (i === 0) edge("wpe", b);

    const ln1 = `${b}.ln1`;
    node(ln1, "layer_norm", "LayerNorm 1", b, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: seqH }],
      parameters: [param(`transformer.h.${i}.ln_1.weight`, wi, providerId), param(`transformer.h.${i}.ln_1.bias`, wi, providerId)],
    });
    edge(b, ln1);

    const attn = `${b}.attn`;
    node(attn, "attention", "Attention", b, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: seqH }],
      metadata: { numHeads: cfg.numHeads, headDim: H / cfg.numHeads },
    });
    edge(ln1, attn);

    const attnWShape = wi[`transformer.h.${i}.attn.c_attn.weight`].shape; // [H, 3H] fused QKV
    const attnBShape = wi[`transformer.h.${i}.attn.c_attn.bias`].shape; // [3H]
    const qkvLabels: Array<[string, NodeType, string]> = [
      ["q", "q_projection", "Q Projection"],
      ["k", "k_projection", "K Projection"],
      ["v", "v_projection", "V Projection"],
    ];
    qkvLabels.forEach(([suffix, type, label], slot) => {
      const id = `${attn}.${suffix}`;
      node(id, type, label, attn, {
        inputs: [{ dims: seqH }],
        outputs: [{ dims: seqH }],
        parameters: [
          param(`transformer.h.${i}.attn.c_attn.weight`, wi, providerId, {
            ranges: [{ start: 0, end: attnWShape[0] }, { start: slot * H, end: (slot + 1) * H }],
          }),
          param(`transformer.h.${i}.attn.c_attn.bias`, wi, providerId, {
            ranges: [{ start: slot * H, end: (slot + 1) * H }],
          }),
        ],
        metadata: { note: "Sliced out of the fused c_attn weight — GPT-2 packs Q, K, V into a single matrix.", fusedShape: attnWShape, fusedBiasShape: attnBShape },
      });
      edge(ln1, id);
    });

    const outp = `${attn}.out`;
    node(outp, "output_projection", "Output Projection", attn, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: seqH }],
      parameters: [param(`transformer.h.${i}.attn.c_proj.weight`, wi, providerId), param(`transformer.h.${i}.attn.c_proj.bias`, wi, providerId)],
    });
    edge(`${attn}.q`, outp);
    edge(`${attn}.k`, outp);
    edge(`${attn}.v`, outp);

    const res1 = `${b}.res1`;
    node(res1, "residual", "Residual Add", b, {
      inputs: [{ dims: seqH }, { dims: seqH }],
      outputs: [{ dims: seqH }],
      metadata: { description: "Adds the block's input back in around Attention." },
    });
    edge(outp, res1);
    edge(b, res1, "skip");

    const ln2 = `${b}.ln2`;
    node(ln2, "layer_norm", "LayerNorm 2", b, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: seqH }],
      parameters: [param(`transformer.h.${i}.ln_2.weight`, wi, providerId), param(`transformer.h.${i}.ln_2.bias`, wi, providerId)],
    });
    edge(res1, ln2);

    const ffn = `${b}.ffn`;
    node(ffn, "ffn", "Feed Forward", b, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: seqH }],
    });
    edge(ln2, ffn);

    const fc = `${ffn}.fc`;
    node(fc, "linear", "Linear (expand)", ffn, {
      inputs: [{ dims: seqH }],
      outputs: [{ dims: ["sequence_length", cfg.intermediateSize] }],
      parameters: [param(`transformer.h.${i}.mlp.c_fc.weight`, wi, providerId), param(`transformer.h.${i}.mlp.c_fc.bias`, wi, providerId)],
    });
    edge(ffn, fc);
    // also wired directly from ln2, so the block-detail (leaf-only) graph view
    // — which skips the purely-organizational "ffn" container — stays connected.
    edge(ln2, fc);

    const act = `${ffn}.act`;
    node(act, "activation", String(cfg.extra.activationFunction ?? "gelu"), ffn, {
      inputs: [{ dims: ["sequence_length", cfg.intermediateSize] }],
      outputs: [{ dims: ["sequence_length", cfg.intermediateSize] }],
    });
    edge(fc, act);

    const proj = `${ffn}.proj`;
    node(proj, "linear", "Linear (project)", ffn, {
      inputs: [{ dims: ["sequence_length", cfg.intermediateSize] }],
      outputs: [{ dims: seqH }],
      parameters: [param(`transformer.h.${i}.mlp.c_proj.weight`, wi, providerId), param(`transformer.h.${i}.mlp.c_proj.bias`, wi, providerId)],
    });
    edge(act, proj);

    const res2 = `${b}.res2`;
    node(res2, "residual", "Residual Add", b, {
      inputs: [{ dims: seqH }, { dims: seqH }],
      outputs: [{ dims: seqH }],
      metadata: { description: "Adds the pre-FFN state back in around the Feed Forward block." },
    });
    edge(proj, res2);
    edge(res1, res2, "skip");

    prevOut = res2;
  }

  const lnf = "ln_f";
  node(lnf, "layer_norm", "Final LayerNorm", "model", {
    inputs: [{ dims: seqH }],
    outputs: [{ dims: seqH }],
    parameters: [param("transformer.ln_f.weight", wi, providerId), param("transformer.ln_f.bias", wi, providerId)],
  });
  edge(prevOut, lnf);

  const lmHeadParams: ParameterRef[] = wi["lm_head.weight"]
    ? [param("lm_head.weight", wi, providerId)]
    : [param("transformer.wte.weight", wi, providerId)];
  node("lm_head", "lm_head", "LM Head", "model", {
    inputs: [{ dims: seqH }],
    outputs: [{ dims: ["sequence_length", cfg.vocabSize] }],
    parameters: lmHeadParams,
    metadata: { tied: !wi["lm_head.weight"], description: !wi["lm_head.weight"] ? "Tied to the token embedding weight (transposed)." : undefined },
  });
  edge(lnf, "lm_head");

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

