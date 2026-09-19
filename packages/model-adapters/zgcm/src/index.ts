import type { LoadProgress, Model, ModelAdapter, ModelMetadata, ModelSource, WeightProvider } from "@tensorium/model-ir";
import { SafetensorsWeightProvider, SyntheticWeightProvider } from "@tensorium/tensor-core";
import { loadSafetensorsMetadata } from "@tensorium/hf-client";
import { buildModelConfig, buildGraph, runInference, type LlamaFamilyRawConfig } from "@tensorium/adapter-llama-family";

// ZGCM-1 (model_type "zgcm", ZgcmForCausalLM; custom modeling_zgcm.py, no
// transformers-native class) is a Llama-family decoder with four real
// differences, each confirmed against modeling_zgcm.py and the real
// safetensors index before writing this:
//  - Hybrid attention: config.json's layer_types marks most layers
//    "sliding_attention" (each token sees only the last `sliding_window`
//    tokens, 128 here) and every 6th "full_attention".
//  - Attention output gate: sliding layers carry a separate
//    self_attn.g_proj ([num_heads*head_dim, hidden]); sigmoid(g_proj(x))
//    multiplies the attention heads' output right before o_proj. Full-
//    attention layers have no g_proj (attention_gate_layers is false there).
//  - Per-head QK-Norm (q_norm/k_norm, head_dim-wide), applied before RoPE.
//  - Partial RoPE with a width of int(head_dim * partial_rotary_factor)
//    rounded down to even (128 × 0.334 → 42, not the 43 a plain round gives),
//    rotating the first 42 dims of each head NeoX-style.
// The block itself is an ordinary pre-norm block; only the weight *names*
// are unusual: the norm before attention is `post_attention_layernorm` and
// the one before the FFN is `post_feedforward_layernorm` (there is no
// `input_layernorm`).
const PROVIDER_ID = "zgcm-weights";

export const ZgcmAdapter: ModelAdapter = {
  id: "zgcm",
  displayName: "ZGCM",

  canLoad(_source, metadata) {
    if (!metadata) return true;
    return metadata.model_type === "zgcm" || (metadata.architectures ?? []).some((a) => a === "ZgcmForCausalLM");
  },

  async loadMetadata(source: ModelSource, onProgress?: (progress: LoadProgress) => void): Promise<ModelMetadata> {
    const { rawConfig, weightIndex, weightsBuffer, structureOnly } = await loadSafetensorsMetadata<LlamaFamilyRawConfig>(source, onProgress);

    return {
      architecture: (rawConfig.architectures && rawConfig.architectures[0]) || "ZgcmForCausalLM",
      config: buildModelConfig(rawConfig, {
        defaultModelType: "zgcm",
        qkNorm: true,
        perLayerSlidingWindow: true,
        attentionOutputGate: true,
        normWeightNames: { preAttention: "post_attention_layernorm", preFfn: "post_feedforward_layernorm" },
        rotaryDimRounding: "floor_even",
      }),
      weightIndex,
      source,
      weightsBuffer,
      structureOnly,
    };
  },

  buildGraph(metadata: ModelMetadata): Model {
    return buildGraph(metadata, PROVIDER_ID);
  },

  getWeightProvider(metadata: ModelMetadata): WeightProvider {
    if (metadata.structureOnly) return new SyntheticWeightProvider(PROVIDER_ID, metadata.weightIndex);
    if (!metadata.weightsBuffer) throw new Error("No weights buffer available on this metadata");
    return new SafetensorsWeightProvider(PROVIDER_ID, metadata.weightsBuffer);
  },

  runInference,
};
