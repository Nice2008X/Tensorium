import type { LoadProgress, Model, ModelAdapter, ModelMetadata, ModelSource, WeightProvider } from "@tensorium/model-ir";
import { SafetensorsWeightProvider, SyntheticWeightProvider } from "@tensorium/tensor-core";
import { loadSafetensorsMetadata } from "@tensorium/hf-client";
import { buildModelConfig, buildGraph, runInference, type LlamaFamilyRawConfig } from "@tensorium/adapter-llama-family";

// GLM-4 (Glm4ForCausalLM, distinct from the older custom ChatGLM code) is a
// Llama-family model with two real structural differences, both confirmed
// against the real safetensors header before writing this (same as every
// other adapter here):
//  - Sandwich norm: each layer carries post_self_attn_layernorm and
//    post_mlp_layernorm weights alongside the usual input_layernorm/
//    post_attention_layernorm pair — an extra RMSNorm on each sub-layer's
//    *output*, right before it joins the residual stream, on top of (not
//    instead of) the existing pre-norms.
//  - Partial rotary: config.json's partial_rotary_factor (0.5 here) means
//    only the first half of each head's dimensions get RoPE; the rest pass
//    through unrotated. Read directly off raw config in adapter-llama-family
//    (it's a real config.json field, not an architecture-inferred trait).
// Fused gate_up_proj is also present (like Phi3/Phi4), so fusedGateUp is
// set too; q/k/v stay unfused, unlike Phi.
const PROVIDER_ID = "glm4-weights";

export const Glm4Adapter: ModelAdapter = {
  id: "glm4",
  displayName: "GLM-4",

  canLoad(_source, metadata) {
    if (!metadata) return true;
    return metadata.model_type === "glm4" || (metadata.architectures ?? []).some((a) => a === "Glm4ForCausalLM");
  },

  async loadMetadata(source: ModelSource, onProgress?: (progress: LoadProgress) => void): Promise<ModelMetadata> {
    const { rawConfig, weightIndex, weightsBuffer, structureOnly } = await loadSafetensorsMetadata<LlamaFamilyRawConfig>(source, onProgress);

    return {
      architecture: (rawConfig.architectures && rawConfig.architectures[0]) || "Glm4ForCausalLM",
      config: buildModelConfig(rawConfig, { defaultModelType: "glm4", sandwichNorm: true, fusedGateUp: true }),
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
