import type { LoadProgress, Model, ModelAdapter, ModelMetadata, ModelSource, WeightProvider } from "@tensorium/model-ir";
import { SafetensorsWeightProvider, SyntheticWeightProvider } from "@tensorium/tensor-core";
import { loadSafetensorsMetadata } from "@tensorium/hf-client";
import { buildModelConfig, buildGraph, runInference, type LlamaFamilyRawConfig } from "@tensorium/adapter-llama-family";

// OLMo (v1) is otherwise Llama-shaped — RoPE, SwiGLU gated FFN, pre-norm
// blocks, no attention/MLP biases — but its norm is a true LayerNorm with
// no learnable weight or bias at all (fixed gamma=1, beta=0), not RMSNorm,
// and it optionally clamps Q/K/V to [-clip_qkv, clip_qkv] right after
// projection (a real config.json field, null on most released checkpoints).
//
// Deliberately scoped to exactly "olmo" / "OlmoForCausalLM": OLMo 2 uses
// its own model_type ("olmo2") and is architecturally different enough
// (RMSNorm applied after each sub-layer instead of before it, plus
// per-head QK-norm, no LayerNorm/clip_qkv at all) to need its own adapter
// rather than a flag on this one.
const PROVIDER_ID = "olmo-weights";

export const OlmoAdapter: ModelAdapter = {
  id: "olmo",
  displayName: "OLMo",

  canLoad(_source, metadata) {
    if (!metadata) return true;
    return metadata.model_type === "olmo" || (metadata.architectures ?? []).some((a) => a === "OlmoForCausalLM");
  },

  async loadMetadata(source: ModelSource, onProgress?: (progress: LoadProgress) => void): Promise<ModelMetadata> {
    const { rawConfig, weightIndex, weightsBuffer, structureOnly } = await loadSafetensorsMetadata<LlamaFamilyRawConfig>(source, onProgress);

    return {
      architecture: (rawConfig.architectures && rawConfig.architectures[0]) || "OlmoForCausalLM",
      config: buildModelConfig(rawConfig, { defaultModelType: "olmo", normType: "layernorm_no_affine" }),
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
