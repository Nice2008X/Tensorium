import type { LoadProgress, Model, ModelAdapter, ModelMetadata, ModelSource, WeightProvider } from "@tensorium/model-ir";
import { SafetensorsWeightProvider, SyntheticWeightProvider } from "@tensorium/tensor-core";
import { loadSafetensorsMetadata } from "@tensorium/hf-client";
import { buildModelConfig, buildGraph, runInference, type LlamaFamilyRawConfig } from "@tensorium/adapter-llama-family";

// Qwen3 is a Llama-family model like Qwen2 (RoPE, standard RMSNorm, SwiGLU,
// GQA, explicit head_dim like Gemma), but its stabilization mechanism
// changed: Qwen2's bias on q/k/v projections is gone (attention_bias:
// false), replaced by QK-Norm — a per-head RMSNorm (one shared [head_dim]
// weight reused across every head) applied to Q and K right after
// projection, before RoPE. Confirmed against the real safetensors header
// (q_norm.weight/k_norm.weight per layer, shape [head_dim], no bias
// tensors at all) before writing this, same as every other adapter here.
const PROVIDER_ID = "qwen3-weights";

export const Qwen3Adapter: ModelAdapter = {
  id: "qwen3",
  displayName: "Qwen3",

  canLoad(_source, metadata) {
    if (!metadata) return true;
    return metadata.model_type === "qwen3" || (metadata.architectures ?? []).some((a) => a === "Qwen3ForCausalLM");
  },

  async loadMetadata(source: ModelSource, onProgress?: (progress: LoadProgress) => void): Promise<ModelMetadata> {
    const { rawConfig, weightIndex, weightsBuffer, structureOnly } = await loadSafetensorsMetadata<LlamaFamilyRawConfig>(source, onProgress);

    return {
      architecture: (rawConfig.architectures && rawConfig.architectures[0]) || "Qwen3ForCausalLM",
      config: buildModelConfig(rawConfig, { defaultModelType: "qwen3", qkNorm: true, qkvBias: false }),
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
