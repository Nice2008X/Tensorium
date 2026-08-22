import type { LoadProgress, Model, ModelAdapter, ModelMetadata, ModelSource, WeightProvider } from "@tensorium/model-ir";
import { SafetensorsWeightProvider, SyntheticWeightProvider } from "@tensorium/tensor-core";
import { loadSafetensorsMetadata } from "@tensorium/hf-client";
import { buildModelConfig, buildGraph, runInference, type LlamaFamilyRawConfig } from "@tensorium/adapter-llama-family";

// Qwen2-MoE (Qwen1.5-MoE) is Qwen2's attention (RoPE, GQA, a Q/K/V bias)
// wrapped around a sparse Mixture-of-Experts FFN instead of one dense gated
// FFN: a router scores every expert per token, the top few actually run
// (weighted-summed by the router's confidence), and one extra "shared"
// expert always runs on every token on top of that, sigmoid-gated.
//
// Deliberately scoped to exactly "qwen2_moe" / "Qwen2MoeForCausalLM": Qwen3
// has its own MoE variant ("qwen3_moe") with QK-Norm instead of a Q/K/V
// bias and no shared expert — different enough to need its own adapter
// rather than another flag here.
const PROVIDER_ID = "qwen-moe-weights";

export const QwenMoeAdapter: ModelAdapter = {
  id: "qwen2_moe",
  displayName: "Qwen2-MoE",

  canLoad(_source, metadata) {
    if (!metadata) return true;
    return metadata.model_type === "qwen2_moe" || (metadata.architectures ?? []).some((a) => a === "Qwen2MoeForCausalLM");
  },

  async loadMetadata(source: ModelSource, onProgress?: (progress: LoadProgress) => void): Promise<ModelMetadata> {
    const { rawConfig, weightIndex, weightsBuffer, structureOnly } = await loadSafetensorsMetadata<LlamaFamilyRawConfig>(source, onProgress);

    return {
      architecture: (rawConfig.architectures && rawConfig.architectures[0]) || "Qwen2MoeForCausalLM",
      config: buildModelConfig(rawConfig, { defaultModelType: "qwen2_moe", qkvBias: true, moe: true }),
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
