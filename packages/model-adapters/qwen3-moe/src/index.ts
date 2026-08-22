import type { LoadProgress, Model, ModelAdapter, ModelMetadata, ModelSource, WeightProvider } from "@tensorium/model-ir";
import { SafetensorsWeightProvider, SyntheticWeightProvider } from "@tensorium/tensor-core";
import { loadSafetensorsMetadata } from "@tensorium/hf-client";
import { buildModelConfig, buildGraph, runInference, type LlamaFamilyRawConfig } from "@tensorium/adapter-llama-family";

// Qwen3-MoE is Qwen3's attention (RoPE, GQA, per-head Q/K-Norm, no Q/K/V
// bias) wrapped around a sparse Mixture-of-Experts FFN — same routing idea
// as Qwen2-MoE (a router picks a few experts per token, weighted-summed),
// but with two real differences: no always-on shared expert, and not every
// layer is sparse — config.json's decoder_sparse_step (2 on real Qwen3-MoE
// checkpoints) skips routing on every other layer, which falls back to a
// plain dense gated FFN instead. Both are real, checkpoint-driven config
// fields adapter-llama-family already threads through generically.
//
// Deliberately scoped to exactly "qwen3_moe" / "Qwen3MoeForCausalLM":
// Qwen3.5-MoE ("qwen3_5_moe") is a different, much larger architecture —
// a hybrid of Mamba-style linear-attention layers and ordinary attention,
// a vision tower, and a multi-token-prediction head — none of which this
// engine models.
const PROVIDER_ID = "qwen3-moe-weights";

export const Qwen3MoeAdapter: ModelAdapter = {
  id: "qwen3_moe",
  displayName: "Qwen3-MoE",

  canLoad(_source, metadata) {
    if (!metadata) return true;
    return metadata.model_type === "qwen3_moe" || (metadata.architectures ?? []).some((a) => a === "Qwen3MoeForCausalLM");
  },

  async loadMetadata(source: ModelSource, onProgress?: (progress: LoadProgress) => void): Promise<ModelMetadata> {
    const { rawConfig, weightIndex, weightsBuffer, structureOnly } = await loadSafetensorsMetadata<LlamaFamilyRawConfig>(source, onProgress);

    return {
      architecture: (rawConfig.architectures && rawConfig.architectures[0]) || "Qwen3MoeForCausalLM",
      config: buildModelConfig(rawConfig, { defaultModelType: "qwen3_moe", qkNorm: true, moe: true }),
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
