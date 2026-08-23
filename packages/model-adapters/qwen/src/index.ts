import type { LoadProgress, Model, ModelAdapter, ModelMetadata, ModelSource, WeightProvider } from "@tensorium/model-ir";
import { SafetensorsWeightProvider, SyntheticWeightProvider } from "@tensorium/tensor-core";
import { loadSafetensorsMetadata } from "@tensorium/hf-client";
import { buildModelConfig, buildGraph, runInference, type LlamaFamilyRawConfig } from "@tensorium/adapter-llama-family";

// Qwen2 (and Qwen2.5, which reuses the same architecture class and config
// shape — only training data/tokenizer differ) is otherwise a Llama-family
// model: RoPE, standard RMSNorm, SwiGLU gated FFN, GQA. Its one real
// architectural difference is a bias on q_proj/k_proj/v_proj (o_proj has
// none) — a deliberate holdover from the original Qwen design, everywhere
// else in this family has no attention/MLP biases at all.
//
// Deliberately scoped to exactly "qwen2" / "Qwen2ForCausalLM": Qwen3 uses
// its own model_type ("qwen3") and adds real differences (QK-norm) this
// adapter doesn't implement, and multimodal Qwen variants (VL, Audio, ...)
// have different weight structures entirely — an exact match means an
// unsupported checkpoint fails loudly with "no adapter found" instead of
// silently producing wrong output.
const PROVIDER_ID = "qwen-weights";

export const QwenAdapter: ModelAdapter = {
  id: "qwen2",
  displayName: "Qwen2",

  canLoad(_source, metadata) {
    if (!metadata) return true;
    return metadata.model_type === "qwen2" || (metadata.architectures ?? []).some((a) => a === "Qwen2ForCausalLM");
  },

  async loadMetadata(source: ModelSource, onProgress?: (progress: LoadProgress) => void): Promise<ModelMetadata> {
    const { rawConfig, weightIndex, weightsBuffer, structureOnly } = await loadSafetensorsMetadata<LlamaFamilyRawConfig>(source, onProgress);

    return {
      architecture: (rawConfig.architectures && rawConfig.architectures[0]) || "Qwen2ForCausalLM",
      config: buildModelConfig(rawConfig, { defaultModelType: "qwen2", qkvBias: true }),
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
