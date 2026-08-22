import type { LoadProgress, Model, ModelAdapter, ModelMetadata, ModelSource, WeightProvider } from "@tensorium/model-ir";
import { SafetensorsWeightProvider, SyntheticWeightProvider } from "@tensorium/tensor-core";
import { loadSafetensorsMetadata } from "@tensorium/hf-client";
import { buildModelConfig, buildGraph, runInference, type LlamaFamilyRawConfig } from "@tensorium/adapter-llama-family";

// Mistral's architecture and config.json field names are essentially
// identical to Llama's (RoPE, RMSNorm, SwiGLU, separate Q/K/V) — the real
// difference in practice is that Mistral models are usually trained with
// grouped-query attention (num_key_value_heads < num_attention_heads) and a
// sliding attention window. The window isn't modeled here (a plain causal
// mask over a short debugging prompt is identical to windowed-causal as
// long as the prompt is shorter than the window, which it always will be
// for the tiny/demo checkpoints this app targets) — everything else is
// exactly the shared Llama-family engine.
const PROVIDER_ID = "mistral-weights";

export const MistralAdapter: ModelAdapter = {
  id: "mistral",
  displayName: "Mistral",

  canLoad(_source, metadata) {
    if (!metadata) return true;
    return metadata.model_type === "mistral" || (metadata.architectures ?? []).some((a) => a.startsWith("Mistral"));
  },

  async loadMetadata(source: ModelSource, onProgress?: (progress: LoadProgress) => void): Promise<ModelMetadata> {
    const { rawConfig, weightIndex, weightsBuffer, structureOnly } = await loadSafetensorsMetadata<LlamaFamilyRawConfig>(source, onProgress);

    return {
      architecture: (rawConfig.architectures && rawConfig.architectures[0]) || "MistralForCausalLM",
      config: buildModelConfig(rawConfig, { defaultModelType: "mistral" }),
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
