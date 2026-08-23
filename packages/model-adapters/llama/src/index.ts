import type { LoadProgress, Model, ModelAdapter, ModelMetadata, ModelSource, WeightProvider } from "@tensorium/model-ir";
import { SafetensorsWeightProvider, SyntheticWeightProvider } from "@tensorium/tensor-core";
import { loadSafetensorsMetadata } from "@tensorium/hf-client";
import { buildModelConfig, buildGraph, runInference, type LlamaFamilyRawConfig } from "@tensorium/adapter-llama-family";

const PROVIDER_ID = "llama-weights";

export const LlamaAdapter: ModelAdapter = {
  id: "llama",
  displayName: "Llama",

  canLoad(_source, metadata) {
    if (!metadata) return true; // optimistic default until config.json is fetched
    return metadata.model_type === "llama" || (metadata.architectures ?? []).some((a) => a.startsWith("Llama"));
  },

  async loadMetadata(source: ModelSource, onProgress?: (progress: LoadProgress) => void): Promise<ModelMetadata> {
    const { rawConfig, weightIndex, weightsBuffer, structureOnly } = await loadSafetensorsMetadata<LlamaFamilyRawConfig>(source, onProgress);

    return {
      architecture: (rawConfig.architectures && rawConfig.architectures[0]) || "LlamaForCausalLM",
      config: buildModelConfig(rawConfig, { defaultModelType: "llama" }),
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
