import type { LoadProgress, Model, ModelAdapter, ModelMetadata, ModelSource, WeightProvider } from "@tensorium/model-ir";
import { SafetensorsWeightProvider, SyntheticWeightProvider } from "@tensorium/tensor-core";
import { loadSafetensorsMetadata } from "@tensorium/hf-client";
import { buildModelConfig, buildGraph, type GPT2RawConfig } from "./graph.js";
import { runInference } from "./inference.js";

const PROVIDER_ID = "gpt2-weights";

export const GPT2Adapter: ModelAdapter = {
  id: "gpt2",
  displayName: "GPT-2",

  canLoad(_source, metadata) {
    if (!metadata) return true; // optimistic default until config.json is fetched
    return metadata.model_type === "gpt2" || (metadata.architectures ?? []).some((a) => a.startsWith("GPT2"));
  },

  async loadMetadata(source: ModelSource, onProgress?: (progress: LoadProgress) => void): Promise<ModelMetadata> {
    const { rawConfig, weightIndex, weightsBuffer, structureOnly } = await loadSafetensorsMetadata<GPT2RawConfig>(source, onProgress);

    return {
      architecture: (rawConfig.architectures && rawConfig.architectures[0]) || "GPT2LMHeadModel",
      config: buildModelConfig(rawConfig, weightIndex),
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

export * from "./graph.js";
export * from "./inference.js";
