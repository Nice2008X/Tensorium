import type { LoadProgress, Model, ModelAdapter, ModelMetadata, ModelSource, WeightProvider } from "@tensorium/model-ir";
import { SafetensorsWeightProvider, SyntheticWeightProvider } from "@tensorium/tensor-core";
import { loadSafetensorsMetadata } from "@tensorium/hf-client";
import { buildModelConfig, buildGraph, runInference, type LlamaFamilyRawConfig } from "@tensorium/adapter-llama-family";

const PROVIDER_ID = "gemma-weights";

export const GemmaAdapter: ModelAdapter = {
  id: "gemma",
  displayName: "Gemma",

  canLoad(_source, metadata) {
    if (!metadata) return true;
    // Deliberately exact match, not a prefix check: Gemma 2/3 add real
    // architectural differences (sandwich norms, alternating attention,
    // logit softcapping) this adapter does not implement. Silently claiming
    // those and producing plausible-but-wrong output would be worse than
    // this adapter declining and falling through to "no adapter found".
    return metadata.model_type === "gemma" || (metadata.architectures ?? []).some((a) => a === "GemmaForCausalLM" || a === "GemmaModel");
  },

  async loadMetadata(source: ModelSource, onProgress?: (progress: LoadProgress) => void): Promise<ModelMetadata> {
    const { rawConfig, weightIndex, weightsBuffer, structureOnly } = await loadSafetensorsMetadata<LlamaFamilyRawConfig>(source, onProgress);

    return {
      architecture: (rawConfig.architectures && rawConfig.architectures[0]) || "GemmaForCausalLM",
      config: buildModelConfig(rawConfig, {
        defaultModelType: "gemma",
        rmsNormVariant: "gemma",
        embeddingScale: "sqrt_hidden",
        tiedByDefault: true,
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
