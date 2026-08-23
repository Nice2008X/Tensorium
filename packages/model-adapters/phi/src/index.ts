import type { LoadProgress, Model, ModelAdapter, ModelMetadata, ModelSource, WeightProvider } from "@tensorium/model-ir";
import { SafetensorsWeightProvider, SyntheticWeightProvider } from "@tensorium/tensor-core";
import { loadSafetensorsMetadata } from "@tensorium/hf-client";
import { buildModelConfig, buildGraph, runInference, type LlamaFamilyRawConfig } from "@tensorium/adapter-llama-family";

// Phi-4 reuses the Phi-3 architecture class outright (architectures:
// ["Phi3ForCausalLM"], model_type: "phi3" — same as how Qwen2.5 reuses
// Qwen2's class), so this one adapter covers both. Otherwise a standard
// Llama-family model — RoPE, RMSNorm, SwiGLU FFN, GQA, no biases — except
// it fuses two pairs of projections that everything else in this family
// keeps separate: Q/K/V into one qkv_proj weight (row-sliced, GQA-sized —
// confirmed against the real safetensors header before writing this), and
// the gated-FFN's gate/up projections into one gate_up_proj weight
// (split into equal halves).
const PROVIDER_ID = "phi-weights";

export const PhiAdapter: ModelAdapter = {
  id: "phi3",
  displayName: "Phi-3/Phi-4",

  canLoad(_source, metadata) {
    if (!metadata) return true;
    return metadata.model_type === "phi3" || (metadata.architectures ?? []).some((a) => a === "Phi3ForCausalLM");
  },

  async loadMetadata(source: ModelSource, onProgress?: (progress: LoadProgress) => void): Promise<ModelMetadata> {
    const { rawConfig, weightIndex, weightsBuffer, structureOnly } = await loadSafetensorsMetadata<LlamaFamilyRawConfig>(source, onProgress);

    return {
      architecture: (rawConfig.architectures && rawConfig.architectures[0]) || "Phi3ForCausalLM",
      config: buildModelConfig(rawConfig, { defaultModelType: "phi3", fusedQkv: true, fusedGateUp: true, tiedByDefault: false }),
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
