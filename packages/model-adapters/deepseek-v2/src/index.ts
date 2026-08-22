import type { LoadProgress, Model, ModelAdapter, ModelMetadata, ModelSource, WeightProvider } from "@tensorium/model-ir";
import { SafetensorsWeightProvider } from "@tensorium/tensor-core";
import { loadSafetensorsMetadata } from "@tensorium/hf-client";
import { buildModelConfig, buildGraph, type DeepseekV2RawConfig } from "./graph.js";
import { runInference } from "./inference.js";

export { type DeepseekV2RawConfig } from "./graph.js";

// DeepSeek-V2 (and the smaller "Lite" checkpoint this adapter targets)
// departs from the rest of this app's supported architectures in two real
// ways rather than being a parameter variant of Llama's shape — see
// graph.ts's doc comment for the full picture:
//
// 1. Multi-head Latent Attention (MLA) instead of GQA: K and V are both
//    reconstructed from one shared low-rank latent, and only a slice of
//    each head actually gets RoPE — the rest carries no positional signal.
// 2. DeepSeekMoE instead of Qwen-style MoE: many small experts, one or more
//    always-on unconditional "shared" experts (no gate), and optionally
//    group-limited routing (experts bucketed into groups; only the
//    best-scoring groups are eligible before top-k runs).
//
// Both are different enough from `@tensorium/adapter-llama-family`'s engine
// that this gets its own graph/inference modules (like GPT-2) rather than
// another set of flags on that shared engine.
const PROVIDER_ID = "deepseek-v2-weights";

export const DeepseekV2Adapter: ModelAdapter = {
  id: "deepseek_v2",
  displayName: "DeepSeek-V2",

  canLoad(_source, metadata) {
    if (!metadata) return true;
    return metadata.model_type === "deepseek_v2" || (metadata.architectures ?? []).some((a) => a === "DeepseekV2ForCausalLM");
  },

  async loadMetadata(source: ModelSource, onProgress?: (progress: LoadProgress) => void): Promise<ModelMetadata> {
    const { rawConfig, weightIndex, weightsBuffer } = await loadSafetensorsMetadata<DeepseekV2RawConfig>(source, onProgress);

    return {
      architecture: (rawConfig.architectures && rawConfig.architectures[0]) || "DeepseekV2ForCausalLM",
      config: buildModelConfig(rawConfig),
      weightIndex,
      source,
      weightsBuffer,
    };
  },

  buildGraph(metadata: ModelMetadata): Model {
    return buildGraph(metadata, PROVIDER_ID);
  },

  getWeightProvider(metadata: ModelMetadata): WeightProvider {
    if (!metadata.weightsBuffer) throw new Error("No weights buffer available on this metadata");
    return new SafetensorsWeightProvider(PROVIDER_ID, metadata.weightsBuffer);
  },

  runInference,
};
