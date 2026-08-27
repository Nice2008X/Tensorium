import type { LoadProgress, Model, ModelAdapter, ModelMetadata, ModelSource, WeightProvider } from "@tensorium/model-ir";
import { SafetensorsWeightProvider, SyntheticWeightProvider } from "@tensorium/tensor-core";
import { loadSafetensorsMetadata } from "@tensorium/hf-client";
import { buildModelConfig, buildGraph, type Qwen35RawConfig } from "./graph.js";
import { runInference } from "./inference.js";

export { type Qwen35RawConfig } from "./graph.js";

// Qwen/Qwen3.5-27B and Qwen/Qwen3.8-27B (model_type: "qwen3_5") are
// genuinely multimodal checkpoints — a hybrid text decoder plus a separate
// vision tower (model.visual.*) and an optional multi-token-prediction head
// (mtp.*), the same shape of file adapter-gemma4 already handles for its
// own vision/audio towers. This adapter covers the text decoder only, for
// the same reason: canLoad/loadMetadata/buildGraph/runInference all work
// exclusively with model.language_model.* weights (plus the top-level
// lm_head.weight) — model.visual.*/mtp.* stay present in weightIndex but
// are never referenced by any node this adapter creates.
//
// The decoder itself is a genuinely new attention mechanism for this app:
// every layer is either ordinary causal GQA ("full_attention", periodic)
// or a linear/recurrent Gated DeltaNet layer ("linear_attention", most
// layers) — a short causal convolution feeding a per-token recurrent state
// update (the "delta rule"), nothing like the softmax attention every other
// adapter here shares. Both layer kinds also gate their output through a
// learned sigmoid/SiLU gate before their respective output projection.
// None of that fits adapter-llama-family's option flags, so — like GPT-2,
// DeepSeek-V2, and Gemma-4 — this gets its own graph/inference modules
// entirely. See graph.ts and inference.ts for the exact math, confirmed
// against the real transformers models/qwen3_5/modeling_qwen3_5.py source
// and a real checkpoint's own safetensors header, not guessed.
const PROVIDER_ID = "qwen3-5-weights";

export const Qwen35Adapter: ModelAdapter = {
  id: "qwen3-5",
  displayName: "Qwen3.5 / Qwen3.8 (text-only)",

  canLoad(_source, metadata) {
    if (!metadata) return true;
    return metadata.model_type === "qwen3_5" || (metadata.architectures ?? []).some((a) => a === "Qwen3_5ForConditionalGeneration");
  },

  async loadMetadata(source: ModelSource, onProgress?: (progress: LoadProgress) => void): Promise<ModelMetadata> {
    const { rawConfig, weightIndex, weightsBuffer, structureOnly } = await loadSafetensorsMetadata<Qwen35RawConfig>(source, onProgress);

    if (!rawConfig.text_config) {
      throw new Error(`This checkpoint's config.json has no "text_config" section — this adapter only knows how to read Qwen3.5/3.8's text decoder, and can't find it here.`);
    }

    return {
      architecture: (rawConfig.architectures && rawConfig.architectures[0]) || rawConfig.model_type || "Qwen3_5ForConditionalGeneration",
      config: buildModelConfig(rawConfig),
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
