import type { LoadProgress, Model, ModelAdapter, ModelMetadata, ModelSource, WeightProvider } from "@tensorium/model-ir";
import { SafetensorsWeightProvider, SyntheticWeightProvider } from "@tensorium/tensor-core";
import { loadSafetensorsMetadata } from "@tensorium/hf-client";
import { buildModelConfig, buildGraph, type Gemma4RawConfig } from "./graph.js";
import { runInference } from "./inference.js";

export { type Gemma4RawConfig } from "./graph.js";

// google/gemma-4-* checkpoints are genuinely multimodal — a text decoder
// plus separate vision and audio towers under model.{language_model,
// vision_tower,audio_tower}.* — the same file every other adapter in this
// app would otherwise ignore entirely (nothing here has ever handled
// anything but a text decoder). This adapter covers the text decoder only:
// canLoad/loadMetadata/buildGraph/runInference all work exclusively with
// model.language_model.* weights; the (much larger, in tensor count)
// vision_tower/audio_tower entries stay present in weightIndex — the
// structure-fetch reads the whole file's header regardless — but are never
// referenced by any node this adapter creates, so they're simply inert.
//
// This model's real architecture departs from every other adapter's shared
// llama-family engine in several concrete ways (confirmed against the real
// `transformers` v5.15.1 source, not guessed — see graph.ts's doc
// comments): plain-weight RMSNorm (not Gemma's classic (1+weight)), a
// fixed attention scale of 1.0 instead of 1/sqrt(head_dim), two different
// head_dim/RoPE configurations alternating by layer (sliding-window vs
// full-context layers), the last several layers reusing an earlier layer's
// frozen K/V instead of computing their own, Per-Layer Embeddings injected
// additively into every layer from a second embedding table, and a real
// learned per-layer output scalar. None of that fits adapter-llama-family's
// existing option flags, so — like GPT-2 and DeepSeek-V2 — this gets its
// own graph/inference modules entirely.
const PROVIDER_ID = "gemma4-weights";

export const Gemma4Adapter: ModelAdapter = {
  id: "gemma4",
  displayName: "Gemma 4 (text-only)",

  canLoad(_source, metadata) {
    if (!metadata) return true;
    return metadata.model_type === "gemma4" || (metadata.architectures ?? []).some((a) => a === "Gemma4ForConditionalGeneration");
  },

  async loadMetadata(source: ModelSource, onProgress?: (progress: LoadProgress) => void): Promise<ModelMetadata> {
    const { rawConfig, weightIndex, weightsBuffer, structureOnly } = await loadSafetensorsMetadata<Gemma4RawConfig>(source, onProgress);

    if (!rawConfig.text_config) {
      throw new Error(`This checkpoint's config.json has no "text_config" section — this adapter only knows how to read Gemma-4's text decoder, and can't find it here.`);
    }

    return {
      architecture: (rawConfig.architectures && rawConfig.architectures[0]) || rawConfig.model_type || "Gemma4ForConditionalGeneration",
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
