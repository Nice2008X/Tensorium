import type { LoadProgress, Model, ModelAdapter, ModelMetadata, ModelSource, WeightProvider } from "@tensorium/model-ir";
import { SafetensorsWeightProvider, SyntheticWeightProvider } from "@tensorium/tensor-core";
import { loadSafetensorsMetadata } from "@tensorium/hf-client";
import { buildModelConfig, buildGraph, type MuseGlimmerRawConfig } from "./graph.js";
import { runInference } from "./inference.js";

export { type MuseGlimmerRawConfig } from "./graph.js";

// meta-models/Muse-Glimmer-30B is genuinely multimodal — a text decoder
// plus separate vision tower/adapter weights under
// model.{language_model,vision_tower,vision_adapter,vision_projection}.*
// in the same file. Like Gemma 4 and Qwen3.5/Qwen3.8, this adapter covers
// the text decoder only: canLoad/loadMetadata/buildGraph/runInference all
// work exclusively with model.language_model.* (+ the top-level lm_head.
// weight) — the vision tower's entries stay present in weightIndex, since
// the structure-fetch reads the whole file's header regardless, but are
// never referenced by any node this adapter creates.
//
// This model's real architecture departs from adapter-llama-family's
// shared engine in several concrete ways all at once (confirmed against
// the checkpoint's own real `modeling_muse_glimmer.py` /
// `modular_muse_glimmer.py`, not guessed): a Gemma-2-style four-norm
// "sandwich" per block, but with the pre/post pairs at two genuinely
// different epsilons; a weightless per-head QK-norm followed by a Q-only
// constant scale factor (on top of, not instead of, the usual 1/sqrt(
// head_dim)); a per-layer NoPE toggle (every 4th layer counting backward
// from the last gets no positional encoding at all, instead of RoPE);
// every layer (not just some) gating its attention output through a
// sigmoid; a weightless RMSNorm applied to the raw embedding lookup
// (instead of Gemma's classic sqrt(hidden_size) scale); and a two-stage
// logit rescale (a constant output_multiplier, then a Gemma-style tanh
// softcap) before the LM head's output. None of that fits
// adapter-llama-family's existing option flags, so — like GPT-2,
// DeepSeek-V2, Gemma 4, and Qwen3.5/Qwen3.8 — this gets its own
// graph/inference modules entirely.
const PROVIDER_ID = "museglimmer-weights";

export const MuseGlimmerAdapter: ModelAdapter = {
  id: "museglimmer",
  displayName: "Muse Glimmer (text-only)",

  canLoad(_source, metadata) {
    if (!metadata) return true;
    return metadata.model_type === "muse_glimmer" || (metadata.architectures ?? []).some((a) => a === "MuseGlimmerForConditionalGeneration");
  },

  async loadMetadata(source: ModelSource, onProgress?: (progress: LoadProgress) => void): Promise<ModelMetadata> {
    const { rawConfig, weightIndex, weightsBuffer, structureOnly } = await loadSafetensorsMetadata<MuseGlimmerRawConfig>(source, onProgress);

    if (!rawConfig.text_config) {
      throw new Error(`This checkpoint's config.json has no "text_config" section — this adapter only knows how to read Muse Glimmer's text decoder, and can't find it here.`);
    }

    return {
      architecture: (rawConfig.architectures && rawConfig.architectures[0]) || rawConfig.model_type || "MuseGlimmerForConditionalGeneration",
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
