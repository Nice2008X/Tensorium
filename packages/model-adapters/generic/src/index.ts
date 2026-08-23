import type { LoadProgress, Model, ModelAdapter, ModelMetadata, ModelSource, WeightProvider } from "@tensorium/model-ir";
import { SafetensorsWeightProvider, SyntheticWeightProvider } from "@tensorium/tensor-core";
import { loadSafetensorsMetadata } from "@tensorium/hf-client";
import { buildModelConfig, buildGraph, runInference, type LlamaFamilyRawConfig } from "@tensorium/adapter-llama-family";
import { detectLlamaFamilyOptions, looksLikeLlamaFamilyBlock } from "./detect.js";

export { detectLlamaFamilyOptions, looksLikeLlamaFamilyBlock } from "./detect.js";

/**
 * The fallback adapter for any checkpoint no named adapter recognizes.
 * Every named adapter's canLoad() is checked first (see apps/web's
 * NAMED_ADAPTERS / GenericAdapter split in adapters.ts) — this only ever
 * runs for a `model_type` genuinely nobody has added a real adapter for
 * yet, and only after the user has explicitly confirmed they want a
 * best-effort load (see App.tsx's UnknownModelDialog). Its id is checked
 * elsewhere in the UI (ModelInfoBar's "best effort" badge) precisely so a
 * model loaded this way stays visibly different from one a named,
 * hand-verified adapter loaded.
 *
 * Reuses adapter-llama-family's graph/inference wholesale — the only thing
 * this adapter contributes over e.g. LlamaAdapter is *how* the family
 * options get decided (detected from the checkpoint's own weight names,
 * see detect.ts, instead of a human stating them). That also means its
 * ceiling is adapter-llama-family's ceiling: an architecture whose
 * differences aren't expressible as that engine's existing options (a
 * different attention mechanism entirely, an interleaved-pairs RoPE layout
 * like DeepSeek-V2's, ...) will load and *look* right — real shapes, real
 * layer count — while silently computing wrong numbers on Run Forward
 * Pass. There's no way to detect that failure mode from here; it's exactly
 * what the confirmation dialog and the ongoing badge exist to disclose.
 */
export const GenericAdapter: ModelAdapter = {
  id: "generic",
  displayName: "Generic (best-effort)",

  // Deliberately unconditional: this only gets consulted by the UI after
  // every named adapter has already declined, and only with the user's
  // explicit go-ahead — canLoad() itself has no way to see the checkpoint's
  // weight names yet (that requires the safetensors header, fetched in
  // loadMetadata below), so the real "does this even look loadable" check
  // happens there instead, where it can actually inspect them.
  canLoad() {
    return true;
  },

  async loadMetadata(source: ModelSource, onProgress?: (progress: LoadProgress) => void): Promise<ModelMetadata> {
    const { rawConfig, weightIndex, weightsBuffer, structureOnly } = await loadSafetensorsMetadata<LlamaFamilyRawConfig>(source, onProgress);

    if (!looksLikeLlamaFamilyBlock(weightIndex)) {
      throw new Error(
        `This checkpoint's weight layout doesn't structurally resemble a supported transformer block (no recognizable attention/feed-forward projections found at model.layers.0) — this app can't load it, even with best-effort detection.`
      );
    }

    const options = detectLlamaFamilyOptions(weightIndex, rawConfig);
    return {
      architecture: (rawConfig.architectures && rawConfig.architectures[0]) || rawConfig.model_type || "UnknownForCausalLM",
      config: buildModelConfig(rawConfig, options),
      weightIndex,
      source,
      weightsBuffer,
      structureOnly,
    };
  },

  buildGraph(metadata: ModelMetadata): Model {
    return buildGraph(metadata, "generic-weights");
  },

  getWeightProvider(metadata: ModelMetadata): WeightProvider {
    if (metadata.structureOnly) return new SyntheticWeightProvider("generic-weights", metadata.weightIndex);
    if (!metadata.weightsBuffer) throw new Error("No weights buffer available on this metadata");
    return new SafetensorsWeightProvider("generic-weights", metadata.weightsBuffer);
  },

  runInference,
};
