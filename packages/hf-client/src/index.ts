import type { LoadProgress, ModelMetadata, ModelSource } from "@tensorium/model-ir";
import { parseSafetensorsHeader } from "@tensorium/tensor-core";
import { fetchCachedArrayBuffer, type ByteProgressCallback } from "./modelCache.js";
import { fetchModelStructure } from "./structure.js";

export { MAX_CACHEABLE_BYTES } from "./modelCache.js";
export type { ByteProgressCallback } from "./modelCache.js";
export { fetchModelStructure } from "./structure.js";
export type { ModelStructure } from "./structure.js";

/**
 * Above this many logical weight bytes, loadSafetensorsMetadata skips
 * downloading the checkpoint's actual tensor data entirely and returns
 * `structureOnly: true` instead — the architecture graph is built from real
 * shapes/dtypes either way (see fetchModelStructure), but a WeightProvider
 * over a checkpoint this large has to fabricate its tensor values rather
 * than read real ones (see tensor-core's SyntheticWeightProvider). 3 GB
 * comfortably covers this app's real hand-typed presets while keeping any
 * checkpoint in the multi-GB range from ever hitting a real multi-gigabyte
 * `fetch()` in a browser tab.
 */
export const STRUCTURE_ONLY_THRESHOLD_BYTES = 3 * 1024 * 1024 * 1024; // 3 GB

export function hfResolveUrl(source: Extract<ModelSource, { kind: "huggingface" }>, file: string): string {
  const revision = source.revision ?? "main";
  return `https://huggingface.co/${source.repo}/resolve/${revision}/${file}`;
}

export async function fetchJson<T>(url: string, onProgress?: ByteProgressCallback): Promise<T> {
  const bytes = await fetchCachedArrayBuffer(url, onProgress);
  return JSON.parse(new TextDecoder().decode(bytes)) as T;
}

export async function fetchArrayBuffer(url: string, onProgress?: ByteProgressCallback): Promise<ArrayBuffer> {
  return fetchCachedArrayBuffer(url, onProgress);
}

/** Reads one file out of a `{ kind: "local" }` source's in-memory file map — the local-loading equivalent of a fetch. */
export function readLocalBytes(source: ModelSource, filename: string): ArrayBuffer {
  if (source.kind !== "local") throw new Error(`readLocalBytes called on a non-local source (${source.kind})`);
  const bytes = source.files[filename];
  if (!bytes) throw new Error(`Missing required local file: ${filename}`);
  return bytes;
}

/** Same as `readLocalBytes`, JSON-parsed. */
export function readLocalJson<T>(source: ModelSource, filename: string): T {
  return JSON.parse(new TextDecoder().decode(readLocalBytes(source, filename))) as T;
}

export interface HfConfigPreview {
  model_type?: string;
  architectures?: string[];
}

/** Reads just enough of config.json to let ModelAdapter.canLoad decide, before any adapter commits to fetching weights. */
export async function peekModelType(source: ModelSource): Promise<HfConfigPreview> {
  if (source.kind === "local") return readLocalJson<HfConfigPreview>(source, "config.json");
  return fetchJson<HfConfigPreview>(hfResolveUrl(source, "config.json"));
}

export interface RawSafetensorsMetadata<TConfig> {
  rawConfig: TConfig;
  weightIndex: ModelMetadata["weightIndex"];
  /** Absent exactly when structureOnly is true — no tensor bytes were ever downloaded, real or otherwise. */
  weightsBuffer?: ArrayBuffer;
  /** See ModelMetadata.structureOnly — an adapter's getWeightProvider() must check this and hand back a SyntheticWeightProvider instead of a SafetensorsWeightProvider when it's true. */
  structureOnly: boolean;
}

function headerToWeightIndex(header: Record<string, { shape: number[]; dtype: string }>): ModelMetadata["weightIndex"] {
  const weightIndex: ModelMetadata["weightIndex"] = {};
  for (const [name, entry] of Object.entries(header)) weightIndex[name] = { shape: entry.shape, dtype: entry.dtype };
  return weightIndex;
}

/**
 * The fetch sequence every safetensors-backed adapter needs: raw config.json
 * (typed however that adapter likes) plus the safetensors file, with its
 * header already parsed into a name -> {shape, dtype} index. Each adapter
 * turns `rawConfig` into its own normalized ModelConfig from here. Works
 * identically for a Hugging Face source (fetched, cache-backed) and a
 * `{ kind: "local" }` source (files the user already picked, just read
 * straight out of memory) — adapters don't need to know which one it was.
 *
 * A Hugging Face source gets one extra step first: fetchModelStructure()
 * learns the checkpoint's true total size (and whether it's sharded) via a
 * handful of small Range requests, *before* committing to downloading any
 * tensor data. Above STRUCTURE_ONLY_THRESHOLD_BYTES — or for any sharded
 * checkpoint, since this function's eager path only ever fetches a single
 * model.safetensors — this returns with `structureOnly: true` and no
 * `weightsBuffer` at all; the returned `weightIndex` still has every real
 * tensor's true shape/dtype, just no downloaded bytes behind them. A local
 * source is never structure-only: its bytes are already sitting in memory
 * (the user picked the file directly), so there's no download to avoid.
 */
export async function loadSafetensorsMetadata<TConfig>(
  source: ModelSource,
  onProgress?: (progress: LoadProgress) => void
): Promise<RawSafetensorsMetadata<TConfig>> {
  onProgress?.({ phase: "config" });
  const rawConfig =
    source.kind === "local" ? readLocalJson<TConfig>(source, "config.json") : await fetchJson<TConfig>(hfResolveUrl(source, "config.json"));

  if (source.kind === "local") {
    const weightsBuffer = readLocalBytes(source, "model.safetensors");
    onProgress?.({ phase: "weights", loadedBytes: weightsBuffer.byteLength, totalBytes: weightsBuffer.byteLength });
    onProgress?.({ phase: "parsing" });
    const weightIndex = headerToWeightIndex(parseSafetensorsHeader(weightsBuffer).header);
    return { rawConfig, weightIndex, weightsBuffer, structureOnly: false };
  }

  onProgress?.({ phase: "structure" });
  const structure = await fetchModelStructure(source);
  if (structure.shardCount > 1 || structure.totalBytes > STRUCTURE_ONLY_THRESHOLD_BYTES) {
    return { rawConfig, weightIndex: structure.weightIndex, structureOnly: true };
  }

  const weightsBuffer = await fetchArrayBuffer(hfResolveUrl(source, "model.safetensors"), (loadedBytes, totalBytes) =>
    onProgress?.({ phase: "weights", loadedBytes, totalBytes })
  );
  onProgress?.({ phase: "parsing" });
  const weightIndex = headerToWeightIndex(parseSafetensorsHeader(weightsBuffer).header);
  return { rawConfig, weightIndex, weightsBuffer, structureOnly: false };
}
