import { DownloadCancelledError, type LoadProgress, type ModelMetadata, type ModelSource, type WeightsBuffer } from "@tensorium/model-ir";
import { parseSafetensorsHeader } from "@tensorium/tensor-core";
import { fetchCachedArrayBuffer, fetchCachedWeights, type ByteProgressCallback } from "./modelCache.js";
import { fetchModelStructure, type ModelStructure } from "./structure.js";

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
 * than read real ones (see tensor-core's SyntheticWeightProvider).
 *
 * A single ArrayBuffer tops out around 2 GiB in a browser (Chromium 153:
 * 2040 MiB allocates, 2046 MiB throws regardless of free RAM), so anything
 * bigger is held as a SegmentedBuffer of 512 MiB pieces instead — this
 * ceiling is now just a sanity bound on total tab memory, not an
 * allocation limit.
 */
export const STRUCTURE_ONLY_THRESHOLD_BYTES = 3 * 1024 * 1024 * 1024; // 3 GB

/** Checkpoints above this size get a "download the weights or structure only?" prompt before anything heavy is fetched. */
export const LARGE_MODEL_WARNING_BYTES = 1024 ** 3; // 1 GB

/** Whether this app can eagerly download a checkpoint's weights at all: only a single file, and only below STRUCTURE_ONLY_THRESHOLD_BYTES. Anything else is structure-only regardless of what the user would prefer. */
export function canDownloadWeights(structure: ModelStructure): boolean {
  return structure.shardCount <= 1 && structure.totalBytes <= STRUCTURE_ONLY_THRESHOLD_BYTES;
}

/** Hard ceilings for the JSON files fetched from a (possibly hostile) repo, enforced while streaming so an oversized response is cut off rather than buffered. Real config.json files are a few KB; the biggest real tokenizer.json files are ~35 MB. */
export const MAX_CONFIG_BYTES = 5 * 1024 * 1024;
export const MAX_TOKENIZER_BYTES = 128 * 1024 * 1024;
export const MAX_INDEX_BYTES = 16 * 1024 * 1024;

// One path/name component: no dots-only names (so no "." / ".."), no slashes,
// no query/fragment/percent characters — nothing that can change what URL
// the string ends up naming.
const SAFE_SEGMENT = /^[A-Za-z0-9_][A-Za-z0-9._+@-]{0,127}$/;
const SAFE_REVISION = /^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$/;
const MAX_REPO_SPEC_LENGTH = 512;

/**
 * A `repo` string is `org/name`, optionally followed by a subfolder path
 * (`org/name/sub/dir`) for repos that keep each checkpoint in its own folder
 * instead of at the root (e.g. `AlexWortega/openjev/qwen3.5-4b-nli-v2`).
 * A real HF repo id never has more than two segments, so this is unambiguous.
 * Every segment is validated (see SAFE_SEGMENT), so a pasted or persisted
 * string can't smuggle in `..`, `?`, `#`, `%2e%2e` or a backslash to make
 * a request land somewhere other than the repo it appears to name.
 */
export function parseHfRepoSpec(spec: string): { repoId: string; subfolder: string[] } {
  const segments = spec.split("/");
  if (spec.length > MAX_REPO_SPEC_LENGTH || segments.length > 12 || !segments.every((s) => SAFE_SEGMENT.test(s))) {
    throw new Error(`"${spec.slice(0, 80)}" isn't a valid Hugging Face repo id — expected "org/name", optionally followed by a subfolder, using only letters, digits, and . _ - + @.`);
  }
  const [org, name, ...subfolder] = segments;
  return { repoId: name === undefined ? org : `${org}/${name}`, subfolder };
}

export function isValidHfRepoSpec(spec: string): boolean {
  try {
    parseHfRepoSpec(spec);
    return true;
  } catch {
    return false;
  }
}

/** The one place a Hugging Face file URL is built: `file` (which may come from a remote index.json, e.g. a shard name) gets the same segment check as the repo, so it can't climb out of the repo either. */
export function hfResolveUrl(source: Extract<ModelSource, { kind: "huggingface" }>, file: string): string {
  const revision = source.revision ?? "main";
  if (!SAFE_REVISION.test(revision) || revision.includes("..")) throw new Error(`Invalid Hugging Face revision "${revision.slice(0, 40)}".`);
  const { repoId, subfolder } = parseHfRepoSpec(source.repo);
  const fileSegments = file.split("/");
  if (!fileSegments.every((s) => SAFE_SEGMENT.test(s))) throw new Error(`Refusing to fetch unsafe file path "${file.slice(0, 80)}".`);
  return `https://huggingface.co/${repoId}/resolve/${revision}/${[...subfolder, ...fileSegments].join("/")}`;
}

/** `maxBytes` is required: every JSON fetched here comes from a repo the user merely named, so there's no safe default size. */
export async function fetchJson<T>(url: string, maxBytes: number, onProgress?: ByteProgressCallback): Promise<T> {
  const bytes = await fetchCachedArrayBuffer(url, onProgress, undefined, maxBytes);
  return JSON.parse(new TextDecoder().decode(bytes)) as T;
}

export async function fetchArrayBuffer(url: string, maxBytes: number, onProgress?: ByteProgressCallback): Promise<ArrayBuffer> {
  return fetchCachedArrayBuffer(url, onProgress, undefined, maxBytes);
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
  return fetchJson<HfConfigPreview>(hfResolveUrl(source, "config.json"), MAX_CONFIG_BYTES);
}

export interface RawSafetensorsMetadata<TConfig> {
  rawConfig: TConfig;
  weightIndex: ModelMetadata["weightIndex"];
  /** Absent exactly when structureOnly is true — no tensor bytes were ever downloaded, real or otherwise. */
  weightsBuffer?: WeightsBuffer;
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
    source.kind === "local" ? readLocalJson<TConfig>(source, "config.json") : await fetchJson<TConfig>(hfResolveUrl(source, "config.json"), MAX_CONFIG_BYTES);

  if (source.kind === "local") {
    const weightsBuffer = readLocalBytes(source, "model.safetensors");
    onProgress?.({ phase: "weights", loadedBytes: weightsBuffer.byteLength, totalBytes: weightsBuffer.byteLength });
    onProgress?.({ phase: "parsing" });
    const weightIndex = headerToWeightIndex(parseSafetensorsHeader(weightsBuffer).header);
    return { rawConfig, weightIndex, weightsBuffer, structureOnly: false };
  }

  onProgress?.({ phase: "structure" });
  const structure = await fetchModelStructure(source);
  if (source.weightsMode === "structure-only" || !canDownloadWeights(structure)) {
    return { rawConfig, weightIndex: structure.weightIndex, structureOnly: true };
  }

  if (source.download?.cancelled) throw new DownloadCancelledError();
  const weightsBuffer = await fetchCachedWeights(
    hfResolveUrl(source, structure.weightsFile),
    (loadedBytes, totalBytes) => onProgress?.({ phase: "weights", loadedBytes, totalBytes }),
    source.download
  );
  onProgress?.({ phase: "parsing" });
  const weightIndex = headerToWeightIndex(parseSafetensorsHeader(weightsBuffer).header);
  return { rawConfig, weightIndex, weightsBuffer, structureOnly: false };
}
