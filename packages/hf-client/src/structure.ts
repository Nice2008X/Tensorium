import type { ModelMetadata, ModelSource } from "@tensorium/model-ir";
import { dtypeSize, numElements } from "@tensorium/model-ir";
import { parseSafetensorsHeader, type SafetensorsEntry } from "@tensorium/tensor-core";
import { hfResolveUrl } from "./index.js";

type HfSource = Extract<ModelSource, { kind: "huggingface" }>;

/**
 * How much of a safetensors file's header we guess-fetch in one shot before
 * checking whether we actually got all of it. Real checkpoints' JSON headers
 * run from a few hundred bytes (a handful of tensors) up to a few hundred KB
 * even for models with thousands of tensors — 256 KB covers the overwhelming
 * majority in a single request; readHeaderOnly() falls back to a second,
 * exactly-sized request for anything bigger instead of guessing again.
 */
const SPECULATIVE_HEADER_BYTES = 256 * 1024;

/** Thrown by fetchByteRange with the real HTTP status attached, so a caller can tell "this file doesn't exist" (404 — a legitimate, expected outcome when probing for an optional shard layout) apart from "this host doesn't support Range requests" (any other non-206 status — a real problem, must not be silently swallowed). */
class RangeRequestError extends Error {
  constructor(
    message: string,
    public status: number
  ) {
    super(message);
  }
}

/**
 * Fetches an exact byte range and throws rather than silently downloading
 * the whole file if the server doesn't honor the Range request — the entire
 * point of this module is reading a checkpoint's structure (shapes/dtypes)
 * without ever pulling down its multi-GB tensor payload, so a server that
 * ignores Range (returns 200 instead of 206) has to be a hard error here,
 * not a quiet fallback to `res.arrayBuffer()` on a possibly enormous body.
 */
async function fetchByteRange(url: string, start: number, end: number): Promise<ArrayBuffer> {
  const res = await fetch(url, { headers: { Range: `bytes=${start}-${end}` } });
  if (res.status !== 206) {
    throw new RangeRequestError(
      `Expected a partial (206) response reading ${url} bytes ${start}-${end}, got ${res.status}. ` +
        `This host may not support HTTP Range requests, which structure-only loading requires.`,
      res.status
    );
  }
  return res.arrayBuffer();
}

/**
 * Reads one safetensors file's header — tensor name -> {shape, dtype} — via
 * one or two small Range requests, never touching the (potentially
 * multi-gigabyte) tensor data that follows it in the same file. This is the
 * mechanism behind "structure only" loading: a model's whole architecture
 * (layer count, hidden sizes, every parameter's exact shape) is fully knowable
 * from just this header, at a cost of a few KB to a few hundred KB of
 * network traffic regardless of how large the actual checkpoint is.
 */
async function readHeaderOnly(url: string): Promise<Record<string, SafetensorsEntry>> {
  const speculative = await fetchByteRange(url, 0, SPECULATIVE_HEADER_BYTES - 1);
  const headerLength = Number(new DataView(speculative).getBigUint64(0, true));
  const totalNeeded = 8 + headerLength;

  const headerBuffer = totalNeeded <= speculative.byteLength ? speculative : await fetchByteRange(url, 0, totalNeeded - 1);

  // parseSafetensorsHeader only ever reads bytes [0, 8+headerLength) — it's
  // safe to hand it a buffer that stops exactly there, with no tensor data
  // behind it, as long as nothing downstream tries to readTensor() from it.
  return parseSafetensorsHeader(headerBuffer).header;
}

/** name -> filename, from a sharded checkpoint's model.safetensors.index.json. */
interface SafetensorsIndexJson {
  weight_map: Record<string, string>;
}

export interface ModelStructure {
  /** Every parameter's shape/dtype, merged across shards if the checkpoint has any — identical shape to ModelMetadata.weightIndex, so it's a drop-in structure-only substitute for it. */
  weightIndex: ModelMetadata["weightIndex"];
  /** Total logical weight bytes across every tensor (sum of numElements(shape) * dtypeSize(dtype)) — the real basis for an "is this checkpoint too big to download" decision, independent of on-disk compression or dtype quirks. */
  totalBytes: number;
  /** How many distinct safetensors files this checkpoint's weights are split across (1 for an unsharded checkpoint). */
  shardCount: number;
}

function toWeightIndex(header: Record<string, SafetensorsEntry>): { weightIndex: ModelMetadata["weightIndex"]; bytes: number } {
  const weightIndex: ModelMetadata["weightIndex"] = {};
  let bytes = 0;
  for (const [name, entry] of Object.entries(header)) {
    weightIndex[name] = { shape: entry.shape, dtype: entry.dtype };
    bytes += numElements(entry.shape) * dtypeSize(entry.dtype);
  }
  return { weightIndex, bytes };
}

/**
 * Learns a Hugging-Face-hosted checkpoint's complete parameter structure —
 * every tensor's name, shape, dtype, and the true total weight size — without
 * downloading any tensor data. Handles both single-file checkpoints
 * (`model.safetensors`) and sharded ones (`model.safetensors.index.json` +
 * `model-NNNNN-of-MMMMM.safetensors`), the same two layouts
 * `loadSafetensorsMetadata` already assumes elsewhere in this app — the only
 * difference here is that every fetch is a small Range request instead of
 * the whole file.
 *
 * Tries the single-file layout first, not the index — every existing preset
 * in this app (and the overwhelming majority of real checkpoints, which
 * aren't large enough to need sharding) is unsharded, so this ordering means
 * the common case never has to probe for an index.json that doesn't exist
 * (a 404 that JS can handle perfectly well, but that the browser still logs
 * to the console as a failed request — not worth incurring on every single
 * load just to check a case that's rare in practice).
 */
export async function fetchModelStructure(source: HfSource): Promise<ModelStructure> {
  try {
    const header = await readHeaderOnly(hfResolveUrl(source, "model.safetensors"));
    const { weightIndex, bytes } = toWeightIndex(header);
    return { weightIndex, totalBytes: bytes, shardCount: 1 };
  } catch (err) {
    if (!(err instanceof RangeRequestError) || err.status !== 404) throw err;
  }

  // No single model.safetensors — fall back to the sharded layout.
  const indexUrl = hfResolveUrl(source, "model.safetensors.index.json");
  const indexRes = await fetch(indexUrl);
  if (!indexRes.ok) {
    throw new Error(`Could not find model.safetensors or model.safetensors.index.json for ${source.repo} — this checkpoint's weight layout isn't one this app recognizes.`);
  }
  const index = (await indexRes.json()) as SafetensorsIndexJson;
  const shardFiles = [...new Set(Object.values(index.weight_map))];

  const merged: Record<string, SafetensorsEntry> = {};
  for (const file of shardFiles) {
    const header = await readHeaderOnly(hfResolveUrl(source, file));
    Object.assign(merged, header);
  }

  const { weightIndex, bytes } = toWeightIndex(merged);
  return { weightIndex, totalBytes: bytes, shardCount: shardFiles.length };
}
