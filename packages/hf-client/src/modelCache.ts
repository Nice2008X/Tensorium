import { DownloadCancelledError, SEGMENT_BYTES, SegmentedBuffer, type DownloadControl, type WeightsBuffer } from "@tensorium/model-ir";

const DB_NAME = "Tensorium-model-cache";
const DB_VERSION = 1;
const STORE_NAME = "files";

/**
 * Files larger than this are never written to the cache — fetched fresh
 * from Hugging Face every time instead. Keeps one oversized checkpoint from
 * burning through a large share of the browser's per-origin storage quota,
 * and avoids a slow structured-clone IndexedDB write on the main thread for
 * something that isn't one of this app's small "tiny-random" presets.
 */
export const MAX_CACHEABLE_BYTES = 50 * 1024 * 1024; // 50 MB

interface CachedFile {
  url: string;
  bytes: ArrayBuffer;
  size: number;
  cachedAt: number;
}

let dbPromise: Promise<IDBDatabase | null> | null = null;

/** Opens (or creates) the cache database. Resolves to null rather than rejecting on any failure — IndexedDB can be unavailable or disabled (private browsing in some browsers, storage restrictions), and that should just mean "no cache", never a load failure. */
function openDb(): Promise<IDBDatabase | null> {
  if (!dbPromise) {
    dbPromise = new Promise((resolve) => {
      if (typeof indexedDB === "undefined") {
        resolve(null);
        return;
      }
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => req.result.createObjectStore(STORE_NAME, { keyPath: "url" });
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    });
  }
  return dbPromise;
}

async function getCached(url: string): Promise<ArrayBuffer | undefined> {
  const db = await openDb();
  if (!db) return undefined;
  return new Promise((resolve) => {
    const req = db.transaction(STORE_NAME, "readonly").objectStore(STORE_NAME).get(url);
    req.onsuccess = () => resolve((req.result as CachedFile | undefined)?.bytes);
    req.onerror = () => resolve(undefined);
  });
}

async function putCached(url: string, bytes: ArrayBuffer): Promise<void> {
  const db = await openDb();
  if (!db) return;
  return new Promise((resolve) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const entry: CachedFile = { url, bytes, size: bytes.byteLength, cachedAt: Date.now() };
    tx.objectStore(STORE_NAME).put(entry);
    // A write failure (quota exceeded despite the size check above, private
    // mode restrictions) should never break loading the model that's
    // already sitting in memory — just means it won't be cached this time.
    tx.oncomplete = () => resolve();
    tx.onerror = () => resolve();
  });
}

/** `loadedBytes` counts up as the response streams in; `totalBytes` is undefined when the server doesn't send Content-Length. */
export type ByteProgressCallback = (loadedBytes: number, totalBytes: number | undefined) => void;

/**
 * Streams `url` into memory, reporting progress, and — when given a
 * DownloadControl — can be paused (request aborted, bytes so far kept) and
 * later resumed by re-requesting only the remainder with a Range header.
 * When the size is known up front the result buffer is allocated once and
 * written into directly, so peak memory is 1x the file rather than the 2x a
 * collect-chunks-then-concatenate approach would need. With `segmented`, a
 * file bigger than SEGMENT_BYTES lands in a SegmentedBuffer instead — a
 * single ArrayBuffer tops out around 2 GiB in a browser.
 *
 * A resume that comes back 200 (server ignored Range) restarts from zero; a
 * 206 whose Content-Range doesn't line up with what we already hold is an
 * error rather than a silently corrupt file.
 */
async function streamDownload(url: string, onProgress?: ByteProgressCallback, control?: DownloadControl, segmented = false): Promise<WeightsBuffer> {
  let loaded = 0;
  let total: number | undefined;
  let target: Uint8Array | SegmentedBuffer | undefined;
  const chunks: Uint8Array[] = [];

  for (;;) {
    await control?.waitUntilRunning();
    const abort = new AbortController();
    control?.attach(abort);
    try {
      const res = await fetch(url, { headers: loaded > 0 ? { Range: `bytes=${loaded}-` } : undefined, signal: abort.signal });
      if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);

      if (res.status === 206) {
        const match = /^bytes (\d+)-\d+\/(\d+)$/.exec(res.headers.get("content-range") ?? "");
        if (!match || Number(match[1]) !== loaded || (total !== undefined && Number(match[2]) !== total)) {
          throw new Error(`Could not resume download of ${url}: the server's byte range doesn't match what was already received.`);
        }
        total = Number(match[2]);
      } else {
        loaded = 0;
        chunks.length = 0;
        target = undefined;
        const header = res.headers.get("content-length");
        total = header ? Number(header) : undefined;
      }
      if (total !== undefined && !target && loaded === 0) {
        try {
          target = segmented && total > SEGMENT_BYTES ? new SegmentedBuffer(total) : new Uint8Array(total);
        } catch {
          throw new Error(`Not enough browser memory to hold this ${(total / 1024 ** 3).toFixed(2)} GB file. Try loading structure only instead.`);
        }
      }
      onProgress?.(loaded, total);

      if (!res.body) {
        const bytes = new Uint8Array(await res.arrayBuffer());
        onProgress?.(bytes.byteLength, bytes.byteLength);
        return bytes.buffer;
      }

      const reader = res.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (target) {
          if (loaded + value.byteLength > target.byteLength) throw new Error(`Received more data than the ${target.byteLength} bytes ${url} advertised.`);
          if (target instanceof SegmentedBuffer) target.write(loaded, value);
          else target.set(value, loaded);
        } else {
          chunks.push(value);
        }
        loaded += value.byteLength;
        onProgress?.(loaded, total);
      }

      if (total !== undefined && loaded !== total) throw new Error(`Download of ${url} ended early (${loaded} of ${total} bytes).`);
      if (target instanceof SegmentedBuffer) return target;
      if (target) return target.buffer as ArrayBuffer;
      if (segmented && loaded > SEGMENT_BYTES) {
        const out = new SegmentedBuffer(loaded);
        let at = 0;
        for (const chunk of chunks) {
          out.write(at, chunk);
          at += chunk.byteLength;
        }
        return out;
      }
      const bytes = new Uint8Array(loaded);
      let offset = 0;
      for (const chunk of chunks) {
        bytes.set(chunk, offset);
        offset += chunk.byteLength;
      }
      return bytes.buffer;
    } catch (err) {
      if (control?.cancelled) throw new DownloadCancelledError();
      if (control?.paused) continue; // aborted by pause(): loop back, wait for resume, then request the rest
      throw err;
    } finally {
      control?.detach(abort);
    }
  }
}

/**
 * Fetches `url` as raw bytes, transparently caching the result in
 * IndexedDB (skipping anything over MAX_CACHEABLE_BYTES) so a re-click of
 * the same preset loads instantly with no network request at all —
 * `fetchJson` and `fetchArrayBuffer` both route through this, so
 * config.json/tokenizer.json/model.safetensors are all covered uniformly.
 *
 * A cache hit reports `onProgress` once, immediately, at 100% — there's no
 * network transfer to time, but callers shouldn't have to special-case that.
 */
export async function fetchCachedArrayBuffer(url: string, onProgress?: ByteProgressCallback, control?: DownloadControl): Promise<ArrayBuffer> {
  return (await fetchCachedBytes(url, onProgress, control, false)) as ArrayBuffer;
}

/** Same as fetchCachedArrayBuffer, but a file too big for one ArrayBuffer comes back as a SegmentedBuffer. Only meant for weight files, which are never cached (far over MAX_CACHEABLE_BYTES). */
export function fetchCachedWeights(url: string, onProgress?: ByteProgressCallback, control?: DownloadControl): Promise<WeightsBuffer> {
  return fetchCachedBytes(url, onProgress, control, true);
}

async function fetchCachedBytes(url: string, onProgress: ByteProgressCallback | undefined, control: DownloadControl | undefined, segmented: boolean): Promise<WeightsBuffer> {
  const cached = await getCached(url);
  if (cached) {
    onProgress?.(cached.byteLength, cached.byteLength);
    return cached;
  }

  let bytes: WeightsBuffer;
  if (onProgress || control) {
    bytes = await streamDownload(url, onProgress, control, segmented);
  } else {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
    bytes = await res.arrayBuffer();
  }
  if (!(bytes instanceof SegmentedBuffer) && bytes.byteLength <= MAX_CACHEABLE_BYTES) await putCached(url, bytes);
  return bytes;
}
