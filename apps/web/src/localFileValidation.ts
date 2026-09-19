import { decodeSafetensorsHeader, readSafetensorsHeaderLength } from "@tensorium/tensor-core";
import { formatBytes } from "./format.js";

export interface FileCheck {
  ok: boolean;
  error?: string;
  warning?: string;
}

const TEXT_SNIFF_BYTES = 4096;
/** Not a hard limit — the browser can technically hold more — just the point past which loading everything into memory as one buffer is likely to be slow or risk running out of memory. */
const WEIGHTS_WARN_BYTES = 300 * 1024 * 1024;

/** Reads only a small prefix and checks it decodes as clean UTF-8 with no embedded NUL bytes — the cheap, reliable way to tell "this is actually text/JSON" from "this is binary content someone renamed to .json", without reading a potentially huge file in full just to reject it. */
async function looksLikeText(file: File): Promise<boolean> {
  const prefix = new Uint8Array(await file.slice(0, TEXT_SNIFF_BYTES).arrayBuffer());
  if (prefix.includes(0)) return false;
  try {
    new TextDecoder("utf-8", { fatal: true }).decode(prefix);
    return true;
  } catch {
    return false;
  }
}

export type JsonFileKind = "config" | "tokenizer";

/** Real config.json files are a few KB; real tokenizer.json files run up to ~35 MB — same ceilings the remote loader enforces (hf-client MAX_CONFIG_BYTES / MAX_TOKENIZER_BYTES). */
const JSON_MAX_BYTES: Record<JsonFileKind, number> = { config: 5 * 1024 * 1024, tokenizer: 128 * 1024 * 1024 };

/** Judged on the extension the user's file actually has, not on the `accept` attribute (which is only a picker hint and doesn't stop a drag-drop or an "All files" pick). */
function hasExtension(file: File, ext: string): boolean {
  return file.name.toLowerCase().endsWith(ext);
}

/** config.json / tokenizer.json: a `.json` file that is small, valid UTF-8 text, and parses as a JSON *object* (not `null`, a number, or an array — adapters index straight into it) — catches a mislabeled binary file (e.g. a `.safetensors` renamed to `.json`) or a truncated/corrupt download before the app commits to loading it. */
export async function checkJsonFile(file: File, kind: JsonFileKind): Promise<FileCheck> {
  if (!hasExtension(file, ".json")) return { ok: false, error: "Expected a .json file." };
  if (file.size === 0) return { ok: false, error: "File is empty." };
  if (file.size > JSON_MAX_BYTES[kind]) {
    return { ok: false, error: `${formatBytes(file.size)} is too large for a ${kind} file — expected JSON text${kind === "config" ? ", typically a few KB" : ""}. Check you picked the right file.` };
  }
  if (!(await looksLikeText(file))) {
    return { ok: false, error: "This doesn't look like a text/JSON file — its content looks binary." };
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(await file.text());
  } catch (err) {
    return { ok: false, error: `Not valid JSON: ${err instanceof Error ? err.message : String(err)}` };
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    return { ok: false, error: `Expected a JSON object at the top level of this ${kind} file.` };
  }
  if (kind === "tokenizer" && (typeof (parsed as { model?: unknown }).model !== "object" || (parsed as { model?: unknown }).model === null)) {
    return { ok: false, error: 'This doesn\'t look like a tokenizer.json (no "model" section).' };
  }
  return { ok: true };
}

/** .safetensors: validated by actually reading and strictly checking its header (the same validator every safetensors read in the app goes through — well-formed entries, shapes that match their byte spans, no tensor pointing past the end of *this* file) rather than trusting the file extension. Only the header-sized prefix is read, never the whole (possibly huge) file. */
export async function checkWeightsFile(file: File): Promise<FileCheck> {
  if (!hasExtension(file, ".safetensors")) return { ok: false, error: "Expected a .safetensors file." };
  if (file.size < 10) return { ok: false, error: "File is too small to be a safetensors file." };
  try {
    const headerLength = readSafetensorsHeaderLength(new Uint8Array(await file.slice(0, 8).arrayBuffer()));
    if (8 + headerLength > file.size) return { ok: false, error: "This doesn't look like a valid safetensors file (it ends before its header does)." };
    decodeSafetensorsHeader(new Uint8Array(await file.slice(8, 8 + headerLength).arrayBuffer()), file.size - 8 - headerLength);
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : "This doesn't look like a valid safetensors file." };
  }
  const warning =
    file.size > WEIGHTS_WARN_BYTES
      ? `This file is ${formatBytes(file.size)} — loading a model this large in the browser may be slow or run out of memory.`
      : undefined;
  return { ok: true, warning };
}
