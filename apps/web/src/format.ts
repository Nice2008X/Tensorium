/**
 * Formats a 0..1 fraction as a percentage string, stepping down to more
 * decimal places for small values so a real-but-tiny probability (e.g. a
 * long tail token at 0.06%) doesn't collapse to a misleading "0.0%".
 */
export function formatPercent(fraction: number): string {
  const pct = fraction * 100;
  const abs = Math.abs(pct);
  if (abs === 0) return "0%";
  if (abs >= 0.1) return `${pct.toFixed(1)}%`;
  if (abs >= 0.01) return `${pct.toFixed(2)}%`;
  if (abs >= 0.001) return `${pct.toFixed(3)}%`;
  if (abs >= 0.0001) return `${pct.toFixed(4)}%`;
  return `${pct.toPrecision(1)}%`;
}

/** Compact parameter/element count, e.g. 1234567 -> "1.23M". */
export function formatCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

/** Compact byte size, e.g. 1234567 -> "1.2 MB". */
export function formatBytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`;
  return `${n} B`;
}

/** Reduces a pasted Hugging Face model URL (e.g. https://huggingface.co/Qwen/Qwen3-1.7B/tree/main?x=1) to its `org/name` repo id — or `org/name/subfolder` for a `/tree/<revision>/<subfolder>` URL (see hf-client's hfResolveUrl). Anything that isn't an HF URL is returned trimmed and otherwise untouched. */
export function normalizeRepoId(input: string): string {
  const trimmed = input.trim();
  const match = trimmed.match(/^(?:https?:\/\/)?(?:www\.)?(?:huggingface\.co|hf\.co)\/(.+)$/i);
  if (!match) return trimmed;
  const segments = match[1]
    .split(/[?#]/)[0]
    .split("/")
    .filter(Boolean);
  if (segments[0] === "models") segments.shift();
  const subfolder = segments[2] === "tree" ? segments.slice(4) : [];
  return [...segments.slice(0, 2), ...subfolder].join("/");
}

const WINDOWS_RESERVED_NAME = /^(con|prn|aux|nul|com[1-9]|lpt[1-9])$/i;

/**
 * Turns an arbitrary display name (a repo id, or a name derived from a user-picked file) into something safe to hand
 * to a browser download: no path separators or characters Windows/macOS reject, no control characters (including
 * bidi overrides that make `evil.exe` render as `exe.live`), no `..` sequences, no leading/trailing dots or spaces,
 * no reserved Windows device names, and a bounded length. Never returns an empty string.
 */
export function sanitizeFileName(name: string, fallback = "model"): string {
  const cleaned = name
    .replace(/[\u0000-\u001f\u007f-\u009f\u202a-\u202e\u2066-\u2069\\/:*?"<>|]+/g, "-")
    .replace(/\.{2,}/g, ".")
    .replace(/^[.\s-]+|[.\s-]+$/g, "")
    .slice(0, 100)
    .replace(/[.\s-]+$/g, "");
  return !cleaned || WINDOWS_RESERVED_NAME.test(cleaned.split(".")[0]) ? fallback : cleaned;
}
