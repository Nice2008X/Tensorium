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

/** Reduces a pasted Hugging Face model URL (e.g. https://huggingface.co/Qwen/Qwen3-1.7B/tree/main?x=1) to its `org/name` repo id. Anything that isn't an HF URL is returned trimmed and otherwise untouched. */
export function normalizeRepoId(input: string): string {
  const trimmed = input.trim();
  const match = trimmed.match(/^(?:https?:\/\/)?(?:www\.)?(?:huggingface\.co|hf\.co)\/(.+)$/i);
  if (!match) return trimmed;
  const segments = match[1]
    .split(/[?#]/)[0]
    .split("/")
    .filter(Boolean);
  if (segments[0] === "models") segments.shift();
  return segments.slice(0, 2).join("/");
}
