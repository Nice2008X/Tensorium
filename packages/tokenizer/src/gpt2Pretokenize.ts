// GPT-2's byte-level pre-tokenization: map each of the 256 possible byte
// values to a unique printable Unicode character (so BPE never has to deal
// with control characters or invalid UTF-8), split the input into "words"
// with GPT-2's regex, then represent each word as a sequence of those
// mapped characters. Matches HF's ByteLevel pre-tokenizer bit-for-bit.

function buildByteToUnicode(): Map<number, string> {
  const bs: number[] = [];
  for (let b = "!".charCodeAt(0); b <= "~".charCodeAt(0); b++) bs.push(b);
  for (let b = "¡".charCodeAt(0); b <= "¬".charCodeAt(0); b++) bs.push(b);
  for (let b = "®".charCodeAt(0); b <= "ÿ".charCodeAt(0); b++) bs.push(b);

  const bset = new Set(bs);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bset.has(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const map = new Map<number, string>();
  for (let i = 0; i < bs.length; i++) map.set(bs[i], String.fromCharCode(cs[i]));
  return map;
}

const BYTE_TO_UNICODE = buildByteToUnicode();
const UNICODE_TO_BYTE = new Map<string, number>([...BYTE_TO_UNICODE.entries()].map(([b, c]) => [c, b]));

// GPT-2's canonical pre-tokenizer regex — the default when a tokenizer.json
// doesn't specify its own Split pattern (some byte-level tokenizers, Qwen's
// among them, use a different regex; see resolveByteLevelSplit below).
export const GPT2_SPLIT_RE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

export type PreTokenSplitStep = { kind: "regex"; regex: RegExp } | { kind: "digits"; individualDigits: boolean };

/**
 * Runs one "Isolated"-behavior split step over the current list of pieces.
 * Isolated means both the matched spans *and* the text between them survive
 * as separate pieces (rather than discarding the gaps) so a later step in
 * the sequence gets a chance to split whatever this one didn't claim.
 * Single-step callers (GPT-2, Qwen) still get identical output to a plain
 * `text.match(regex)`, since their one regex is exhaustive — every
 * character falls into some alternation branch, so there are no gaps left.
 */
function applySplitStep(pieces: string[], step: PreTokenSplitStep): string[] {
  const regex =
    step.kind === "digits"
      ? new RegExp(step.individualDigits ? "[0-9]" : "[0-9]+", "g")
      : new RegExp(step.regex.source, step.regex.flags.includes("g") ? step.regex.flags : step.regex.flags + "g");

  const out: string[] = [];
  for (const piece of pieces) {
    let last = 0;
    for (const m of piece.matchAll(regex)) {
      if (m.index! > last) out.push(piece.slice(last, m.index));
      out.push(m[0]);
      last = m.index! + m[0].length;
    }
    if (last < piece.length) out.push(piece.slice(last));
  }
  return out.filter((p) => p.length > 0);
}

/** Splits text into byte-level "words", each represented as an array of byte-mapped Unicode characters (the BPE merge algorithm's initial symbols). */
export function gpt2Pretokenize(text: string, splitSteps: PreTokenSplitStep[] = [{ kind: "regex", regex: GPT2_SPLIT_RE }]): string[][] {
  let words = [text];
  for (const step of splitSteps) words = applySplitStep(words, step);
  return words.map((word) => {
    const bytes = new TextEncoder().encode(word);
    return Array.from(bytes, (b) => BYTE_TO_UNICODE.get(b)!);
  });
}

/** Reverses byte-level mapping back to real UTF-8 text, for decoding/display. */
export function gpt2ByteDecode(mapped: string): string {
  const bytes = Uint8Array.from(Array.from(mapped, (ch) => UNICODE_TO_BYTE.get(ch) ?? 0));
  return new TextDecoder("utf-8", { fatal: false }).decode(bytes);
}

export interface PreTokenizerSpec {
  type?: string;
  pretokenizers?: PreTokenizerSpec[];
  pattern?: { Regex?: string; String?: string };
  individual_digits?: boolean;
}

export interface ByteLevelResolution {
  isByteLevel: boolean;
  /** null = use the default GPT2_SPLIT_RE; some byte-level tokenizers ship their own Split/Digits step(s) instead. */
  splitSteps: PreTokenSplitStep[] | null;
}

// Rust/Oniguruma inline case-insensitive groups ("(?i:...)") aren't valid JS
// regex syntax. The only place this has shown up in practice is wrapping the
// apostrophe-contraction alternatives ('s/'t/'re/...), and nothing else in
// these patterns is a case-sensitive literal, so converting the scoped flag
// to a global `i` flag is behavior-preserving here rather than a narrow
// one-off hack.
const MAX_SPLIT_REGEX_LENGTH = 2048;

function toJsRegex(pattern: string): RegExp | null {
  // Real split patterns are a few hundred characters. A tokenizer.json from an untrusted repo could ship a huge or catastrophically-backtracking one, and this regex runs on the main thread over every prompt; refuse the oversized ones (the caller falls back to its default splitter).
  if (pattern.length > MAX_SPLIT_REGEX_LENGTH) return null;
  const jsPattern = pattern.replace(/\(\?i:/g, "(?:");
  try {
    return new RegExp(jsPattern, "gui");
  } catch {
    return null;
  }
}

/**
 * Not every byte-level tokenizer.json declares `pre_tokenizer: {"type":
 * "ByteLevel"}` directly the way GPT-2's does — Qwen's is a `Sequence` of a
 * custom regex `Split` step followed by a `ByteLevel` step; DeepSeek's goes
 * further still, chaining five `Split` steps (newlines, letters, punctuation,
 * trailing whitespace, CJK) and a `Digits` step ahead of `ByteLevel`, each
 * covering only its own slice of characters and leaving the rest for the
 * next step (see gpt2Pretokenize's "Isolated" handling). This walks that
 * shape and returns every Split/Digits step in order, instead of assuming
 * there's exactly one that covers the whole input by itself.
 */
export function resolveByteLevel(spec: PreTokenizerSpec | null | undefined): ByteLevelResolution {
  if (!spec) return { isByteLevel: false, splitSteps: null };
  const steps = spec.type === "Sequence" ? (spec.pretokenizers ?? []) : [spec];
  if (!steps.some((s) => s.type === "ByteLevel")) return { isByteLevel: false, splitSteps: null };

  const splitSteps: PreTokenSplitStep[] = [];
  for (const s of steps) {
    if (s.type === "Split" && s.pattern?.Regex) {
      const regex = toJsRegex(s.pattern.Regex);
      if (regex) splitSteps.push({ kind: "regex", regex });
    } else if (s.type === "Digits") {
      splitSteps.push({ kind: "digits", individualDigits: s.individual_digits ?? false });
    }
  }
  return { isByteLevel: true, splitSteps: splitSteps.length > 0 ? splitSteps : null };
}
