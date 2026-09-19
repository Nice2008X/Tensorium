import type { ModelSource } from "@tensorium/model-ir";
import { fetchJson, hfResolveUrl, MAX_TOKENIZER_BYTES, readLocalJson, type ByteProgressCallback } from "@tensorium/hf-client";
import { bpeMerge } from "./bpe.js";
import { gpt2ByteDecode, gpt2Pretokenize, resolveByteLevel, type PreTokenizerSpec } from "./gpt2Pretokenize.js";
import { normalizerHasPrepend, normalizerRequiresNFC, spBpeDecodePieces, spBpePretokenize, type NormalizerSpec } from "./llamaPretokenize.js";

export interface EncodeResult {
  ids: number[];
  /** raw vocab strings — mostly useful for debugging the tokenizer itself */
  tokens: string[];
  /** human-readable per-token text, suitable for UI chips (spaces/newlines restored) */
  displayTokens: string[];
}

export interface Tokenizer {
  encode(text: string): EncodeResult;
  decode(ids: number[]): string;
  decodeToken(id: number): string;
}

interface RawTokenizerJson {
  model: {
    type: string;
    vocab: Record<string, number>;
    merges: Array<string | [string, string]>;
  };
  pre_tokenizer?: PreTokenizerSpec | null;
  normalizer?: NormalizerSpec | null;
}

/**
 * Loads a Hugging Face "fast tokenizer" (tokenizer.json) and returns a BPE
 * encoder/decoder. Detects which pre-tokenization scheme it needs from the
 * file itself — GPT-2's flat ByteLevel, Qwen's Sequence-of-[custom regex
 * Split, ByteLevel] (same byte-level family, different word-splitting
 * rules and an NFC normalizer), or the SentencePiece-style scheme Llama,
 * Mistral, and Gemma use — rather than assuming any one shape. Every case
 * here was added because a real tokenizer.json didn't match the simpler
 * assumption that came before it (see resolveByteLevel's and
 * normalizerHasPrepend's doc comments for the specific ones).
 */
export async function loadTokenizer(source: ModelSource, onProgress?: ByteProgressCallback): Promise<Tokenizer> {
  const raw =
    source.kind === "local"
      ? readLocalJson<RawTokenizerJson>(source, "tokenizer.json")
      : await fetchJson<RawTokenizerJson>(hfResolveUrl(source, "tokenizer.json"), MAX_TOKENIZER_BYTES, onProgress);
  if (raw.model.type !== "BPE") {
    throw new Error(`Unsupported tokenizer model type: "${raw.model.type}" (only BPE fast tokenizers are supported)`);
  }

  const vocab = new Map<string, number>(Object.entries(raw.model.vocab));
  const idToToken = new Map<number, string>([...vocab.entries()].map(([token, id]) => [id, token]));

  const mergeRank = new Map<string, number>();
  raw.model.merges.forEach((m, i) => {
    const key = Array.isArray(m) ? `${m[0]} ${m[1]}` : m;
    mergeRank.set(key, i);
  });

  const { isByteLevel, splitSteps } = resolveByteLevel(raw.pre_tokenizer);
  const needsNFC = isByteLevel && normalizerRequiresNFC(raw.normalizer);
  const stripsArtificialLeadingSpace = !isByteLevel && normalizerHasPrepend(raw.normalizer);

  function encode(text: string): EncodeResult {
    const normalized = needsNFC ? text.normalize("NFC") : text;
    const wordSymbolGroups = isByteLevel
      ? gpt2Pretokenize(normalized, splitSteps ?? undefined)
      : [spBpePretokenize(text, vocab, raw.normalizer)];

    const ids: number[] = [];
    const tokens: string[] = [];
    for (const symbols of wordSymbolGroups) {
      for (const piece of bpeMerge(symbols, mergeRank)) {
        const id = vocab.get(piece);
        if (id === undefined) continue; // shouldn't happen for well-formed BPE output
        ids.push(id);
        tokens.push(piece);
      }
    }

    // per-token display pieces keep their natural leading space (if any) —
    // only the *whole-sequence* decode() strips the single artificial one
    // that a Prepend normalizer step adds at the very start.
    const displayTokens = tokens.map((t) => (isByteLevel ? gpt2ByteDecode(t) : spBpeDecodePieces([t])));
    return { ids, tokens, displayTokens };
  }

  function decode(ids: number[]): string {
    const pieces = ids.map((id) => idToToken.get(id) ?? "");
    if (isByteLevel) return pieces.map(gpt2ByteDecode).join("");
    const text = spBpeDecodePieces(pieces);
    return stripsArtificialLeadingSpace && text.startsWith(" ") ? text.slice(1) : text;
  }

  function decodeToken(id: number): string {
    const piece = idToToken.get(id) ?? "";
    return isByteLevel ? gpt2ByteDecode(piece) : spBpeDecodePieces([piece]);
  }

  return { encode, decode, decodeToken };
}
