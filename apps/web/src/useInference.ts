import { useCallback, useState } from "react";
import type { ActivationCapture, Model, ModelAdapter, WeightProvider } from "@tensorium/model-ir";
import type { Tokenizer } from "@tensorium/tokenizer";

export interface InferenceState {
  status: "idle" | "running" | "ready" | "error";
  error?: string;
  result?: ActivationCapture;
  displayTokens?: string[];
  /** Wall-clock time of the `adapter.runInference` call itself (not tokenization) — a real browser measurement, not an estimate. */
  elapsedMs?: number;
}

export function useInference(model: Model | undefined, weightProvider: WeightProvider | undefined, adapter: ModelAdapter | undefined, tokenizer: Tokenizer | undefined) {
  const [state, setState] = useState<InferenceState>({ status: "idle" });

  const run = useCallback(
    async (prompt: string) => {
      if (!model || !weightProvider || !tokenizer) return;
      if (!adapter?.runInference) {
        setState({ status: "error", error: `${adapter?.displayName ?? "This adapter"} does not support running inference yet.` });
        return;
      }
      setState({ status: "running" });
      try {
        const { ids, displayTokens } = tokenizer.encode(prompt);
        if (ids.length === 0) throw new Error("Prompt tokenized to zero tokens — try a non-empty prompt.");
        const start = performance.now();
        const result = await adapter.runInference(model, weightProvider, ids);
        const elapsedMs = performance.now() - start;
        setState({ status: "ready", result, displayTokens, elapsedMs });
      } catch (err) {
        setState({ status: "error", error: err instanceof Error ? err.message : String(err) });
      }
    },
    [model, weightProvider, adapter, tokenizer]
  );

  const reset = useCallback(() => setState({ status: "idle" }), []);

  return { state, run, reset };
}
