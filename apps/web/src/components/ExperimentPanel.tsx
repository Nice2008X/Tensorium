import { useState } from "react";
import type { ActivationCapture, Intervention, Model, ModelAdapter, ModelNode, WeightProvider } from "@tensorium/model-ir";
import type { Tokenizer } from "@tensorium/tokenizer";
import { outputTokenLabel, topKFromLogits, type RankedToken } from "../logits.js";

interface Props {
  model: Model;
  weightProvider: WeightProvider;
  adapter: ModelAdapter;
  tokenizer: Tokenizer;
  selectedNode: ModelNode | null;
  mainTokenIds: number[];
  mainResult: ActivationCapture;
  promptBResult: ActivationCapture | null;
  /** Reported whenever this panel's own background computation starts/stops — lets the app show a busy cursor while it runs. */
  onBusyChange?: (busy: boolean) => void;
}

type Operation = "zero" | "zero_head" | "replace";

export function ExperimentPanel({ model, weightProvider, adapter, tokenizer, selectedNode, mainTokenIds, mainResult, promptBResult, onBusyChange }: Props) {
  const [operation, setOperation] = useState<Operation>("zero");
  const [headIndex, setHeadIndex] = useState(0);
  const [tokenScope, setTokenScope] = useState<"all" | number>("all");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<{ before: RankedToken[]; after: RankedToken[] } | null>(null);

  if (!selectedNode) {
    return <div className="empty-hint">Select a component in the graph or tree, then choose an intervention to run below.</div>;
  }
  if (!adapter.runInference) {
    return <div className="empty-hint">{adapter.displayName} does not support running experiments.</div>;
  }

  const isAttention = selectedNode.type === "attention";
  const numHeads = Number(selectedNode.metadata.numHeads ?? model.config.numHeads);
  const canReplace = !!promptBResult?.activations[selectedNode.id];
  const S = mainTokenIds.length;
  const displayTokens = mainTokenIds.map((id) => tokenizer.decodeToken(id));

  async function run() {
    setRunning(true);
    setError(null);
    onBusyChange?.(true);
    try {
      const intervention: Intervention = {
        nodeId: selectedNode!.id,
        operation,
        tokenIndex: tokenScope === "all" ? undefined : tokenScope,
        headIndex: operation === "zero_head" ? headIndex : undefined,
        replacementValue: operation === "replace" ? promptBResult?.activations[selectedNode!.id] : undefined,
      };
      const before = topKFromLogits(mainResult.logits, S - 1);
      const result = await adapter.runInference!(model, weightProvider, mainTokenIds, [intervention]);
      const after = topKFromLogits(result.logits, S - 1);
      setOutcome({ before, after });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
      onBusyChange?.(false);
    }
  }

  return (
    <div className="experiment-panel">
      <div className="experiment-target">
        Target: <strong>{selectedNode.name}</strong> <span className="experiment-node-id">({selectedNode.id})</span>
      </div>

      <div className="experiment-controls">
        <label className="experiment-field">
          Operation
          <select value={operation} onChange={(e) => setOperation(e.target.value as Operation)}>
            <option value="zero">Zero out (ablate)</option>
            <option value="zero_head" disabled={!isAttention}>
              Zero one attention head{!isAttention ? " (attention nodes only)" : ""}
            </option>
            <option value="replace" disabled={!canReplace}>
              Patch in from Prompt B{!canReplace ? " (run Prompt B with this node selected first)" : ""}
            </option>
          </select>
        </label>

        {operation === "zero_head" && isAttention && (
          <label className="experiment-field">
            Head
            <select value={headIndex} onChange={(e) => setHeadIndex(Number(e.target.value))}>
              {Array.from({ length: numHeads }, (_, h) => (
                <option key={h} value={h}>
                  Head {h}
                </option>
              ))}
            </select>
          </label>
        )}

        <label className="experiment-field">
          Token position
          <select value={tokenScope} onChange={(e) => setTokenScope(e.target.value === "all" ? "all" : Number(e.target.value))}>
            <option value="all">All positions</option>
            {displayTokens.map((t, i) => (
              <option key={i} value={i}>
                {i}: "{t}"
              </option>
            ))}
          </select>
        </label>

        <button onClick={run} disabled={running}>
          {running && <span className="spinner spinner-inline" />}
          {running ? "Running…" : "Run Experiment"}
        </button>
      </div>

      {error && <div className="inference-error">{error}</div>}

      {outcome && (
        <div className="experiment-result">
          <div className="experiment-compare-col">
            <div className="experiment-compare-title">Before</div>
            {outcome.before.map((t) => (
              <RankRow key={t.tokenId} token={t} label={outputTokenLabel(tokenizer, model.config.classLabels, t.tokenId)} />
            ))}
          </div>
          <div className="experiment-compare-col">
            <div className="experiment-compare-title">After</div>
            {outcome.after.map((t) => {
              const beforeMatch = outcome.before.find((b) => b.tokenId === t.tokenId);
              const delta = beforeMatch ? t.prob - beforeMatch.prob : t.prob;
              return <RankRow key={t.tokenId} token={t} label={outputTokenLabel(tokenizer, model.config.classLabels, t.tokenId)} delta={delta} />;
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function RankRow({ token, label, delta }: { token: RankedToken; label: string; delta?: number }) {
  const display = label.trim() || `#${token.tokenId}`;
  return (
    <div className="rank-row">
      <span className="rank-token">{display}</span>
      <span className="rank-prob">{(token.prob * 100).toFixed(1)}%</span>
      {delta !== undefined && (
        <span className={"rank-delta" + (delta >= 0 ? " rank-delta-up" : " rank-delta-down")}>
          {delta >= 0 ? "▲" : "▼"} {Math.abs(delta * 100).toFixed(1)}pp
        </span>
      )}
    </div>
  );
}
