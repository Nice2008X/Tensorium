import type { Model } from "@tensorium/model-ir";
import { totalParameterCount, totalParameterBytes } from "@tensorium/model-ir";
import { formatBytes, formatCount } from "../format.js";

/**
 * Three loosely-themed clusters instead of one flat run of 10 same-weight
 * stats — "what model" / "what shape" / "what it costs to run" reads faster
 * at a glance than a row where params and dtype carry equal visual weight.
 * Nothing here is dropped, just grouped; each group renders as one
 * `·`-joined line rather than its own boxes.
 */
export function ModelInfoBar({ model, structureOnly, bestEffort }: { model: Model; structureOnly?: boolean; bestEffort?: boolean }) {
  const params = totalParameterCount(model);
  const bytes = totalParameterBytes(model);
  const dtype = Object.values(model.nodes).find((n) => n.parameters.length > 0)?.parameters[0]?.dtype ?? "—";
  const weightsLabel = structureOnly ? "weights, real, not downloaded" : "weights in browser";

  const groups: [string, string][] = [
    ["Model", `${model.architecture} · ${formatCount(params)} params · ${model.config.numLayers} layers`],
    [
      "Architecture",
      `${model.config.numHeads} heads · ${model.config.hiddenSize} hidden · ${model.config.intermediateSize} MLP · ${model.config.vocabSize.toLocaleString()} vocab`,
    ],
    ["Runtime", `${dtype} · ${formatBytes(bytes)} ${weightsLabel} · ${model.config.contextLength.toLocaleString()} context`],
  ];

  return (
    <div className="model-info-bar">
      <div className="model-info-name">
        {model.name}
        {structureOnly && (
          <span
            className="model-info-structure-badge"
            title="This checkpoint is too large (or sharded) to download in a browser tab — the architecture and every tensor's shape/dtype above are exact, but no real weight bytes were ever fetched. A forward pass runs against randomly generated values instead, if enabled in Settings."
          >
            Structure only
          </span>
        )}
        {bestEffort && (
          <span
            className="model-info-structure-badge"
            title="No named adapter recognized this checkpoint's model type — its structure (layer count, projection shapes, MoE routing, ...) was auto-detected from its weight names instead of hand-verified. The graph should be accurate, but Run Forward Pass may compute incorrect numbers for any architecture detail this app couldn't detect automatically."
          >
            Best effort
          </span>
        )}
      </div>
      {/* Only the last (Runtime) group is allowed to shrink+ellipsize — see
          .model-info-group:last-child — so a narrow window (or a long model
          name eating into the available width) degrades to "F32 · 32.8 MB…"
          instead of the row running underneath .top-right-controls. */}
      <div className="model-info-groups">
        {groups.map(([label, value]) => (
          <div key={label} className="model-info-group">
            <div className="model-info-group-label">{label}</div>
            <div className="model-info-group-value">{value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
