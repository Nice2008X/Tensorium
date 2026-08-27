import type { Model } from "@tensorium/model-ir";
import { totalParameterCount, totalParameterBytes } from "@tensorium/model-ir";
import { formatBytes, formatCount } from "../format.js";

export function ModelInfoBar({ model, structureOnly, bestEffort }: { model: Model; structureOnly?: boolean; bestEffort?: boolean }) {
  const params = totalParameterCount(model);
  const bytes = totalParameterBytes(model);
  const dtype = Object.values(model.nodes).find((n) => n.parameters.length > 0)?.parameters[0]?.dtype ?? "—";

  const stats: [string, string][] = [
    ["Architecture", model.architecture],
    ["Parameters", formatCount(params)],
    ["Layers", String(model.config.numLayers)],
    ["Attention heads", String(model.config.numHeads)],
    ["Hidden size", String(model.config.hiddenSize)],
    ["Intermediate size", String(model.config.intermediateSize)],
    ["Vocabulary", model.config.vocabSize.toLocaleString()],
    ["Context length", model.config.contextLength.toLocaleString()],
    ["Dtype", dtype],
    [structureOnly ? "Weights (real, not downloaded)" : "Weights (in browser)", formatBytes(bytes)],
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
      <div className="model-info-stats">
        {stats.map(([label, value]) => (
          <div key={label} className="model-info-stat">
            <div className="model-info-stat-value">{value}</div>
            <div className="model-info-stat-label">{label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
