import type { TensorStats } from "@tensorium/tensor-core";

export function Histogram({ stats }: { stats: TensorStats }) {
  const max = Math.max(1, ...stats.histogram.map((b) => b.count));
  return (
    <div className="histogram">
      {stats.histogram.map((b, i) => (
        <div
          key={i}
          className="histogram-bar"
          style={{ height: `${(b.count / max) * 100}%` }}
          title={`[${b.binStart.toFixed(4)}, ${b.binEnd.toFixed(4)}) — ${b.count}`}
        />
      ))}
    </div>
  );
}
