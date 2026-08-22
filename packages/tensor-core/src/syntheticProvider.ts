import type { ParameterRef, Tensor, TensorSlice, WeightProvider } from "@tensorium/model-ir";
import { dtypeSize, numElements } from "@tensorium/model-ir";

/**
 * Small-magnitude values in roughly the range real transformer weights
 * actually occupy after initialization (many real config.json files
 * literally set `initializer_range: 0.02`) — chosen purely so a synthetic
 * checkpoint's heatmaps/histograms look plausible rather than wildly out of
 * scale, not to resemble any specific real distribution.
 */
const SYNTHETIC_WEIGHT_SCALE = 0.02;

/**
 * WeightProvider for a checkpoint whose real tensor bytes were never
 * downloaded — either it's too large (see hf-client's
 * STRUCTURE_ONLY_THRESHOLD_BYTES) or sharded (this app's eager loader only
 * ever fetches a single model.safetensors). `listParameters`/
 * `getParameterInfo` report every tensor's *real* shape and dtype (known
 * from the safetensors header alone, per hf-client's fetchModelStructure);
 * only `loadTensor` fabricates its data, uniformly at random.
 *
 * Every real WeightProvider's `runInference` call pattern (reload each
 * weight it needs, every call) means the SAME tensor gets requested
 * repeatedly within one loaded session — e.g. Prompt A vs Prompt B, or
 * Token Attribution's per-token re-runs. Caching by (parameterId, slice),
 * exactly like SafetensorsWeightProvider does, is what keeps a given
 * tensor's random values stable across those repeated calls; without it,
 * "the same" weight would silently change value on every read, which would
 * make the forward pass internally inconsistent, not just synthetic.
 */
export class SyntheticWeightProvider implements WeightProvider {
  id: string;
  private index: Record<string, { shape: number[]; dtype: string }>;
  private tensorCache = new Map<string, Tensor>();

  constructor(id: string, weightIndex: Record<string, { shape: number[]; dtype: string }>) {
    this.id = id;
    this.index = weightIndex;
  }

  async listParameters(): Promise<ParameterRef[]> {
    return Object.entries(this.index).map(([name, entry]) => this.toParameterRef(name, entry));
  }

  async getParameterInfo(parameterId: string): Promise<ParameterRef> {
    const entry = this.index[parameterId];
    if (!entry) throw new Error(`Unknown parameter: ${parameterId}`);
    return this.toParameterRef(parameterId, entry);
  }

  async loadTensor(parameterId: string, options?: TensorSlice): Promise<Tensor> {
    const cacheKey = options ? `${parameterId}::${JSON.stringify(options)}` : parameterId;
    const cached = this.tensorCache.get(cacheKey);
    if (cached) return cached;

    const entry = this.index[parameterId];
    if (!entry) throw new Error(`Unknown parameter: ${parameterId}`);

    // Same range-clamping a real slice goes through in readTensor() — only
    // the output shape matters here, since there's no real underlying data
    // to index into.
    const outShape = entry.shape.map((dimSize, i) => {
      const r = options?.ranges?.[i];
      const start = Math.max(0, Math.min(r?.start ?? 0, dimSize));
      const end = Math.max(start, Math.min(r?.end ?? dimSize, dimSize));
      return end - start;
    });

    const total = outShape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(total);
    for (let i = 0; i < total; i++) data[i] = (Math.random() * 2 - 1) * SYNTHETIC_WEIGHT_SCALE;

    const tensor: Tensor = { shape: outShape, dtype: entry.dtype, data };
    this.tensorCache.set(cacheKey, tensor);
    return tensor;
  }

  private toParameterRef(name: string, entry: { shape: number[]; dtype: string }): ParameterRef {
    const n = numElements(entry.shape);
    return {
      name,
      shape: entry.shape,
      dtype: entry.dtype,
      numElements: n,
      bytes: n * dtypeSize(entry.dtype),
      providerId: this.id,
      logicalShape: entry.shape,
    };
  }
}
