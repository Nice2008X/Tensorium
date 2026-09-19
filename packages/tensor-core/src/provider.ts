import type { ParameterRef, Tensor, TensorSlice, WeightProvider, WeightsBuffer } from "@tensorium/model-ir";
import { dtypeSize, numElements } from "@tensorium/model-ir";
import { parseSafetensorsHeader, readTensor, type SafetensorsFile } from "./safetensors.js";

/**
 * WeightProvider backed by an in-memory safetensors buffer.
 *
 * The whole file is downloaded once (fine for tiny models — a few hundred KB
 * to a few MB), but tensor *decoding* stays lazy: listParameters()/
 * getParameterInfo() only ever read the safetensors header, and loadTensor()
 * decodes just the requested slice. For multi-GB checkpoints the same
 * interface would be backed by a server that does true byte-range HTTP reads
 * instead of an in-memory buffer — nothing above this provider would change.
 *
 * loadTensor() results are cached by (parameterId, slice) for the lifetime
 * of this provider (i.e. the lifetime of the loaded model). Every adapter's
 * runInference() reloads every weight it needs on each call — fine for one
 * forward pass, but Token Attribution and Experiment interventions call it
 * repeatedly against the *same* unchanged weights, and decoding is a real
 * per-element byte-format conversion (see readTensor), not a cheap slice.
 * Safe to cache the returned Tensor object itself rather than a copy: every
 * consumer (tensorToMatrix/tensorToVector, this app's canvas renderers)
 * only ever reads a Tensor's `data`, never writes back into it.
 */
export class SafetensorsWeightProvider implements WeightProvider {
  id: string;
  private file: SafetensorsFile;
  private tensorCache = new Map<string, Tensor>();

  constructor(id: string, buffer: WeightsBuffer) {
    this.id = id;
    this.file = parseSafetensorsHeader(buffer);
  }

  async listParameters(): Promise<ParameterRef[]> {
    return Object.entries(this.file.header).map(([name, entry]) => this.toParameterRef(name, entry));
  }

  async getParameterInfo(parameterId: string): Promise<ParameterRef> {
    const entry = this.file.header[parameterId];
    if (!entry) throw new Error(`Unknown parameter: ${parameterId}`);
    return this.toParameterRef(parameterId, entry);
  }

  async loadTensor(parameterId: string, options?: TensorSlice): Promise<Tensor> {
    // A given call site always requests the same slice of a given name (a
    // node's ParameterRef is fixed at graph-build time), so this key is
    // stable across repeated calls — it just needs to distinguish e.g.
    // GPT-2's c_attn sliced into q/k/v from the unsliced tensor itself.
    const cacheKey = options ? `${parameterId}::${JSON.stringify(options)}` : parameterId;
    const cached = this.tensorCache.get(cacheKey);
    if (cached) return cached;

    const tensor = readTensor(this.file, parameterId, options);
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
