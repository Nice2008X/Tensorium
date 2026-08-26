// Model IR — the normalized representation every visualization and every
// model adapter agrees on. Nothing in here knows about GPT-2, Llama, or any
// specific architecture; adapters translate a real model INTO this shape.

/**
 * The vocabulary of node kinds the UI knows how to render.
 * Adding a new architecture should rarely require adding a new NodeType —
 * prefer composing existing types (e.g. RoPE can be a "custom" node with a
 * distinguishing `metadata.op`, or promoted here if it recurs across models).
 */
export type NodeType =
  | "model"
  | "block_group"
  | "input"
  | "embedding"
  | "positional_embedding"
  | "transformer_block"
  | "layer_norm"
  | "rms_norm"
  | "attention"
  | "linear_attention"
  | "q_projection"
  | "k_projection"
  | "v_projection"
  | "qkv_projection"
  | "output_projection"
  | "rope"
  | "ffn"
  | "linear"
  | "activation"
  | "elementwise_mul"
  | "residual"
  | "router"
  | "moe_experts"
  | "lm_head"
  | "output";

export interface TensorSpec {
  /** Dimension sizes; a string entry is a symbolic/dynamic dim, e.g. "sequence_length". */
  dims: Array<number | string>;
}

export interface ParameterRef {
  /** Fully-qualified name as it appears in the weight file (e.g. HF state-dict key). */
  name: string;
  /** Full shape of the underlying weight-file tensor (before any `slice` below is applied). */
  shape: number[];
  dtype: string;
  /** Element count / byte size of the full underlying tensor. */
  numElements: number;
  bytes: number;
  /** Which WeightProvider owns this parameter's actual data. */
  providerId: string;
  /**
   * Set when this ref stands for a sub-range of `name`'s tensor rather than
   * the whole thing — e.g. GPT-2 fuses Q/K/V into one c_attn matrix, so the
   * "Q projection" node's ParameterRef points at c_attn with a column slice.
   * Consumers should pass this straight through to WeightProvider.loadTensor.
   */
  slice?: TensorSlice;
  /** The effective shape after `slice` is applied (== shape when slice is absent). */
  logicalShape: number[];
}

export interface ModelNode {
  id: string;
  type: NodeType;
  name: string;

  inputs: TensorSpec[];
  outputs: TensorSpec[];

  /** Weights/biases owned directly by this node (empty for structural/container nodes). */
  parameters: ParameterRef[];

  /** Child node ids, for the logical hierarchy (Model -> Block -> Attention -> Q ...). */
  children: string[];
  parentId: string | null;

  metadata: Record<string, unknown>;
}

export interface ModelEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
}

export interface ModelConfig {
  modelType: string;
  numLayers: number;
  numHeads: number;
  hiddenSize: number;
  intermediateSize: number;
  vocabSize: number;
  contextLength: number;
  /** Anything architecture-specific that doesn't fit the common fields above. */
  extra: Record<string, unknown>;
}

export interface Model {
  id: string;
  name: string;
  architecture: string;
  config: ModelConfig;

  inputs: TensorSpec[];
  outputs: TensorSpec[];

  /** Flat lookup of every node in the hierarchy, keyed by id. */
  nodes: Record<string, ModelNode>;
  /** Execution-order edges between node ids — the "what flows into what" graph. */
  edges: ModelEdge[];

  rootId: string;
}

// ---------------------------------------------------------------------------
// Weight access — deliberately separate from the graph above. A node knows
// *which* parameters it owns (name/shape/dtype); it does not hold the actual
// bytes. Those are fetched lazily through a WeightProvider, on demand.
// ---------------------------------------------------------------------------

export interface TensorSlice {
  /** Per-dimension [start, end) ranges. Omitted dims are taken in full. */
  ranges?: Array<{ start: number; end: number }>;
}

export interface Tensor {
  shape: number[];
  dtype: string;
  /** Always materialized as float64 for the UI's sake, regardless of source dtype. */
  data: Float64Array;
}

export interface WeightProvider {
  id: string;
  listParameters(): Promise<ParameterRef[]>;
  getParameterInfo(parameterId: string): Promise<ParameterRef>;
  loadTensor(parameterId: string, options?: TensorSlice): Promise<Tensor>;
}

// ---------------------------------------------------------------------------
// Model adapters — the only thing that has to change to support a new
// architecture. Everything above and everything in the UI stays fixed.
// ---------------------------------------------------------------------------

export type ModelSource =
  | { kind: "huggingface"; repo: string; revision?: string }
  | { kind: "local"; name: string; files: Record<string, ArrayBuffer> };

/** A human-readable label for a source — the HF repo id, or the display name chosen when the local files were picked. */
export function modelSourceLabel(source: ModelSource): string {
  return source.kind === "huggingface" ? source.repo : source.name;
}

export interface ModelMetadata {
  architecture: string;
  config: ModelConfig;
  /** name -> {shape, dtype}, taken from the weight file's header, no tensor data loaded. */
  weightIndex: Record<string, { shape: number[]; dtype: string }>;
  source: ModelSource;
  /** Raw bytes backing the WeightProvider this metadata will build, if already fetched. */
  weightsBuffer?: ArrayBuffer;
  /**
   * Set when this checkpoint's real weight bytes were deliberately never
   * downloaded — either because the checkpoint is too large (see
   * hf-client's STRUCTURE_ONLY_THRESHOLD_BYTES) or sharded (this app's
   * eager loader only ever fetches a single model.safetensors, so a
   * sharded checkpoint of any size has no eager path). `weightIndex` still
   * has every real tensor's true shape/dtype in this case — only the
   * actual numbers are synthetic — so the architecture graph is completely
   * real; getWeightProvider() must return a SyntheticWeightProvider
   * instead of a SafetensorsWeightProvider when this is true.
   */
  structureOnly?: boolean;
}

/**
 * The result of an actual forward pass — Mode B from the project notes
 * (Runtime -> Input -> Forward pass -> Activations -> Visualization), kept
 * fully separate from Mode A (static architecture/weight visualization,
 * which needs none of this).
 */
export interface ActivationCapture {
  tokenIds: number[];
  tokens: string[];
  /** nodeId -> the tensor that actually flowed through that node during this forward pass. */
  activations: Record<string, Tensor>;
  /** nodeId -> per-head attention weights [numHeads, queryPos, keyPos], softmax already applied. Attention nodes only. */
  attentionWeights: Record<string, Tensor>;
  /** [sequence_length, vocabSize] */
  logits: Tensor;
}

/**
 * A single edit to the forward pass, applied at the point a node's
 * activation is computed and threaded through everything downstream — this
 * is the mechanism behind ablation, cross-prompt activation patching, and
 * occlusion-based attribution (they're all just different Interventions).
 *
 * Deliberately simpler than "arbitrary tensor surgery": every op targets one
 * node's activation, optionally narrowed to one token position and/or one
 * attention head. That covers "zero this component", "zero this attention
 * head", and "replace this activation with the one captured from a
 * different run" — the three things the interpretability workflows in this
 * app actually need — without a general tensor-patching DSL.
 */
export interface Intervention {
  /** Which node's activation to intervene on. */
  nodeId: string;
  operation: "zero" | "zero_head" | "scale" | "replace";
  /** Restrict to one sequence position; omitted = every position. */
  tokenIndex?: number;
  /** Required for "zero_head" — only meaningful on an "attention" node. */
  headIndex?: number;
  /** Required for "scale". */
  scale?: number;
  /** Required for "replace" — typically another run's activation for the same nodeId. */
  replacementValue?: Tensor;
}

/**
 * Reported during `loadMetadata` so the UI can show something more useful
 * than an inert spinner. `loadedBytes`/`totalBytes` are only meaningful for
 * the phases that actually stream bytes over the network ("weights",
 * "tokenizer") — `totalBytes` itself is further only known when the server
 * sends a Content-Length header, which HF's CDN does for these files but
 * isn't guaranteed in general, so consumers must treat it as optional.
 */
export interface LoadProgress {
  phase: "config" | "structure" | "weights" | "parsing" | "building" | "tokenizer";
  loadedBytes?: number;
  totalBytes?: number;
}

export interface ModelAdapter {
  id: string;
  displayName: string;
  canLoad(source: ModelSource, metadata?: { architectures?: string[]; model_type?: string }): boolean;
  loadMetadata(source: ModelSource, onProgress?: (progress: LoadProgress) => void): Promise<ModelMetadata>;
  buildGraph(metadata: ModelMetadata): Model;
  getWeightProvider(metadata: ModelMetadata): WeightProvider;
  /**
   * Optional: runs a real forward pass over the weights this adapter's
   * WeightProvider exposes, capturing intermediate tensors. Computing actual
   * model math is inherently architecture-specific — unlike everything else
   * in this interface, there's no way to make this generic — so an adapter
   * that only supports Mode A (static visualization) can simply omit it.
   *
   * `interventions`, when given, are applied as each targeted node's
   * activation is computed, so everything downstream sees the edited value
   * — this is what makes ablation/patching an actual re-execution rather
   * than a cosmetic overlay on the original run's numbers.
   */
  runInference?(model: Model, weightProvider: WeightProvider, tokenIds: number[], interventions?: Intervention[]): Promise<ActivationCapture>;
}

// ---------------------------------------------------------------------------
// Small helpers shared by every adapter/consumer.
// ---------------------------------------------------------------------------

/**
 * Sums each *underlying* weight-file tensor exactly once. Necessary because
 * several nodes can share one tensor via `slice` (e.g. GPT-2's fused c_attn
 * backs its Q, K, and V projection nodes) — naively summing every node's
 * ParameterRef would count that tensor three times over.
 */
function uniqueParameters(model: Model): ParameterRef[] {
  const seen = new Map<string, ParameterRef>();
  for (const node of Object.values(model.nodes)) {
    for (const p of node.parameters) if (!seen.has(p.name)) seen.set(p.name, p);
  }
  return [...seen.values()];
}

export function totalParameterCount(model: Model): number {
  return uniqueParameters(model).reduce((sum, p) => sum + p.numElements, 0);
}

export function totalParameterBytes(model: Model): number {
  return uniqueParameters(model).reduce((sum, p) => sum + p.bytes, 0);
}

export function getChildren(model: Model, nodeId: string): ModelNode[] {
  const node = model.nodes[nodeId];
  if (!node) return [];
  return node.children.map((id) => model.nodes[id]).filter((n): n is ModelNode => !!n);
}

export function dtypeSize(dtype: string): number {
  switch (dtype) {
    case "F64":
    case "I64":
      return 8;
    case "F32":
    case "I32":
      return 4;
    case "F16":
    case "BF16":
    case "I16":
      return 2;
    case "I8":
    case "U8":
    case "BOOL":
      return 1;
    default:
      return 4;
  }
}

export function numElements(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}
