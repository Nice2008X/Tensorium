import { readBytes, type Tensor, type TensorSlice, type WeightsBuffer } from "@tensorium/model-ir";

export interface SafetensorsEntry {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

export interface SafetensorsFile {
  header: Record<string, SafetensorsEntry>;
  /** Byte offset (within the original ArrayBuffer) where tensor data begins. */
  dataStart: number;
  buffer: WeightsBuffer;
}

/** Same ceiling the reference safetensors implementation enforces — a header claiming more is corrupt or hostile, and would otherwise be read into memory whole. */
export const MAX_SAFETENSORS_HEADER_BYTES = 100_000_000;
const MAX_TENSOR_RANK = 16;
const MAX_TENSOR_NAME_LENGTH = 4096;
const DTYPE_PATTERN = /^[A-Za-z0-9_]{1,16}$/;
const CONTROL_CHARS = /[\u0000-\u001f\u007f]/;

/** Element sizes for the dtypes whose `data_offsets` span can be cross-checked against `shape`; an unrecognized dtype is still accepted (structure-only viewing works for any), just not size-checked. */
const KNOWN_DTYPE_BYTES: Record<string, number> = {
  F64: 8, I64: 8, U64: 8, F32: 4, I32: 4, U32: 4, F16: 2, BF16: 2, I16: 2, U16: 2, I8: 1, U8: 1, BOOL: 1, F8_E4M3: 1, F8_E5M2: 1,
};

/** Reads the 8-byte little-endian header length and rejects an absurd one *before* anything tries to allocate or fetch that many bytes. */
export function readSafetensorsHeaderLength(prefix: Uint8Array): number {
  if (prefix.byteLength < 8) throw new Error("Not a safetensors file: shorter than its 8-byte header-length prefix.");
  const length = new DataView(prefix.buffer, prefix.byteOffset, 8).getBigUint64(0, true);
  if (length < 2n || length > BigInt(MAX_SAFETENSORS_HEADER_BYTES)) {
    throw new Error(`Not a valid safetensors file: its header claims ${length} bytes (expected 2 to ${MAX_SAFETENSORS_HEADER_BYTES}).`);
  }
  return Number(length);
}

/**
 * Decodes and strictly validates a safetensors JSON header — the only
 * gatekeeper between an untrusted file (a local upload or any Hugging Face
 * repo's checkpoint) and every consumer of tensor names/shapes/offsets.
 * Rejects anything that isn't a plain object of well-formed entries, a
 * `__proto__` tensor name (assigning it would swap the header's prototype
 * instead of adding an entry), non-integer/negative/oversized shapes,
 * offsets that don't match the shape's byte size or run past the data
 * region, and non-UTF-8 text. `dataLength` is the size of the tensor-data
 * region when the whole file is known; omit it when only the header itself
 * was fetched.
 */
export function decodeSafetensorsHeader(headerBytes: Uint8Array, dataLength?: number): Record<string, SafetensorsEntry> {
  let raw: unknown;
  try {
    raw = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(headerBytes));
  } catch {
    throw new Error("Not a valid safetensors file: its header isn't valid UTF-8 JSON.");
  }
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) throw new Error("Not a valid safetensors file: its header isn't a JSON object.");

  const header: Record<string, SafetensorsEntry> = {};
  for (const [name, entry] of Object.entries(raw)) {
    if (name === "__metadata__") continue;
    const fail = (why: string): never => {
      throw new Error(`Invalid safetensors header: tensor "${name.slice(0, 80)}" ${why}.`);
    };
    if (name === "__proto__" || name.length === 0 || name.length > MAX_TENSOR_NAME_LENGTH || CONTROL_CHARS.test(name)) fail("has an unacceptable name");
    if (typeof entry !== "object" || entry === null || Array.isArray(entry)) fail("isn't an object");
    const { dtype, shape, data_offsets: offsets } = entry as Record<string, unknown>;
    if (typeof dtype !== "string" || !DTYPE_PATTERN.test(dtype)) fail("has a malformed dtype");
    if (!Array.isArray(shape) || shape.length > MAX_TENSOR_RANK || !shape.every((d) => Number.isSafeInteger(d) && d >= 0)) fail("has a malformed shape");
    if (!Array.isArray(offsets) || offsets.length !== 2 || !offsets.every((o) => Number.isSafeInteger(o) && o >= 0) || offsets[0] > offsets[1]) fail("has malformed data_offsets");

    const dims = shape as number[];
    const [begin, end] = offsets as [number, number];
    let elements = 1;
    for (const d of dims) {
      elements *= d;
      if (!Number.isSafeInteger(elements)) fail("has an overflowing shape");
    }
    const elementBytes = KNOWN_DTYPE_BYTES[dtype as string];
    if (elementBytes !== undefined && end - begin !== elements * elementBytes) fail("has data_offsets that don't match its shape and dtype");
    if (dataLength !== undefined && end > dataLength) fail("points past the end of the file");
    header[name] = { dtype: dtype as string, shape: dims, data_offsets: [begin, end] };
  }
  return header;
}

/**
 * safetensors layout: [8-byte LE header length][UTF-8 JSON header][raw tensor bytes].
 * Header data_offsets are relative to the byte right after the header.
 * Pass `headerOnly` when `buffer` holds just the header (a structure-only
 * Range fetch), so tensor offsets aren't bounds-checked against bytes that
 * were deliberately never downloaded.
 */
export function parseSafetensorsHeader(buffer: WeightsBuffer, options: { headerOnly?: boolean } = {}): SafetensorsFile {
  const headerLength = readSafetensorsHeaderLength(readBytes(buffer, 0, Math.min(8, buffer.byteLength)));
  if (8 + headerLength > buffer.byteLength) throw new Error("Not a valid safetensors file: it ends before its header does (truncated).");
  const dataStart = 8 + headerLength;
  const header = decodeSafetensorsHeader(readBytes(buffer, 8, headerLength), options.headerOnly ? undefined : buffer.byteLength - dataStart);
  return { header, dataStart, buffer };
}

function bytesPerElement(dtype: string): number {
  switch (dtype) {
    case "F64":
    case "I64":
    case "U64":
      return 8;
    case "F32":
    case "I32":
    case "U32":
      return 4;
    case "F16":
    case "BF16":
    case "I16":
    case "U16":
      return 2;
    case "I8":
    case "U8":
    case "BOOL":
      return 1;
    default:
      throw new Error(`Unsupported dtype: ${dtype}`);
  }
}

function decodeFloat16(bits: number): number {
  const sign = (bits & 0x8000) >> 15;
  const exponent = (bits & 0x7c00) >> 10;
  const fraction = bits & 0x03ff;
  let value: number;
  if (exponent === 0) {
    value = fraction * Math.pow(2, -24);
  } else if (exponent === 0x1f) {
    value = fraction ? NaN : Infinity;
  } else {
    value = (1 + fraction / 1024) * Math.pow(2, exponent - 15);
  }
  return sign ? -value : value;
}

function decodeBFloat16(bits: number): number {
  // bfloat16 is just the top 16 bits of a float32.
  const f32Bits = bits << 16;
  const buf = new ArrayBuffer(4);
  new Uint32Array(buf)[0] = f32Bits;
  return new Float32Array(buf)[0];
}

/** Reads a single scalar at a flat (row-major) index within one tensor's own data region. */
function readElement(view: DataView, byteOffset: number, dtype: string, flatIndex: number): number {
  const size = bytesPerElement(dtype);
  const at = byteOffset + flatIndex * size;
  switch (dtype) {
    case "F32":
      return view.getFloat32(at, true);
    case "F64":
      return view.getFloat64(at, true);
    case "F16":
      return decodeFloat16(view.getUint16(at, true));
    case "BF16":
      return decodeBFloat16(view.getUint16(at, true));
    case "I64":
      return Number(view.getBigInt64(at, true));
    case "U64":
      return Number(view.getBigUint64(at, true));
    case "I32":
      return view.getInt32(at, true);
    case "U32":
      return view.getUint32(at, true);
    case "I16":
      return view.getInt16(at, true);
    case "U16":
      return view.getUint16(at, true);
    case "I8":
      return view.getInt8(at);
    case "U8":
      return view.getUint8(at);
    case "BOOL":
      return view.getUint8(at) ? 1 : 0;
    default:
      throw new Error(`Unsupported dtype: ${dtype}`);
  }
}

function rowMajorStrides(shape: number[]): number[] {
  const strides = new Array(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

/**
 * Reads a (possibly sliced) tensor out of a safetensors buffer.
 * Never materializes more than the requested slice.
 */
export function readTensor(file: SafetensorsFile, name: string, slice?: TensorSlice): Tensor {
  const entry = file.header[name];
  if (!entry) throw new Error(`Unknown tensor: ${name}`);

  const { shape, dtype } = entry;
  // One DataView over just this tensor's bytes: zero-copy for a plain
  // ArrayBuffer (or a tensor inside one segment), a copy of only this
  // tensor when it straddles a SegmentedBuffer segment boundary.
  const byteOffset = 0;
  const tensorBytes = readBytes(file.buffer, file.dataStart + entry.data_offsets[0], entry.data_offsets[1] - entry.data_offsets[0]);
  const view = new DataView(tensorBytes.buffer, tensorBytes.byteOffset, tensorBytes.byteLength);
  const strides = rowMajorStrides(shape);

  const ranges = shape.map((dimSize, i) => {
    const r = slice?.ranges?.[i];
    const start = Math.max(0, Math.min(r?.start ?? 0, dimSize));
    const end = Math.max(start, Math.min(r?.end ?? dimSize, dimSize));
    return { start, end };
  });
  const outShape = ranges.map((r) => r.end - r.start);
  const total = outShape.reduce((a, b) => a * b, 1);

  const out = new Float64Array(total);
  const idx = new Array(shape.length).fill(0).map((_, i) => ranges[i].start);
  for (let n = 0; n < total; n++) {
    let flat = 0;
    for (let d = 0; d < shape.length; d++) flat += idx[d] * strides[d];
    out[n] = readElement(view, byteOffset, dtype, flat);

    // odometer increment over outShape
    for (let d = shape.length - 1; d >= 0; d--) {
      idx[d]++;
      if (idx[d] < ranges[d].end) break;
      idx[d] = ranges[d].start;
    }
  }

  return { shape: outShape, dtype, data: out };
}
