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

/**
 * safetensors layout: [8-byte LE header length][UTF-8 JSON header][raw tensor bytes].
 * Header data_offsets are relative to the byte right after the header.
 */
export function parseSafetensorsHeader(buffer: WeightsBuffer): SafetensorsFile {
  const lengthBytes = readBytes(buffer, 0, 8);
  const headerLength = Number(new DataView(lengthBytes.buffer, lengthBytes.byteOffset, 8).getBigUint64(0, true));
  const headerBytes = readBytes(buffer, 8, headerLength);
  const headerJson = new TextDecoder("utf-8").decode(headerBytes);
  const raw = JSON.parse(headerJson) as Record<string, SafetensorsEntry | Record<string, unknown>>;

  const header: Record<string, SafetensorsEntry> = {};
  for (const [name, entry] of Object.entries(raw)) {
    if (name === "__metadata__") continue;
    header[name] = entry as SafetensorsEntry;
  }

  return { header, dataStart: 8 + headerLength, buffer };
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
