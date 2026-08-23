import type { ModelAdapter } from "@tensorium/model-ir";
import { GPT2Adapter } from "@tensorium/adapter-gpt2";
import { LlamaAdapter } from "@tensorium/adapter-llama";
import { MistralAdapter } from "@tensorium/adapter-mistral";
import { GemmaAdapter } from "@tensorium/adapter-gemma";
import { QwenAdapter } from "@tensorium/adapter-qwen";
import { Qwen3Adapter } from "@tensorium/adapter-qwen3";
import { PhiAdapter } from "@tensorium/adapter-phi";
import { Glm4Adapter } from "@tensorium/adapter-glm4";
import { OlmoAdapter } from "@tensorium/adapter-olmo";
import { QwenMoeAdapter } from "@tensorium/adapter-qwen-moe";
import { Qwen3MoeAdapter } from "@tensorium/adapter-qwen3-moe";
import { DeepseekV2Adapter } from "@tensorium/adapter-deepseek-v2";

/**
 * Every architecture the explorer supports. Adding a new one means writing
 * a new adapter package and adding it here — nothing else in the app
 * changes. Mistral, Gemma, Qwen2, Qwen3, and Phi-3/4 are all thin wrappers
 * around `@tensorium/adapter-llama-family` (same graph shape and
 * forward pass as Llama, parameterized by their real differences — GQA
 * ratio for Mistral; explicit head_dim, a (1+weight) RMSNorm, and
 * embedding scaling for Gemma; a bias on Q/K/V projections for Qwen2; a
 * per-head QK-Norm for Qwen3; fused Q/K/V and gate/up projections for
 * Phi; a sandwich norm (extra post-sub-layer RMSNorm) and partial rotary
 * (only a leading slice of each head gets RoPE) for GLM-4; a non-parametric
 * LayerNorm (no learnable weight or bias) and an optional Q/K/V clamp for
 * OLMo; a sparse Mixture-of-Experts FFN (a router picks a few experts per
 * token instead of running one dense FFN on every token — every layer for
 * Qwen2-MoE, every other layer for Qwen3-MoE, which keeps its dense layers
 * as plain gated FFNs) for Qwen2-MoE/Qwen3-MoE — rather than separate
 * copies of ~400 lines each). DeepSeek-V2(-Lite) is the one exception that
 * gets its own adapter package instead of another llama-family flag: its
 * Multi-head Latent Attention (K/V reconstructed from one shared low-rank
 * latent, only part of each head gets RoPE) and DeepSeekMoE (fine-grained
 * experts, unconditional always-on shared experts, optional group-limited
 * routing) are structurally different enough from GQA + Qwen-style MoE to
 * need their own graph/inference modules, the same way GPT-2 does.
 */
export const ADAPTERS: ModelAdapter[] = [
  GPT2Adapter,
  LlamaAdapter,
  MistralAdapter,
  GemmaAdapter,
  QwenAdapter,
  Qwen3Adapter,
  PhiAdapter,
  Glm4Adapter,
  OlmoAdapter,
  QwenMoeAdapter,
  Qwen3MoeAdapter,
  DeepseekV2Adapter,
];

// NOTE: for a checkpoint small enough to fit comfortably in a browser tab
// (a few hundred KB to a few MB — every preset below except the "isLarge"
// ones), the WeightProvider still just downloads the whole safetensors file
// up front; simplest thing that works at this size. Above ~3GB, or for any
// sharded checkpoint (multiple `model-NNNNN-of-MMMMM.safetensors` files),
// loadSafetensorsMetadata instead reads only the checkpoint's structure —
// every real tensor's name/shape/dtype — via small HTTP Range requests
// (see hf-client's fetchModelStructure), and getWeightProvider() hands back
// a SyntheticWeightProvider that fabricates tensor values on demand rather
// than ever downloading the real (multi-gigabyte) ones. No backend
// involved — every fetch here is a direct browser -> huggingface.co request.
export const PRESET_MODELS = [
  { repo: "hf-internal-testing/tiny-random-gpt2", label: "GPT-2 · tiny-random-gpt2 (5 layers, 4 heads, hidden=32)", isMoE: false, isLarge: false },
  { repo: "hf-internal-testing/tiny-random-LlamaForCausalLM", label: "Llama · tiny-random-LlamaForCausalLM (2 layers, 4 heads, hidden=16)", isMoE: false, isLarge: false },
  { repo: "yujiepan/mistral-tiny-random", label: "Mistral · mistral-tiny-random (2 layers, GQA 4:2 heads, hidden=8)", isMoE: false, isLarge: false },
  { repo: "fxmarty/tiny-random-GemmaForCausalLM", label: "Gemma · tiny-random-GemmaForCausalLM (1 layer, 2 heads, hidden=32)", isMoE: false, isLarge: false },
  { repo: "yujiepan/qwen2-tiny-random", label: "Qwen2 · qwen2-tiny-random (2 layers, GQA 4:2 heads, Q/K/V bias)", isMoE: false, isLarge: false },
  { repo: "tiny-random/qwen3", label: "Qwen3 · tiny-random/qwen3 (2 layers, GQA 2:1 heads, QK-Norm)", isMoE: false, isLarge: false },
  { repo: "tiny-random/phi-4", label: "Phi-4 · tiny-random/phi-4 (2 layers, GQA 2:1 heads, fused QKV + gate/up)", isMoE: false, isLarge: false },
  { repo: "tiny-random/glm-4", label: "GLM-4 · tiny-random/glm-4 (2 layers, sandwich norm, partial rotary)", isMoE: false, isLarge: false },
  { repo: "katuni4ka/tiny-random-olmo-hf", label: "OLMo · tiny-random-olmo-hf (2 layers, 2 heads, hidden=64, non-parametric LayerNorm)", isMoE: false, isLarge: false },
  { repo: "katuni4ka/tiny-random-qwen1.5-moe", label: "Qwen2-MoE · tiny-random-qwen1.5-moe (4 layers, 8 experts, top-4 + shared expert)", isMoE: true, isLarge: false },
  { repo: "tiny-random/qwen3-moe", label: "Qwen3-MoE · tiny-random/qwen3-moe (2 layers, 1 dense + 1 MoE, QK-Norm, top-2 of 8)", isMoE: true, isLarge: false },
  {
    repo: "yujiepan/deepseek-v2-0628-tiny-random",
    label: "DeepSeek-V2 · deepseek-v2-0628-tiny-random (2 layers, 1 dense + 1 MoE, MLA, group-limited routing)",
    isMoE: true,
    isLarge: false,
  },
  // Real, full-size checkpoints — structure-only (see the note above): the
  // architecture graph and every tensor's true shape/dtype are exact, but
  // no real weight bytes are ever downloaded, so a forward pass on these
  // runs against synthetic random values, not the model's real behavior.
  {
    repo: "Qwen/Qwen2.5-3B-Instruct",
    label: "Qwen2.5-3B-Instruct (real) · 36 layers, GQA 16:2 heads, sharded, 5.75 GB",
    isMoE: false,
    isLarge: true,
  },
  {
    repo: "Qwen/Qwen2.5-7B-Instruct",
    label: "Qwen2.5-7B-Instruct (real) · 28 layers, GQA 28:4 heads, sharded, 14.2 GB",
    isMoE: false,
    isLarge: true,
  },
  {
    repo: "deepseek-ai/DeepSeek-V2-Lite",
    label: "DeepSeek-V2-Lite (real) · 27 layers, MLA, 64-expert DeepSeekMoE, sharded, 29.3 GB",
    isMoE: true,
    isLarge: true,
  },
];

// NOTE on DeepSeek LLM: architecturally it's plain Llama (LlamaAdapter loads
// it with zero extra code — verified against yujiepan/deepseek-llm-tiny-random's
// real config and safetensors), so any DeepSeek-LLM checkpoint with sane
// dimensions works today by typing its repo id into the loader directly.
// That specific tiny-random fixture isn't listed as a preset chip, though:
// its hidden_size=2 / num_attention_heads=2 gives head_dim=1, which is odd —
// RoPE rotates head_dim in (x, y) pairs and has no defined behavior for an
// odd dimension (the same wall a real PyTorch RoPE implementation would hit),
// so it fails to load every time with a clear error rather than silently
// producing NaN. Not worth a one-click preset that's guaranteed to fail.
