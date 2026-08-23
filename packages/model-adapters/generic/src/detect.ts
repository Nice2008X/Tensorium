import type { ModelMetadata } from "@tensorium/model-ir";
import type { LlamaFamilyOptions, LlamaFamilyRawConfig } from "@tensorium/adapter-llama-family";

type WeightIndex = ModelMetadata["weightIndex"];

/**
 * Every option `adapter-llama-family`'s named wrappers (Llama, Mistral,
 * Qwen2/3, Phi, GLM-4, OLMo, Qwen2/3-MoE, ...) currently hand-code is, with
 * two exceptions, directly visible in the checkpoint's own weight names —
 * a wrapper adapter only exists today because *something* has to state
 * these before the checkpoint is fetched, and a human is the simplest
 * something. This reads the same facts out of the already-fetched
 * `weightIndex` instead, off layer 0 (every layer in a real checkpoint
 * shares the same shape family, so one layer is representative).
 *
 * The two exceptions — Gemma's `(1 + weight)` RMSNorm scaling and its
 * √hidden_size embedding scaling — are pure modeling-code convention with
 * no trace in either config.json or the weight names/shapes, so they can't
 * be detected at all; this always assumes the standard (non-Gemma)
 * behavior. In practice this rarely matters: GemmaAdapter's own canLoad()
 * already claims anything actually named "gemma", so this function only
 * ever runs for checkpoints no named adapter recognized in the first
 * place — a *new* checkpoint happening to need Gemma's specific quirks
 * without also being identifiable as Gemma is the unlikely case this
 * can't cover, not the common one.
 */
export function detectLlamaFamilyOptions(weightIndex: WeightIndex, rawConfig: LlamaFamilyRawConfig): LlamaFamilyOptions {
  const L = "model.layers.0";
  const has = (name: string) => !!weightIndex[name];

  const fusedQkv = has(`${L}.self_attn.qkv_proj.weight`);
  const fusedGateUp = has(`${L}.mlp.gate_up_proj.weight`);
  const qkvBias = !fusedQkv && has(`${L}.self_attn.q_proj.bias`);
  const qkNorm = has(`${L}.self_attn.q_norm.weight`);
  const sandwichNorm = has(`${L}.post_self_attn_layernorm.weight`);
  const moe = has(`${L}.mlp.gate.weight`) || has(`${L}.mlp.experts.0.gate_proj.weight`);
  // A checkpoint with a genuinely parameter-free norm (OLMo v1's variant)
  // simply has no input_layernorm.weight tensor at all in the safetensors
  // file — confirmed against a real OLMo checkpoint, not assumed.
  const normType = has(`${L}.input_layernorm.weight`) ? "rmsnorm" : "layernorm_no_affine";

  return {
    defaultModelType: rawConfig.model_type ?? "generic",
    qkvBias,
    qkNorm,
    fusedQkv,
    fusedGateUp,
    sandwichNorm,
    normType,
    moe,
    rmsNormVariant: "standard",
    embeddingScale: "none",
  };
}

/**
 * The minimum a checkpoint needs to plausibly be "a llama-shaped decoder
 * block" at all — some form of Q/K/V attention projection (fused or not)
 * and some form of gated-MLP-or-MoE feed-forward. A checkpoint that fails
 * this isn't just missing a quirk this adapter doesn't know about; it's a
 * different kind of model entirely (an encoder, a diffusion U-Net, a vision
 * tower, ...) that no amount of flag-guessing will make loadable here.
 */
export function looksLikeLlamaFamilyBlock(weightIndex: WeightIndex): boolean {
  const L = "model.layers.0";
  const has = (name: string) => !!weightIndex[name];
  const hasAttention = has(`${L}.self_attn.qkv_proj.weight`) || has(`${L}.self_attn.q_proj.weight`);
  const hasFfn = has(`${L}.mlp.gate_up_proj.weight`) || has(`${L}.mlp.gate_proj.weight`) || has(`${L}.mlp.gate.weight`) || has(`${L}.mlp.experts.0.gate_proj.weight`);
  return hasAttention && hasFfn;
}
