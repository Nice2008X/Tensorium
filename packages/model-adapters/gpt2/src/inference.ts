import type { ActivationCapture, Intervention, Model, WeightProvider } from "@tensorium/model-ir";
import {
  addMatrices,
  applyActivation,
  applyHeadIntervention,
  applyInterventions,
  causalSelfAttention,
  embed,
  layerNorm,
  linear,
  matrixToTensor,
  tensorToMatrix,
  tensorToVector,
  type Matrix,
} from "@tensorium/nn-ops";

export async function runInference(model: Model, weightProvider: WeightProvider, tokenIds: number[], interventions?: Intervention[]): Promise<ActivationCapture> {
  const cfg = model.config;
  const numHeads = cfg.numHeads;
  const headDim = cfg.hiddenSize / numHeads;
  const eps = Number(cfg.extra.layerNormEpsilon ?? 1e-5);
  const activationKind = String(cfg.extra.activationFunction ?? "gelu_new");

  const activations: ActivationCapture["activations"] = {};
  const attentionWeights: ActivationCapture["attentionWeights"] = {};
  // Token IDs are real per-run data too (the tokenizer's output for this
  // exact prompt) — capturing them under the root "input" node id lets the
  // Token/Positional Embedding nodes' "input" view show real values instead
  // of reporting nothing captured, since "input" otherwise never appears on
  // the left side of a `record()` call below.
  activations["input"] = matrixToTensor(
    tokenIds.map((id) => [id]),
    "I32"
  );

  const loadMatrix = async (name: string): Promise<Matrix> => tensorToMatrix(await weightProvider.loadTensor(name));
  const loadVector = async (name: string): Promise<number[]> => tensorToVector(await weightProvider.loadTensor(name));
  // Applies any Interventions targeting this node, records the (possibly
  // edited) result, and returns it — every call site below uses the
  // *returned* value downstream, which is what makes an intervention an
  // actual re-execution rather than a display-only overlay.
  const record = (nodeId: string, m: Matrix): Matrix => {
    const patched = applyInterventions(nodeId, m, interventions);
    activations[nodeId] = matrixToTensor(patched);
    return patched;
  };

  const wte = await loadMatrix("transformer.wte.weight");
  const wpe = await loadMatrix("transformer.wpe.weight");
  const tokenEmbed = record("wte", embed(tokenIds, wte));
  const posEmbed = record("wpe", tokenIds.map((_, pos) => wpe[pos]));
  let x = addMatrices(tokenEmbed, posEmbed);

  for (let i = 0; i < cfg.numLayers; i++) {
    const b = `block.${i}`;
    const blockInput = x;

    const ln1g = await loadVector(`transformer.h.${i}.ln_1.weight`);
    const ln1b = await loadVector(`transformer.h.${i}.ln_1.bias`);
    const ln1Out = record(`${b}.ln1`, layerNorm(x, ln1g, ln1b, eps));

    const cAttnW = await loadMatrix(`transformer.h.${i}.attn.c_attn.weight`); // [H, 3H], in_out (Conv1D)
    const cAttnB = await loadVector(`transformer.h.${i}.attn.c_attn.bias`);
    const H = cfg.hiddenSize;
    const qW = cAttnW.map((row) => row.slice(0, H));
    const kW = cAttnW.map((row) => row.slice(H, 2 * H));
    const vW = cAttnW.map((row) => row.slice(2 * H, 3 * H));
    const qB = cAttnB.slice(0, H);
    const kB = cAttnB.slice(H, 2 * H);
    const vB = cAttnB.slice(2 * H, 3 * H);

    const q = record(`${b}.attn.q`, linear(ln1Out, qW, qB, "in_out"));
    const k = record(`${b}.attn.k`, linear(ln1Out, kW, kB, "in_out"));
    const v = record(`${b}.attn.v`, linear(ln1Out, vW, vB, "in_out"));

    const { output: attnHeadsRaw, attentionWeights: headWeights } = causalSelfAttention(q, k, v, numHeads, numHeads, headDim);
    const attnRaw = applyHeadIntervention(`${b}.attn`, attnHeadsRaw, interventions, headDim);
    attentionWeights[`${b}.attn`] = headsToTensor(headWeights);

    const cProjW = await loadMatrix(`transformer.h.${i}.attn.c_proj.weight`);
    const cProjB = await loadVector(`transformer.h.${i}.attn.c_proj.bias`);
    const attnOutRaw = linear(attnRaw, cProjW, cProjB, "in_out");
    const attnProjected = record(`${b}.attn.out`, attnOutRaw);
    // "block.N.attn" is the container users actually click in the tree/graph
    // (its leaf children aren't shown there — see graph.ts) so an
    // intervention targeting it must also flow downstream, not just get
    // recorded for display.
    const attnOut = record(`${b}.attn`, attnProjected);

    const res1 = record(`${b}.res1`, addMatrices(attnOut, blockInput));

    const ln2g = await loadVector(`transformer.h.${i}.ln_2.weight`);
    const ln2b = await loadVector(`transformer.h.${i}.ln_2.bias`);
    const ln2Out = record(`${b}.ln2`, layerNorm(res1, ln2g, ln2b, eps));

    const fcW = await loadMatrix(`transformer.h.${i}.mlp.c_fc.weight`);
    const fcB = await loadVector(`transformer.h.${i}.mlp.c_fc.bias`);
    const fcOut = record(`${b}.ffn.fc`, linear(ln2Out, fcW, fcB, "in_out"));

    const actOut = record(`${b}.ffn.act`, applyActivation(fcOut, activationKind));

    const projW = await loadMatrix(`transformer.h.${i}.mlp.c_proj.weight`);
    const projB = await loadVector(`transformer.h.${i}.mlp.c_proj.bias`);
    const ffnProjected = record(`${b}.ffn.proj`, linear(actOut, projW, projB, "in_out"));
    const ffnOut = record(`${b}.ffn`, ffnProjected);

    const res2 = record(`${b}.res2`, addMatrices(ffnOut, res1));
    // Same reasoning as the attn/ffn aliases above: "block.N" is the node
    // users actually click at the architecture level, so it must be a real
    // intervention point that feeds block N+1, not a discarded display copy.
    x = record(b, res2);
  }

  const lnfg = await loadVector("transformer.ln_f.weight");
  const lnfb = await loadVector("transformer.ln_f.bias");
  const lnfOut = record("ln_f", layerNorm(x, lnfg, lnfb, eps));

  const lmHeadRef = model.nodes["lm_head"].parameters[0];
  const lmHeadW = await loadMatrix(lmHeadRef.name); // [vocab, hidden], nn.Linear layout (out_in)
  const logits = record("lm_head", linear(lnfOut, lmHeadW, null, "out_in"));

  return {
    tokenIds,
    tokens: tokenIds.map((id) => `#${id}`), // adapter-agnostic placeholder; the app fills in real strings via the tokenizer
    activations,
    attentionWeights,
    logits: matrixToTensor(logits),
  };
}

function headsToTensor(headWeights: number[][][]): import("@tensorium/model-ir").Tensor {
  const numHeads = headWeights.length;
  const S = headWeights[0]?.length ?? 0;
  const data = new Float64Array(numHeads * S * S);
  let idx = 0;
  for (let h = 0; h < numHeads; h++) {
    for (let i = 0; i < S; i++) {
      for (let j = 0; j < S; j++) {
        const w = headWeights[h][i][j];
        data[idx++] = Number.isFinite(w) ? w : 0;
      }
    }
  }
  return { shape: [numHeads, S, S], dtype: "F32", data };
}
