# Tensorium

**[Live demo →](https://nice2008x.github.io/Tensorium/)**

An interactive, in-browser explorer and debugger for large language model
internals. Point it at a Hugging Face repo that ships `safetensors`
weights for one of the [supported architectures](#supported-architectures)
(GPT-2, Llama, Mistral, Gemma, Gemma 4, Qwen2, Qwen3, Qwen3.5/Qwen3.8,
Phi-3/4, GLM-4, OLMo, Qwen2-MoE, Qwen3-MoE, or DeepSeek-V2) and it parses
the model's real config and weights, renders its architecture as a
navigable graph — laid out by [ELK](https://eclipse.dev/elk/), the same
layered-graph engine used by professional diagramming tools, not an ad
hoc heuristic — runs an actual forward pass in the browser (no backend,
no GPU), and lets you inspect every tensor, watch predictions form layer
by layer, and run causal interventions (ablate a head, patch in an
activation from another prompt) to see what actually drives the model's
output.

Everything runs client-side. Weights are fetched directly from the
Hugging Face CDN and executed with a small dependency-free numeric engine
written in TypeScript. Fetched files are cached in the browser's IndexedDB
(keyed by their exact URL), so reloading the same repo later needs no
network round trip at all — files over 50 MB are never written to that
cache and always come straight from Hugging Face instead.

![Screenshot of Tensorium: DeepSeek-V2's Multi-head Latent Attention internals, ELK-routed, with a scope box grouping the Q/KV down/up-projections and decoupled RoPE, the model tree, the Inspector panel with the attention formula, and the Tensor Explorer's activation heatmap](docs/screenshot.png)

Note: the built-in presets are tiny, randomly-initialized test checkpoints,
not real trained models — predictions won't be coherent. This tool is for
exploring architecture and mechanics, not model quality.

## Features

- **Load any compatible Hugging Face model** by repo id — no upload, no
  server-side processing. A dozen-plus architecture families are
  supported out of the box (see [Supported
  architectures](#supported-architectures)), including a few that need
  real Model IR extensions beyond a Llama-shaped GQA + gated-MLP
  block — Multi-head Latent Attention, a hybrid linear/recurrent
  attention decoder, a per-layer-input-projection design — not just
  another config flag. Alternatively, pick a local `config.json` +
  `.safetensors` (+ optional `tokenizer.json`) straight off disk — each
  file is content-sniffed before loading (not just trusted by extension)
  to catch a mislabeled or corrupt file immediately, with a size warning
  for very large weight files; any structure-only model over 20 GB has
  its synthetic forward pass disabled outright (the option to allow one
  is disabled too), since a forward pass through fabricated random
  weights that size isn't worth the browser memory it costs. A **Save
  model** button downloads the loaded model's exact original bytes back
  out, behind a confirmation dialog that names every file and its size
  before anything downloads.
- **Architecture graph** — the model rendered as a node graph (via [React
  Flow](https://reactflow.dev/) for rendering, [ELK](https://eclipse.dev/elk/)
  for layout — a real layered-graph engine with explicit per-node port
  ordering and real edge routing, not a hand-rolled approximation) at two
  levels of detail: the full architecture, and a double-click-to-expand
  view of a single transformer block's internal wiring (attention
  projections, norms, MLP, residual adds, gates). Opens at a true 100%
  zoom by default (with a live zoom-percentage readout), centered near
  the top of the graph — the on-canvas fit-to-screen button and the `0`
  shortcut zoom out to see the whole thing at once. Selecting a container
  node (e.g. Attention) draws a scope box around its leaf components; a
  graph control can collapse repeated same-type chains (e.g. 5
  transformer blocks) into a single stacked node for a more condensed
  view, and toggle back to the expanded chain on demand. A built-in
  export button renders the full graph (not just the visible viewport)
  to a PNG.
- **Model tree** — a classic collapsible tree view of every module and
  parameter, alongside the graph.
- **Inspector** — click any component for a plain-language explanation of
  what it does, its input/output shapes, its parameters, and (where it can
  be determined unambiguously) an Input Construction breakdown showing the
  math behind how its input was assembled (e.g. token embedding +
  positional embedding).
- **Tensor Explorer** — browse every weight tensor and every activation
  captured from the last forward pass, rendered as a heatmap, a raw
  matrix, or a value histogram. An **Input/Output tab** shows exactly what
  flowed into and out of the selected node for that run — including which
  of several upstream sources for a multi-input node (e.g. a residual add)
  — deep-linked directly from the Inspector's "This run" section. Supports
  windowing into large tensors and side-by-side A/B/diff comparison across
  two prompts.
- **Logit Lens** — project every layer's hidden state through the final
  norm and LM head to watch the model's next-token prediction sharpen (or
  change its mind) layer by layer.
- **Token Attribution** — occlusion-based attribution: mask each input
  token in turn and measure how much the prediction shifts, to see which
  tokens actually mattered.
- **Experiment panel** — causal interventions on a real forward pass:
  zero out a component, zero a single attention head, or patch in an
  activation captured from a second prompt, and see the before/after
  effect on the output distribution.
- **Themes and language** — dark, light, pastel, and sepia themes, plus a
  UI translated into nine languages, both configurable from the settings
  panel (top-right gear icon).
- **Resizable, collapsible layout** — drag the bottom panel's top edge, or
  the model tree's and Inspector's own side edges, to resize them (within
  sane min/max bounds); every panel collapses independently, and every
  size/collapse preference persists across reloads.
- **Status footer** — an always-visible strip showing the current
  run/idle/error state, the selected node's breadcrumb, its real output
  shape and dtype (or the static declared shape before any run), zoom
  level, and a reminder that every computation here runs on the CPU, not a
  GPU. While a forward pass is in flight, it shows real per-layer progress
  (not a simulated animation) — useful on a large structure-only model,
  where each layer's weight load is a genuine, sometimes-slow read.

## Screenshots

<table>
<tr>
<td width="50%">

**Loading a model**
![The loader screen, offering GPT-2/Llama/Mistral/Gemma/Qwen/GLM-4/OLMo presets grouped into dense, MoE, and real-bigger-model tabs](docs/screenshot-loader.png)

</td>
<td width="50%">

**A hybrid linear/recurrent decoder**
Qwen3.5's Gated DeltaNet block — a short causal convolution feeding a per-token recurrent state update, nothing like softmax attention — laid out by ELK alongside the ordinary residual/FFN path.
![Qwen3.5's Gated DeltaNet block: QKV Projection fanning out to a causal convolution and the β/decay gate projections](docs/screenshot-gated-deltanet.png)

</td>
</tr>
<tr>
<td width="50%">

**The full architecture, at a glance**
The top-level graph — real 100% zoom by default, with the current zoom level always readable in the status footer — before diving into any one block's internals.
![Qwen3-MoE's top-level architecture graph: token embedding into two transformer blocks into the LM head](docs/screenshot-architecture-overview.png)

</td>
<td width="50%">

**Logit Lens**
Every layer's hidden state projected through the final norm and LM head, watching the next-token prediction sharpen (or change its mind) layer by layer.
![The Logit Lens panel showing the evolving next-token prediction across the embedding, each transformer block, and the final layer](docs/screenshot-logit-lens.png)

</td>
</tr>
</table>

**See exactly what fed a node, one click away** — the Inspector's "This run" section deep-links straight into Tensor Explorer's Input/Output tab, landing on the right upstream source (here, the previous block's own output, correctly resolved rather than showing the block container's own — different — result) with real captured data, not a placeholder.
![Tensor Explorer's Input/Output tab showing LayerNorm 1's real input tensor, deep-linked from the Inspector, with the status footer visible at the bottom](docs/screenshot-input-output.png)

**Causal interventions** — zero out a component or patch in an activation from a second prompt, and see the effect on the output distribution.
![The Experiment panel with RMSNorm (pre-attention) selected, offering a Zero out (ablate) / Patch-in operation and a token-position scope](docs/screenshot-experiment.png)

## How it works

The core design is a normalized **Model IR** (intermediate representation)
that every architecture is translated into, so the UI never has to know
whether it's looking at GPT-2, Llama, or Qwen — only the small
architecture-specific *adapter* that produced the graph does.

```
Hugging Face repo (config.json + *.safetensors)
        │
        ▼
  Model Adapter            canLoad() picks the right adapter for the repo's
  (per architecture)       model_type / architectures field
        │
        ▼
  Model IR                 Model / ModelNode / ParameterRef / WeightProvider
  (packages/model-ir)      — a normalized graph + lazy tensor access, the
        │                    same shape regardless of source architecture
        ▼
  nn-ops + adapter's        a real forward pass (matmul, RMSNorm/LayerNorm,
  runInference()             RoPE, GQA/MQA attention, SwiGLU/GELU MLPs...),
                              capturing every intermediate activation
        │
        ▼
  React UI                 graph view, tensor explorer, logit lens, token
  (apps/web)                attribution, and intervention experiments — all
                              driven purely by the IR + captured activations
```

A `ParameterRef` can also be a named *slice* of a larger underlying
tensor — the mechanism that lets one shared engine model both
non-fused checkpoints and checkpoints that fuse multiple projections into
one weight (GPT-2's `c_attn`, Phi's `qkv_proj`/`gate_up_proj`) without
special-casing the rest of the pipeline.

Most of the supported architectures (Llama, Mistral, Gemma, Qwen2/2.5,
Qwen3, Phi-3/4, GLM-4, OLMo, Qwen2-MoE, Qwen3-MoE) are RoPE + gated-MLP
"Llama-shaped" models that differ only in a handful of concrete details
(GQA ratio, an explicit `head_dim`, a bias here, a fused projection there,
a non-parametric LayerNorm instead of RMSNorm for OLMo, or — the two MoE
variants' case — swapping the dense FFN for a router plus a bank of
sparsely-activated expert FFNs on some or all layers). Rather than
duplicating the graph-building and forward-pass code per architecture,
they're all thin wrappers around one shared, option-parameterized engine
(`adapter-llama-family`) — see [Adding a new
architecture](#adding-a-new-architecture).

Three architectures are structurally different enough that they get their
own adapter package instead of another `adapter-llama-family` option:
**DeepSeek-V2**'s Multi-head Latent Attention (K/V reconstructed from one
shared low-rank latent, with only part of each head rotated) and
DeepSeekMoE; **Gemma 4**'s alternating sliding/global attention with two
different head_dim/RoPE configurations, per-layer frozen K/V reuse on
some layers, and a per-layer embedding table (text decoder only — its
vision/audio towers aren't loaded); and **Qwen3.5/Qwen3.8**'s hybrid
decoder, where most layers run a linear/recurrent Gated DeltaNet
mechanism (a short causal convolution feeding a per-token recurrent
state update — nothing like softmax attention) interleaved with periodic
ordinary GQA layers (text decoder only here too — its vision tower isn't
loaded).

## Project layout

```
packages/
  model-ir/              Normalized graph types: Model, ModelNode, ParameterRef,
                          WeightProvider, ActivationCapture, Intervention.
  tensor-core/            Safetensors parsing, dtype decoding, tensor statistics,
                          the WeightProvider implementation.
  hf-client/              Shared Hugging Face fetch helpers (config.json + the
                          safetensors header).
  nn-ops/                 Dependency-free numeric primitives for running a real
                          forward pass in JS: matmul, LayerNorm/RMSNorm, RoPE,
                          causal/GQA attention, GELU/SiLU, and intervention
                          application (ablation/patching).
  tokenizer/              From-scratch BPE tokenizer reading Hugging Face's
                          tokenizer.json (handles GPT-2, Llama/SentencePiece-style,
                          and Qwen's tokenizer.json shapes).
  interpretability/       Logit lens and occlusion-based token attribution, built
                          on top of runInference()'s captured activations.
  model-adapters/
    gpt2/                 Architecture-specific: config -> Model IR, weight name
                          mapping, forward pass.
    llama-family/          Shared engine for every Llama-shaped architecture
                          (RoPE, RMSNorm, gated FFN, GQA), parameterized by each
                          architecture's real differences.
    llama/                 Thin wrapper over llama-family.
    mistral/                Thin wrapper over llama-family (real GQA ratios).
    gemma/                  Wrapper with explicit head_dim, a (1+weight) RMSNorm
                          variant, and √hidden_size embedding scaling.
    qwen/                   Wrapper with a bias on Q/K/V projections.
    qwen3/                  Wrapper with QK-Norm (per-head RMSNorm on Q/K before
                          RoPE).
    phi/                    Wrapper with fused qkv_proj and gate_up_proj
                          projections (via ParameterRef slicing).
    glm4/                   Wrapper with a sandwich norm (extra RMSNorm after
                          each sub-layer's output, before the residual add)
                          and partial rotary (RoPE applied to only a leading
                          slice of each head).
    deepseek-v2/            Own adapter: Multi-head Latent Attention (shared
                          low-rank K/V latent, partial RoPE) and DeepSeekMoE
                          (fine-grained experts, always-on shared experts,
                          optional group-limited routing).
    gemma4/                 Own adapter: alternating sliding/global attention,
                          two head_dim/RoPE configurations, per-layer frozen
                          K/V reuse, a per-layer embedding table. Text decoder
                          only — vision/audio towers aren't loaded.
    qwen3-5/                Own adapter: hybrid decoder interleaving a
                          linear/recurrent Gated DeltaNet mechanism with
                          periodic ordinary GQA layers. Text decoder only —
                          the vision tower isn't loaded.
apps/
  web/                    React UI: tree / architecture graph (React Flow for
                          rendering, ELK for layout) / inspector / tensor
                          explorer / inference panel / logit lens / token
                          attribution / experiment panel / settings (themes +
                          language).
```

## Supported architectures

| Architecture | Adapter | Example checkpoint used as a built-in preset |
|---|---|---|
| GPT-2 | `adapter-gpt2` | [`hf-internal-testing/tiny-random-gpt2`](https://huggingface.co/hf-internal-testing/tiny-random-gpt2) |
| Llama | `adapter-llama` | [`hf-internal-testing/tiny-random-LlamaForCausalLM`](https://huggingface.co/hf-internal-testing/tiny-random-LlamaForCausalLM) |
| Mistral | `adapter-mistral` | [`yujiepan/mistral-tiny-random`](https://huggingface.co/yujiepan/mistral-tiny-random) |
| Gemma | `adapter-gemma` | [`fxmarty/tiny-random-GemmaForCausalLM`](https://huggingface.co/fxmarty/tiny-random-GemmaForCausalLM) |
| Qwen2 / 2.5 | `adapter-qwen` | [`yujiepan/qwen2-tiny-random`](https://huggingface.co/yujiepan/qwen2-tiny-random) |
| Qwen3 | `adapter-qwen3` | [`tiny-random/qwen3`](https://huggingface.co/tiny-random/qwen3) |
| Phi-3 / Phi-4 | `adapter-phi` | [`tiny-random/phi-4`](https://huggingface.co/tiny-random/phi-4) |
| GLM-4 | `adapter-glm4` | [`tiny-random/glm-4`](https://huggingface.co/tiny-random/glm-4) |
| OLMo | `adapter-olmo` | [`katuni4ka/tiny-random-olmo-hf`](https://huggingface.co/katuni4ka/tiny-random-olmo-hf) |
| Qwen2-MoE | `adapter-qwen-moe` | [`katuni4ka/tiny-random-qwen1.5-moe`](https://huggingface.co/katuni4ka/tiny-random-qwen1.5-moe) |
| Qwen3-MoE | `adapter-qwen3-moe` | [`tiny-random/qwen3-moe`](https://huggingface.co/tiny-random/qwen3-moe) |
| DeepSeek-V2 (MLA + DeepSeekMoE) | `adapter-deepseek-v2` | [`yujiepan/deepseek-v2-0628-tiny-random`](https://huggingface.co/yujiepan/deepseek-v2-0628-tiny-random) |
| Gemma 4 (text decoder only) | `adapter-gemma4` | [`google/gemma-4-E2B`](https://huggingface.co/google/gemma-4-E2B) (real, structure-only) |
| Qwen3.5 / Qwen3.8 (text decoder only) | `adapter-qwen3-5` | [`tiny-random/qwen3.5`](https://huggingface.co/tiny-random/qwen3.5) |

These are all deliberately tiny (randomly-initialized, few-layer) test
checkpoints, chosen so the full model can be loaded and explored instantly
in a browser tab. Any other repo with the same `model_type` and a
`model.safetensors` file will work too — type its repo id into the loader
instead of picking a preset. DeepSeek-LLM, for example, needs no adapter of
its own: its `config.json` reports `model_type: "llama"` (it predates
DeepSeek's MoE/latent-attention architectures), so `adapter-llama` already
loads it — see
[`yujiepan/deepseek-llm-tiny-random`](https://huggingface.co/yujiepan/deepseek-llm-tiny-random).

Sparse Mixture-of-Experts is supported (Qwen2-MoE, Qwen3-MoE, and
DeepSeek-V2's DeepSeekMoE: a router, a bank of expert FFNs, and — for
checkpoints that have one — an always-on shared expert, all real graph
nodes with a real forward pass, not a placeholder; a checkpoint doesn't
even have to route every layer — Qwen3-MoE's
`decoder_sparse_step`/`mlp_only_layers` fields, honored exactly as
config.json states them, leave some layers as plain dense FFNs).
DeepSeek-V2's Multi-head Latent Attention and Qwen3.5/Qwen3.8's hybrid
linear/recurrent attention decoder are both supported too, as real Model
IR extensions rather than another `adapter-llama-family` flag.

For a genuinely multimodal checkpoint (Gemma 4, Qwen3.5/Qwen3.8), only
the text decoder is loaded — the vision (and, for Gemma 4, audio) towers
are parsed out of the checkpoint's structure but never rendered or run,
since neither this app's graph/Inspector nor its forward-pass machinery
understand a non-transformer tower yet. What's still fully out of scope
is full-size checkpoints without any adapter path at all (a production
MoE like Mixtral or DeepSeek-V3), other multimodal/vision architectures
(GLM-4.5V, GLM-5, Qwen-VL, Llama 4's vision tower), and genuine
state-space/Mamba architectures (Bamba, Phi-4-flash) — the vision-tower
case needs real Model IR extensions (patch embedding, a non-causal
attention variant, image-token splicing into the text sequence) this
project doesn't have yet, and a production-scale MoE needs genuinely
large-checkpoint streaming this MVP's upfront-download `WeightProvider`
doesn't do.

## Getting started

### Prerequisites

- Node.js 22.12+ (see `.nvmrc`)
- npm (this is an npm-workspaces monorepo)

### Install

```bash
npm install
```

### Run in development

```bash
npm run dev
```

Open the printed `http://localhost:5173` URL. A tiny GPT-2 checkpoint
loads by default; pick any other preset from the loader screen, or type a
Hugging Face repo id directly.

### Build for production

```bash
npm run build
```

### Type-check the whole workspace

```bash
npm run typecheck
```

### Run with Docker

No Node.js install needed — this builds the static production bundle in
a `node:22-alpine` stage and serves it with `nginx:alpine`:

```bash
docker build -t tensorium .
docker run --rm -p 8080:80 tensorium
```

Then open `http://localhost:8080`. Since the whole app is a static
client-side bundle, this container has no backend and holds no state —
it's just a file server.

## Usage

1. **Load a model** — pick a preset (sorted alphabetically) or type a
   `org/model-name` repo id and click Load.
2. **Explore the architecture** — click any node in the graph or tree to
   inspect it; double-click a transformer block to drop into its internal
   wiring, and use the breadcrumb to step back out.
3. **Run a forward pass** — type a prompt and click *Run Forward Pass* to
   populate activations throughout the graph. Add a second prompt via *+
   Compare with another prompt* to unlock A/B/diff views — Prompt B gets
   its own next-token prediction panel, and a source toggle inside Logit
   Lens and Token Attribution switches between analyzing Prompt A or B.
4. **Analyze** — switch between the four bottom tabs:
   - **Tensor Explorer** for raw weights/activations,
   - **Logit Lens** for the evolving next-token prediction,
   - **Token Attribution** for which input tokens mattered,
   - **Experiment** for causal interventions (zero/patch a component or
     head and see the effect on the output).
5. **Customize** — open the settings panel (gear icon, top-right) to
   switch theme or UI language.

## Adding a new architecture

1. Write a new package under `packages/model-adapters/`, implementing the
   `ModelAdapter` interface from `@tensorium/model-ir` (`canLoad`,
   `loadMetadata`, `buildGraph`, `getWeightProvider`, `runInference`).
2. If the architecture is Llama-shaped (RoPE + RMSNorm + gated FFN — most
   are), it's very likely a thin wrapper over `adapter-llama-family`
   rather than new graph/inference code — see the `mistral` package for
   the minimal case and `gemma`/`phi` for ones with real option overrides
   (embedding scaling, fused projections, QK-Norm, etc). Otherwise,
   implement `buildGraph`/`runInference` directly with
   `@tensorium/nn-ops`'s primitives — see `deepseek-v2`, `gemma4`, and
   `qwen3-5` for real examples of architectures structurally different
   enough to need this (Multi-head Latent Attention, per-layer frozen
   K/V reuse, a linear/recurrent attention mechanism).
3. Register the adapter (and, optionally, a preset checkpoint) in
   `apps/web/src/adapters.ts`.

Nothing in the UI needs to change: the graph renderer, inspector, tensor
viewer, logit lens, and token attribution are all driven by the generic
`NodeType`/`ActivationCapture`/`Intervention` contracts in the Model IR,
not by which adapter produced them.

## Verifying correctness

Every adapter's forward pass, tokenizer, and intervention behavior is
checked against real output from Python/PyTorch and Hugging Face's
`transformers` library — not just "does it run." Numeric outputs
typically match reference values to ~1e-6/1e-7 for F32/BF16 checkpoints,
and interventions are checked against genuine PyTorch forward hooks
(register_forward_hook / register_forward_pre_hook), not just this
project's own reimplementation of the same idea.

## Known limitations

- Weights are downloaded as one in-memory buffer per model — fine for the
  tiny checkpoints this app targets, but a multi-GB checkpoint needs a
  backend doing true HTTP range reads behind the same `WeightProvider`
  interface, which isn't implemented yet.
- Inference is a single forward pass, not autoregressive generation.
- The tokenizer doesn't handle special/added tokens, and has a known gap
  with unusual (doubled) whitespace against SentencePiece-style
  normalizers.
- The Gemma adapter is scoped to Gemma 1 only; Gemma 2/3 add real
  architectural differences (sandwich norms, alternating attention, logit
  softcapping) it doesn't implement.
- Mistral's sliding-window attention isn't modeled.
- Only OLMo (v1) is supported; OLMo 2 uses a different block topology
  (RMSNorm applied after each sub-layer instead of before it, plus
  per-head QK-norm) that needs its own adapter.
- A genuinely multimodal checkpoint's vision/audio towers are never
  loaded — only the text decoder (see [Supported
  architectures](#supported-architectures)) — and a production-scale MoE
  checkpoint too large for this MVP's upfront-download `WeightProvider`
  isn't supported either.
- Only Qwen2-MoE, Qwen3-MoE, and DeepSeek-V2's DeepSeekMoE variants of
  sparse MoE are implemented; other MoE families (Mixtral-style) need
  their own adapter.
- Any structure-only model over 20 GB (see the preset list) can't run
  even a synthetic forward pass — the option to allow one is disabled
  outright above that size, since fabricating and multiplying random
  values through tens of gigabytes of "weights" isn't worth the browser
  memory or time it'd cost for a model whose real weights were never
  downloaded in the first place.
- Token attribution is occlusion-based only — no gradient-based
  attribution.
- The IndexedDB cache is per-browser, not shared across users or devices —
  it just saves a repeat visitor's own re-downloads, not bandwidth across
  everyone loading the app.
- No persistence: experiments, comparisons, and logit-lens runs live only
  in browser state for the current session.

## Credits

This project only exists because of the open model architectures and
tiny test checkpoints Hugging Face and its community publish. Particular
thanks to the maintainers of the preset checkpoints used above:
[`hf-internal-testing`](https://huggingface.co/hf-internal-testing),
[`yujiepan`](https://huggingface.co/yujiepan),
[`fxmarty`](https://huggingface.co/fxmarty), and the
[`tiny-random`](https://huggingface.co/tiny-random) org — and to Hugging
Face for `transformers`, `safetensors`, and the model hub that makes a
config.json + safetensors pair a reliable, inspectable source of truth
for a given architecture.

The architecture graph is laid out with [ELK](https://eclipse.dev/elk/)
(via [`elkjs`](https://github.com/kieler/elkjs)) and rendered with [React
Flow](https://reactflow.dev/).

## License

[MIT](LICENSE)
