import { useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent } from "react";
import type { Model, Tensor } from "@tensorium/model-ir";
import { totalParameterBytes } from "@tensorium/model-ir";
import { useModel } from "./useModel.js";
import { useInference } from "./useInference.js";
import { useLocalStorageState } from "./useLocalStorageState.js";
import { useTheme } from "./components/ThemeSwitcher.js";
import { useTranslation } from "./components/LanguageContext.js";
import { SettingsButton, SettingsPanel } from "./components/SettingsPanel.js";
import { ModelLoader } from "./components/ModelLoader.js";
import { LoadProgressBar } from "./components/LoadProgressBar.js";
import { LoadModelPanel } from "./components/LoadModelPanel.js";
import { SaveModelDialog, type SaveModelFile } from "./components/SaveModelDialog.js";
import { UnknownModelDialog } from "./components/UnknownModelDialog.js";
import { ModelInfoBar } from "./components/ModelInfoBar.js";
import { ModelTree } from "./components/ModelTree.js";
import { ArchitectureGraph, type GraphView } from "./components/ArchitectureGraph.js";
import { Inspector } from "./components/Inspector.js";
import { TensorExplorer, type TensorSourceRequest } from "./components/TensorExplorer.js";
import { InferencePanel } from "./components/InferencePanel.js";
import { PredictionPanel } from "./components/PredictionPanel.js";
import { LogitLensPanel } from "./components/LogitLensPanel.js";
import { TokenAttributionPanel } from "./components/TokenAttributionPanel.js";
import { ExperimentPanel } from "./components/ExperimentPanel.js";

type BottomTab = "tensor" | "logitlens" | "attribution" | "experiment";

const BOTTOM_PANEL_DEFAULT_HEIGHT = 360;
const BOTTOM_PANEL_MIN_HEIGHT = 160;
/** Leaves at least this much vertical space for the tree/graph/inspector row above, however tall the window is. */
const BOTTOM_PANEL_TOP_RESERVE = 240;

const TREE_PANEL_DEFAULT_WIDTH = 260;
// Below this, tree rows (indented 14px per depth) and long node names start
// truncating illegibly rather than just looking cramped.
const TREE_PANEL_MIN_WIDTH = 160;
const INSPECTOR_PANEL_DEFAULT_WIDTH = 320;
// Below this, the Inspector's io-rows (a label and a monospace value on one
// line — see .io-row) start wrapping instead of staying legible.
const INSPECTOR_PANEL_MIN_WIDTH = 220;
/** Leaves at least this much horizontal space for the graph pane, however wide the tree/inspector panels are dragged. */
const GRAPH_PANEL_MIN_WIDTH = 320;
/** The two 6px resize handles plus each pane's own 1px border (×3 panes) — real chrome the max-width math below would otherwise ignore, letting the graph pane end up a bit under GRAPH_PANEL_MIN_WIDTH rather than at least that wide. */
const PANE_CHROME_ALLOWANCE = 24;

/** Real (on-disk) weight-byte ceiling past which a structure-only model's synthetic forward pass is refused outright, not just opt-in-gated — see `oversizedForForwardPass` below. */
const MAX_FORWARD_PASS_WEIGHT_BYTES = 20 * 1024 ** 3;

function computeActivationMagnitude(t: Tensor): number {
  let sum = 0;
  for (let i = 0; i < t.data.length; i++) sum += t.data[i] * t.data[i];
  return Math.sqrt(sum);
}

/** Triggers a browser "Save As" for one file's raw bytes — no server round-trip, just a Blob + an off-DOM `<a download>` click. */
function downloadBytes(bytes: ArrayBuffer, filename: string) {
  const url = URL.createObjectURL(new Blob([bytes]));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/** The transformer block that owns `nodeId` — itself if it is one, else the nearest ancestor — or null for a top-level node (embeddings, final norm, LM head, ...) that isn't inside any block. */
function containingBlockId(model: Model, nodeId: string): string | null {
  let cur: string | null = nodeId;
  while (cur) {
    if (model.nodes[cur].type === "transformer_block") return cur;
    cur = model.nodes[cur].parentId ?? null;
  }
  return null;
}

export function App() {
  const { state, load, loadLocalFiles, reset, restoring, progress, confirmUnknownModel } = useModel();
  const { theme, setTheme } = useTheme();
  const { t } = useTranslation();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [loadModelOpen, setLoadModelOpen] = useState(false);
  // Persisted (not plain useState) so a refresh that restores the last
  // Hugging-Face-sourced model — see useModel's `restoring` — can also land
  // back on the same node/view instead of resetting to the top-level
  // architecture view every time.
  const [selectedId, setSelectedId] = useLocalStorageState<string | null>("chart:selected-id", null);
  const [view, setView] = useLocalStorageState<GraphView>("chart:view", { kind: "architecture" });
  const [selectedTokenIndex, setSelectedTokenIndex] = useState<number | null>(null);
  const [compareEnabled, setCompareEnabled] = useState(false);
  const [bottomTab, setBottomTab] = useState<BottomTab>("tensor");
  const [analysisBusy, setAnalysisBusy] = useState(false);
  const [tensorSourceRequest, setTensorSourceRequest] = useState<TensorSourceRequest | null>(null);
  /** Which prompt's tokens Token Attribution attributes — controlled here (rather than as the panel's own local state) so the Prediction panel's "Why?" link can request the matching side instead of always landing back on Prompt A. */
  const [attributionSource, setAttributionSource] = useState<"A" | "B">("A");
  const [treeCollapsed, setTreeCollapsed] = useLocalStorageState("panel:tree-collapsed", false);
  const [inspectorCollapsed, setInspectorCollapsed] = useLocalStorageState("panel:inspector-collapsed", false);
  const [bottomCollapsed, setBottomCollapsed] = useLocalStorageState("panel:bottom-collapsed", false);
  const [bottomHeight, setBottomHeight] = useLocalStorageState("panel:bottom-height", BOTTOM_PANEL_DEFAULT_HEIGHT);
  const [resizingBottom, setResizingBottom] = useState(false);
  const [treeWidth, setTreeWidth] = useLocalStorageState("panel:tree-width", TREE_PANEL_DEFAULT_WIDTH);
  const [inspectorWidth, setInspectorWidth] = useLocalStorageState("panel:inspector-width", INSPECTOR_PANEL_DEFAULT_WIDTH);
  const [resizingTree, setResizingTree] = useState(false);
  const [resizingInspector, setResizingInspector] = useState(false);
  // Mirrors ArchitectureGraph's own on-canvas zoom badge — kept here too so
  // the status footer can show it without either component owning the
  // other's state.
  const [zoomPercent, setZoomPercent] = useState(100);
  const [predictionCollapsed, setPredictionCollapsed] = useLocalStorageState("panel:prediction-collapsed", false);
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  // Off by default: a structure-only model's real weights were deliberately
  // never downloaded (too large or sharded — see hf-client's
  // fetchModelStructure), so running inference against it means fabricated
  // random weights, not the model's real behavior. Gating this behind an
  // explicit opt-in keeps that from being a silent trap.
  const [allowSyntheticForwardPass, setAllowSyntheticForwardPass] = useLocalStorageState("settings:allow-synthetic-forward-pass", false);
  // Plain (non-persisted) state, not a setting: dismissing the structure-only
  // note just hides it for the currently-loaded model — reset below whenever
  // a different model loads, so a genuinely new large model still gets the
  // warning even if a previous one's was dismissed.
  const [structureOnlyNoteDismissed, setStructureOnlyNoteDismissed] = useState(false);
  // Turning synthetic forward passes ON — whether from Settings or the
  // note's own "Enable anyway" — should surface the note again even if it
  // was previously dismissed, since enabling it is exactly the moment a
  // "this uses fake weights and may be slow" reminder (now with a quick
  // "Disable" action) is most useful. Turning it back off doesn't need the
  // same treatment: the note is already on screen at that point.
  const setAllowSyntheticForwardPassAndReveal = (next: boolean) => {
    setAllowSyntheticForwardPass(next);
    if (next) setStructureOnlyNoteDismissed(false);
  };

  const inference = useInference(state.model, state.weightProvider, state.adapter, state.tokenizer);
  const promptB = useInference(state.model, state.weightProvider, state.adapter, state.tokenizer);

  // Captured once, on the very first render — true only when this mount
  // started out restoring a persisted model after a page refresh (see
  // useModel's `restoring`). While true, the reset-on-model-change effect
  // below is skipped entirely (including its mount-time firing before the
  // restored model has even landed) so the persisted selectedId/view aren't
  // wiped out from under the restore. It flips to false — permanently, for
  // the rest of this mount — the moment `state.model` is first observed
  // defined, i.e. right when the restored model actually lands, so any
  // *later* switch to a genuinely different model resets as normal.
  const skipResetRef = useRef(restoring);

  // A different model can have completely different node ids (fewer/more
  // blocks, different architecture) — stale selection/view referencing the
  // old model's ids would otherwise crash ArchitectureGraph's breadcrumb.
  useEffect(() => {
    if (skipResetRef.current) {
      if (state.model) skipResetRef.current = false;
      return;
    }
    setSelectedId(null);
    setView({ kind: "architecture" });
    setSelectedTokenIndex(null);
    setBottomTab("tensor");
    setAttributionSource("A");
    inference.reset();
    promptB.reset();
    setStructureOnlyNoteDismissed(false);
  }, [state.model]);

  // Per-node activation magnitude (L2 norm) from the last run — computed
  // once here and shared by the model tree's per-row ticks and the
  // Inspector's "This run" section, rather than each recomputing it.
  const activationMagnitudeById = useMemo(() => {
    const result = inference.state.result;
    if (!result) return undefined;
    const map: Record<string, number> = {};
    for (const nodeId in result.activations) map[nodeId] = computeActivationMagnitude(result.activations[nodeId]);
    return map;
  }, [inference.state.result]);

  if (state.status !== "ready" || !state.model || !state.weightProvider) {
    // While the mount-time restore is in flight, show a plain spinner
    // instead of the full model-picker form — otherwise a refresh on the
    // chart page would visibly flash the home screen's title/input/preset
    // chips before snapping back to the chart, which reads as "it navigated
    // to home and back" even though nothing was actually reset.
    const showRestoring = restoring && state.status !== "error";
    return (
      <div className="app-loader-screen">
        <div className="top-right-controls">
          <SettingsButton open={settingsOpen} onToggle={() => setSettingsOpen((v) => !v)} />
          <SettingsPanel
            open={settingsOpen}
            onClose={() => setSettingsOpen(false)}
            theme={theme}
            onThemeChange={setTheme}
            allowSyntheticForwardPass={allowSyntheticForwardPass}
            onAllowSyntheticForwardPassChange={setAllowSyntheticForwardPassAndReveal}
          />
        </div>
        {showRestoring ? (
          <div className="app-restoring">
            <div className="app-restoring-spinner" />
            <div className="app-restoring-text">{t("app.restoringSession")}</div>
            {progress && <LoadProgressBar progress={progress} />}
          </div>
        ) : (
          <ModelLoader
            status={state.status === "confirm-unknown" ? "loading" : state.status}
            error={state.error}
            progress={progress}
            onLoad={load}
            onLoadLocal={loadLocalFiles}
          />
        )}
        <UnknownModelDialog
          open={state.status === "confirm-unknown"}
          modelType={state.pendingPreview?.model_type}
          architectures={state.pendingPreview?.architectures}
          onCancel={() => confirmUnknownModel(false)}
          onConfirm={() => confirmUnknownModel(true)}
        />
        <div className="app-loader-disclaimer">{t("loader.disclaimer")}</div>
      </div>
    );
  }

  const model = state.model;
  const structureOnly = state.metadata?.structureOnly === true;
  // Beyond this many real (on-disk) weight bytes, a synthetic forward pass
  // isn't just gated behind opt-in, it's refused outright: at 4x that in
  // materialized Float64Array weights (see estimatedForwardPassBytes below),
  // the tab is essentially guaranteed to freeze or crash, so there's no
  // responsible "enable anyway" for a checkpoint this large.
  const oversizedForForwardPass = structureOnly && totalParameterBytes(model) > MAX_FORWARD_PASS_WEIGHT_BYTES;
  const forwardPassBlocked = structureOnly && (oversizedForForwardPass || !allowSyntheticForwardPass);
  // Rough floor, not a precise prediction: every tensor this app touches is
  // materialized as a Float64Array (8 bytes/element) regardless of its
  // on-disk dtype (BF16/F16 = 2 bytes/element on every structure-only preset
  // today), and a full forward pass has to load every layer's weights, not
  // just the ones a user happens to click in Tensor Explorer — so 4x the
  // checkpoint's real declared size is the minimum it'll actually hold in
  // memory once cached, before counting per-array JS engine overhead on top.
  const estimatedForwardPassBytes = totalParameterBytes(model) * 4;
  const selectedNode = selectedId ? model.nodes[selectedId] ?? null : null;
  // Defends against the one render where a just-finished model load has
  // landed but `view`/`selectedId` haven't been reset to match yet (the
  // effect above runs after this render, not before it) — without this, a
  // stale blockId from the previous model would crash ArchitectureGraph.
  const safeView: GraphView = view.kind === "block" && !model.nodes[view.blockId] ? { kind: "architecture" } : view;

  // Shared by the graph's double-click-to-expand gesture and the tree's
  // double-click on any row: jump to whichever graph view actually contains
  // this node (its own block's detail view, or back out to the top-level
  // architecture view for a node — e.g. Logits — that isn't inside a block
  // at all) and select it there, instead of only handling the
  // transformer-block case and leaving other double-clicks a no-op.
  const navigateToNode = (nodeId: string) => {
    const blockId = containingBlockId(model, nodeId);
    setView(blockId ? { kind: "block", blockId } : { kind: "architecture" });
    setSelectedId(nodeId);
  };

  // A stale selectedTokenIndex from a previous, possibly-longer prompt would
  // otherwise silently point past the new run's token count (App only
  // resets it on model change, not on every re-run) — clear it so every
  // panel's `?? tokenIds.length - 1` default kicks in fresh for the new
  // result, matching "just ran, so show me the last position" intuition.
  const runPromptA = (prompt: string) => {
    setSelectedTokenIndex(null);
    setPredictionCollapsed(false);
    inference.run(prompt);
  };

  // Switching tabs while the bottom panel is collapsed should actually show
  // the tab, not just change which one is "active" behind a collapsed strip
  // — every place that jumps to a specific bottom tab (the tab bar itself,
  // Inspector's quick actions, Prediction panel's "Why?" link) goes through
  // this instead of setBottomTab directly.
  const selectBottomTab = (tab: BottomTab) => {
    setBottomTab(tab);
    setBottomCollapsed(false);
  };

  const requestTensorSource = (value: "weights" | "activations") => {
    selectBottomTab("tensor");
    setTensorSourceRequest({ value, nonce: Date.now() });
  };

  // Inspector's "This run" input/output links — jumps to Tensor Explorer's
  // Input/Output tab, pre-selected to whichever side (and, for an input,
  // which upstream source) the user actually clicked.
  const requestTensorIO = (io: "input" | "output", sourceId?: string) => {
    selectBottomTab("tensor");
    setTensorSourceRequest({ value: "io", io, sourceId, nonce: Date.now() });
  };

  const viewWhy = (source: "A" | "B") => {
    setAttributionSource(source);
    selectBottomTab("attribution");
  };

  // Every file downloaded exactly as fetched/picked (state.rawFiles holds
  // the original bytes, not anything re-serialized from the parsed/decoded
  // in-memory model) — this is what lets the files saved here be loaded
  // straight back in via "Local files" with no round-trip loss.
  const safeModelName = model.name.replace(/[\\/:*?"<>|]+/g, "-");
  const saveModelFiles: SaveModelFile[] = [
    state.rawFiles?.weightsBytes && { filename: `${safeModelName}.safetensors`, bytes: state.rawFiles.weightsBytes.byteLength },
    state.rawFiles?.configBytes && { filename: `${safeModelName}.config.json`, bytes: state.rawFiles.configBytes.byteLength },
    state.rawFiles?.tokenizerBytes && { filename: `${safeModelName}.tokenizer.json`, bytes: state.rawFiles.tokenizerBytes.byteLength },
  ].filter((f): f is SaveModelFile => !!f);
  const canSaveModel = !!state.rawFiles?.weightsBytes;
  const confirmSaveModel = () => {
    if (state.rawFiles?.weightsBytes) downloadBytes(state.rawFiles.weightsBytes, `${safeModelName}.safetensors`);
    if (state.rawFiles?.configBytes) downloadBytes(state.rawFiles.configBytes, `${safeModelName}.config.json`);
    if (state.rawFiles?.tokenizerBytes) downloadBytes(state.rawFiles.tokenizerBytes, `${safeModelName}.tokenizer.json`);
    setSaveDialogOpen(false);
  };

  const hasResult = inference.state.status === "ready" && !!inference.state.result;
  const hasResultB = compareEnabled && promptB.state.status === "ready" && !!promptB.state.result;
  const analysisTabsEnabled = hasResult && !!state.adapter?.runInference;
  const currentRepo = state.source?.kind === "huggingface" ? state.source.repo : undefined;

  // Status-footer fields — all read from state this component already
  // holds, nothing computed just for display here.
  const runningA = inference.state.status === "running";
  const runningB = promptB.state.status === "running";
  const footerBusy = analysisBusy || runningA || runningB;
  // Real per-layer progress from whichever run is actually in flight (see
  // InferenceProgress) — prefers A when both happen to be running at once,
  // which is rare enough not to need a combined display.
  const footerProgress = runningA ? inference.state.progress : runningB ? promptB.state.progress : undefined;
  const footerRunningLabel = runningA && runningB ? t("footer.runningBoth") : runningB ? t("footer.runningB") : t("inference.running");
  const footerStatus: { label: string; tone: "ready" | "busy" | "error" } =
    inference.state.status === "error" || promptB.state.status === "error"
      ? { label: t("footer.error"), tone: "error" }
      : footerBusy
        ? { label: analysisBusy ? t("footer.analyzing") : footerRunningLabel, tone: "busy" }
        : { label: t("footer.ready"), tone: "ready" };
  // The selected node's own block if it's inside one, else whichever block
  // the graph is currently showing the detail view of, else no block
  // context at all (top-level architecture view, nothing selected).
  const footerBlockId = selectedId ? containingBlockId(model, selectedId) : safeView.kind === "block" ? safeView.blockId : null;
  const footerBlockLabel = footerBlockId ? model.nodes[footerBlockId]?.name : t("footer.architectureView");
  // Selecting a node should change *what's shown*, not just the numbers
  // next to it — "Transformer Block 0" alone doesn't say whether you're
  // looking at its Attention or its FFN. Folds the block context and the
  // node's own name into one breadcrumb-style title (just the block name
  // when nothing's selected, or when the selection *is* the block itself).
  const footerTitle = selectedNode && footerBlockId !== selectedNode.id ? `${footerBlockLabel} › ${selectedNode.name}` : selectedNode?.name ?? footerBlockLabel;
  const footerTensor = selectedId ? inference.state.result?.activations[selectedId] : undefined;
  // Shape falls back to the node's statically-declared output spec (real —
  // every adapter fills this in at graph-build time, not a run artifact —
  // just symbolic, e.g. "sequence_length" instead of the run's actual
  // token count) rather than a bare dash whenever there's no captured
  // activation yet. Dtype has no such static equivalent for most node
  // types — model-ir's TensorSpec carries dims only, not a dtype — so it
  // only falls back for a node that owns weights, where the weight's own
  // dtype is a real (if imperfect) stand-in; otherwise it stays "—".
  const footerShape = footerTensor
    ? `[${footerTensor.shape.join(" × ")}]`
    : selectedNode?.outputs[0]
      ? `[${selectedNode.outputs[0].dims.join(", ")}]`
      : "—";
  const footerDtype = footerTensor?.dtype ?? selectedNode?.parameters[0]?.dtype ?? "—";
  const footerNorm = selectedId ? activationMagnitudeById?.[selectedId] : undefined;

  // Derived, not a separate stored flag: "max frame" just means every
  // surrounding panel is currently collapsed. The prediction panel only
  // counts when it's actually rendered (a result exists) — otherwise its
  // stored collapse preference shouldn't stop the other three panels from
  // reading as "already maximized".
  const isMaxFrame = treeCollapsed && inspectorCollapsed && bottomCollapsed && (!hasResult || predictionCollapsed);
  const toggleMaxFrame = () => {
    const next = !isMaxFrame;
    setTreeCollapsed(next);
    setInspectorCollapsed(next);
    setBottomCollapsed(next);
    setPredictionCollapsed(next);
  };

  // Drag-to-resize for the bottom panel. Height is tracked in state (not
  // just read from the DOM after drag) so it can be persisted; the max
  // clamp is computed live off window.innerHeight rather than a fixed
  // constant so it stays sane across window resizes.
  const handleBottomResizeStart = (e: ReactMouseEvent) => {
    e.preventDefault();
    const startY = e.clientY;
    const startHeight = bottomHeight;
    setResizingBottom(true);
    const onMove = (moveEvent: MouseEvent) => {
      const maxHeight = Math.max(BOTTOM_PANEL_MIN_HEIGHT, window.innerHeight - BOTTOM_PANEL_TOP_RESERVE);
      const next = startHeight + (startY - moveEvent.clientY);
      setBottomHeight(Math.min(maxHeight, Math.max(BOTTOM_PANEL_MIN_HEIGHT, Math.round(next))));
    };
    const onUp = () => {
      setResizingBottom(false);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  // Drag-to-resize for the tree and inspector panels — same pattern as the
  // bottom panel above, mirrored horizontally. The max clamp for each also
  // accounts for the *other* side panel's current width (full or collapsed
  // to its 36px strip), so dragging one out doesn't silently starve the
  // graph pane below GRAPH_PANEL_MIN_WIDTH just because the other panel
  // also happens to be wide.
  const handleTreeResizeStart = (e: ReactMouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startWidth = treeWidth;
    setResizingTree(true);
    const onMove = (moveEvent: MouseEvent) => {
      const inspectorSpace = inspectorCollapsed ? 36 : inspectorWidth;
      const maxWidth = Math.max(TREE_PANEL_MIN_WIDTH, window.innerWidth - inspectorSpace - GRAPH_PANEL_MIN_WIDTH - PANE_CHROME_ALLOWANCE);
      const next = startWidth + (moveEvent.clientX - startX);
      setTreeWidth(Math.min(maxWidth, Math.max(TREE_PANEL_MIN_WIDTH, Math.round(next))));
    };
    const onUp = () => {
      setResizingTree(false);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  const handleInspectorResizeStart = (e: ReactMouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startWidth = inspectorWidth;
    setResizingInspector(true);
    const onMove = (moveEvent: MouseEvent) => {
      const treeSpace = treeCollapsed ? 36 : treeWidth;
      const maxWidth = Math.max(INSPECTOR_PANEL_MIN_WIDTH, window.innerWidth - treeSpace - GRAPH_PANEL_MIN_WIDTH - PANE_CHROME_ALLOWANCE);
      const next = startWidth + (startX - moveEvent.clientX);
      setInspectorWidth(Math.min(maxWidth, Math.max(INSPECTOR_PANEL_MIN_WIDTH, Math.round(next))));
    };
    const onUp = () => {
      setResizingInspector(false);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  return (
    <div
      className={
        "app" +
        (analysisBusy ? " app-busy" : "") +
        (resizingBottom ? " app-resizing-panel" : "") +
        (resizingTree || resizingInspector ? " app-resizing-col-panel" : "")
      }
    >
      <ModelInfoBar model={model} structureOnly={structureOnly} bestEffort={state.adapter?.id === "generic"} />
      <div className="top-right-controls">
        <div className="control-group">
          <button className="close-model" onClick={reset} title={t("app.closeModel")}>
            {t("app.closeModel")}
          </button>
          <button className="load-different" onClick={() => setLoadModelOpen((v) => !v)}>
            {t("app.loadDifferentModel")}
          </button>
          <LoadModelPanel
            open={loadModelOpen}
            onClose={() => setLoadModelOpen(false)}
            status={state.status}
            error={state.error}
            excludeRepo={currentRepo}
            onLoad={load}
            onLoadLocal={loadLocalFiles}
          />
          <button className="save-model" onClick={() => setSaveDialogOpen(true)} disabled={!canSaveModel} title={t("app.saveModel")}>
            {t("app.saveModel")}
          </button>
          <SaveModelDialog open={saveDialogOpen} files={saveModelFiles} onCancel={() => setSaveDialogOpen(false)} onConfirm={confirmSaveModel} />
        </div>
        <div className="control-group">
          <SettingsButton open={settingsOpen} onToggle={() => setSettingsOpen((v) => !v)} />
          <SettingsPanel
            open={settingsOpen}
            onClose={() => setSettingsOpen(false)}
            theme={theme}
            onThemeChange={setTheme}
            allowSyntheticForwardPass={allowSyntheticForwardPass}
            onAllowSyntheticForwardPassChange={setAllowSyntheticForwardPassAndReveal}
            forwardPassSettingDisabled={oversizedForForwardPass}
          />
        </div>
      </div>
      <InferencePanel
        supported={!!state.tokenizer}
        state={inference.state}
        onRun={runPromptA}
        selectedTokenIndex={selectedTokenIndex}
        onSelectToken={setSelectedTokenIndex}
        compareEnabled={compareEnabled}
        onToggleCompare={() => setCompareEnabled((v) => !v)}
        promptBState={promptB.state}
        onRunB={promptB.run}
        structureOnly={structureOnly}
        forwardPassBlocked={forwardPassBlocked}
        oversizedForForwardPass={oversizedForForwardPass}
        onEnableForwardPass={() => setAllowSyntheticForwardPassAndReveal(true)}
        onDisableForwardPass={() => setAllowSyntheticForwardPass(false)}
        estimatedForwardPassBytes={estimatedForwardPassBytes}
        noteDismissed={structureOnlyNoteDismissed}
        onDismissNote={() => setStructureOnlyNoteDismissed(true)}
      />
      {hasResult && state.tokenizer && (
        <div className="prediction-panels-row">
          <PredictionPanel
            result={inference.state.result!}
            tokenizer={state.tokenizer}
            selectedTokenIndex={selectedTokenIndex}
            onViewWhy={() => viewWhy("A")}
            collapsed={predictionCollapsed}
            onToggleCollapsed={() => setPredictionCollapsed((v) => !v)}
            promptLabel={hasResultB ? t("inference.promptA") : undefined}
          />
          {hasResultB && (
            <PredictionPanel
              result={promptB.state.result!}
              tokenizer={state.tokenizer}
              selectedTokenIndex={selectedTokenIndex}
              onViewWhy={() => viewWhy("B")}
              collapsed={predictionCollapsed}
              onToggleCollapsed={() => setPredictionCollapsed((v) => !v)}
              promptLabel={t("inference.promptB")}
            />
          )}
        </div>
      )}
      <div className="app-main-row">
        <aside
          className={"pane pane-tree" + (treeCollapsed ? " collapsed" : "") + (resizingTree ? " resizing" : "")}
          style={treeCollapsed ? undefined : { width: treeWidth }}
        >
          <div className="pane-header">
            {!treeCollapsed && <span className="pane-header-title">{t("app.modelTree")}</span>}
            <button className="pane-collapse-btn" onClick={() => setTreeCollapsed((v) => !v)} title={treeCollapsed ? t("app.expandTree") : t("app.collapseTree")}>
              {treeCollapsed ? "›" : "‹"}
            </button>
          </div>
          {treeCollapsed ? (
            <span className="pane-vertical-label">{t("app.modelTree")}</span>
          ) : (
            <div className="pane-tree-body">
              <ModelTree
                model={model}
                selectedId={selectedId}
                onSelect={setSelectedId}
                onNavigate={navigateToNode}
                activationMagnitudeById={activationMagnitudeById}
              />
            </div>
          )}
        </aside>
        {!treeCollapsed && (
          <div
            className={"pane-resize-handle pane-resize-handle-vertical" + (resizingTree ? " resizing" : "")}
            onMouseDown={handleTreeResizeStart}
            title={t("app.resizePanel")}
          />
        )}
        <main className="pane pane-graph">
          <ArchitectureGraph
            model={model}
            view={safeView}
            selectedId={selectedId}
            onSelect={setSelectedId}
            onEnterBlock={navigateToNode}
            onExitBlock={() => setView({ kind: "architecture" })}
            isMaxFrame={isMaxFrame}
            onToggleMaxFrame={toggleMaxFrame}
            onZoomChange={setZoomPercent}
          />
        </main>
        {!inspectorCollapsed && (
          <div
            className={"pane-resize-handle pane-resize-handle-vertical" + (resizingInspector ? " resizing" : "")}
            onMouseDown={handleInspectorResizeStart}
            title={t("app.resizePanel")}
          />
        )}
        <aside
          className={"pane pane-inspector" + (inspectorCollapsed ? " collapsed" : "") + (resizingInspector ? " resizing" : "")}
          style={inspectorCollapsed ? undefined : { width: inspectorWidth }}
        >
          <div className="pane-header">
            <button className="pane-collapse-btn" onClick={() => setInspectorCollapsed((v) => !v)} title={inspectorCollapsed ? t("app.expandInspector") : t("app.collapseInspector")}>
              {inspectorCollapsed ? "‹" : "›"}
            </button>
            {!inspectorCollapsed && <span className="pane-header-title">{t("app.inspector")}</span>}
          </div>
          {inspectorCollapsed ? (
            <span className="pane-vertical-label">{t("app.inspector")}</span>
          ) : (
            <div className="pane-inspector-body">
              <Inspector
                model={model}
                node={selectedNode}
                activationShape={selectedId ? inference.state.result?.activations[selectedId]?.shape : undefined}
                activationMagnitude={selectedId ? activationMagnitudeById?.[selectedId] : undefined}
                onViewActivation={() => requestTensorSource("activations")}
                onViewWeights={() => requestTensorSource("weights")}
                onViewInput={(sourceId) => requestTensorIO("input", sourceId)}
                onViewOutput={() => requestTensorIO("output")}
                inferenceResult={inference.state.result}
                tokenizer={state.tokenizer}
                elapsedMs={inference.state.elapsedMs}
                onDeselect={() => setSelectedId(null)}
              />
            </div>
          )}
        </aside>
      </div>
      <section
        className={"pane pane-tensor" + (bottomCollapsed ? " collapsed" : "") + (resizingBottom ? " resizing" : "")}
        style={bottomCollapsed ? undefined : { height: bottomHeight }}
      >
        {!bottomCollapsed && (
          <div
            className="pane-tensor-resize-handle"
            onMouseDown={handleBottomResizeStart}
            title={t("app.resizePanel")}
          />
        )}
        <div className="bottom-tabs">
          <button className={bottomTab === "tensor" ? "active" : ""} onClick={() => selectBottomTab("tensor")}>
            {t("app.tensorExplorer")}
          </button>
          <button className={bottomTab === "logitlens" ? "active" : ""} disabled={!analysisTabsEnabled} onClick={() => selectBottomTab("logitlens")} title={!analysisTabsEnabled ? t("app.runForwardPassFirst") : undefined}>
            {t("app.logitLens")}
          </button>
          <button className={bottomTab === "attribution" ? "active" : ""} disabled={!analysisTabsEnabled} onClick={() => selectBottomTab("attribution")} title={!analysisTabsEnabled ? t("app.runForwardPassFirst") : undefined}>
            {t("app.tokenAttribution")}
          </button>
          <button className={bottomTab === "experiment" ? "active" : ""} disabled={!analysisTabsEnabled} onClick={() => selectBottomTab("experiment")} title={!analysisTabsEnabled ? t("app.runForwardPassFirst") : undefined}>
            {t("app.experiment")}
          </button>
          <span className="bottom-tabs-spacer" />
          <button className="bottom-collapse-btn" onClick={() => setBottomCollapsed((v) => !v)} title={bottomCollapsed ? t("app.expandPanel") : t("app.collapsePanel")}>
            {bottomCollapsed ? "▴" : "▾"}
          </button>
        </div>

        {!bottomCollapsed && bottomTab === "tensor" && (
          <TensorExplorer
            model={model}
            weightProvider={state.weightProvider}
            selectedNode={selectedNode}
            inference={inference.state}
            selectedTokenIndex={selectedTokenIndex}
            promptBInference={promptB.state}
            sourceRequest={tensorSourceRequest}
            structureOnly={structureOnly}
          />
        )}
        {!bottomCollapsed && bottomTab === "logitlens" && analysisTabsEnabled && state.tokenizer && (
          <LogitLensPanel
            model={model}
            weightProvider={state.weightProvider}
            capture={inference.state.result!}
            promptBCapture={hasResultB ? promptB.state.result : undefined}
            tokenizer={state.tokenizer}
            selectedTokenIndex={selectedTokenIndex}
            onSelectToken={setSelectedTokenIndex}
            onBusyChange={setAnalysisBusy}
          />
        )}
        {!bottomCollapsed && bottomTab === "attribution" && analysisTabsEnabled && state.tokenizer && (
          <TokenAttributionPanel
            model={model}
            weightProvider={state.weightProvider}
            adapter={state.adapter!}
            tokenIds={inference.state.result!.tokenIds}
            promptBTokenIds={hasResultB ? promptB.state.result!.tokenIds : undefined}
            tokenizer={state.tokenizer}
            selectedTokenIndex={selectedTokenIndex}
            source={attributionSource}
            onSourceChange={setAttributionSource}
            onSelectNode={setSelectedId}
            onBusyChange={setAnalysisBusy}
          />
        )}
        {!bottomCollapsed && bottomTab === "experiment" && analysisTabsEnabled && state.tokenizer && (
          <ExperimentPanel
            model={model}
            weightProvider={state.weightProvider}
            adapter={state.adapter!}
            tokenizer={state.tokenizer}
            selectedNode={selectedNode}
            mainTokenIds={inference.state.result!.tokenIds}
            mainResult={inference.state.result!}
            promptBResult={promptB.state.result ?? null}
            onBusyChange={setAnalysisBusy}
          />
        )}
      </section>
      <div className="status-footer">
        <span className={"status-footer-dot status-footer-dot-" + footerStatus.tone} />
        <span className="status-footer-item status-footer-status">{footerStatus.label}</span>
        {/* Real per-layer progress (see InferenceProgress) — only while a
            forward pass is actually running and the adapter reports it, not
            a simulated/timed fill. */}
        {footerBusy && footerProgress && (
          <span
            className="status-footer-item status-footer-progress"
            title={t("footer.layerProgress").replace("{completed}", String(footerProgress.completed)).replace("{total}", String(footerProgress.total))}
          >
            <span className="status-footer-progress-track">
              <span className="status-footer-progress-fill" style={{ width: `${Math.round((footerProgress.completed / footerProgress.total) * 100)}%` }} />
            </span>
            <span className="status-footer-progress-label">
              {footerProgress.completed}/{footerProgress.total}
            </span>
          </span>
        )}
        <span className="status-footer-sep" />
        <span className="status-footer-item status-footer-title" title={footerTitle}>
          {footerTitle}
        </span>
        <span className="status-footer-sep" />
        <span className="status-footer-item">{footerShape}</span>
        <span className="status-footer-item">{footerDtype}</span>
        {/* No static equivalent for a norm — it's only ever a real run
            artifact — so this simply doesn't render rather than showing an
            empty "— norm" next to the two fields above that now do have a
            static, always-real fallback. */}
        {footerNorm !== undefined && <span className="status-footer-item">{t("footer.norm").replace("{value}", footerNorm.toFixed(4))}</span>}
        <span className="status-footer-sep" />
        <span className="status-footer-item">{t("footer.zoom").replace("{percent}", String(zoomPercent))}</span>
        <span className="status-footer-spacer" />
        <span className="status-footer-item status-footer-compute" title={t("footer.cpuTooltip")}>
          {t("footer.cpu")}
        </span>
      </div>
    </div>
  );
}
