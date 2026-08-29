import { useState } from "react";
import type { InferenceState } from "../useInference.js";
import { useTranslation } from "./LanguageContext.js";
import { formatBytes } from "../format.js";

interface Props {
  supported: boolean;
  state: InferenceState;
  onRun: (prompt: string) => void;
  selectedTokenIndex: number | null;
  onSelectToken: (i: number) => void;
  compareEnabled: boolean;
  onToggleCompare: () => void;
  promptBState: InferenceState;
  onRunB: (prompt: string) => void;
  /** True when this model's real weights were never downloaded (structure-only) — the note stays visible (until dismissed) regardless of whether synthetic runs are currently allowed, so it's always clear these numbers aren't real. */
  structureOnly: boolean;
  /** True when structureOnly and the user hasn't opted into running against synthetic weights — Run/Run Prompt B/Compare all stay disabled until either changes. */
  forwardPassBlocked: boolean;
  /** True when this model's real weight bytes exceed the hard 20GB ceiling — forwardPassBlocked is then always true regardless of the setting, and the "Enable anyway" action is hidden entirely rather than offering a toggle that can't actually help. */
  oversizedForForwardPass: boolean;
  onEnableForwardPass: () => void;
  onDisableForwardPass: () => void;
  /** Rough floor for how much browser memory running this model would hold once every layer's weights get cached — shown in the note so enabling (or leaving enabled) is an informed choice, not a leap in the dark. */
  estimatedForwardPassBytes: number;
  /** Hides the structure-only note for the current model only — reset by the caller whenever a different model loads. */
  noteDismissed: boolean;
  onDismissNote: () => void;
}

export function InferencePanel({
  supported,
  state,
  onRun,
  selectedTokenIndex,
  onSelectToken,
  compareEnabled,
  onToggleCompare,
  promptBState,
  onRunB,
  structureOnly,
  forwardPassBlocked,
  oversizedForForwardPass,
  onEnableForwardPass,
  onDisableForwardPass,
  estimatedForwardPassBytes,
  noteDismissed,
  onDismissNote,
}: Props) {
  const { t } = useTranslation();
  const [prompt, setPrompt] = useState("The cat sat on the");
  const [promptB, setPromptB] = useState("The dog sat on the");

  if (!supported) {
    return (
      <div className="inference-panel inference-panel-disabled">
        {t("inference.noTokenizer")}
      </div>
    );
  }

  return (
    <div className="inference-panel">
      <form
        className="inference-form"
        onSubmit={(e) => {
          e.preventDefault();
          onRun(prompt);
        }}
      >
        <span className="inference-label">{t("inference.promptA")}</span>
        <input value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder={t("inference.placeholderA")} />
        <button type="submit" disabled={state.status === "running" || forwardPassBlocked}>
          {state.status === "running" ? t("inference.running") : t("inference.run")}
        </button>
        <button
          type="button"
          className="compare-toggle"
          onClick={onToggleCompare}
          disabled={forwardPassBlocked}
          title={forwardPassBlocked ? t("inference.structureOnlyBlocked").replace("{memory}", formatBytes(estimatedForwardPassBytes)) : undefined}
        >
          {compareEnabled ? t("inference.hidePromptB") : t("inference.comparePromptB")}
        </button>
      </form>

      {structureOnly && !noteDismissed && (
        <div className="inference-structure-only-note">
          <span>{t("inference.structureOnlyBlocked").replace("{memory}", formatBytes(estimatedForwardPassBytes))}</span>
          {!oversizedForForwardPass && (
            <button type="button" onClick={forwardPassBlocked ? onEnableForwardPass : onDisableForwardPass}>
              {forwardPassBlocked ? t("inference.enableSyntheticForwardPass") : t("inference.disableSyntheticForwardPass")}
            </button>
          )}
          <button type="button" className="inference-structure-only-note-close" onClick={onDismissNote} aria-label={t("inference.dismissNote")} title={t("inference.dismissNote")}>
            ×
          </button>
        </div>
      )}

      {state.status === "error" && <div className="inference-error">{state.error}</div>}

      {state.status === "ready" && state.displayTokens && (
        <div className="token-chips">
          {state.displayTokens.map((t, i) => {
            const id = state.result?.tokenIds[i];
            return (
              <button
                key={i}
                className={"token-chip" + (i === selectedTokenIndex ? " selected" : "")}
                onClick={() => onSelectToken(i)}
                title={`position ${i}${id !== undefined ? ` · token id ${id}` : ""}`}
              >
                <span className="token-chip-text">{t.trim() === "" ? "·".repeat(Math.max(1, t.length)) : t}</span>
                {id !== undefined && <span className="token-chip-id">{id}</span>}
              </button>
            );
          })}
        </div>
      )}

      {compareEnabled && (
        <form
          className="inference-form prompt-b-form"
          onSubmit={(e) => {
            e.preventDefault();
            onRunB(promptB);
          }}
        >
          <span className="inference-label">{t("inference.promptB")}</span>
          <input value={promptB} onChange={(e) => setPromptB(e.target.value)} placeholder={t("inference.placeholderB")} />
          <button type="submit" disabled={promptBState.status === "running" || forwardPassBlocked}>
            {promptBState.status === "running" ? t("inference.running") : t("inference.runB")}
          </button>
        </form>
      )}
      {compareEnabled && promptBState.status === "error" && <div className="inference-error">{promptBState.error}</div>}
      {compareEnabled && promptBState.status === "ready" && promptBState.displayTokens && (
        <div className="token-chips token-chips-b">
          {promptBState.displayTokens.map((t, i) => {
            const id = promptBState.result?.tokenIds[i];
            return (
              <span key={i} className="token-chip token-chip-readonly" title={id !== undefined ? `position ${i} · token id ${id}` : undefined}>
                <span className="token-chip-text">{t.trim() === "" ? "·".repeat(Math.max(1, t.length)) : t}</span>
                {id !== undefined && <span className="token-chip-id">{id}</span>}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}
