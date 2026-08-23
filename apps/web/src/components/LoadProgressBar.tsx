import type { LoadProgress } from "@tensorium/model-ir";
import { formatBytes } from "../format.js";
import { useTranslation } from "./LanguageContext.js";
import type { TranslationKey } from "../i18n.js";

const PHASE_LABEL_KEY: Record<LoadProgress["phase"], TranslationKey> = {
  config: "loader.progress.config",
  structure: "loader.progress.structure",
  weights: "loader.progress.weights",
  parsing: "loader.progress.parsing",
  building: "loader.progress.building",
  tokenizer: "loader.progress.tokenizer",
};

/**
 * Renders a byte-accurate progress bar for the phases that actually stream
 * bytes over the network ("weights", "tokenizer") — those carry both
 * `loadedBytes` and `totalBytes` whenever the server sent a Content-Length.
 * Every other phase (config peek, header parsing, graph building) is over
 * in milliseconds with no meaningful fraction to show, so it renders as an
 * indeterminate (animated, no fixed width) bar with just the phase label.
 */
export function LoadProgressBar({ progress }: { progress: LoadProgress }) {
  const { t } = useTranslation();
  const { phase, loadedBytes, totalBytes } = progress;
  const pct = totalBytes && loadedBytes !== undefined ? Math.min(100, (loadedBytes / totalBytes) * 100) : undefined;

  return (
    <div className="load-progress">
      <div className="load-progress-label">
        <span>{t(PHASE_LABEL_KEY[phase])}</span>
        {loadedBytes !== undefined && totalBytes !== undefined && (
          <span className="load-progress-bytes">
            {formatBytes(loadedBytes)} / {formatBytes(totalBytes)}
          </span>
        )}
      </div>
      <div className={"load-progress-track" + (pct === undefined ? " indeterminate" : "")}>
        <div className="load-progress-fill" style={pct === undefined ? undefined : { width: `${pct}%` }} />
      </div>
    </div>
  );
}
