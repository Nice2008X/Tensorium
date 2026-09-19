import { useEffect } from "react";
import type { WeightsMode } from "@tensorium/model-ir";
import { formatBytes } from "../format.js";
import { useTranslation } from "./LanguageContext.js";

interface Props {
  open: boolean;
  totalBytes?: number;
  onChoose: (mode: WeightsMode | null) => void;
}

/** Weights are held as 8-byte floats once a forward pass touches them — the same 4x-of-BF16 floor App.tsx uses for its own memory estimate. */
const FORWARD_PASS_MEMORY_FACTOR = 4;

/**
 * Shown before a big (over LARGE_MODEL_WARNING_BYTES) single-file checkpoint
 * loads: downloading the real weights is a multi-GB transfer that can make
 * the tab sluggish once a forward pass runs, so the user picks it
 * deliberately instead of it happening silently. Neither option is
 * pre-emphasised as "the" default; Escape/backdrop/Cancel all load nothing.
 */
export function LargeModelDialog({ open, totalBytes, onChoose }: Props) {
  const { t } = useTranslation();

  useEffect(() => {
    if (!open) return;
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onChoose(null);
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [open, onChoose]);

  if (!open || totalBytes === undefined) return null;

  const size = formatBytes(totalBytes);
  const memory = formatBytes(totalBytes * FORWARD_PASS_MEMORY_FACTOR);

  return (
    <div
      className="save-model-dialog-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onChoose(null);
      }}
    >
      <div className="save-model-dialog large-model-dialog" role="dialog" aria-modal="true" aria-label={t("app.largeModelDialogTitle")}>
        <div className="save-model-dialog-header">
          <span>{t("app.largeModelDialogTitle")}</span>
          <button className="save-model-dialog-close" onClick={() => onChoose(null)} aria-label={t("loader.close")} title={t("loader.close")}>
            ×
          </button>
        </div>
        <p className="save-model-dialog-desc">{t("app.largeModelDialogDesc").replace("{size}", size)}</p>
        <button className="large-model-option" onClick={() => onChoose("download")}>
          <span className="large-model-option-title">{t("app.largeModelDownloadTitle").replace("{size}", size)}</span>
          <span className="large-model-option-desc">{t("app.largeModelDownloadDesc")}</span>
          <span className="large-model-option-warning">{t("app.largeModelDownloadWarning").replace("{memory}", memory)}</span>
        </button>
        <button className="large-model-option" onClick={() => onChoose("structure-only")}>
          <span className="large-model-option-title">{t("app.largeModelStructureTitle")}</span>
          <span className="large-model-option-desc">{t("app.largeModelStructureDesc")}</span>
        </button>
        <div className="save-model-dialog-actions">
          <button className="save-model-dialog-cancel" onClick={() => onChoose(null)}>
            {t("app.cancel")}
          </button>
        </div>
      </div>
    </div>
  );
}
