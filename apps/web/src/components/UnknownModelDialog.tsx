import { useEffect } from "react";
import { useTranslation } from "./LanguageContext.js";

interface Props {
  open: boolean;
  modelType?: string;
  architectures?: string[];
  onCancel: () => void;
  onConfirm: () => void;
}

/**
 * Gates GenericAdapter's best-effort load behind an explicit choice — shown
 * whenever every named (hand-verified) adapter has declined a checkpoint
 * but its weight layout still structurally resembles a supported
 * transformer block. Defaults to Cancel in every sense that matters: no
 * button is auto-focused into looking primary, Escape/backdrop-click both
 * cancel, and nothing loads unless "Load anyway" is clicked deliberately.
 */
export function UnknownModelDialog({ open, modelType, architectures, onCancel, onConfirm }: Props) {
  const { t } = useTranslation();

  useEffect(() => {
    if (!open) return;
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onCancel();
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [open, onCancel]);

  if (!open) return null;

  const detected = [modelType, ...(architectures ?? [])].filter(Boolean).join(" · ") || "—";

  return (
    <div
      className="save-model-dialog-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onCancel();
      }}
    >
      <div className="save-model-dialog" role="dialog" aria-modal="true" aria-label={t("app.unknownModelDialogTitle")}>
        <div className="save-model-dialog-header">
          <span>{t("app.unknownModelDialogTitle")}</span>
          <button className="save-model-dialog-close" onClick={onCancel} aria-label={t("loader.close")} title={t("loader.close")}>
            ×
          </button>
        </div>
        <p className="save-model-dialog-desc">{t("app.unknownModelDialogDesc")}</p>
        <div className="unknown-model-dialog-detected">
          <span className="unknown-model-dialog-detected-label">{t("app.unknownModelDialogModelType")}</span>
          <span className="unknown-model-dialog-detected-value">{detected}</span>
        </div>
        <div className="save-model-dialog-actions">
          <button className="save-model-dialog-cancel" onClick={onCancel}>
            {t("app.cancel")}
          </button>
          <button className="unknown-model-dialog-confirm" onClick={onConfirm}>
            {t("app.unknownModelDialogConfirm")}
          </button>
        </div>
      </div>
    </div>
  );
}
