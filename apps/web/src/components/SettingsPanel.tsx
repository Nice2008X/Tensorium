import { useEffect, useRef } from "react";
import { useTranslation } from "./LanguageContext.js";
import { ThemeSwitcher, type Theme } from "./ThemeSwitcher.js";
import { LANGUAGES, type Language } from "../i18n.js";

interface Props {
  open: boolean;
  onClose: () => void;
  theme: Theme;
  onThemeChange: (t: Theme) => void;
  allowSyntheticForwardPass: boolean;
  onAllowSyntheticForwardPassChange: (v: boolean) => void;
  /** True when the currently loaded model's real weight bytes exceed the hard 20GB ceiling — the checkbox is disabled outright, since enabling it couldn't do anything for this model (forwardPassBlocked stays true regardless). Defaults to false — no model, or a model under the ceiling, never disables this. */
  forwardPassSettingDisabled?: boolean;
}

export function SettingsButton({ open, onToggle }: { open: boolean; onToggle: () => void }) {
  const { t } = useTranslation();
  return (
    <button className={"settings-toggle-btn" + (open ? " active" : "")} onClick={onToggle} title={t("app.settings")} aria-label={t("app.settings")} aria-expanded={open}>
      ⚙
    </button>
  );
}

export function SettingsPanel({
  open,
  onClose,
  theme,
  onThemeChange,
  allowSyntheticForwardPass,
  onAllowSyntheticForwardPassChange,
  forwardPassSettingDisabled = false,
}: Props) {
  const { t, language, setLanguage } = useTranslation();
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    function onPointerDown(e: PointerEvent) {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) onClose();
    }
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("pointerdown", onPointerDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("pointerdown", onPointerDown);
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="settings-panel" ref={panelRef}>
      <div className="settings-panel-header">
        <span>{t("settings.title")}</span>
        <button className="settings-panel-close" onClick={onClose} aria-label={t("settings.close")} title={t("settings.close")}>
          ×
        </button>
      </div>
      <div className="settings-section">
        <div className="settings-section-title">{t("settings.theme")}</div>
        <ThemeSwitcher theme={theme} onChange={onThemeChange} />
      </div>
      <div className="settings-section">
        <div className="settings-section-title">{t("settings.language")}</div>
        <select className="settings-language-select" value={language} onChange={(e) => setLanguage(e.target.value as Language)}>
          {LANGUAGES.map((l) => (
            <option key={l.id} value={l.id}>
              {l.nativeLabel}
            </option>
          ))}
        </select>
      </div>
      <div className="settings-section">
        <label className={"settings-checkbox-row" + (forwardPassSettingDisabled ? " disabled" : "")}>
          <input
            type="checkbox"
            checked={allowSyntheticForwardPass}
            disabled={forwardPassSettingDisabled}
            onChange={(e) => onAllowSyntheticForwardPassChange(e.target.checked)}
          />
          {t("settings.structureOnlyForwardPass")}
        </label>
        <div className="settings-section-desc">{t(forwardPassSettingDisabled ? "settings.structureOnlyForwardPassDisabledDesc" : "settings.structureOnlyForwardPassDesc")}</div>
      </div>
    </div>
  );
}
