import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { LoadProgress } from "@tensorium/model-ir";
import { PRESET_MODELS } from "../adapters.js";
import { useTranslation } from "./LanguageContext.js";
import { checkJsonFile, checkWeightsFile, type FileCheck } from "../localFileValidation.js";
import { formatBytes } from "../format.js";
import { LoadProgressBar } from "./LoadProgressBar.js";

export interface LocalModelFiles {
  name: string;
  config: File;
  weights: File;
  tokenizer?: File;
}

interface Props {
  status: "idle" | "loading" | "ready" | "error";
  error?: string;
  progress?: LoadProgress;
  onLoad: (repo: string) => void;
  onLoadLocal: (files: LocalModelFiles) => void;
  /** Presets matching this repo id are left out of the list — used when re-opening the loader for a model that's already loaded, so it isn't offered back as if it were a fresh option. */
  excludeRepo?: string;
  /** Drops the built-in title/subtitle and card chrome (background/border/padding) — used when this is embedded inside a panel that already provides its own header, e.g. the "load a different model" popover. */
  embedded?: boolean;
}

type SourceMode = "huggingface" | "local";

/** Strips a trailing `.safetensors` so a file named `model.safetensors` becomes the model's display name `model`, not a name with a stray extension. */
function defaultModelName(filename: string): string {
  return filename.replace(/\.safetensors$/i, "");
}

interface Validation {
  checking: boolean;
  check: FileCheck | null;
}
const IDLE_VALIDATION: Validation = { checking: false, check: null };
/** How many presets a preset tab shows per page — configurable in one place. */
const PRESETS_PER_PAGE = 5;

interface PresetGroupSpec {
  key: string;
  title: ReactNode;
  presets: { repo: string; label: string }[];
}

/**
 * Splits preset chips into tabs — one per group (e.g. "without MoE" vs.
 * "with MoE (Mixture-of-Experts)") — instead of showing every chip at
 * once. Only one group is visible at a time, so the list stays scannable
 * as more architectures (and more presets per group) get added, without
 * needing an internal scrollbar. Each tab's label always states its group
 * name and count, so which group you're looking at is never ambiguous.
 */
function PresetTabs({ groups, onPick }: { groups: PresetGroupSpec[]; onPick: (repo: string) => void }) {
  const nonEmptyGroups = useMemo(() => groups.filter((g) => g.presets.length > 0), [groups]);
  const [activeKey, setActiveKey] = useState(nonEmptyGroups[0]?.key);
  const active = nonEmptyGroups.find((g) => g.key === activeKey) ?? nonEmptyGroups[0];
  // Remembered per tab (keyed by group key) rather than one shared number,
  // so paging into "Real bigger models" doesn't also jump "Models with
  // MoE" to that same page when you switch back to it.
  const [pageByGroup, setPageByGroup] = useState<Record<string, number>>({});
  if (!active) return null;
  const totalPages = Math.max(1, Math.ceil(active.presets.length / PRESETS_PER_PAGE));
  // Clamped rather than reset via an effect — handles a page that no
  // longer exists (e.g. the preset list shrank) without an extra render.
  const page = Math.min(pageByGroup[active.key] ?? 1, totalPages);
  const pagePresets = active.presets.slice((page - 1) * PRESETS_PER_PAGE, page * PRESETS_PER_PAGE);
  return (
    <div className="model-loader-preset-tabs-wrap">
      <div className="model-loader-source-tabs model-loader-preset-tabs">
        {nonEmptyGroups.map((g) => (
          <button key={g.key} type="button" className={g.key === active.key ? "active" : ""} onClick={() => setActiveKey(g.key)}>
            {g.title} <span className="model-loader-preset-tab-count">({g.presets.length})</span>
          </button>
        ))}
      </div>
      <div className="model-loader-presets">
        {pagePresets.map((p) => (
          <PresetChip key={p.repo} label={p.label} onClick={() => onPick(p.repo)} />
        ))}
      </div>
      {totalPages > 1 && (
        <div className="model-loader-preset-pagination">
          {Array.from({ length: totalPages }, (_, i) => i + 1).map((n) => (
            <button
              key={n}
              type="button"
              className={"model-loader-preset-page" + (n === page ? " active" : "")}
              onClick={() => setPageByGroup((prev) => ({ ...prev, [active.key]: n }))}
            >
              {n}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * A tooltip repeating text that's already fully visible is just noise —
 * this only sets `title` once the CSS ellipsis (see `.preset-chip` in
 * styles.css) has actually cut the label off, detected by comparing the
 * button's scrollWidth (the full untruncated content) against its
 * clientWidth (what's actually visible) after layout. Re-checked on
 * window resize since the card's available width can change (the
 * embedded "load a different model" popover isn't a fixed size the way
 * the first-load screen's card is).
 */
function PresetChip({ label, onClick }: { label: string; onClick: () => void }) {
  const ref = useRef<HTMLButtonElement>(null);
  const [truncated, setTruncated] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const check = () => setTruncated(el.scrollWidth > el.clientWidth);
    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, [label]);
  return (
    <button ref={ref} className="preset-chip" onClick={onClick} title={truncated ? label : undefined}>
      {label}
    </button>
  );
}

/** Inline feedback under a file input: a "checking…" state while content-sniffing runs, then the validator's error/warning, or a plain size confirmation once a file passes. Content is sniffed (not just the file extension) so a mislabeled or corrupt file is caught here instead of failing deep inside the model adapter later. */
function FileRowStatus({ file, validation }: { file: File | null; validation: Validation }) {
  if (!file) return null;
  if (validation.checking) return <span className="model-loader-file-status checking">Checking file…</span>;
  if (validation.check?.error) return <span className="model-loader-file-status error">{validation.check.error}</span>;
  if (validation.check?.warning) return <span className="model-loader-file-status warning">{validation.check.warning}</span>;
  if (validation.check?.ok) return <span className="model-loader-file-status ok">{formatBytes(file.size)}</span>;
  return null;
}

export function ModelLoader({ status, error, progress, onLoad, onLoadLocal, excludeRepo, embedded }: Props) {
  const { t } = useTranslation();
  const sortedPresets = useMemo(
    () => PRESET_MODELS.filter((p) => p.repo !== excludeRepo).sort((a, b) => a.label.localeCompare(b.label)),
    [excludeRepo]
  );
  const presetsWithoutMoe = useMemo(() => sortedPresets.filter((p) => !p.isMoE && !p.isLarge), [sortedPresets]);
  const presetsWithMoe = useMemo(() => sortedPresets.filter((p) => p.isMoE && !p.isLarge), [sortedPresets]);
  const presetsLarge = useMemo(() => sortedPresets.filter((p) => p.isLarge), [sortedPresets]);
  const [repo, setRepo] = useState(() => PRESET_MODELS.find((p) => p.repo !== excludeRepo)?.repo ?? "");
  const [mode, setMode] = useState<SourceMode>("huggingface");
  const [configFile, setConfigFile] = useState<File | null>(null);
  const [weightsFile, setWeightsFile] = useState<File | null>(null);
  const [tokenizerFile, setTokenizerFile] = useState<File | null>(null);
  const [configValidation, setConfigValidation] = useState<Validation>(IDLE_VALIDATION);
  const [weightsValidation, setWeightsValidation] = useState<Validation>(IDLE_VALIDATION);
  const [tokenizerValidation, setTokenizerValidation] = useState<Validation>(IDLE_VALIDATION);
  // Guards against a stale validation result clobbering state if the user
  // swaps a file again before the previous (async, content-sniffing) check
  // finishes — only the result matching the most recent pick for that field
  // is applied.
  const configNonce = useRef(0);
  const weightsNonce = useRef(0);
  const tokenizerNonce = useRef(0);

  function pickFile(
    file: File | null,
    setFile: (f: File | null) => void,
    setValidation: (v: Validation) => void,
    nonceRef: { current: number },
    validate: (f: File) => Promise<FileCheck>
  ) {
    setFile(file);
    const nonce = ++nonceRef.current;
    if (!file) {
      setValidation(IDLE_VALIDATION);
      return;
    }
    setValidation({ checking: true, check: null });
    validate(file).then((check) => {
      if (nonceRef.current === nonce) setValidation({ checking: false, check });
    });
  }

  const canLoadLocal =
    !!configFile &&
    configValidation.check?.ok === true &&
    !!weightsFile &&
    weightsValidation.check?.ok === true &&
    (!tokenizerFile || tokenizerValidation.check?.ok === true) &&
    status !== "loading";

  return (
    <div className={"model-loader" + (embedded ? " embedded" : "")}>
      {!embedded && (
        <>
          <div className="model-loader-title">{t("loader.title")}</div>
          <div className="model-loader-sub">{t("loader.subtitle")}</div>
          <div className="model-loader-limitation">{t("loader.limitationNote")}</div>
        </>
      )}
      <div className="model-loader-source-tabs">
        <button type="button" className={mode === "huggingface" ? "active" : ""} onClick={() => setMode("huggingface")}>
          {t("loader.sourceHf")}
        </button>
        <button type="button" className={mode === "local" ? "active" : ""} onClick={() => setMode("local")}>
          {t("loader.sourceLocal")}
        </button>
      </div>

      {mode === "huggingface" ? (
        <>
          <form
            className="model-loader-form"
            onSubmit={(e) => {
              e.preventDefault();
              onLoad(repo);
            }}
          >
            <input value={repo} onChange={(e) => setRepo(e.target.value)} placeholder={t("loader.inputPlaceholder")} />
            <button type="submit" disabled={status === "loading"}>
              {status === "loading" ? t("loader.loading") : t("loader.load")}
            </button>
          </form>
          <PresetTabs
            groups={[
              { key: "without-moe", title: t("loader.presetsWithoutMoe"), presets: presetsWithoutMoe },
              { key: "with-moe", title: t("loader.presetsWithMoe"), presets: presetsWithMoe },
              { key: "large", title: t("loader.presetsLarge"), presets: presetsLarge },
            ]}
            onPick={(repo) => {
              setRepo(repo);
              onLoad(repo);
            }}
          />
        </>
      ) : (
        <form
          className="model-loader-local-form"
          onSubmit={(e) => {
            e.preventDefault();
            if (!canLoadLocal || !configFile || !weightsFile) return;
            onLoadLocal({ name: defaultModelName(weightsFile.name), config: configFile, weights: weightsFile, tokenizer: tokenizerFile ?? undefined });
          }}
        >
          <p className="model-loader-local-hint">{t("loader.localHint")}</p>
          <label className="model-loader-file-row">
            <span className="model-loader-file-label">{t("loader.localConfig")}</span>
            <span className="model-loader-file-desc">{t("loader.localConfigDesc")}</span>
            <input
              type="file"
              accept=".json"
              onChange={(e) => pickFile(e.target.files?.[0] ?? null, setConfigFile, setConfigValidation, configNonce, checkJsonFile)}
            />
            <FileRowStatus file={configFile} validation={configValidation} />
          </label>
          <label className="model-loader-file-row">
            <span className="model-loader-file-label">{t("loader.localWeights")}</span>
            <span className="model-loader-file-desc">{t("loader.localWeightsDesc")}</span>
            <input
              type="file"
              accept=".safetensors"
              onChange={(e) => pickFile(e.target.files?.[0] ?? null, setWeightsFile, setWeightsValidation, weightsNonce, checkWeightsFile)}
            />
            <FileRowStatus file={weightsFile} validation={weightsValidation} />
          </label>
          <label className="model-loader-file-row">
            <span className="model-loader-file-label">{t("loader.localTokenizer")}</span>
            <span className="model-loader-file-desc">{t("loader.localTokenizerDesc")}</span>
            <input
              type="file"
              accept=".json"
              onChange={(e) => pickFile(e.target.files?.[0] ?? null, setTokenizerFile, setTokenizerValidation, tokenizerNonce, checkJsonFile)}
            />
            <FileRowStatus file={tokenizerFile} validation={tokenizerValidation} />
          </label>
          <button type="submit" disabled={!canLoadLocal}>
            {status === "loading" ? t("loader.loading") : t("loader.load")}
          </button>
        </form>
      )}
      {status === "loading" && progress && <LoadProgressBar progress={progress} />}
      {status === "error" && <div className="model-loader-error">{error}</div>}
    </div>
  );
}
