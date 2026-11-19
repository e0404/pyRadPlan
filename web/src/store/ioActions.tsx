/**
 * Flows of package C: loading files, saving objects (download), snapshots, base data, log level,
 * UI mode, and the window-level hooks (warnings after a run, close-while-busy prompt).
 * The Qt counterparts are named in each function.
 */
import { BackendError, type ExportFormat, type ExportObject, type SessionState } from "../api";
import { ndarray } from "../api/codec";
import type { BaseDataKind, BaseDataAddResult, ExportSaveParams, LoadKind } from "../api/protocol/io";
import { exportFormats, feature, isImageFile, loadAccept, structureFormatOf, unavailableImport } from "../components/io/capabilities";
import { describeDownload, downloadFile } from "../components/io/download";
import { askImportImage } from "../components/io/ImportImageDialog";
import { askYesNo, messageBox, promptText, showCritical, showInfo, showWarning } from "../components/io/messageBox";
import { RESERVED_TAGS, sanitizeTag } from "../components/io/models";
import { askSaveFile, askSaveResult, type QuantityChoice } from "../components/io/SaveDialogs";
import { log, storeInternals, useApp, type LogEntry } from "./appStore";
import { bumpBaseData, useIo, type LogThreshold } from "./ioStore";
import { resolvePendingChanges } from "./pendingChanges";
import { setUiMode, type UiMode } from "./uiMode";

const { call, reportError } = storeInternals;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

type RunResult<T> = { ok: true; value: T } | { ok: false };

/**
 * Busy/progress wrapper like the store's `run` (which only reports success): the value of `fn` is
 * returned, `handled` may consume an error before the generic error dialog.
 */
async function runIo<T>(label: string, fn: () => Promise<T>, handled?: (err: unknown) => boolean): Promise<RunResult<T>> {
  if (useApp.getState().busy) return { ok: false };
  useApp.setState({ busy: label, progress: { label, fraction: null }, statusMessage: `${label}…` });
  const started = performance.now();
  try {
    const value = await fn();
    useApp.setState({ statusMessage: `${label} finished (${((performance.now() - started) / 1000).toFixed(1)} s)` });
    return { ok: true, value };
  } catch (err) {
    if (!handled?.(err)) reportError(label, err);
    return { ok: false };
  } finally {
    useApp.setState({ busy: null, progress: null });
  }
}

const errorText = (err: unknown): string => (err instanceof Error ? err.message : String(err));

/** Open the browser's file chooser; resolves with the file, or `null` when cancelled. Needs a user gesture. */
export function pickFile(accept: string): Promise<File | null> {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = accept;
    input.style.display = "none";
    input.dataset.testid = "file-picker";
    const finish = (file: File | null) => {
      input.remove();
      resolve(file);
    };
    input.addEventListener("change", () => finish(input.files?.[0] ?? null));
    input.addEventListener("cancel", () => finish(null));
    document.body.appendChild(input);
    input.click();
  });
}

async function fileBytes(file: File) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  return ndarray(bytes, [bytes.length]);
}

// ---------------------------------------------------------------------------
// Loading (File menu, Workflow widget)
// ---------------------------------------------------------------------------

/** Qt `WorkflowWidget._on_load_file` after the file dialog. */
export async function loadFile(file: File): Promise<boolean> {
  const { info, state } = useApp.getState();
  let kind: LoadKind | undefined;
  const blocked = unavailableImport(info, file.name);
  if (blocked) {
    await showWarning("Load", blocked.reason ?? `${blocked.label} files cannot be loaded by this backend.`);
    return false;
  }
  if (isImageFile(file.name)) {
    const io = feature(info, "image_file_io");
    if (!io.available) {
      await showWarning("Load", io.reason ?? "Image files cannot be loaded by this backend.");
      return false;
    }
    const choice = await askImportImage(file.name, Boolean(state?.has.ct));
    if (!choice) return false;
    // The result name of a dose import has no protocol parameter yet (`data.load_file` takes `kind` only).
    kind = choice.kind;
  }
  const data = await fileBytes(file);
  log("info", `Uploading ${file.name} (${(data.data.length / 1024).toFixed(0)} KiB)`);
  return storeInternals.run(
    `Loading ${file.name}`,
    "data.load_file",
    { filename: file.name, data, ...(kind ? { kind } : {}) },
    "data",
  );
}

/** The File menu / Load File button: file chooser, then {@link loadFile}. */
export async function chooseAndLoadFile(): Promise<boolean> {
  const file = await pickFile(loadAccept(useApp.getState().info));
  return file ? loadFile(file) : false;
}

/** Qt `_on_load_dij` after the file dialog. */
export async function loadDijFile(file: File): Promise<boolean> {
  const data = await fileBytes(file);
  const result = await runIo(
    "Loading dose influence",
    () => call("data.load_dij", { filename: file.name, data }),
    (err) => {
      if (err instanceof BackendError && err.code === "no_dij") {
        void showWarning("Load Dij", "No dose-influence matrix found in the file.");
        return true;
      }
      return false;
    },
  );
  if (!result.ok) return false;
  useApp.setState({ state: result.value });
  return true;
}

export async function chooseAndLoadDij(): Promise<boolean> {
  const file = await pickFile(".mat");
  return file ? loadDijFile(file) : false;
}

// ---------------------------------------------------------------------------
// Saving (File menu; Qt `_workflow_widget.py:1489-1701`)
// ---------------------------------------------------------------------------

async function performSave(title: string, params: ExportSaveParams): Promise<boolean> {
  const out = await runIo(`${title}`, async () => {
    const result = await call("export.save", params);
    return { result, saved: downloadFile(result) };
  });
  if (!out.ok) return false;
  const text = describeDownload(out.value.result, out.value.saved);
  log("info", `${title}: saved ${text}`);
  useApp.setState({ statusMessage: `${title}: saved ${text}` });
  void showInfo(title, `Saved:\n${text}`, "saved-dialog");
  return true;
}

function formatsFor(object: ExportObject): ExportFormat[] {
  return exportFormats(useApp.getState().info, object);
}

export async function savePlan(): Promise<boolean> {
  const title = "Save Plan";
  if (!useApp.getState().state?.has.pln) {
    await showWarning(title, "No pln available to save.");
    return false;
  }
  if (!(await resolvePendingChanges(title))) return false;
  const choice = await askSaveFile({ title, formats: formatsFor("pln"), basename: "pln" });
  return choice ? performSave(title, { object: "pln", format: choice.format.format, basename: choice.basename }) : false;
}

export async function saveDij(): Promise<boolean> {
  const title = "Save Dij";
  if (!useApp.getState().state?.has.dij) {
    await showWarning(title, "No dij available to save.");
    return false;
  }
  const choice = await askSaveFile({ title, formats: formatsFor("dij"), basename: "dij" });
  return choice ? performSave(title, { object: "dij", format: choice.format.format, basename: choice.basename }) : false;
}

export async function saveCst(): Promise<boolean> {
  const title = "Save CST";
  const state = useApp.getState().state;
  if (!state?.has.cst) {
    await showWarning(title, "No cst available to save.");
    return false;
  }
  const choice = await askSaveFile({ title, formats: formatsFor("cst"), basename: "cst" });
  if (!choice) return false;
  const hasObjectives = state.structures.some((s) => (s.num_objectives ?? 0) > 0);
  if (hasObjectives && choice.format.keeps_objectives === false) {
    const answer = await messageBox({
      title,
      icon: "warning",
      text: `The ${choice.format.label} format cannot store objectives; only the structure masks will be exported. Continue?`,
      buttons: [
        { id: "ok", label: "OK", primary: true },
        { id: "cancel", label: "Cancel", reject: true },
      ],
      testId: "objectives-warning",
    });
    if (answer !== "ok") return false;
  }
  const structure_format = structureFormatOf(choice.format);
  return performSave(title, {
    object: "cst",
    format: choice.format.format,
    ...(structure_format ? { structure_format } : {}),
    basename: choice.basename,
  });
}

/** Exportable quantities of the result: "<label> (<key>)" like Qt's Save Result checklist. */
export function resultQuantityChoices(state: SessionState | null): QuantityChoice[] {
  if (!state) return [];
  const groups = state.results?.groups;
  if (groups) {
    return groups.flatMap((g) =>
      g.components
        .filter((c) => c.kind === "full")
        .map((c) => ({ key: c.key, text: g.tag ? c.key : `${g.meta_label} (${c.key})` })),
    );
  }
  return state.quantities.map((key) => ({ key, text: key }));
}

/** How several quantities are packed (PROTOCOL.md v1 amendment 3): `mat` is one `resultGUI`, the others zip. */
function resultArchiveNote(format: ExportFormat): string {
  return format.format === "mat"
    ? "All selected quantities are written into one resultGUI file."
    : "Several quantities are saved together in a zip archive.";
}

export async function saveResult(): Promise<boolean> {
  const title = "Save Result";
  const state = useApp.getState().state;
  if (!state?.has.result) {
    await showWarning(title, "No result available to save.");
    return false;
  }
  const quantities = resultQuantityChoices(state);
  if (!quantities.length) {
    await showWarning(title, "The result has no exportable image quantities.");
    return false;
  }
  const keys = await askSaveResult(quantities);
  if (!keys) return false;
  const choice = await askSaveFile({
    title,
    formats: formatsFor("result"),
    basename: keys.length === 1 ? keys[0] : "result",
    note: keys.length > 1 ? resultArchiveNote : undefined,
  });
  return choice
    ? performSave(title, { object: "result", format: choice.format.format, quantities: keys, basename: choice.basename })
    : false;
}

export async function saveWorkspace(): Promise<boolean> {
  const title = "Save Workspace";
  if (!useApp.getState().state?.has.ct) {
    await showWarning(title, "Load a CT before saving a workspace.");
    return false;
  }
  if (!(await resolvePendingChanges(title))) return false;
  const choice = await askSaveFile({ title, formats: formatsFor("workspace"), basename: "workspace" });
  return choice ? performSave(title, { object: "workspace", format: choice.format.format, basename: choice.basename }) : false;
}

// ---------------------------------------------------------------------------
// Snapshot Result (Qt `_on_save_result`)
// ---------------------------------------------------------------------------

export async function snapshotResult(): Promise<boolean> {
  const title = "Snapshot Result";
  const { state } = useApp.getState();
  const plan = state?.plan;
  const fallback = plan ? `${plan.radiation_mode}_${plan.num_of_fractions}fx` : "result";
  const text = await promptText({
    title,
    label: "Name for this result snapshot:",
    value: state?.results?.default_tag || fallback,
  });
  if (text === null) return false;
  const tag = sanitizeTag(text);
  if (!tag) return false;
  if ((RESERVED_TAGS as readonly string[]).includes(tag)) {
    await showWarning(title, `'${tag}' names per-${tag} results and cannot be used as a snapshot name.`);
    return false;
  }
  const out = await runIo(title, () => call("workflow.snapshot", { tag }));
  if (!out.ok) return false;
  useApp.setState({ state: out.value.state });
  log("info", `Saved result snapshot '${out.value.tag}'`);
  await storeInternals.reloadQuantities();
  return true;
}

// ---------------------------------------------------------------------------
// Base data (Qt `menus/_base_data.py`)
// ---------------------------------------------------------------------------

export const BASE_DATA_LABELS: Record<Exclude<BaseDataKind, "phantoms">, string> = {
  machines: "machine",
  rbe_tables: "RBE table",
  stopping_power_tables: "stopping power table",
};

function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** Base data was added or removed (Qt `base_data_changed`): the plan widgets refetch their options. */
async function baseDataChanged(): Promise<void> {
  bumpBaseData();
  try {
    // `state.revisions.base_data` is what the plan widgets watch.
    useApp.setState({ state: await call("session.state", {}) });
  } catch {
    /* the next state event carries it */
  }
}

/**
 * Qt `add_base_data_interactively`: validate the file, resolve name clashes by asking, add it.
 * Resolves with the added row, or `null` when cancelled or invalid.
 */
export async function addBaseDataFile(kind: Exclude<BaseDataKind, "phantoms">, file: File): Promise<BaseDataAddResult | null> {
  const label = BASE_DATA_LABELS[kind];
  const title = `Add ${capitalize(label)}`;
  const data = await fileBytes(file);
  let probe: BaseDataAddResult;
  try {
    probe = await call("base_data.add", { kind, filename: file.name, data, dry_run: true });
  } catch (err) {
    await showCritical(title, `Could not read a ${label} from\n${file.name}\n\n${errorText(err)}`);
    return null;
  }
  let name = probe.suggested_name;
  let overwrite = false;
  for (;;) {
    try {
      const added = await call("base_data.add", {
        kind,
        filename: file.name,
        data,
        name,
        radiation_mode: probe.radiation_mode,
        overwrite,
      });
      await baseDataChanged();
      return added;
    } catch (err) {
      if (!(err instanceof BackendError && err.code === "name_clash")) {
        await showCritical(title, errorText(err));
        return null;
      }
    }
    const clash = (await call("base_data.add", { kind, filename: file.name, data, name, radiation_mode: probe.radiation_mode, dry_run: true })).clash;
    let prompt: string;
    if (clash?.source === "user") {
      const answer = await messageBox({
        title,
        icon: "question",
        text: `A ${label} named '${name}' was already added.\nReplace it?`,
        buttons: [
          { id: "yes", label: "Yes" },
          { id: "no", label: "No", primary: true },
          { id: "cancel", label: "Cancel", reject: true },
        ],
        testId: "replace-dialog",
      });
      if (answer === "cancel") return null;
      if (answer === "yes") {
        overwrite = true;
        continue;
      }
      prompt = `Name to add the ${label} under:`;
    } else {
      prompt = `A ${label} named '${name}' ships with pyRadPlan.\nName to add the ${label} under:`;
    }
    const entered = await promptText({ title, label: <PromptLines text={prompt} />, value: name });
    const trimmed = entered?.trim();
    if (!trimmed) return null;
    name = trimmed;
    overwrite = false;
  }
}

/** The Base Data menu's Add entries: file chooser, add, confirmation (Qt `BaseDataMenu.add`). */
export async function chooseAndAddBaseData(kind: Exclude<BaseDataKind, "phantoms">): Promise<boolean> {
  const file = await pickFile(".mat");
  if (!file) return false;
  const added = await addBaseDataFile(kind, file);
  if (!added) return false;
  await showInfo("Base Data", `Added the ${BASE_DATA_LABELS[kind]} '${added.row?.file ?? added.suggested_name}'.`, "base-data-added");
  return true;
}

export async function removeBaseDataFile(kind: Exclude<BaseDataKind, "phantoms">, file: string, name: string): Promise<boolean> {
  const confirmed = await askYesNo(
    "Remove Base Data",
    `Remove the ${BASE_DATA_LABELS[kind]} '${name}'?\nIts file is deleted from your base data folder.`,
    "remove-dialog",
  );
  if (!confirmed) return false;
  try {
    await call("base_data.remove", { kind, file });
  } catch (err) {
    await showCritical("Remove Base Data", errorText(err));
    return false;
  }
  await baseDataChanged();
  return true;
}

/** Fetch the phantom list for the File menu (a v0 backend falls back to `info.phantoms`). */
export async function refreshPhantoms(): Promise<void> {
  const { backend, session, info } = useApp.getState();
  if (!backend || !session || !info?.capabilities) return;
  try {
    const listing = await call("base_data.list", { kind: "phantoms" });
    useIo.setState({ phantoms: { rows: listing.rows, folders: listing.folders } });
  } catch {
    useIo.setState({ phantoms: null });
  }
}

// ---------------------------------------------------------------------------
// UI mode, log level, settings
// ---------------------------------------------------------------------------

/** Qt `FileMenu._on_ui_mode`: set the mode and mirror it to the backend's `gui.ui_mode`. */
export async function chooseUiMode(mode: UiMode): Promise<void> {
  setUiMode(mode);
  const { backend, session, info } = useApp.getState();
  if (!backend || !session || !info?.capabilities) return;
  try {
    await call("app.settings_set", { section: "gui", values: { ui_mode: mode } });
  } catch {
    /* the mode is held by the frontend; a backend without a gui section does not matter */
  }
}

/** Qt log combo: filter the console and set the backend's `pyRadPlan` logger level. */
export async function setLogLevel(level: LogThreshold): Promise<void> {
  useIo.setState({ logLevel: level });
  const { session, info } = useApp.getState();
  if (!session || !info?.methods?.includes("session.log_level")) return;
  try {
    await call("session.log_level", { level });
  } catch (err) {
    log("warning", `Could not set the log level: ${errorText(err)}`);
  }
}

/** Merge the backend's log records after the newest one the console has (`session.history`). */
export async function fetchLogHistory(): Promise<void> {
  const { session, info, logs } = useApp.getState();
  if (!session || !info?.methods?.includes("session.history")) return;
  const since = logs.reduce((m, l) => Math.max(m, l.seq ?? 0), 0);
  try {
    const history = await call("session.history", { since });
    if (!history.log.length) return;
    const known = new Set(logs.map((l) => l.seq).filter((s): s is number => s !== undefined));
    const entries: LogEntry[] = history.log
      .filter((r) => r.seq !== undefined && !known.has(r.seq) && !r.repeat)
      .map((r) => ({ id: -1_000_000 - (r.seq ?? 0), time: new Date(), level: r.level, message: r.message, logger: r.logger, seq: r.seq }));
    if (entries.length) useApp.setState((s) => ({ logs: [...entries, ...s.logs].slice(-2000) }));
  } catch {
    /* history is a convenience */
  }
}

/** Adopt the backend's `gui.ui_mode` (a configured native backend may start in the educational mode). */
export async function adoptBackendUiMode(): Promise<void> {
  const { info, session } = useApp.getState();
  if (!session || !info?.methods?.includes("app.settings")) return;
  try {
    const { sections } = await call("app.settings", {});
    const mode = sections.find((s) => s.id === "gui")?.form.values.ui_mode;
    if (mode === "default" || mode === "educational") setUiMode(mode);
  } catch {
    /* keep the frontend default */
  }
}

// ---------------------------------------------------------------------------
// Window-level hooks
// ---------------------------------------------------------------------------

/**
 * Warnings the backend logs while an operation runs are shown once afterwards (Qt
 * `_show_pending_warnings`), and the browser asks before the page is closed while busy (Qt
 * `closeEvent`: "Computation running … Quit anyway?"). Returns the cleanup function.
 */
export function installWindowHooks(): () => void {
  const pending: string[] = [];
  let attached: unknown = null;
  let detach: () => void = () => {};

  const attach = (backend: ReturnType<typeof useApp.getState>["backend"]) => {
    if (backend === attached) return;
    detach();
    attached = backend;
    detach = backend
      ? backend.subscribe((event) => {
          // Only algorithm warning reports, as Qt's WarningReport (not every logged warning).
          if (event.event === "log" && event.report && useApp.getState().busy !== null && !pending.includes(event.message)) {
            pending.push(event.message);
          }
        })
      : () => {};
  };
  attach(useApp.getState().backend);

  const checkSession = (s: ReturnType<typeof useApp.getState>) => {
    if (s.session && s.info && s.session !== handledSession) {
      handledSession = s.session;
      void onSession();
    }
  };
  const unsubscribe = useApp.subscribe((s, prev) => {
    attach(s.backend);
    if (prev.busy === null && s.busy !== null) pending.length = 0;
    if (prev.busy !== null && s.busy === null && pending.length) {
      void showWarning("Warning", pending.splice(0).join("\n\n"), "run-warnings");
    }
    checkSession(s);
  });
  checkSession(useApp.getState());

  const beforeUnload = (event: BeforeUnloadEvent) => {
    if (useApp.getState().busy === null) return;
    event.preventDefault();
    event.returnValue = "A computation is still running. Quit anyway?";
  };
  window.addEventListener("beforeunload", beforeUnload);

  return () => {
    unsubscribe();
    detach();
    window.removeEventListener("beforeunload", beforeUnload);
  };
}

let handledSession: string | null = null;

/** Work to do once per session: log level, history, phantom list, backend UI mode. */
async function onSession(): Promise<void> {
  await setLogLevel(useIo.getState().logLevel);
  await Promise.all([fetchLogHistory(), refreshPhantoms(), adoptBackendUiMode()]);
}

function PromptLines(props: { text: string }) {
  return <span style={{ whiteSpace: "pre-wrap" }}>{props.text}</span>;
}
