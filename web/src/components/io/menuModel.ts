/**
 * Pure model of the menu bar (Qt `menus/_file.py`, `_view.py`, `_base_data.py`, `_settings.py`,
 * `_help.py`): which entries exist, whether they are enabled and why not. Enabling follows the
 * session `actions` and `app.info.capabilities`, not knowledge of the runtime.
 */
import type { AppInfo, ExportObject, SessionState } from "../../api";
import type { PhantomListing } from "../../store/ioStore";
import { UI_MODES, UI_MODE_TIPS, type UiMode } from "../../store/uiMode";
import { exportBlockedReason, feature } from "./capabilities";

export interface MenuItem {
  kind: "item";
  id: string;
  label: string;
  enabled: boolean;
  tooltip?: string;
  /** Present on checkable entries. */
  checked?: boolean;
  /** Entry of an exclusive group (shown with a bullet instead of a check mark). */
  radio?: boolean;
  /** Handler key and argument. */
  action: string;
  arg?: string;
}

export interface MenuSeparator {
  kind: "separator";
  id: string;
}

export interface MenuSubmenu {
  kind: "submenu";
  id: string;
  label: string;
  enabled: boolean;
  tooltip?: string;
  entries: MenuEntry[];
}

export type MenuEntry = MenuItem | MenuSeparator | MenuSubmenu;

export interface MenuDefinition {
  id: string;
  /** Title with the Qt mnemonic marked by `&`. */
  title: string;
  /** Qt disables a whole menu (the File menu while busy). */
  enabled: boolean;
  entries: MenuEntry[];
}

export interface MenuInput {
  info: AppInfo | null;
  state: SessionState | null;
  busy: string | null;
  phantoms: PhantomListing | null;
  uiMode: UiMode;
  dark: boolean;
  showLog: boolean;
}

export const DOCUMENTATION_URL = "https://pyradplan.readthedocs.io";
export const SOURCE_URL = "https://github.com/e0404/pyRadPlan";

const sep = (id: string): MenuSeparator => ({ kind: "separator", id });

const item = (
  id: string,
  label: string,
  action: string,
  options: Partial<Omit<MenuItem, "kind" | "id" | "label" | "action">> = {},
): MenuItem => ({ kind: "item", id, label, action, enabled: true, ...options });

/** Tooltip text with the reason a stub is disabled appended. */
function withReason(tooltip: string | undefined, reason: string | null | undefined): string | undefined {
  if (!reason) return tooltip;
  return tooltip ? `${tooltip}\n\nNot available: ${reason}` : `Not available: ${reason}`;
}

const SAVE_ENTRIES: Array<{ id: string; label: string; object: ExportObject; action: string; tooltip?: string }> = [
  { id: "save_pln", label: "Save Plan…", object: "pln", action: "savePlan" },
  { id: "save_dij", label: "Save Dij…", object: "dij", action: "saveDij" },
  { id: "save_cst", label: "Save CST…", object: "cst", action: "saveCst", tooltip: "Includes the structures' objectives" },
  { id: "save_result", label: "Save Result…", object: "result", action: "saveResult" },
  { id: "save_workspace", label: "Save Workspace…", object: "workspace", action: "saveWorkspace" },
];

const HAS_KEY: Record<ExportObject, keyof SessionState["has"]> = {
  pln: "pln",
  dij: "dij",
  cst: "cst",
  result: "result",
  workspace: "ct",
};

/** Whether the session has the object a save entry writes (`actions.save_*`, else `has`). */
export function canSave(state: SessionState | null, entryId: string, object: ExportObject): boolean {
  if (!state) return false;
  const flag = state.actions?.[entryId];
  return typeof flag === "boolean" ? flag : state.has[HAS_KEY[object]];
}

export function phantomEntries(info: AppInfo | null, listing: PhantomListing | null, enabled: boolean): MenuEntry[] {
  const entries: MenuEntry[] = [];
  const rows: Array<{ name: string; file: string; source: string; path?: string }> =
    listing?.rows ?? (info?.phantoms ?? []).map((name) => ({ name, file: `${name}.mat`, source: "bundled" }));
  let previous: string | null = null;
  for (const row of rows) {
    if (previous !== null && row.source !== previous) entries.push(sep(`phantom-sep-${row.source}`));
    previous = row.source;
    const name = row.name.replace(/\.(mat|npz)$/i, "");
    const folder = listing?.folders.find((f) => f.source === row.source)?.path;
    entries.push(
      item(`phantom:${name}`, name, "loadPhantom", {
        arg: name,
        enabled,
        tooltip: row.source === "bundled" ? "Shipped with pyRadPlan" : (row.path ?? (folder ? `${folder}/${row.file}` : row.file)),
      }),
    );
  }
  return entries;
}

export function buildMenus(input: MenuInput): MenuDefinition[] {
  const { info, state, busy } = input;
  const idle = state !== null && busy === null;
  const dicom = feature(info, "dicom_import");
  const imageIo = feature(info, "image_file_io");
  const folderReason = !dicom.available
    ? dicom.reason
    : !imageIo.available
      ? imageIo.reason
      : "Folder import is not available in the web interface.";
  const openFolder = feature(info, "open_folder");

  const phantomMenu: MenuSubmenu = {
    kind: "submenu",
    id: "phantoms",
    label: "Open Example Phantom",
    enabled: idle,
    entries: [
      ...phantomEntries(info, input.phantoms, idle),
      sep("phantom-sep-folder"),
      item("open_phantom_folder", "Open Phantom Folder", "openFolder", {
        enabled: openFolder.available,
        tooltip: withReason("Phantoms (.mat, .npz) copied to your phantom folder are listed here", openFolder.reason),
      }),
    ],
  };

  const saveItems = SAVE_ENTRIES.map((entry) => {
    const blocked = exportBlockedReason(info, entry.object);
    const present = canSave(state, entry.id, entry.object);
    return item(entry.id, entry.label, entry.action, {
      enabled: idle && present && blocked === null,
      tooltip: withReason(entry.tooltip, blocked),
    });
  });

  const file: MenuDefinition = {
    id: "file",
    title: "&File",
    enabled: busy === null,
    entries: [
      item("load_file", "Load File…", "loadFile", { enabled: idle }),
      item("load_folder", "Load Folder…", "loadFolder", {
        enabled: false,
        tooltip: withReason(
          "Import a folder of DICOM data, or a folder of image files (CT + binary structure masks)",
          folderReason,
        ),
      }),
      phantomMenu,
      sep("sep-save"),
      ...saveItems.slice(0, 4),
      sep("sep-workspace"),
      saveItems[4],
      sep("sep-ui-mode"),
      {
        kind: "submenu",
        id: "ui_mode",
        label: "UI Mode",
        enabled: true,
        entries: UI_MODES.map((mode) =>
          item(`ui_mode:${mode}`, mode.charAt(0).toUpperCase() + mode.slice(1), "uiMode", {
            arg: mode,
            radio: true,
            checked: input.uiMode === mode,
            tooltip: UI_MODE_TIPS[mode],
          }),
        ),
      },
      sep("sep-exit"),
      item("exit", "Exit", "exit", {
        enabled: false,
        tooltip: "A web page cannot close itself: close the browser tab or window to exit.",
      }),
    ],
  };

  const viewStubs: Array<[string, string]> = [
    ["Screenshot", "Take a screenshot of the current view"],
    ["Zoom In", "Zoom in"],
    ["Zoom Out", "Zoom out"],
    ["Pan", "Pan"],
    ["Data Cursor", "Data cursor"],
    ["Toggle Legend", "Toggle legend"],
    ["Toggle Colorbar", "Toggle colorbar"],
  ];
  const view: MenuDefinition = {
    id: "view",
    title: "&View",
    enabled: true,
    entries: [
      item("log_panel", "Log Panel", "toggleLog", { checked: input.showLog }),
      sep("sep-view-stubs"),
      ...viewStubs.map(([label, tooltip]) => item(`view:${label}`, label, "none", { enabled: false, tooltip })),
      sep("sep-dark"),
      item("dark_mode", "Dark Mode", "toggleDark", { checked: input.dark, tooltip: "Switch between light and dark mode" }),
    ],
  };

  const persistence = feature(info, "base_data_persistence");
  const baseData: MenuDefinition = {
    id: "base_data",
    title: "&Base Data",
    enabled: true,
    entries: [
      item("show_base_data", "Show Available Base Data…", "showBaseData", {
        tooltip: "Machines, RBE tables and stopping power tables shipped with pyRadPlan and added by you",
      }),
      sep("sep-add"),
      item("add_machine", "Add Machine File…", "addMachine", {
        tooltip: persistence.available
          ? "Make a machine file (matRad *.mat) available by name, also in later sessions"
          : `Make a machine file (matRad *.mat) available by name.\n${persistence.reason ?? ""}`.trim(),
      }),
      item("add_rbe", "Add RBE Table…", "addRbe", {
        tooltip: "Make an α/β RBE table available to the 'α/β tables + fluence spectra' model",
      }),
      sep("sep-folder"),
      item("open_data_folder", "Open Data Folder", "openFolder", {
        enabled: openFolder.available,
        tooltip: withReason(undefined, openFolder.reason),
      }),
    ],
  };

  const ai = feature(info, "ai");
  const settings: MenuDefinition = {
    id: "settings",
    title: "&Settings",
    enabled: true,
    entries: [
      item("settings_xp", "XP (Backend)…", "settings", { arg: "xp" }),
      item("settings_ai", "AI…", "settings", { arg: "ai", tooltip: withReason(undefined, ai.available ? null : ai.reason) }),
      item("settings_gui", "GUI…", "settings", { arg: "gui" }),
      sep("sep-prefs"),
      item("preferences", "Preferences…", "settings"),
    ],
  };

  const help: MenuDefinition = {
    id: "help",
    title: "&Help",
    enabled: true,
    entries: [
      item("documentation", "Documentation", "openUrl", { arg: DOCUMENTATION_URL }),
      item("source_code", "Source Code", "openUrl", { arg: SOURCE_URL }),
      sep("sep-licenses"),
      item("licenses", "Third-party Licenses…", "licenses"),
      item("about", "About pyRadPlan…", "about"),
    ],
  };

  return [file, view, baseData, settings, help];
}

/** Look an entry up by id (tests). */
export function findEntry(entries: MenuEntry[], id: string): MenuEntry | undefined {
  for (const entry of entries) {
    if (entry.id === id) return entry;
    if (entry.kind === "submenu") {
      const found = findEntry(entry.entries, id);
      if (found) return found;
    }
  }
  return undefined;
}
