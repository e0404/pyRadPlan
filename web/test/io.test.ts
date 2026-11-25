import { describe, expect, it } from "vitest";
// The mock backend module comes first: `mock/io.ts` and `mock/mockBackend.ts` import each other.
import { MockBackend } from "../src/api/mock/mockBackend";
import { BackendError } from "../src/api/backend";
import { ndarray } from "../src/api/codec";
import type { AppInfo, FileResult, SessionState } from "../src/api/protocol";
import type { LicenseDetail, LicenseRow, SettingsResult } from "../src/api/protocol/io";
import { mockCapabilities } from "../src/api/mock/io";
import { crc32, zip } from "../src/api/mock/zip";
import { exportBlockedReason, feature, loadAccept, structureFormatOf, unavailableImport } from "../src/components/io/capabilities";
import { describeDownload, downloadFile, safeFilename, type DownloadEnv } from "../src/components/io/download";
import { buildMenus, canSave, findEntry, type MenuInput } from "../src/components/io/menuModel";
import { baseNameOf, formatLogLine, sanitizeTag, splashPhase, statusWithFurtherResults, visibleLogs } from "../src/components/io/models";
import { orderSections } from "../src/components/settings/model";
import { lockedByUiMode } from "../src/store/uiMode";

function fakeEnv() {
  const calls: string[] = [];
  const anchor = {
    href: "",
    download: "",
    style: { display: "" },
    click: () => calls.push(`click ${anchor.download} ${anchor.href}`),
    remove: () => calls.push("remove"),
  };
  const deferred: Array<() => void> = [];
  const env: DownloadEnv = {
    createObjectURL: (blob) => {
      calls.push(`url ${blob.type} ${blob.size}`);
      return "blob:fake";
    },
    revokeObjectURL: (url) => calls.push(`revoke ${url}`),
    createAnchor: () => anchor,
    append: () => calls.push("append"),
    defer: (fn) => void deferred.push(fn),
  };
  return { env, calls, anchor, deferred };
}

function fileResult(filename: string, bytes: Uint8Array, mime = "application/octet-stream", files?: string[]): FileResult {
  return { filename, mime, data: ndarray(bytes, [bytes.length]), ...(files ? { files } : {}) };
}

describe("downloadFile", () => {
  it("offers the bytes through an anchor with the download attribute", () => {
    const { env, calls, anchor, deferred } = fakeEnv();
    const saved = downloadFile(fileResult("plan.mat", new Uint8Array([1, 2, 3]), "application/x-matlab-data"), env);
    expect(saved).toEqual({ filename: "plan.mat", mime: "application/x-matlab-data", bytes: 3 });
    expect(anchor.download).toBe("plan.mat");
    expect(anchor.href).toBe("blob:fake");
    expect(calls).toEqual(["url application/x-matlab-data 3", "append", "click plan.mat blob:fake", "remove"]);
    deferred[0]();
    expect(calls.at(-1)).toBe("revoke blob:fake");
  });

  it("strips directories and forbidden characters from the file name", () => {
    expect(safeFilename("../../etc/passwd")).toBe("passwd");
    expect(safeFilename("C:\\temp\\a<b>.zip")).toBe("a_b_.zip");
    expect(safeFilename("")).toBe("download");
  });

  it("describes archives with their member count", () => {
    const zipped = fileResult("result.zip", new Uint8Array(2), "application/zip", ["a.npz", "b.npz"]);
    expect(describeDownload(zipped, { filename: "result.zip", mime: "application/zip", bytes: 2 })).toBe("result.zip (2 files)");
    const single = fileResult("pln.mat", new Uint8Array(2));
    expect(describeDownload(single, { filename: "pln.mat", mime: "", bytes: 2 })).toBe("pln.mat");
  });
});

describe("zip writer", () => {
  it("writes a readable stored archive", () => {
    const data = zip([{ name: "a.txt", data: new TextEncoder().encode("hello") }]);
    expect(String.fromCharCode(data[0], data[1])).toBe("PK");
    expect(crc32(new TextEncoder().encode("123456789"))).toBe(0xcbf43926);
    const eocd = new DataView(data.buffer, data.byteLength - 22, 22);
    expect(eocd.getUint32(0, true)).toBe(0x06054b50);
    expect(eocd.getUint16(10, true)).toBe(1);
  });
});

const web = mockCapabilities("pyodide");
const native = mockCapabilities("native");

function info(capabilities = web, extra: Partial<AppInfo> = {}): AppInfo {
  return {
    version: "0.9.0",
    python: "3.12.1",
    backend: "mock",
    phantoms: ["TG119", "Liver"],
    machines: {},
    solvers: [],
    engines: {},
    capabilities,
    ...extra,
  };
}

function session(has: Partial<SessionState["has"]> = {}, actions: Record<string, boolean> = {}): SessionState {
  return {
    session: "s1",
    has: { ct: false, cst: false, pln: false, stf: false, dij: false, result: false, ...has },
    stages: { data: "missing", plan: "missing", stf: "missing", dij: "missing", optimization: "missing" },
    actions: { generate_stf: false, calc_dose_influence: false, optimize: false, recalc: false, ...actions },
    ct: null,
    plan: null,
    quantities: [],
    structures: [],
    status: "",
  } as SessionState;
}

function menuInput(overrides: Partial<MenuInput> = {}): MenuInput {
  return { info: info(), state: session(), busy: null, phantoms: null, uiMode: "default", dark: false, showLog: true, ...overrides };
}

const fileMenu = (input: MenuInput) => buildMenus(input).find((m) => m.id === "file")!;
const enabled = (input: MenuInput, id: string) => {
  const entry = findEntry(buildMenus(input).flatMap((m) => m.entries), id);
  return entry && entry.kind !== "separator" ? entry.enabled : undefined;
};

describe("menu model", () => {
  it("has the Qt menus in order", () => {
    expect(buildMenus(menuInput()).map((m) => m.title)).toEqual(["&File", "&View", "&Base Data", "&Settings", "&Help"]);
  });

  it("enables the save entries from the session actions", () => {
    const state = session({ ct: true, cst: true, pln: true }, { save_pln: true, save_cst: true, save_workspace: true, save_dij: false, save_result: false });
    const input = menuInput({ state });
    expect(enabled(input, "save_pln")).toBe(true);
    expect(enabled(input, "save_cst")).toBe(true);
    expect(enabled(input, "save_workspace")).toBe(true);
    expect(enabled(input, "save_dij")).toBe(false);
    expect(enabled(input, "save_result")).toBe(false);
  });

  it("falls back to `has` when the backend sends no save actions", () => {
    expect(canSave(session({ pln: true }), "save_pln", "pln")).toBe(true);
    expect(canSave(session({ ct: true }), "save_workspace", "workspace")).toBe(true);
    expect(canSave(session({ ct: true }), "save_dij", "dij")).toBe(false);
    expect(canSave(null, "save_pln", "pln")).toBe(false);
  });

  it("locks the File menu while busy, like Qt, but not Base Data and Settings", () => {
    const state = session({ ct: true, cst: true, pln: true }, { save_pln: true });
    const input = menuInput({ state, busy: "Optimization" });
    expect(fileMenu(input).enabled).toBe(false);
    expect(enabled(input, "save_pln")).toBe(false);
    expect(enabled(input, "load_file")).toBe(false);
    expect(enabled(input, "show_base_data")).toBe(true);
    expect(enabled(input, "preferences")).toBe(true);
    expect(buildMenus(input).find((m) => m.id === "base_data")!.enabled).toBe(true);
  });

  it("disables loading and saving without a session", () => {
    expect(enabled(menuInput({ state: null }), "load_file")).toBe(false);
    expect(enabled(menuInput({ state: null }), "phantoms")).toBe(false);
  });

  it("shows capability reasons on the stubs", () => {
    const input = menuInput();
    const folder = findEntry(fileMenu(input).entries, "load_folder");
    expect(folder?.kind === "item" && folder.enabled).toBe(false);
    expect(folder?.kind === "item" && folder.tooltip).toContain("SimpleITK");
    const phantomFolder = findEntry(fileMenu(input).entries, "open_phantom_folder");
    expect(phantomFolder?.kind === "item" && phantomFolder.enabled).toBe(false);
    expect(phantomFolder?.kind === "item" && phantomFolder.tooltip).toContain("browser cannot open folders");
    const nativeInput = menuInput({ info: info(native) });
    const nativeFolder = findEntry(fileMenu(nativeInput).entries, "open_phantom_folder");
    expect(nativeFolder?.kind === "item" && nativeFolder.enabled).toBe(true);
    // Load Folder has no protocol method: a stub even when the runtime could read folders.
    expect(enabled(nativeInput, "load_folder")).toBe(false);
  });

  it("disables a save entry whose formats are all unavailable and says why", () => {
    const blocked = structuredClone(web);
    blocked.formats.export = blocked.formats.export.map((f) => (f.object === "dij" ? { ...f, available: false, reason: "no writer" } : f));
    const input = menuInput({ info: info(blocked), state: session({ dij: true }, { save_dij: true }) });
    const entry = findEntry(fileMenu(input).entries, "save_dij");
    expect(entry?.kind === "item" && entry.enabled).toBe(false);
    expect(entry?.kind === "item" && entry.tooltip).toContain("no writer");
  });

  it("treats a backend without capabilities as unable to export", () => {
    const v0 = { ...info(), capabilities: undefined } as AppInfo;
    expect(exportBlockedReason(v0, "pln")).toContain("does not export");
    expect(enabled(menuInput({ info: v0, state: session({ pln: true }, { save_pln: true }) }), "save_pln")).toBe(false);
    expect(feature(v0, "ai").available).toBe(true);
  });

  it("lists the phantoms grouped by source with Qt tooltips", () => {
    const phantoms = {
      rows: [
        { name: "TG119", file: "TG119.mat", source: "bundled" as const, removable: false },
        { name: "Mine", file: "Mine.mat", source: "user" as const, removable: true },
      ],
      folders: [{ source: "user" as const, label: "Added by you", path: "/data/phantoms" }],
    };
    const menu = findEntry(fileMenu(menuInput({ phantoms })).entries, "phantoms");
    expect(menu?.kind).toBe("submenu");
    const entries = menu?.kind === "submenu" ? menu.entries : [];
    expect(entries.map((e) => e.id)).toEqual(["phantom:TG119", "phantom-sep-user", "phantom:Mine", "phantom-sep-folder", "open_phantom_folder"]);
    const tg = entries[0];
    expect(tg.kind === "item" && tg.tooltip).toBe("Shipped with pyRadPlan");
    const mine = entries[2];
    expect(mine.kind === "item" && mine.tooltip).toBe("/data/phantoms/Mine.mat");
  });

  it("falls back to info.phantoms without a listing", () => {
    const menu = findEntry(fileMenu(menuInput()).entries, "phantoms");
    const names = menu?.kind === "submenu" ? menu.entries.filter((e) => e.kind === "item" && e.action === "loadPhantom").map((e) => e.id) : [];
    expect(names).toEqual(["phantom:TG119", "phantom:Liver"]);
  });

  it("checks the UI mode, dark mode and log panel entries", () => {
    const input = menuInput({ uiMode: "educational", dark: true, showLog: false });
    const modes = findEntry(fileMenu(input).entries, "ui_mode");
    const checked = modes?.kind === "submenu" ? modes.entries.filter((e) => e.kind === "item" && e.checked).map((e) => e.id) : [];
    expect(checked).toEqual(["ui_mode:educational"]);
    const view = buildMenus(input).find((m) => m.id === "view")!;
    expect(view.entries.find((e) => e.id === "dark_mode")).toMatchObject({ checked: true });
    expect(view.entries.find((e) => e.id === "log_panel")).toMatchObject({ checked: false });
    expect(view.entries.filter((e) => e.kind === "item" && !e.enabled).length).toBe(7);
  });
});

describe("capabilities helpers", () => {
  it("builds the load accept list from available import formats", () => {
    expect(loadAccept(info(web))).toBe(".mat,.npz,.h5,.hdf5");
    expect(loadAccept(info(native))).toContain(".nii.gz");
    expect(loadAccept(info(native))).toContain(".pkl");
    expect(loadAccept(null)).toBe(".mat");
  });

  it("refuses a pickle where the backend cannot load it, with the backend's reason", () => {
    expect(web.features.pickle_load).toMatchObject({ available: false });
    expect(unavailableImport(info(web), "case.PKL")).toMatchObject({ format: "pickle", reason: web.features.pickle_load!.reason });
    expect(unavailableImport(info(web), "case.pickle")?.format).toBe("pickle");
    expect(unavailableImport(info(web), "case.mat")).toBeUndefined();
    expect(unavailableImport(info(native), "case.pkl")).toBeUndefined();
    expect(unavailableImport(null, "case.pkl")).toBeUndefined();
  });

  it("derives the structure format of the two DICOM entries", () => {
    const dcm = web.formats.export.filter((f) => f.object === "cst" && f.format === "dcm");
    expect(dcm.map(structureFormatOf)).toEqual(["rtstruct", "seg"]);
    expect(structureFormatOf({ ...dcm[0], label: "DICOM something", structure_format: "seg" } as never)).toBe("seg");
    expect(structureFormatOf({ ...dcm[1], label: "DICOM RTSTRUCT" })).toBe("seg");
    expect(structureFormatOf(web.formats.export.find((f) => f.object === "pln")!)).toBeUndefined();
    const nrrd = web.formats.export.filter((f) => f.object === "cst" && f.format === "nrrd");
    expect(nrrd.map(structureFormatOf)).toEqual([undefined, "layers"]);
  });

  it("strips a typed extension from the base name", () => {
    expect(baseNameOf("plan.mat", ".mat")).toBe("plan");
    expect(baseNameOf(" plan ", ".mat")).toBe("plan");
    expect(baseNameOf("result.zip", ".zip")).toBe("result");
  });
});

describe("small helpers", () => {
  it("sanitizes snapshot names like Qt", () => {
    expect(sanitizeTag("  my snap! ")).toBe("my_snap");
    expect(sanitizeTag("__a--b__")).toBe("a_b");
    expect(sanitizeTag("  ")).toBe("");
  });

  it("appends the number of further results to the result status", () => {
    expect(statusWithFurtherResults("Optimization result available", ["a"])).toBe("Optimization result available · 1 further result");
    expect(statusWithFurtherResults("Optimization result available", ["a", "b"])).toBe("Optimization result available · 2 further results");
    expect(statusWithFurtherResults("Ready for optimization", ["a"])).toBe("Ready for optimization");
    expect(statusWithFurtherResults("Optimization result available", [])).toBe("Optimization result available");
    expect(statusWithFurtherResults("Optimization result available · 1 further result", ["a"])).toBe("Optimization result available · 1 further result");
  });

  it("locks advanced fields only in the educational mode", () => {
    expect(lockedByUiMode({ advanced: true }, "educational")).toBe(true);
    expect(lockedByUiMode({ advanced: true }, "default")).toBe(false);
    expect(lockedByUiMode({}, "educational")).toBe(false);
  });

  it("formats and filters log lines like the Qt console", () => {
    const t = new Date(2026, 0, 5, 7, 8, 9);
    expect(formatLogLine({ time: t, level: "warning", message: "x", logger: "pyRadPlan.dose" })).toBe("07:08:09 [WARNING] pyRadPlan.dose: x");
    expect(formatLogLine({ time: t, level: "info", message: "x" })).toBe("07:08:09 [INFO] x");
    const logs = (["debug", "info", "warning", "error", "critical"] as const).map((level, id) => ({ id, time: t, level, message: level }));
    expect(visibleLogs(logs, "warning").map((l) => l.level)).toEqual(["warning", "error", "critical"]);
    expect(visibleLogs(logs, "debug")).toHaveLength(5);
  });

  it("maps boot stages to the Qt splash phases", () => {
    const boot = (message: string, fraction: number | null) => ({ phase: "starting" as const, message, fraction });
    expect(splashPhase(boot("Loading Python runtime…", 0.02), false, null, false)).toMatchObject({ message: "Loading Python runtime…", step: 1, total: 5 });
    expect(splashPhase(boot("Loading numpy, scipy and the other Pyodide packages…", 0.3), false, null, false)).toMatchObject({ message: "Loading core…", step: 2 });
    expect(splashPhase(boot("Importing pyRadPlan…", 0.9), false, null, false)).toMatchObject({ message: "Loading pyRadPlan…", step: 3 });
    expect(splashPhase({ phase: "ready", message: "", fraction: 1 }, false, null, false).message).toBe("Building main window…");
    expect(splashPhase({ phase: "ready", message: "", fraction: 1 }, true, "TG119", true)).toMatchObject({ message: "Loading patient TG119…", total: 6 });
  });

  it("orders the settings sections like the Qt tabs", () => {
    const section = (id: "general" | "xp" | "ai" | "gui") => ({
      id,
      title: id,
      form: { id: `settings:${id}`, title: id, fields: [{ name: "a", label: "A", type: "bool" as const }, { name: "b", label: "B", type: "bool" as const, editable: false }], values: { a: false, b: false }, explicit: [] },
    });
    const ordered = orderSections([section("ai"), section("xp"), section("general")], "educational");
    expect(ordered.map((s) => s.id)).toEqual(["general", "xp", "ai", "gui"]);
    expect(ordered[3].form.values.ui_mode).toBe("educational");
  });
});

async function newSession(search = "") {
  void search;
  const backend = new MockBackend({ speed: 0 });
  await backend.ready();
  const { session } = await backend.request("session.create", {});
  return { backend, session };
}

describe("mock handlers of package C", () => {
  it("adds capabilities to app.info and lists its methods", async () => {
    const { backend } = await newSession();
    const result = await backend.request("app.info", {});
    expect(result.runtime).toBe("pyodide");
    expect(result.capabilities?.features.image_file_io?.available).toBe(false);
    expect(result.methods).toEqual(expect.arrayContaining(["export.save", "base_data.list", "app.settings", "app.licenses", "session.log_level", "data.load_dij"]));
  });

  it("flags the save actions from the session objects", async () => {
    const { backend, session } = await newSession();
    let state = await backend.request("session.state", {}, session);
    expect(state.actions).toMatchObject({ load_dij: true, save_pln: false, save_cst: false, save_workspace: false, save_result: false });
    state = await backend.request("data.load_phantom", { name: "TG119" }, session);
    expect(state.actions).toMatchObject({ save_pln: true, save_cst: true, save_workspace: true, save_dij: false, save_result: false });
    state = await backend.request("workflow.run_all", {}, session);
    expect(state.actions).toMatchObject({ save_dij: true, save_result: true, snapshot: true });
  });

  it("exports small real files and refuses unavailable formats", async () => {
    const { backend, session } = await newSession();
    await backend.request("data.load_phantom", { name: "TG119" }, session);
    await backend.request("workflow.run_all", {}, session);
    const mat = await backend.request("export.save", { object: "pln", format: "mat" }, session);
    expect(mat.filename).toBe("pln.mat");
    expect(new TextDecoder().decode(mat.data.data.subarray(0, 15))).toBe("MATLAB 5.0 MAT-");
    const named = await backend.request("export.save", { object: "workspace", format: "pickle", basename: "case" }, session);
    expect(named.filename).toBe("case.pkl");
    expect(named.data.data[0]).toBe(0x80);
    const dicom = await backend.request("export.save", { object: "cst", format: "dcm", structure_format: "seg" }, session);
    expect(dicom.mime).toBe("application/zip");
    expect(dicom.files).toContain("SEG.dcm");
    const rts = await backend.request("export.save", { object: "cst", format: "dcm", structure_format: "rtstruct" }, session);
    expect(rts.files).toContain("RTSTRUCT.dcm");
    const resultMat = await backend.request("export.save", { object: "result", format: "mat", quantities: ["physical_dose"] }, session);
    expect(resultMat.filename).toBe("physical_dose.mat");
    const result = await backend.request("export.save", { object: "result", format: "npz", quantities: ["physical_dose"] }, session);
    expect(result.filename).toBe("physical_dose.npz");
    await expect(backend.request("export.save", { object: "result", format: "nifti" }, session)).rejects.toMatchObject({ type: "PreconditionError", code: "not_available" });
    await expect(backend.request("export.save", { object: "result", format: "npz", quantities: [] }, session)).rejects.toBeInstanceOf(BackendError);
  });

  it("refuses to export what the session lacks", async () => {
    const { backend, session } = await newSession();
    await expect(backend.request("export.save", { object: "pln", format: "mat" }, session)).rejects.toMatchObject({ code: "no_data" });
  });

  it("loads a dij and reports a file without one", async () => {
    const { backend, session } = await newSession();
    await backend.request("data.load_phantom", { name: "TG119" }, session);
    const bytes = new Uint8Array([1, 2, 3]);
    await expect(backend.request("data.load_dij", { filename: "nodij.mat", data: ndarray(bytes, [3]) }, session)).rejects.toMatchObject({ code: "no_dij" });
    const state = await backend.request("data.load_dij", { filename: "dij.mat", data: ndarray(bytes, [3]) }, session);
    expect(state.stages.dij).toBe("done");
    expect(state.has.stf).toBe(true);
  });

  it("refuses image files without image IO and unknown extensions", async () => {
    const { backend, session } = await newSession();
    const data = ndarray(new Uint8Array([1]), [1]);
    await expect(backend.request("data.load_file", { filename: "ct.nii.gz", data }, session)).rejects.toMatchObject({ code: "not_available" });
    await expect(backend.request("data.load_file", { filename: "case.mat", data, kind: "dose" }, session)).rejects.toMatchObject({ code: "not_available" });
    await expect(backend.request("data.load_file", { filename: "notes.txt", data }, session)).rejects.toMatchObject({ type: "ValueError" });
    await expect(backend.request("data.load_file", { filename: "case.pkl", data }, session)).rejects.toMatchObject({ type: "PreconditionError", code: "not_available" });
    const state = await backend.request("data.load_file", { filename: "case.mat", data }, session);
    expect(state.has.ct).toBe(true);
  });

  it("lists, adds, clashes and removes base data", async () => {
    const { backend, session } = await newSession();
    const machines = await backend.request("base_data.list", { kind: "machines" }, session);
    expect(machines.rows.map((r) => r.file)).toContain("photons_Generic.mat");
    expect(machines.folders.map((f) => f.source)).toEqual(["bundled", "user"]);
    const data = ndarray(new Uint8Array([1, 2]), [2]);
    const probe = await backend.request("base_data.add", { kind: "machines", filename: "photons_Generic.mat", data, dry_run: true }, session);
    expect(probe).toMatchObject({ suggested_name: "Generic", radiation_mode: "photons", clash: { file: "photons_Generic.mat", source: "bundled" } });
    await expect(backend.request("base_data.add", { kind: "machines", filename: "photons_Generic.mat", data }, session)).rejects.toMatchObject({ code: "name_clash" });
    const added = await backend.request("base_data.add", { kind: "machines", filename: "photons_Generic.mat", data, name: "UnitTestLinac" }, session);
    expect(added.row).toMatchObject({ file: "photons_UnitTestLinac.mat", source: "user", removable: true });
    await expect(backend.request("base_data.add", { kind: "machines", filename: "x.mat", data, name: "UnitTestLinac", radiation_mode: "photons" }, session)).rejects.toMatchObject({ code: "name_clash" });
    const replaced = await backend.request("base_data.add", { kind: "machines", filename: "x.mat", data, name: "UnitTestLinac", radiation_mode: "photons", overwrite: true }, session);
    expect(replaced.row?.file).toBe("photons_UnitTestLinac.mat");
    await expect(backend.request("base_data.remove", { kind: "machines", file: "photons_Generic.mat" }, session)).rejects.toMatchObject({ code: "not_removable" });
    await backend.request("base_data.remove", { kind: "machines", file: "photons_UnitTestLinac.mat" }, session);
    const after = await backend.request("base_data.list", { kind: "machines" }, session);
    expect(after.rows.some((r) => r.file === "photons_UnitTestLinac.mat")).toBe(false);
    await expect(backend.request("base_data.add", { kind: "machines", filename: "broken.mat", data, dry_run: true }, session)).rejects.toMatchObject({ type: "ValueError" });
  });

  it("serves settings with locked fields and validates writes", async () => {
    const { backend, session } = await newSession();
    const { sections } = (await backend.request("app.settings", {}, session)) as SettingsResult;
    expect(sections.map((s) => s.id)).toEqual(["general", "xp", "ai", "gui"]);
    const xp = sections.find((s) => s.id === "xp")!;
    expect(xp.form.fields.every((f) => f.editable === false && f.reason)).toBe(true);
    const gui = sections.find((s) => s.id === "gui")!;
    expect(gui.form.fields[0].choices?.map((c) => c.value)).toEqual(["default", "educational"]);
    const set = await backend.request("app.settings_set", { section: "gui", values: { ui_mode: "educational" } }, session);
    expect(set.sections.find((s) => s.id === "gui")?.form.values.ui_mode).toBe("educational");
    await expect(backend.request("app.settings_set", { section: "xp", values: { prefer_gpu: false } }, session)).rejects.toMatchObject({ type: "ValidationError", code: "invalid" });
    await expect(backend.request("app.settings_set", { section: "gui", values: { ui_mode: "bogus" } }, session)).rejects.toMatchObject({ type: "ValidationError" });
    await backend.request("app.settings_set", { section: "gui", values: { ui_mode: "default" } }, session);
  });

  it("serves licenses with and without text", async () => {
    const { backend, session } = await newSession();
    const rows = (await backend.request("app.licenses", {}, session)) as LicenseRow[];
    expect(rows.find((r) => r.name === "numpy")).toMatchObject({ license: "BSD-3-Clause" });
    const numpy = (await backend.request("app.licenses", { name: "numpy" }, session)) as LicenseDetail;
    expect(numpy.text).toContain("BSD");
    const tqdm = (await backend.request("app.licenses", { name: "tqdm" }, session)) as LicenseDetail;
    expect(tqdm.text).toBeNull();
    await expect(backend.request("app.licenses", { name: "nope" }, session)).rejects.toMatchObject({ type: "KeyError" });
  });

  it("filters log events by the session level and replays them as history", async () => {
    const { backend, session } = await newSession();
    const seen: number[] = [];
    backend.subscribe((e) => {
      if (e.event === "log") seen.push((e as { seq?: number }).seq ?? -1);
    });
    await backend.request("data.load_phantom", { name: "TG119" }, session);
    expect(seen.length).toBeGreaterThan(0);
    expect(seen.every((s) => s > 0)).toBe(true);
    const all = await backend.request("session.history", {}, session);
    expect(all.log.length).toBe(seen.length);
    const since = await backend.request("session.history", { since: seen[0] }, session);
    expect(since.log.length).toBe(seen.length - 1);
    await backend.request("session.log_level", { level: "warning" }, session);
    const before = seen.length;
    await backend.request("workflow.run_all", {}, session);
    expect(seen.length).toBe(before);
    await expect(backend.request("session.log_level", { level: "loud" as never }, session)).rejects.toMatchObject({ type: "ValueError" });
  });

  it("snapshots the active result under a sanitized name and refuses the name of the active result", async () => {
    const { backend, session } = await newSession();
    await backend.request("data.load_phantom", { name: "TG119" }, session);
    await expect(backend.request("workflow.snapshot", { tag: "x" }, session)).rejects.toBeInstanceOf(BackendError);
    await backend.request("workflow.run_all", {}, session);
    const out = await backend.request("workflow.snapshot", { tag: " my snap " }, session);
    expect(out.tag).toBe("my_snap");
    expect(out.state.results?.tags).toEqual(["my_snap"]);
    await expect(backend.request("workflow.snapshot", { tag: "current" }, session)).rejects.toMatchObject({ type: "ValueError" });
    expect((await backend.request("workflow.snapshot", { tag: "scenario" }, session)).tag).toBe("scenario");
  });

  it("adds a loaded dose as a result of its own, without an active result", async () => {
    const { backend, session } = await newSession();
    const data = { dtype: "uint8" as const, shape: [4], data: new Uint8Array([1, 2, 3, 4]) };
    await backend.request("data.load_phantom", { name: "TG119" }, session);
    let state = await backend.request("data.load_file", { filename: "reference dose.npz", data }, session);
    // The patient stays; the dose is a result next to the (missing) active one.
    expect(state.has).toMatchObject({ ct: true, result: false });
    expect(state.results).toMatchObject({ names: ["reference_dose"], active: "current", tags: ["reference_dose"], default: "reference_dose/physical_dose" });
    expect(state.results?.groups.map((g) => [g.label, g.tag])).toEqual([["dose (reference_dose)", "reference_dose"]]);
    expect(state.actions).toMatchObject({ snapshot: false, recalc: false, save_result: true });
    await expect(backend.request("workflow.snapshot", { tag: "x" }, session)).rejects.toMatchObject({ type: "PreconditionError" });
    const volume = await backend.request("view.volume", { name: "reference_dose/physical_dose" }, session);
    expect(volume.unit).toBe("Gy");
    // The name is made unique.
    state = await backend.request("data.load_file", { filename: "reference dose.npz", data }, session);
    expect(state.results?.names).toEqual(["reference_dose", "reference_dose_2"]);
    // Export offers the full quantities of all results by their keys.
    const saved = await backend.request("export.save", { object: "result", format: "npz" }, session);
    expect(saved.files).toEqual(["reference_dose_physical_dose.npz", "reference_dose_2_physical_dose.npz"]);
    // Optimizing adds the active result and keeps the imported doses.
    state = await backend.request("workflow.run_all", {}, session);
    expect(state.results).toMatchObject({ names: ["reference_dose", "reference_dose_2", "current"], tags: ["reference_dose", "reference_dose_2"], default: "physical_dose" });
    expect(state.results?.groups[0].tag).toBe("");
    expect(state.actions).toMatchObject({ snapshot: true, save_result: true });
    // The doses of all results can be referenced by objectives.
    const catalog = await backend.request("objectives.catalog", {}, session);
    expect(catalog.reference_images.map((c) => c.value)).toEqual(["physical_dose", "reference_dose/physical_dose", "reference_dose_2/physical_dose"]);
    // A new patient replaces all results.
    state = await backend.request("data.load_phantom", { name: "TG119" }, session);
    expect(state.results).toMatchObject({ names: [], tags: [], groups: [], default: null });
  });
});
