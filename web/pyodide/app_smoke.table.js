// Method table driven by app_smoke.mjs, in order. Entry fields:
//   method    protocol method name
//   params    object, or (ctx) => object; {$file: "path"} values become uint8 buffers
//             (path relative to web/pyodide)
//   session   false for methods without a session (default: the session of session.create)
//   v         protocol version: v0 failures fail the run, v1 failures are reported as pending
//             (pass --strict to fail on them too)
//   expect    optional (result, response, ctx) => true | string (the failure reason)
//   after     optional (result, ctx) => void, e.g. to remember a session id
//   setup     true: also run with --only v1 (session and workflow the later entries need)
//   needs     optional file path; the entry is skipped when it does not exist
//   python    instead of a method: Python code run in the worker runtime whose last line
//             is the (JSON-able) result; files listed in `files` are copied to /tmp/smoke/
//   expectError  optional {type, code?}: the entry must fail with that error
//   note      optional text shown next to the result

const prod = (shape) => shape.reduce((a, b) => a * b, 1);
const has = (result, ...keys) =>
  keys.every((k) => result && k in result) || `missing ${keys.filter((k) => !(k in (result ?? {})))}`;

export const PLAN = {
  gantry_angles: [0, 72, 144, 216, 288],
  couch_angles: [0, 0, 0, 0, 0],
  bixel_width: 10.0,
  dose_grid_resolution: [8.0, 8.0, 8.0],
};

export default [
  // ---- v0 pipeline -------------------------------------------------------------------------
  { v: 0, method: "app.info", session: false, expect: (r) => r.backend === "numpy" || `backend ${r.backend}` },
  {
    v: 0,
    method: "session.create",
    setup: true,
    session: false,
    params: {},
    expect: (r) => typeof r.session === "string" || "no session id",
    after: (r, ctx) => {
      ctx.session = r.session;
    },
  },
  {
    v: 0,
    method: "data.load_phantom",
    setup: true,
    params: { name: "TG119" },
    expect: (r) => (r.has.ct && r.has.cst) || "ct/cst missing",
  },
  { v: 0, setup: true, method: "plan.set", params: PLAN, expect: (r) => r.has.pln || "no plan" },
  { v: 0, method: "plan.get", expect: (r) => r.bixel_width === 10 || `bixel_width ${r.bixel_width}` },
  {
    v: 0,
    method: "workflow.run_all",
    setup: true,
    expect: (r) => (r.has.stf && r.has.dij && r.has.result) || JSON.stringify(r.has),
  },
  {
    v: 0,
    method: "view.volume",
    params: { name: "physical_dose" },
    expect: (r, resp) =>
      resp.buffers[r.data.$array]?.nbytes === 4 * prod(r.data.shape) ||
      `buffer size ${resp.buffers[r.data.$array]?.nbytes}`,
  },
  { v: 0, method: "view.volume", params: { name: "ct" }, expect: (r) => r.unit !== undefined || "no unit" },
  {
    v: 0,
    method: "view.structure_masks",
    expect: (r) => r.masks.length === 3 || `${r.masks.length} masks`,
  },
  { v: 0, method: "analysis.dvh", expect: (r) => r.curves.length === 3 || `${r.curves.length} curves` },
  { v: 0, method: "session.state", expect: (r) => r.stages.optimization === "done" || r.stages },

  {
    v: 0,
    python: `
from pyRadPlan.io import load_data
data = load_data("/tmp/smoke/tg119_native.pkl")
{"ct": data["ct"].cube_hu.GetSize(), "vois": [v.name for v in data["cst"].vois]}`,
    files: ["results/tg119_native.pkl"],
    needs: "results/tg119_native.pkl",
    note: "unpickle a native (real SimpleITK) workspace with pyRadPlan.io",
    expect: (r) => (r.ct.join() === "167,167,129" && r.vois.length === 3) || JSON.stringify(r),
  },

  // ---- v1 (web/PROTOCOL.md, v1 section); extend as the backend grows ---------------------------
  { v: 1, method: "app.info", session: false, expect: (r) => has(r, "runtime", "capabilities") },
  {
    v: 1,
    method: "session.create",
    session: false,
    params: { profile: "web" },
    expect: (r) => has(r, "session", "options"),
  },
  {
    v: 1,
    method: "app.licenses",
    session: false,
    note: "Pyodide distribution packages, vendored wheels and the shim are listed",
    expect: (r) => {
      const names = r.map((row) => row.name.toLowerCase());
      const missing = ["pyradplan", "pyradplan-sitk-shim", "numpy", "scipy", "pydicom"].filter(
        (n) => !names.includes(n),
      );
      const unknown = r.filter((row) => row.license === "unknown").map((row) => row.name);
      return (!missing.length && !unknown.length) || `missing ${missing}, unknown license ${unknown}`;
    },
  },
  {
    v: 1,
    method: "app.licenses",
    session: false,
    params: { name: "pyradplan-sitk-shim" },
    note: "the SimpleITK shim is labelled as pyRadPlan's stand-in",
    expect: (r) =>
      (r.license === "BSD-3-Clause" &&
        r.text.includes("part of pyRadPlan, not SimpleITK") &&
        r.text.includes("Copyright (c) 2024-2026, DKFZ")) ||
      JSON.stringify(r).slice(0, 160),
  },
  {
    v: 1,
    method: "app.licenses",
    session: false,
    params: { name: "scipy" },
    note: "a Pyodide distribution package carries its license text",
    expect: (r) => (r.text ?? "").includes("Copyright") || JSON.stringify(r).slice(0, 160),
  },
  { v: 1, method: "app.settings", session: false, expect: (r) => has(r, "sections") },
  { v: 1, method: "session.history", expect: (r) => has(r, "log") },
  { v: 1, method: "session.log_level", params: { level: "info" } },
  { v: 1, method: "plan.get", params: { detail: true } },
  { v: 1, method: "plan.options" },
  { v: 1, method: "plan.form", params: { kind: "solver", name: "scipy" } },
  { v: 1, method: "plan.tissue_check" },
  { v: 1, method: "base_data.list", params: { kind: "machines" }, expect: (r) => has(r, "rows") },
  {
    v: 1,
    method: "cst.get",
    expect: (r) => has(r, "structures"),
    after: (r, ctx) => {
      ctx.coreTissue = r.structures.find((st) => st.name === "Core").tissue;
    },
  },
  { v: 1, method: "tissue.schema" },
  { v: 1, method: "objectives.catalog", expect: (r) => has(r, "objectives") },
  { v: 1, method: "workflow.snapshot", params: { tag: "smoke" } },
  { v: 1, method: "view.volume", params: { name: "ct", encoding: "auto" } },
  { v: 1, method: "view.structure_masks", params: { encoding: "bits" } },
  { v: 1, method: "view.beams", expect: (r) => has(r, "beams") },
  { v: 1, method: "analysis.dvh", params: { range: "shared" } },
  { v: 1, method: "analysis.qi", params: { quantity: "physical_dose" } },
  { v: 1, method: "analysis.compare_units", params: { a: "physical_dose", b: "physical_dose" } },
  // Sequencing (amendments 27 and 28): settings, the run on the active result, its sequence view.
  { v: 1, method: "plan.form", params: { kind: "sequencer", name: "xia" }, expect: (r) => has(r, "fields", "values") },
  {
    v: 1,
    method: "view.sequence",
    note: "no sequence before sequencing",
    expectError: { type: "PreconditionError", code: "no_sequence" },
  },
  {
    v: 1,
    method: "plan.set",
    note: "sequencing settings outdate no stage",
    params: { sequencing: { run: true, sequencer: "xia", values: { num_of_levels: 3 } } },
    expect: (r) =>
      (r.sequencing.available && !r.sequencing.sequenced && r.stages.optimization === "done") ||
      JSON.stringify([r.sequencing, r.stages]),
  },
  {
    v: 1,
    method: "workflow.sequence",
    expect: (r) =>
      (r.sequencing.sequenced && r.quantities.includes("physical_dose_sequenced")) ||
      JSON.stringify([r.sequencing, r.quantities]),
  },
  {
    v: 1,
    method: "view.sequence",
    expect: (r, resp) => {
      const bad = r.beams.filter((beam) => {
        const map = beam.fluence?.delivered;
        return !(
          beam.shapes.length &&
          beam.weights.length === beam.shapes.length &&
          map &&
          resp.buffers[map.$array]?.nbytes === 4 * prod(map.shape) &&
          beam.target?.mask
        );
      });
      return (
        (r.sequencer === "xia" && r.num_of_levels === 3 && r.beams.length === PLAN.gantry_angles.length && !bad.length) ||
        `${r.sequencer}, ${r.num_of_levels} levels, ${r.beams.length} beams, incomplete: ${bad.map((b) => b.label)}`
      );
    },
  },
  {
    v: 1,
    method: "plan.set",
    note: "sequencing off again for the entries below",
    params: { sequencing: { run: false, sequencer: null } },
    expect: (r) => !r.sequencing.available || JSON.stringify(r.sequencing),
  },
  {
    v: 1,
    method: "data.load_file",
    note: "loading a pickle is refused in the browser (unpickling runs code from the file)",
    needs: "results/tg119_native.pkl",
    params: { filename: "tg119_native.pkl", data: { $file: "results/tg119_native.pkl" } },
    expectError: { type: "PreconditionError", code: "not_available" },
  },
  { v: 1, method: "export.save", params: { object: "workspace", format: "pickle" } },

  // ---- v1 writes -----------------------------------------------------------------------------
  {
    v: 1,
    method: "objectives.add",
    params: { structure: "Core" },
    expect: (r) => has(r, "state", "structure"),
    after: (r, ctx) => {
      ctx.objective = r.structure.objectives.length - 1;
    },
  },
  {
    v: 1,
    method: "objectives.update",
    params: (ctx) => ({ structure: "Core", index: ctx.objective, changes: { priority: 123 } }),
    expect: (r, _resp, ctx) => r.structure.objectives[ctx.objective].priority === 123 || "priority not written",
  },
  {
    v: 1,
    method: "objectives.change_type",
    params: (ctx) => ({ structure: "Core", index: ctx.objective, name: "Min DVH" }),
    expect: (r, _resp, ctx) => {
      const o = r.structure.objectives[ctx.objective];
      return (o.name === "Min DVH" && o.priority === 123) || JSON.stringify(o);
    },
  },
  {
    v: 1,
    method: "objectives.update",
    note: "out-of-range parameter -> ValidationError/invalid",
    params: (ctx) => ({ structure: "Core", index: ctx.objective, changes: { parameters: { v_min: 150 } } }),
    expectError: { type: "ValidationError", code: "invalid" },
  },
  { v: 1, method: "objectives.remove", params: (ctx) => ({ structure: "Core", index: ctx.objective }) },
  { v: 1, method: "objectives.set_quantity_all", params: { quantity: "physical_dose" } },
  {
    v: 1,
    method: "tissue.evaluate",
    note: "against an unapplied carbon / LEM plan form (draft)",
    params: (ctx) => ({
      rows: [{ structure: "Core", tissue: ctx.coreTissue }],
      draft: { radiation_mode: "carbon", machine: "Generic", bio_model: "LEM" },
    }),
    expect: (r) => (r.rows.length === 1 && /LEM/.test(r.rows[0].comparison?.closest_class ?? "")) || JSON.stringify(r.rows[0].comparison),
  },
  {
    v: 1,
    method: "tissue.apply",
    params: (ctx) => ({ rows: [{ structure: "Core", tissue: ctx.coreTissue }] }),
    expect: (r) => has(r, "state"),
  },
  {
    v: 1,
    method: "cst.update_structure",
    params: { name: "Core", changes: { color: [1, 2, 3], overlap_priority: 3 } },
    expect: (r) => (r.structure.overlap_priority === 3 && r.structure.color.join() === "1,2,3") || JSON.stringify(r.structure),
  },
  {
    v: 1,
    method: "base_data.add",
    session: false,
    params: {
      kind: "machines",
      filename: "VHEE_Focused.mat",
      name: "SmokeFocused",
      data: { $file: "../../pyRadPlan/data/machines/VHEE_Focused.mat" },
    },
    expect: (r) => has(r, "row"),
    after: (r, ctx) => {
      ctx.addedMachine = r.row.file;
    },
  },
  {
    v: 1,
    method: "base_data.list",
    session: false,
    params: { kind: "machines" },
    expect: (r, _resp, ctx) => r.rows.some((row) => row.file === ctx.addedMachine) || "added machine not listed",
  },
  { v: 1, method: "base_data.remove", session: false, params: (ctx) => ({ kind: "machines", file: ctx.addedMachine }) },
  {
    v: 1,
    python: (ctx) => `
import json
import pyRadPlan.app as app
def call(method, **params):
    reply = app.dispatch({"id": 1, "session": "${ctx.session}", "method": method, "params": params})
    return reply["result"] if reply["ok"] else {"error": reply["error"]["message"]}
none = json.dumps({"model": "none", "parameters": {}}, sort_keys=True)
followed = call("plan.options", radiation_mode="protons", current_bio={"radiation_mode": "photons", "model": "none", "option": none})
pln = app.default_manager().get("${ctx.session}").workspace.pln
pln.prop_opt = {**pln.prop_opt, "convert_dose_objectives": False}
values = call("plan.get", detail=True)["problem"]["values"]
edited = call("plan.set", problem={"name": "nonlin_fluence", "values": {**values, "p_norm": 3.0}})
call("plan.set", scenarios={"model": "wcScen", "values": {}})
switched = call("plan.set", radiation_mode="protons")
pln = app.default_manager().get("${ctx.session}").workspace.pln
result = {"selected": followed["bio_selection"], "default": followed["default_bio_model"], "values": values, "edited": "error" not in edited, "prop_opt": {k: v for k, v in pln.prop_opt.items() if k != "solver"}, "scenario": pln.scenarios.short_name, "mode": pln.radiation_mode}
call("plan.set", radiation_mode="photons", scenarios={"model": "nomScen", "values": {}})
result`,
    note: "bio_selection follows the mode default; convert_dose_objectives survives a web edit; a mode change keeps scenario + problem (v1 amendment 22)",
    expect: (r) =>
      (r.selected.option === r.default &&
        r.selected.replaced &&
        !("convert_dose_objectives" in r.values) &&
        r.edited &&
        r.prop_opt.convert_dose_objectives === false &&
        r.prop_opt.p_norm === 3 &&
        r.scenario === "wcScen" &&
        r.mode === "protons") ||
      JSON.stringify(r),
  },
  { v: 1, method: "plan.set", params: PLAN, note: "coarse plan for the exports" },
  {
    v: 1,
    method: "workflow.run_all",
    expect: (r) => (r.has.dij && r.has.result) || JSON.stringify(r.has),
  },
  ...[
    ["pln", "mat"],
    ["pln", "pickle"],
    ["dij", "mat"],
    ["dij", "pickle"],
    ["cst", "mat"],
    ["cst", "pickle"],
    ["cst", "npz"],
    ["cst", "dcm", "rtstruct"],
    ["cst", "dcm", "seg"],
    ["result", "npz"],
    ["result", "mat"],
    ["result", "dcm"],
  ].map(([object, format, structure_format]) => ({
    v: 1,
    method: "export.save",
    note: [object, format, structure_format].filter(Boolean).join(" "),
    params: { object, format, ...(structure_format ? { structure_format } : {}) },
    expect: (r, resp) => resp.buffers[r.data.$array]?.nbytes > 1000 || `${resp.buffers[r.data.$array]?.nbytes} bytes`,
  })),
  {
    v: 1,
    python: (ctx) => `
import pyRadPlan.app as app
saved = app.dispatch({"id": 1, "session": "${ctx.session}", "method": "export.save", "params": {"object": "dij", "format": "mat"}})
data = bytes(saved["buffers"][0])
loaded = app.dispatch({"id": 2, "session": "${ctx.session}", "method": "data.load_dij", "params": {"filename": "dij.mat", "data": {"$array": 0, "dtype": "uint8", "shape": [len(data)]}}, "buffers": [data]})
{"ok": loaded["ok"], "dij": loaded["result"]["stages"]["dij"] if loaded["ok"] else loaded["error"]}`,
    note: "data.load_dij of an exported dij.mat",
    expect: (r) => (r.ok && r.dij === "done") || JSON.stringify(r),
  },
];
