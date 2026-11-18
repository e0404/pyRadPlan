// Full GUI click-through against the real backend: Pyodide + pyRadPlan in the web worker, TG119
// with the web profile defaults (5 x 5 x 5 mm dose grid, "total" doses). Stages run in one page;
// a failing stage is screenshotted (pyodide-fail-<stage>.png), recorded and the run continues.
// Needs the wheels in public/pyrp/ (`npm run build` in web/pyodide) and network access to the
// Pyodide CDN. Usage: npm run e2e:pyodide [-- --headed] [-- --preview] [-- --only=a,b]
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { build, createServer, preview } from "vite";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const repo = join(root, "..");
const shots = join(here, "screenshots");
mkdirSync(shots, { recursive: true });
const headed = process.argv.includes("--headed");
const usePreview = process.argv.includes("--preview");
const prefix = usePreview ? "pyodide-prod-" : "pyodide-";

let server;
if (usePreview) {
  await build({ root, logLevel: "warn" });
  server = await preview({ root, preview: { port: 4198, strictPort: false }, logLevel: "warn" });
} else {
  server = await createServer({ root, server: { port: 5198, strictPort: false }, logLevel: "warn" });
  await server.listen();
}
const base = server.resolvedUrls.local[0];
console.log(`${usePreview ? "preview" : "dev"} server: ${base}`);
const browser = await chromium.launch({
  channel: "msedge",
  headless: !headed,
  args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
});
const context = await browser.newContext({ viewport: { width: 1680, height: 960 }, acceptDownloads: true });
const page = await context.newPage();
const problems = [];
page.on("pageerror", (err) => problems.push(`pageerror: ${err.message}`));
page.on("console", (msg) => {
  if (msg.type() === "error") problems.push(`console: ${msg.text()}`);
});

const tid = (id) => page.getByTestId(id);
const t0 = Date.now();
const elapsed = () => `${((Date.now() - t0) / 1000).toFixed(1)} s`;
const metrics = {};
const failures = [];
const passes = [];
let step = 0;

async function shot(name, locator) {
  step += 1;
  const file = join(shots, `${prefix}${String(step).padStart(2, "0")}-${name}.png`);
  await page.waitForTimeout(250);
  if (locator) await locator.screenshot({ path: file });
  else await page.screenshot({ path: file });
  console.log(`[${elapsed()}] screenshot ${file}`);
}

function expect(cond, message) {
  if (!cond) throw new Error(`check failed: ${message}`);
  passes.push(message);
  console.log(`ok: ${message}`);
}

const state = () => page.evaluate(() => window.__pyrpApp.useApp.getState().state);
const store = (fn) => page.evaluate(fn);

async function waitIdle(timeout = 30 * 60 * 1000) {
  await page.waitForTimeout(120);
  await page.waitForFunction(() => document.querySelector('[data-testid="btn-stop"]')?.disabled === true, null, { timeout });
}

async function closeStatusWindow() {
  if (await tid("opt-close").count()) await tid("opt-close").click();
}

async function openMenu(id) {
  await tid(`menu-${id}`).click();
  await tid(`popup-${id}`).waitFor();
}

async function clickItem(menu, item) {
  await openMenu(menu);
  await tid(`menu-item-${item}`).click();
}

async function pick(testId, text) {
  await tid(testId).click();
  const popup = tid(`${testId}-popup`).or(page.locator(".combo-popup")).first();
  await popup.waitFor();
  await popup.locator(".combo-item", { hasText: text }).first().click();
}

async function fillSpin(testId, value) {
  await tid(testId).fill(String(value));
  await tid(testId).press("Tab");
}

async function download(trigger, timeout = 10 * 60 * 1000) {
  const [dl] = await Promise.all([page.waitForEvent("download", { timeout }), trigger()]);
  const stream = await dl.createReadStream();
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  return { name: dl.suggestedFilename(), data: Buffer.concat(chunks) };
}

async function chooseFile(trigger, file) {
  const [chooser] = await Promise.all([page.waitForEvent("filechooser"), trigger()]);
  await chooser.setFiles(file);
}

async function dismissDialogs() {
  for (let i = 0; i < 4; i++) {
    const open = await page.locator(".modal-backdrop, [role=dialog]").count();
    if (!open) break;
    await page.keyboard.press("Escape");
    await page.waitForTimeout(100);
  }
  await closeStatusWindow();
}

/** Click a workflow button and wait for its operation; offers the pending plan edits first (Qt). */
async function runStep(testId, label) {
  await tid(testId).click();
  await page.waitForTimeout(300);
  if (await tid("pending-changes-dialog").count()) await tid("msg-apply").click();
  await page.waitForFunction(() => document.querySelector('[data-testid="btn-stop"]')?.disabled === false, null, { timeout: 60000 }).catch(() => undefined);
  await waitIdle();
  await page.waitForTimeout(300);
  if (await tid("run-warnings").count()) {
    const text = ((await tid("run-warnings").textContent()) ?? "").replace(/\s+/g, " ").slice(0, 300);
    console.log(`run warnings (${label}): ${text}`);
    metrics[`warnings_${label}`] = text;
    await tid("msg-ok").click();
  }
  if (await tid("error-dialog").count()) throw new Error(`${label} failed: ${await tid("error-dialog").textContent()}`);
}

/** Calc. Dose Influence, then Optimize (the Qt workflow buttons). */
async function computeResult(label) {
  const start = Date.now();
  await runStep("btn-dij", `${label}_dij`);
  expect((await tid("dot-dij").getAttribute("data-status")) === "done", `dose influence (${label}) in ${((Date.now() - start) / 1000).toFixed(1)} s`);
  await runStep("btn-optimize", `${label}_optimize`);
  const seconds = (Date.now() - start) / 1000;
  metrics[`result_${label}_s`] = +seconds.toFixed(1);
  expect((await tid("dot-result").getAttribute("data-status")) === "done", `dose influence + optimization (${label}) in ${seconds.toFixed(1)} s`);
}

async function memory(label) {
  const mb = await page
    .evaluate(async () => {
      if (!performance.measureUserAgentSpecificMemory) return null;
      const r = await performance.measureUserAgentSpecificMemory();
      return Math.round(r.bytes / 1048576);
    })
    .catch(() => null);
  if (mb !== null) metrics[`memory_mb_${label}`] = mb;
  const wasm = await page
    .evaluate(async () => {
      const backend = window.__pyrpApp.useApp.getState().backend;
      return backend?.wasmMemory ? await backend.wasmMemory() : null;
    })
    .catch(() => null);
  if (wasm) metrics[`wasm_heap_mb_${label}`] = Math.round(wasm / 1048576);
}

const only = (process.argv.find((a) => a.startsWith("--only=")) ?? "").slice(7).split(",").filter(Boolean);
async function stage(name, fn) {
  if (only.length && !only.includes(name) && !["boot", "load"].includes(name)) return;
  const start = Date.now();
  console.log(`\n=== ${name} [${elapsed()}]`);
  try {
    await fn();
    console.log(`=== ${name} done in ${((Date.now() - start) / 1000).toFixed(1)} s`);
  } catch (err) {
    failures.push(`${name}: ${err.message.split("\n")[0]}`);
    console.error(`!!! ${name} failed:`, err.message);
    await page.screenshot({ path: join(shots, `${prefix}fail-${name}.png`) }).catch(() => undefined);
    await dismissDialogs().catch(() => undefined);
    await waitIdle(120000).catch(() => undefined);
  }
}

const saved = {};

try {
  await stage("boot", async () => {
    const start = Date.now();
    await page.goto(`${base}?backend=pyodide`);
    await tid("boot-splash").waitFor({ state: "detached", timeout: 15 * 60 * 1000 });
    await page.waitForFunction(() => window.__pyrpApp?.useApp.getState().session);
    expect(!(await tid("btn-run-all").count()) && !(await tid("phantom-select").count()), "workflow widget has the Qt controls only (no Run All, no phantom selector)");
    metrics.boot_s = +((Date.now() - start) / 1000).toFixed(1);
    const info = await store(() => window.__pyrpApp.useApp.getState().info);
    expect(info.runtime === "pyodide", `runtime pyodide, pyRadPlan ${info.version}, Python ${info.python}, boot ${metrics.boot_s} s`);
    expect(info.default_profile === "web", "web profile is the default under Pyodide");
    await memory("boot");
  });

  await stage("load", async () => {
    await clickItem("file", "phantoms");
    await tid("menu-item-phantom:TG119").click();
    await page.waitForFunction(() => document.querySelector('[data-testid="dot-load"]')?.getAttribute("data-status") === "done", null, { timeout: 120000 });
    await waitIdle();
    await page.waitForFunction(() => document.querySelector('[data-testid="plan-res-x"]')?.value === "5.00");
    const s = await state();
    expect(s.options?.profile === "web" && s.plan.dose_grid_resolution.join(" ") === "5 5 5", `web profile plan: ${s.plan.gantry_angles.length} beam(s), dose grid ${s.plan.dose_grid_resolution.join(" ")} mm`);
    await page.waitForFunction(() => document.querySelectorAll('[data-testid^="obj-row-"]').length > 0);
    await shot("loaded");
  });

  await stage("run-default", async () => {
    await computeResult("default_plan");
    await shot("status-window-default");
    await closeStatusWindow();
    await memory("after_default_run");
  });

  await stage("plan", async () => {
    await tid("plan-num-beams").fill("5");
    await tid("plan-num-beams").press("Enter");
    expect((await tid("plan-gantry").inputValue()) === "0 72 144 216 288", "beam count spin sets 5 equidistant beams");
    await tid("plan-engine-config").click();
    await tid("engine-dialog").waitFor();
    await page.waitForTimeout(300);
    const engineFields = await tid("engine-dialog").locator("[data-testid^='engine-dialog-form-']").count();
    expect(engineFields > 3, `engine dialog shows the real descriptors (${engineFields} fields)`);
    await shot("engine-dialog", tid("engine-dialog"));
    await page.keyboard.press("Escape");
    await tid("plan-scenario-config").click();
    await tid("scenario-dialog").waitFor();
    await shot("scenario-dialog", tid("scenario-dialog"));
    await page.keyboard.press("Escape");
    await tid("plan-problem-config").click();
    await tid("optimization-dialog").waitFor();
    await tid("solver-form").waitFor();
    await tid("solver-combo").click();
    await tid("solver-combo-popup").waitFor();
    const solvers = await tid("solver-combo-popup").locator(".combo-item").allTextContents();
    await shot("optimization-dialog", tid("optimization-dialog"));
    await page.keyboard.press("Escape");
    expect(solvers.length >= 2, `solvers listed: ${solvers.join(" | ")}`);
    await page.keyboard.press("Escape");
    await tid("plan-apply").click();
    await page.waitForFunction(() => /Plan applied/.test(document.querySelector('[data-testid="plan-note"]')?.textContent ?? ""), null, { timeout: 60000 });
    expect((await state()).plan.gantry_angles.length === 5, "plan applied with 5 beams");
  });

  await stage("objectives", async () => {
    const count = async () => Number(((await tid("obj-count").textContent()) ?? "0").split(" ")[0]);
    const before = await count();
    await pick("obj-voi", "Core");
    await tid("obj-add").click();
    await page.waitForFunction((n) => Number(document.querySelector('[data-testid="obj-count"]')?.textContent?.split(" ")[0]) === n + 1, before);
    const key = await page.evaluate(() => {
      const rows = [...document.querySelectorAll('[data-testid^="obj-Core-"][data-testid$="-type"]')];
      return rows.at(-1)?.getAttribute("data-testid")?.replace(/-type$/, "");
    });
    await pick(`${key}-type`, "Min DVH");
    await tid(`${key}-v_min`).waitFor();
    await fillSpin(`${key}-v_min`, "80");
    await fillSpin(`${key}-penalty`, "500");
    await page.waitForTimeout(800);
    const detail = await page.evaluate(async () => {
      const { backend, session } = window.__pyrpApp.useApp.getState();
      const r = await backend.request("cst.get", { names: ["Core"] }, session);
      return r.structures[0].objectives;
    });
    const added = detail.at(-1);
    expect(added.name === "Min DVH" && added.priority === 500 && Number(added.parameters.v_min) === 80, `objective added, type changed, edited: ${JSON.stringify(added)}`);
    await shot("objectives", tid("objectives"));
    await tid(`${key}-remove`).click();
    await page.waitForFunction((n) => Number(document.querySelector('[data-testid="obj-count"]')?.textContent?.split(" ")[0]) === n, before);
    expect(true, "objective removed");
  });

  await stage("run-5-beams", async () => {
    const run = computeResult("5_beams_5mm");
    await tid("opt-status-window").waitFor({ timeout: 30 * 60 * 1000 });
    await page.waitForFunction(() => /iter \d+/.test(document.querySelector('[data-testid="opt-summary"]')?.textContent ?? ""), null, { timeout: 30 * 60 * 1000 });
    await shot("status-window-running");
    await run;
    await shot("status-window-finished");
    await closeStatusWindow();
    await memory("after_5_beams");
  });

  await stage("viewer", async () => {
    await page.waitForFunction(() => window.__pyrpApp.useApp.getState().beams?.beams.length === 5);
    await tid("toggle-beams").check();
    await tid("toggle-isocenter").check();
    await tid("toggle-isolines").check();
    await page.mouse.move(5, 5);
    await shot("viewer-axial-beams-isolines");
    const comps = await tid("quantity-select-component").locator("option").allTextContents();
    expect(comps.includes("beam 2"), `per-beam components: ${comps.join(", ")}`);
    await tid("quantity-select-component").selectOption("physical_dose_beam 2");
    await page.waitForFunction(() => window.__pyrpApp.useApp.getState().quantities["physical_dose_beam 2"], null, { timeout: 60000 });
    const box = await tid("viewer-canvas").boundingBox();
    await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.6);
    await tid("viewer-readout").waitFor();
    await shot("viewer-beam-component");
    await tid("quantity-select-component").selectOption("physical_dose");
    for (const plane of ["sagittal", "coronal"]) {
      await tid("plane-select").selectOption(plane);
      await page.mouse.move(5, 5);
      await shot(`viewer-${plane}`);
    }
    await tid("plane-select").selectOption("axial");
  });

  await stage("snapshot-analysis", async () => {
    await page.waitForFunction(() => document.querySelector('[data-testid="btn-snapshot"]')?.disabled === false);
    await tid("btn-snapshot").click();
    await tid("prompt-dialog").waitFor();
    await tid("prompt-input").fill("five beams");
    await tid("prompt-ok").click();
    await page.waitForFunction(() => window.__pyrpApp.useApp.getState().state.results?.tags.includes("five_beams"), null, { timeout: 120000 });
    expect(true, "snapshot five_beams saved");
    await tid("btn-dvh").click();
    await tid("analysis-window").waitFor();
    await page.waitForFunction(() => document.querySelectorAll('[data-testid="qi-table"] tbody tr').length > 0, null, { timeout: 120000 });
    await tid("analysis-compare-group").selectOption({ label: "dose (five_beams)" });
    await tid("analysis-compare-component").selectOption({ label: "beam 0" });
    await tid("analysis-show-dx").check();
    const headers = await tid("qi-table").locator("th").allTextContents();
    expect(headers.includes("D95 [Gy]"), `QI table: ${headers.join(" | ")}`);
    await page.waitForTimeout(1500);
    await shot("analysis");
    await tid("analysis-close").click();
  });

  await stage("voi-edit", async () => {
    await page.evaluate(() => {
      const input = document.querySelector('[data-testid="voi-color-input"]');
      document.querySelector('[data-testid="voi-color-Core"]').click();
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(input, "#ff8000");
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await page.waitForFunction(() => window.__pyrpApp.useApp.getState().state.structures.find((s) => s.name === "Core")?.color.join(",") === "255,128,0", null, { timeout: 60000 });
    await tid("voi-name-Core").click();
    await tid("edit-voi-dialog").waitFor();
    await tid("edit-voi-priority").fill("4");
    await page.waitForTimeout(800);
    await shot("edit-voi", tid("edit-voi-dialog"));
    await tid("edit-voi-ok").click();
    await tid("edit-voi-dialog").waitFor({ state: "detached", timeout: 60000 });
    const core = (await state()).structures.find((s) => s.name === "Core");
    expect(core.color.join(",") === "255,128,0" && core.overlap_priority === 4, "VOI colour and overlap priority written to the cst");
    // Priority 4 puts Core under BODY (3), which empties it; restore it so later runs optimize a real plan.
    await tid("voi-name-Core").click();
    await tid("edit-voi-dialog").waitFor();
    await tid("edit-voi-priority").fill("2");
    await tid("edit-voi-ok").click();
    await tid("edit-voi-dialog").waitFor({ state: "detached", timeout: 60000 });
    expect((await state()).structures.find((s) => s.name === "Core").overlap_priority === 2, "VOI overlap priority restored");
  });

  await stage("saves", async () => {
    const formats = await store(() => window.__pyrpApp.useApp.getState().info.capabilities.formats.export);
    const item = { pln: "save_pln", dij: "save_dij", cst: "save_cst", result: "save_result", workspace: "save_workspace" };
    for (const f of formats.filter((x) => x.available)) {
      const label = `${f.object}:${f.label}`;
      const start = Date.now();
      await clickItem("file", item[f.object]);
      if (f.object === "result") {
        await tid("save-result-dialog").waitFor();
        await tid("save-result-all").click();
        await tid("save-result-ok").click();
      }
      await tid("save-dialog").waitFor();
      await tid("save-format").selectOption({ label: f.label });
      let file;
      try {
        file = await download(async () => {
          await tid("save-ok").click();
          await page.waitForTimeout(300);
          if (await tid("objectives-warning").count()) await tid("msg-ok").click();
        });
      } catch (err) {
        throw new Error(`${label}: no download (${err.message.split("\n")[0]})`);
      }
      await page.waitForTimeout(200);
      if (await tid("saved-dialog").count()) await tid("msg-ok").click();
      expect(file.data.length > 1000, `saved ${label} -> ${file.name} (${(file.data.length / 1024).toFixed(0)} KiB, ${((Date.now() - start) / 1000).toFixed(1)} s)`);
      const path = join(shots, "..", "downloads", file.name);
      mkdirSync(dirname(path), { recursive: true });
      writeFileSync(path, file.data);
      saved[`${f.object}.${f.format}${f.structure_format ? `.${f.structure_format}` : ""}`] = path;
    }
    await shot("after-saves");
  });

  await stage("reload", async () => {
    const ws = saved["workspace.pickle"];
    expect(Boolean(ws), "workspace pickle saved");
    await tid("file-input").setInputFiles(ws);
    await page.waitForTimeout(500);
    await waitIdle();
    if (await tid("error-dialog").count()) throw new Error(`workspace reload: ${await tid("error-dialog").textContent()}`);
    if (await tid("run-warnings").count()) await tid("msg-ok").click();
    const s = await state();
    expect(s.has.ct && s.has.dij && s.has.result && s.results.tags.includes("five_beams"), "workspace pickle reloads ct, dij, result and snapshot tags");
    const mat = saved["pln.mat"];
    await tid("file-input").setInputFiles(mat);
    await page.waitForTimeout(500);
    await waitIdle();
    if (await tid("run-warnings").count()) await tid("msg-ok").click();
    if (await tid("error-dialog").count()) throw new Error(`plan .mat reload: ${await tid("error-dialog").textContent()}`);
    expect((await state()).plan.gantry_angles.length === 5, "plan .mat reloads through Load File");
    const dij = saved["dij.mat"];
    await chooseFile(() => tid("btn-load-dij").click(), dij);
    await page.waitForTimeout(500);
    await waitIdle();
    if (await tid("error-dialog").count()) throw new Error(`Load Dij: ${await tid("error-dialog").textContent()}`);
    if (await tid("message-box").count()) throw new Error(`Load Dij: ${await tid("message-box").textContent()}`);
    expect((await state()).stages.dij === "done", "Load Dij loads stf + dij from the saved .mat");
    await shot("after-reload");
  });

  await stage("base-data", async () => {
    await clickItem("base_data", "show_base_data");
    await tid("base-data-dialog").waitFor();
    await tid("base-data-row-photons_Generic.mat").waitFor({ timeout: 60000 });
    await shot("base-data", tid("base-data-dialog"));
    await chooseFile(() => tid("base-data-add-machines").click(), join(repo, "pyRadPlan", "data", "machines", "VHEE_Focused.mat"));
    await tid("prompt-dialog").waitFor({ timeout: 60000 });
    await tid("prompt-input").fill("WebFocused");
    await tid("prompt-ok").click();
    await tid("base-data-row-VHEE_WebFocused.mat").waitFor({ timeout: 60000 });
    expect(true, "machine added under a new name");
    await tid("base-data-row-VHEE_WebFocused.mat").click();
    await tid("base-data-remove-machines").click();
    await tid("remove-dialog").waitFor();
    await tid("msg-yes").click();
    await tid("base-data-row-VHEE_WebFocused.mat").waitFor({ state: "detached" });
    expect(true, "added machine removed");
    await page.keyboard.press("Escape");
  });

  await stage("settings-help-log", async () => {
    await clickItem("settings", "preferences");
    await tid("preferences-dialog").waitFor();
    await tid("preferences-tab-xp").click();
    await shot("preferences-xp", tid("preferences-dialog"));
    await tid("preferences-cancel").click();
    await clickItem("help", "licenses");
    await tid("licenses-dialog").waitFor();
    await tid("license-row-numpy").waitFor({ timeout: 60000 });
    await tid("license-row-numpy").click();
    await page.waitForFunction(() => (document.querySelector('[data-testid="license-text"]')?.value ?? "").length > 100, null, { timeout: 60000 });
    await shot("licenses", tid("licenses-dialog"));
    await page.keyboard.press("Escape");
    await clickItem("help", "about");
    await tid("about-dialog").waitFor();
    await shot("about", tid("about-dialog"));
    await tid("about-close").click();
    await tid("log-level").selectOption("debug");
    await page.waitForTimeout(500);
    expect((await store(() => window.__pyrpApp.useApp.getState().logs.length)) > 0, "log level debug set");
    await tid("log-level").selectOption("info");
    await clickItem("view", "dark_mode");
    await page.waitForTimeout(300);
    expect((await page.evaluate(() => document.documentElement.dataset.theme)) === "dark", "dark mode");
    await shot("dark-mode");
    await clickItem("view", "dark_mode");
    await openMenu("file");
    await tid("menu-item-ui_mode").hover();
    await tid("menu-item-ui_mode:educational").click();
    expect(await tid("plan-engine").isDisabled(), "educational mode locks the engine");
    await shot("educational", tid("panel-plan"));
    await openMenu("file");
    await tid("menu-item-ui_mode").hover();
    await tid("menu-item-ui_mode:default").click();
  });

  await stage("robust", async () => {
    // Small robust case: 1 beam, worst-case grid scenarios, a robust objective, few iterations.
    await tid("plan-gantry").fill("0");
    await pick("plan-scenario", "Worst-case Grid");
    await tid("plan-apply").click();
    await page.waitForFunction(() => /Plan applied/.test(document.querySelector('[data-testid="plan-note"]')?.textContent ?? ""), null, { timeout: 60000 });
    const robustKey = await page.evaluate(() => document.querySelector('[data-testid^="obj-OuterTarget-"][data-testid$="-robustness"]')?.getAttribute("data-testid"));
    await pick(robustKey, "Voxel-wise worst case");
    await page.waitForTimeout(800);
    const objectives = await page.evaluate(async () => {
      const { backend, session } = window.__pyrpApp.useApp.getState();
      return (await backend.request("cst.get", { names: ["OuterTarget"] }, session)).structures[0].objectives.map((o) => o.robustness);
    });
    expect(objectives.includes("voxel_worst_case"), `robust objective set: ${objectives.join(", ")}`);
    const start = Date.now();
    await runStep("btn-dij", "robust_dij");
    await page.evaluate(() => window.__pyrpApp.actions.optimize({ max_iter: 15 }));
    await waitIdle();
    await closeStatusWindow();
    metrics.result_robust_1_beam_s = +((Date.now() - start) / 1000).toFixed(1);
    const s = await state();
    const labels = s.results.scenario_labels;
    expect(labels.length > 1 && labels[0] === "nominal", `robust result has ${labels.length} scenarios (${metrics.result_robust_1_beam_s} s)`);
    const comps = await tid("quantity-select-component").locator("option").allTextContents();
    expect(comps.includes(`scenario 1: ${labels[1]}`), `scenario components in the viewer selector: ${comps.slice(0, 4).join(" | ")} …`);
    await tid("quantity-select-component").selectOption("physical_dose_scenario 1");
    await page.waitForFunction(() => window.__pyrpApp.useApp.getState().quantities["physical_dose_scenario 1"], null, { timeout: 60000 });
    const box = await tid("viewer-canvas").boundingBox();
    await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.6);
    await shot("robust-scenario-viewer");
    await tid("btn-dvh").click();
    await tid("analysis-window").waitFor();
    await page.waitForFunction(() => document.querySelectorAll('[data-testid="qi-table"] tbody tr').length > 0, null, { timeout: 120000 });
    expect((await tid("analysis-primary-component").inputValue()) === "physical_dose_scenario 1", "analysis opens on the viewer's scenario");
    await tid("analysis-all-scenarios").check();
    await page.waitForTimeout(2500);
    const qi = await page.evaluate(async () => {
      const { backend, session } = window.__pyrpApp.useApp.getState();
      return (await backend.request("analysis.qi", { quantity: "physical_dose_scenario 1" }, session)).name;
    });
    expect(qi === "dose, scenario 1", "QI of a scenario quantity");
    await shot("robust-analysis");
    await tid("analysis-close").click();
    await memory("after_robust");
    // Back to the nominal 5-beam plan for the next stages.
    await page.keyboard.press("Escape");
    await pick("plan-scenario", "Nominal Scenario");
    await pick(robustKey, "None");
    await page.waitForTimeout(500);
    await tid("plan-gantry").fill("0 72 144 216 288");
    await tid("plan-apply").click();
    await page.waitForFunction(() => /Plan applied/.test(document.querySelector('[data-testid="plan-note"]')?.textContent ?? ""), null, { timeout: 60000 });
  });

  await stage("protons", async () => {
    // A worst-case scenario model applied for photons survives the switch to protons (v1
    // amendment 22: the backend keeps it; the web sends no scenario on the mode change).
    await pick("plan-scenario", "Worst-case Grid");
    await tid("plan-apply").click();
    await page.waitForFunction(() => /Plan applied/.test(document.querySelector('[data-testid="plan-note"]')?.textContent ?? ""), null, { timeout: 60000 });
    await pick("plan-mode", "protons");
    await page.waitForTimeout(1000);
    const backendPick = await page.evaluate(async () => {
      const { backend, session } = window.__pyrpApp.useApp.getState();
      const opts = await backend.request("plan.options", { radiation_mode: "protons" }, session);
      return { option: opts.bio_selection?.option, default: opts.default_bio_model };
    });
    const shownBio = await tid("plan-bio").getAttribute("data-value");
    expect(shownBio === backendPick.option && shownBio === backendPick.default, `the proton bio model is the backend's selection, the mode default (${shownBio})`);
    await tid("plan-bio").click();
    await tid("plan-bio-popup").waitFor();
    const bio = await tid("plan-bio-popup").locator(".combo-item").allTextContents();
    await shot("proton-bio-models", tid("panel-plan"));
    await page.keyboard.press("Escape");
    expect(bio.length >= 2, `proton bio models: ${bio.join(" | ")}`);
    await tid("plan-apply").click();
    await page.waitForFunction(() => /Plan applied/.test(document.querySelector('[data-testid="plan-note"]')?.textContent ?? ""), null, { timeout: 120000 });
    const switched = await page.evaluate(async () => {
      const { backend, session } = window.__pyrpApp.useApp.getState();
      return backend.request("plan.get", { detail: true }, session);
    });
    expect(
      switched.radiation_mode === "protons" && switched.scenario.model === "wcScen" && switched.bio.option === backendPick.default,
      `photons -> protons keeps the scenario model (${switched.scenario.model}) and applies the selected bio model (${switched.bio.model})`,
    );
    await pick("plan-scenario", "Nominal Scenario");
    await tid("plan-apply").click();
    await page.waitForFunction(
      () => /Plan applied/.test(document.querySelector('[data-testid="plan-note"]')?.textContent ?? "") && window.__pyrpApp.useApp.getState().state?.plan?.radiation_mode === "protons",
      null,
      { timeout: 120000 },
    );
    await tid("plan-tissue").click();
    await tid("tissue-dialog").waitFor();
    await page.waitForTimeout(2000);
    await shot("tissue-dialog", tid("tissue-dialog"));
    await page.keyboard.press("Escape");
    await computeResult("protons_5_beams");
    await closeStatusWindow();
    await page.waitForTimeout(1500);
    const ranges = await store(() => Object.fromEntries(Object.entries(window.__pyrpApp.useApp.getState().quantities).map(([k, v]) => [k, v.range])));
    expect((ranges.rbe_x_dose?.[1] ?? 0) > 0, `proton result has an RBE-weighted dose: ${JSON.stringify(ranges)}`);
    await shot("proton-dose");
    await memory("after_protons");
  });

  await stage("stop", async () => {
    await pick("plan-mode", "photons");
    await tid("plan-gantry").fill("0 40 80 120 160 200 240 280 320");
    await tid("plan-apply").click();
    await page.waitForFunction(() => /Plan applied/.test(document.querySelector('[data-testid="plan-note"]')?.textContent ?? ""), null, { timeout: 120000 });
    const applied = (await state()).plan;
    expect(applied.radiation_mode === "photons" && applied.engine !== "HongPB", `Apply right after a mode change sends the new mode's engine (${applied.engine})`);
    await tid("btn-dij").click();
    await page.waitForFunction(() => /Beam/.test(document.querySelector('[data-testid="workflow-status"]')?.textContent ?? ""), null, { timeout: 120000 });
    await tid("btn-stop").click();
    const start = Date.now();
    await page.waitForFunction(() => /stopped/.test(document.querySelector('[data-testid="statusbar"]')?.textContent ?? ""), null, { timeout: 60000 });
    metrics.stop_latency_ms = Date.now() - start;
    await waitIdle();
    expect((await tid("dot-dij").getAttribute("data-status")) !== "done", `Stop interrupts the dose calculation (${metrics.stop_latency_ms} ms)`);
    await shot("stopped");
  });

  await stage("stop-optimization", async () => {
    await tid("plan-gantry").fill("0 120 240");
    await tid("plan-apply").click();
    await page.waitForFunction(() => /Plan applied/.test(document.querySelector('[data-testid="plan-note"]')?.textContent ?? ""), null, { timeout: 120000 });
    await runStep("btn-dij", "stop_optimization_dij");
    await tid("btn-optimize").click();
    await page.waitForTimeout(300);
    if (await tid("pending-changes-dialog").count()) await tid("msg-apply").click();
    await page.waitForFunction(
      () => Number(/iteration (\d+)/.exec(window.__pyrpApp.useApp.getState().progress?.detail ?? "")?.[1] ?? 0) >= 2,
      null,
      { timeout: 120000 },
    );
    await tid("btn-stop").click();
    await waitIdle();
    await page.waitForTimeout(300);
    if (await tid("run-warnings").count()) await tid("msg-ok").click();
    const run = await page.evaluate(async () => {
      const { backend, session } = window.__pyrpApp.useApp.getState();
      return (await backend.request("session.history", {}, session)).optimization;
    });
    expect(run?.outcome === "stopped", `Stop ends the optimization (${run?.outcome} after ${run?.points.length} iterations)`);
    expect((await tid("dot-result").getAttribute("data-status")) === "done", "the stopped optimization keeps its last iterate");
    await closeStatusWindow();
    await shot("optimization-stopped");
  });
} finally {
  console.log(`\nmetrics: ${JSON.stringify(metrics)}`);
  console.log(`passed checks: ${passes.length}`);
  if (problems.length) {
    console.log(`browser problems:\n${[...new Set(problems)].join("\n")}`);
    failures.push("browser reported console errors");
  }
  if (failures.length) {
    console.log(`FAILED stages:\n${failures.join("\n")}`);
    process.exitCode = 1;
  } else console.log(`pyodide e2e passed in ${elapsed()}`);
  writeFileSync(join(shots, `${prefix}metrics.json`), JSON.stringify({ metrics, failures, passes }, null, 2));
  await browser.close();
  await server.close();
}
