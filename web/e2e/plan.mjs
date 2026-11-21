// End-to-end check of package A (plan form, configuration dialogs, optimization dialog,
// tissue dialog, objectives table, optimization status window) against the mock backend.
// Starts its own Vite dev server on port 5201; screenshots go to e2e/screenshots/plan-*.png.
// Usage: npm run e2e:plan [-- --headed] [-- --chromium]
import { mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { createServer } from "vite";
import { acknowledgeNotice } from "./notice.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const shots = join(here, "screenshots");
mkdirSync(shots, { recursive: true });

const headed = process.argv.includes("--headed");
const useBundledChromium = process.argv.includes("--chromium");

const server = await createServer({ root, server: { port: 5201, strictPort: true, hmr: false }, logLevel: "warn" });
await server.listen();
const base = server.resolvedUrls.local[0];
console.log(`dev server: ${base}`);

const browser = await chromium.launch({
  ...(useBundledChromium ? {} : { channel: "msedge" }),
  headless: !headed,
  args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
});
const context = await browser.newContext({ viewport: { width: 1680, height: 1000 }, deviceScaleFactor: 1 });
await acknowledgeNotice(context, base);
const page = await context.newPage();
const problems = [];
page.on("pageerror", (err) => problems.push(`pageerror: ${err.message}`));
page.on("console", (msg) => {
  if (msg.type() === "error") problems.push(`console: ${msg.text()}`);
});

let step = 0;
async function shot(name, locator) {
  step += 1;
  const file = join(shots, `plan-${String(step).padStart(2, "0")}-${name}.png`);
  await page.waitForTimeout(200);
  if (locator) await locator.screenshot({ path: file });
  else await page.screenshot({ path: file });
  console.log(`screenshot ${file}`);
}

const tid = (id) => page.getByTestId(id);

function expect(cond, message) {
  if (!cond) throw new Error(`check failed: ${message}`);
  console.log(`ok: ${message}`);
}

async function waitIdle() {
  await page.waitForFunction(() => {
    const btn = document.querySelector('[data-testid="btn-stop"]');
    return btn && btn.disabled;
  });
}

async function note() {
  return (await tid("plan-note").textContent()) ?? "";
}

/** Pick an entry of one of the Qt-like combo boxes by its visible text. */
async function pick(testId, text) {
  await tid(testId).click();
  const popup = tid(`${testId}-popup`).or(page.locator(".combo-popup")).first();
  await popup.waitFor();
  await popup.locator(".combo-item", { hasText: text }).first().click();
}

async function fillSpin(testId, value) {
  const input = tid(testId);
  await input.fill(String(value));
  await input.press("Tab");
}

try {
  await page.goto(`${base}?backend=mock&speed=0.4`);
  await tid("boot-splash").waitFor({ state: "detached", timeout: 20000 });
  await page.waitForFunction(() => window.__pyrpApp?.useApp.getState().session);
  await tid("menu-file").click();
  await tid("menu-item-phantoms").click();
  await tid("menu-item-phantom:TG119").click();
  await waitIdle();
  await page.waitForFunction(() => document.querySelectorAll('[data-testid^="obj-row-"]').length > 0);
  await page.waitForFunction(() => document.querySelector('[data-testid="plan-res-x"]')?.value === "5.00");
  expect((await note()) === "Default plan applied.", "default plan applied on load (web profile: 5 mm grid)");
  await shot("loaded");
  await shot("plan-widget", tid("panel-plan"));

  // ---- beam angles: count spin, couch sync, presets
  await tid("plan-num-beams").fill("5");
  await tid("plan-num-beams").press("Enter");
  expect((await tid("plan-gantry").inputValue()) === "0 72 144 216 288", "beam count sets equidistant gantry angles");
  expect((await tid("plan-couch").inputValue()) === "0 0 0 0 0", "couch angles follow the beam count");
  expect((await note()).includes("Modified"), "modified note shown");
  expect((await tid("plan-gantry").getAttribute("class")).includes("modified"), "gantry field framed as modified");
  await tid("plan-presets").click();
  await tid("plan-presets-popup").waitFor();
  await shot("preset-menu", tid("panel-plan"));
  await tid("plan-presets-popup").locator(".combo-item", { hasText: "7-field IMRT" }).click();
  expect((await tid("plan-gantry").inputValue()).split(" ").length === 7, "preset sets 7 gantry angles");
  expect((await tid("plan-apply").textContent()) === "Apply *", "Apply shows the pending marker");

  // ---- dose engine "…": generic form with nullable numbers, validation
  await tid("plan-engine-config").click();
  await tid("engine-dialog").waitFor();
  await fillSpin("engine-dialog-form-kernel_cutoff", "-5");
  const engineStatus = await tid("engine-dialog-status").textContent();
  expect(/kernel_cutoff: Input should be greater than 0/.test(engineStatus), `live validation: ${engineStatus}`);
  await shot("engine-dialog-invalid", tid("engine-dialog"));
  await fillSpin("engine-dialog-form-kernel_cutoff", "350");
  await fillSpin("engine-dialog-form-force_penumbra", "4.5");
  await shot("engine-dialog", tid("engine-dialog"));
  await tid("engine-dialog-ok").click();
  expect((await tid("plan-engine-config").getAttribute("class")).includes("modified"), "engine … framed as modified");

  // ---- scenario model + its form (nested groups, number lists)
  await pick("plan-scenario", "Worst-case Grid");
  await tid("plan-scenario-config").click();
  await tid("scenario-dialog").waitFor();
  await tid("scenario-dialog-form-setup").locator(".cf-toggle").click();
  await tid("scenario-dialog-form-range").locator(".cf-toggle").click();
  await tid("scenario-dialog-form-setup.sd").fill("3 3");
  const scenStatus = await tid("scenario-dialog-status").textContent();
  expect(/List should have 3 items/.test(scenStatus), `number list length validated: ${scenStatus}`);
  await tid("scenario-dialog-form-setup.sd").fill("3 3 4");
  await fillSpin("scenario-dialog-form-range.rel_sd", "2.5");
  await shot("scenario-dialog", tid("scenario-dialog"));
  await tid("scenario-dialog-ok").click();

  // ---- optimization dialog: problem form, solver combo (IPOPT unavailable), solver form
  await tid("plan-problem-config").click();
  await tid("optimization-dialog").waitFor();
  await tid("solver-form").waitFor();
  await shot("optimization-dialog-scipy", tid("optimization-dialog"));
  await tid("solver-combo").click();
  await tid("solver-combo-popup").waitFor();
  const ipopt = tid("solver-combo-popup").locator(".combo-item", { hasText: "Interior Point Optimizer" });
  expect((await ipopt.getAttribute("aria-disabled")) === "true", "IPOPT listed disabled");
  expect(/ipyopt is not installed/.test((await ipopt.getAttribute("title")) ?? ""), "IPOPT entry explains why");
  await shot("solver-popup", tid("optimization-dialog"));
  await page.keyboard.press("Escape");
  await fillSpin("solver-form-max_iter", "120");
  await shot("optimization-dialog-scipy-edited", tid("optimization-dialog"));
  await tid("optimization-dialog-ok").click();
  expect((await tid("plan-problem-config").getAttribute("class")).includes("modified"), "problem … framed as modified");

  // ---- server-side validation on Apply
  await fillSpin("plan-bixel", "250");
  await tid("plan-apply").click();
  await page.waitForFunction(() => /bixel_width/.test(document.querySelector('[data-testid="plan-note"]')?.textContent ?? ""));
  expect((await tid("plan-note").getAttribute("class")).includes("error"), "invalid value reported in red");
  await shot("apply-error", tid("panel-plan"));
  await fillSpin("plan-bixel", "5");
  await tid("plan-apply").click();
  await page.waitForFunction(() => document.querySelector('[data-testid="plan-note"]')?.textContent === "Plan applied.");
  expect((await tid("plan-apply").textContent()) === "Apply", "applied form is clean");

  // ---- carbon: bio models re-listed, tissue marker
  await pick("plan-mode", "carbon");
  await page.waitForFunction(() => /LEM/.test(document.querySelector('[data-testid="plan-bio"]')?.textContent ?? ""));
  expect((await tid("plan-tissue-marker").count()) === 0, "no tissue marker: the recorded phantom has no issues with the bundled base data");
  await tid("plan-bio").click();
  await tid("plan-bio-popup").waitFor();
  await shot("bio-popup", tid("panel-plan"));
  await page.keyboard.press("Escape");
  await tid("plan-bio-config").click();
  await tid("bio-dialog").waitFor();
  await shot("bio-dialog", tid("bio-dialog"));
  await tid("bio-dialog-cancel").click();
  await tid("plan-apply").click();
  await page.waitForFunction(() => document.querySelector('[data-testid="plan-note"]')?.textContent === "Plan applied.");

  // ---- tissue dialog: edit a value and apply (the base data checks are the app layer's, not mocked)
  await tid("plan-tissue").click();
  await tid("tissue-dialog").waitFor();
  await tid("tissue-name-Core").waitFor();
  await shot("tissue-dialog", tid("tissue-dialog"));
  await fillSpin("tissue-Core-alpha_x", "0.12");
  await tid("tissue-ok").click();
  await tid("tissue-dialog").waitFor({ state: "detached" });
  await page.waitForFunction(() => window.__pyrpApp.useApp.getState().structureDetails[0]?.tissue.values.alpha_x === 0.12);
  expect(true, "tissue values applied by structure position");

  // ---- objectives
  await shot("objectives", tid("objectives"));
  const before = Number((await tid("obj-count").textContent()).split(" ")[0]);
  await pick("obj-voi", "Core (OAR)");
  await tid("obj-add").click();
  await page.waitForFunction((n) => Number(document.querySelector('[data-testid="obj-count"]')?.textContent?.split(" ")[0]) === n + 1, before);
  expect(true, "objective added");
  const newKey = "Core-1";
  await pick(`obj-${newKey}-type`, "Min DVH");
  await tid(`obj-Core-1-v_min`).waitFor();
  await fillSpin(`obj-Core-1-v_min`, "150");
  expect((await tid("obj-status").textContent()).includes("Ignored invalid value"), "out-of-range parameter rejected client-side");
  await fillSpin(`obj-Core-1-v_min`, "80");
  await fillSpin(`obj-${newKey}-penalty`, "500");
  await pick(`obj-${newKey}-robustness`, "Objective-wise worst case");
  await page.waitForFunction(() => document.querySelector('[data-testid="obj-Core-1-robustness"]')?.getAttribute("data-value") === "objective_worst_case");
  await pick(`obj-${newKey}-quantity`, "RBExDose");
  await page.waitForFunction(() => document.querySelector('[data-testid="obj-Core-1-quantity"]')?.getAttribute("data-value") === "rbe_x_dose");
  await shot("objectives-edited", tid("objectives"));
  await tid("obj-all-quantities").click();
  await tid("obj-all-quantities-popup").waitFor();
  await shot("objectives-all-quantities", tid("objectives"));
  await tid("obj-all-quantities-popup").locator(".combo-item", { hasText: "dose" }).first().click();
  await page.waitForFunction(() => [...document.querySelectorAll('[data-testid^="obj-"][data-testid$="-quantity"]')].every((e) => e.getAttribute("data-value") === "physical_dose"));
  expect(true, "change all quantities");
  await tid(`obj-${newKey}-remove`).click();
  await page.waitForFunction((n) => Number(document.querySelector('[data-testid="obj-count"]')?.textContent?.split(" ")[0]) === n, before);
  expect(true, "objective removed");

  // ---- optimize: status window
  await tid("btn-dij").click();
  await waitIdle();
  await tid("btn-optimize").click();
  await tid("opt-status-window").waitFor();
  await page.waitForFunction(() => /iter \d+/.test(document.querySelector('[data-testid="opt-summary"]')?.textContent ?? ""));
  await page.waitForTimeout(400);
  expect((await tid("opt-pause").isDisabled()) === true, "Pause disabled in the browser");
  await shot("status-window-running");
  await waitIdle();
  await page.waitForFunction(() => document.querySelector('[data-testid="opt-status-window"]')?.getAttribute("data-running") === "false");
  expect((await tid("opt-stop").isDisabled()) === true, "finalized: Stop disabled, curve kept");
  await shot("status-window-finished", tid("opt-status-window"));
  await tid("opt-close").click();

  // ---- history after reload is restored (mock: same page session only, so check the restore call)
  const summary = await page.evaluate(async () => {
    const mod = await import("/src/store/planSlice.ts");
    mod.usePlanStore.setState({ status: { ...mod.usePlanStore.getState().status, points: [], last: null } });
    await mod.restoreOptimizationHistory();
    return mod.usePlanStore.getState().status.points.length;
  });
  expect(summary > 5, `optimization history restored (${summary} points)`);

  // ---- educational mode locks the advanced plan fields
  await page.evaluate(async () => {
    const mod = await import("/src/store/uiMode.ts");
    mod.setUiMode("educational");
  });
  expect(await tid("plan-engine").isDisabled(), "educational: engine locked");
  expect(!(await tid("plan-gantry").isDisabled()), "educational: gantry editable");
  await shot("educational", tid("panel-plan"));
  await page.evaluate(async () => (await import("/src/store/uiMode.ts")).setUiMode("default"));

  // ---- Discard
  await tid("plan-gantry").fill("0 90");
  await tid("plan-discard").click();
  expect((await note()) === "Changes discarded.", "discard resets the form");

  // ---- sequencing: plan row (photons only, "…" form, modified frames) and the workflow button
  const sequenceTip = async () => (await tid("btn-sequence").getAttribute("title")) ?? "";
  expect(await tid("plan-sequencing").isDisabled(), "carbon: Run sequencing disabled");
  expect((await tid("btn-sequence").isDisabled()) && /only available for photon plans/.test(await sequenceTip()), "carbon: Run Sequencing disabled with the reason as tooltip");
  await pick("plan-mode", "photons");
  await page.waitForFunction(() => document.querySelector('[data-testid="plan-sequencing"]')?.disabled === false);
  expect(await tid("plan-sequencer").isDisabled(), "sequencer combo disabled until Run sequencing is ticked");
  await tid("plan-apply").click();
  await page.waitForFunction(() => document.querySelector('[data-testid="plan-note"]')?.textContent === "Plan applied.");
  await tid("btn-dij").click();
  await waitIdle();
  await tid("btn-optimize").click();
  await waitIdle();
  await page.waitForFunction(() => document.querySelector('[data-testid="dot-result"]')?.getAttribute("data-status") === "done");
  await tid("opt-close").click();
  expect((await tid("btn-sequence").isDisabled()) && /Enable 'Run sequencing' in the Plan panel first/.test(await sequenceTip()), "Run Sequencing asks for the plan setting first");

  await tid("plan-sequencing").check();
  expect(((await tid("plan-sequencing").locator("xpath=..").getAttribute("class")) ?? "").includes("modified"), "Run sequencing framed as modified");
  const firstSequencer = await tid("plan-sequencer").getAttribute("data-value");
  expect(firstSequencer === "engel", `first sequencer of the mode selected (${firstSequencer})`);
  await pick("plan-sequencer", "xia_leaf_sequencing");
  await tid("plan-sequencer-config").click();
  await tid("sequencer-dialog").waitFor();
  await fillSpin("sequencer-dialog-form-num_of_levels", "0");
  expect(/num_of_levels/.test((await tid("sequencer-dialog-status").textContent()) ?? ""), "sequencer form validates the number of levels");
  await fillSpin("sequencer-dialog-form-num_of_levels", "7");
  await shot("sequencer-dialog", tid("sequencer-dialog"));
  await tid("sequencer-dialog-ok").click();
  expect((await tid("plan-sequencer-config").getAttribute("class")).includes("modified"), "sequencer … framed as modified");
  await shot("sequencing-row", tid("panel-plan"));
  await page.evaluate(async () => (await import("/src/store/uiMode.ts")).setUiMode("educational"));
  expect(await tid("plan-sequencing").isDisabled(), "educational: sequencing locked");
  await page.evaluate(async () => (await import("/src/store/uiMode.ts")).setUiMode("default"));
  await tid("plan-apply").click();
  await page.waitForFunction(() => document.querySelector('[data-testid="btn-sequence"]')?.disabled === false);
  const sequencingDetail = await page.evaluate(async () => (await import("/src/store/planSlice.ts")).usePlanStore.getState().detail?.sequencing);
  expect(JSON.stringify(sequencingDetail) === JSON.stringify({ run: true, sequencer: "xia", values: { num_of_levels: 7 } }), "sequencing settings applied to the plan");
  expect((await tid("dot-dij").getAttribute("data-status")) === "done" && (await tid("dot-result").getAttribute("data-status")) === "done", "sequencing settings outdate neither dose influence nor result");
  expect((await tid("plan-apply").textContent()) === "Apply", "applied form is clean");

  // Unapplied sequencing edits are offered before the run, as before the other workflow steps.
  await pick("plan-sequencer", "siochi_leaf_sequencing");
  await tid("btn-sequence").click();
  await tid("pending-changes-dialog").waitFor();
  expect(!((await tid("pending-changes-dialog").textContent()) ?? "").includes("outdates the dose influence"), "Run Sequencing offers unapplied plan edits without the recalculation note");
  await tid("msg-apply").click();
  await page.waitForFunction(() => window.__pyrpApp.useApp.getState().state?.results?.groups.some((g) => g.base === "physical_dose_sequenced"));
  await waitIdle();
  expect((await tid("opt-status-window").count()) === 0, "sequencing does not open the optimization status window");

  // ---- sequence window (Qt "pyRadPlan Sequence Viewer"): opened by the run, one tab per beam
  const sequenceLoaded = () => page.waitForFunction(() => Number(document.querySelector('[data-testid="sequence-window"]')?.dataset.beams) > 0);
  await tid("sequence-window").waitFor();
  await sequenceLoaded();
  /** The window is not modal: dragged aside by its title bar, the panels below stay usable. */
  const dragSequenceWindow = async () => {
    const before = await tid("sequence-window").boundingBox();
    await page.mouse.move(before.x + 200, before.y + 12);
    await page.mouse.down();
    await page.mouse.move(before.x + 580, before.y + 62, { steps: 4 });
    await page.mouse.up();
    const after = await tid("sequence-window").boundingBox();
    expect(after.x === before.x + 380 && after.y === before.y + 50, "the sequence window is dragged by its title bar");
  };
  const gantry = await page.evaluate(() => window.__pyrpApp.useApp.getState().state.plan.gantry_angles);
  const tabs = await tid("sequence-window").locator(".sequence-tabs .tab").allTextContents();
  expect(tabs.length === gantry.length && tabs[0] === `Beam 0 (${gantry[0]}°)`, `Run Sequencing opens the sequence window with a tab per beam (${tabs.join(", ")})`);
  expect(/^Sequencer: siochi \(\d+ levels\)$/.test((await tid("sequence-sequencer").textContent()) ?? ""), "sequencer and levels in the header");
  const viewOptions = async () => [
    await tid("sequence-coordinate").inputValue(),
    await tid("sequence-fluence").inputValue(),
    await tid("sequence-mlcBorders").isChecked(),
    await tid("sequence-target").isChecked(),
    await tid("sequence-bixelGrid").isChecked(),
  ];
  expect((await viewOptions()).join() === "leaf_num,delivered,true,true,false", "view options start at the Qt defaults");
  const shapes = Number(await tid("sequence-beam-0").getAttribute("data-shapes"));
  const shapeLabel = async (beam = 0) => (await tid(`sequence-beam-${beam}-label`).textContent()) ?? "";
  expect(shapes > 1 && (await shapeLabel()) === `1 / ${shapes}` && (await tid("sequence-beam-0").getAttribute("data-fluence")) === "true", `beam 0 shows shape 1 of ${shapes} next to its fluence`);
  await shot("sequence-window", tid("sequence-window"));
  await tid("sequence-beam-0-slider").fill("2");
  expect((await shapeLabel()) === `3 / ${shapes}` && (await tid("sequence-beam-0-canvas").getAttribute("data-shape")) === "2", "the slider selects the shape");
  await tid("sequence-coordinate").selectOption("physical");
  await tid("sequence-fluence").selectOption("difference");
  await tid("sequence-bixelGrid").check();
  await shot("sequence-window-difference", tid("sequence-window"));

  // Playback: dwell by weight, stops on the last shape without Loop, wraps with it.
  await fillSpin("sequence-beam-0-speed", "10");
  await tid("sequence-beam-0-loop").uncheck();
  await tid("sequence-beam-0-play").click();
  expect((await tid("sequence-beam-0-play").textContent()) === "Pause", "Play turns into Pause");
  await page.waitForFunction(() => document.querySelector('[data-testid="sequence-beam-0-play"]')?.textContent === "Play");
  expect((await shapeLabel()) === `${shapes} / ${shapes}`, "playback without Loop stops on the last shape");
  await tid("sequence-beam-0-loop").check();
  await tid("sequence-beam-0-play").click();
  await page.waitForFunction(() => document.querySelector('[data-testid="sequence-beam-0-label"]')?.textContent?.startsWith("1 /"));
  await tid("sequence-beam-0-play").click();
  expect((await tid("sequence-beam-0-play").textContent()) === "Play", "Loop wraps to the first shape; Pause stops");

  // Every beam keeps its own player; Show Sequence selects the beam of the shown per-beam quantity.
  await tid("sequence-tab-1").click();
  expect((await tid("sequence-beam-1").isVisible()) && !(await tid("sequence-beam-0").isVisible()) && (await shapeLabel(1)).startsWith("1 /"), "the second beam's tab has its own player");
  await tid("sequence-close").click();
  expect((await tid("sequence-window").count()) === 0 && (await tid("btn-sequence-view").isEnabled()), "the window closes; Show Sequence stays enabled");
  await page.evaluate(async () => (await import("/src/store/revisions.ts")).selectQuantity("physical_dose_beam 2"));
  await tid("btn-sequence-view").click();
  await sequenceLoaded();
  expect(((await tid("sequence-tab-2").getAttribute("class")) ?? "").includes("active"), "Show Sequence opens on the beam of the shown per-beam quantity");
  expect((await viewOptions()).join() === "leaf_num,delivered,true,true,false", "a reopened window starts at the defaults");
  await page.evaluate(async () => (await import("/src/store/revisions.ts")).selectQuantity("physical_dose"));

  // A new sequencing run refreshes the open window and resets its view options.
  await tid("sequence-fluence").selectOption("optimized");
  await tid("sequence-beam-2-slider").fill("1");
  await dragSequenceWindow();
  await tid("btn-sequence").click();
  await page.waitForFunction(() => document.querySelector('[data-testid="sequence-fluence"]')?.value === "delivered");
  await waitIdle();
  expect(((await tid("sequence-tab-0").getAttribute("class")) ?? "").includes("active") && (await shapeLabel()).startsWith("1 /"), "a new sequence starts on its first beam and shape");
  await tid("sequence-close").click();
  const sequencedGroups = await tid("quantity-select-group").locator("option").allTextContents();
  expect(sequencedGroups.join() === "dose,dose (sequenced)", `sequenced dose listed in the quantity selector (${sequencedGroups.join(", ")})`);
  const sequencedPlan = await page.evaluate(async () => (await import("/src/store/planSlice.ts")).usePlanStore.getState().detail?.sequencing?.sequencer);
  expect(sequencedPlan === "siochi", "the run used the sequencer applied from the dialog");
  await tid("quantity-select-group").selectOption({ label: "dose (sequenced)" });
  await page.waitForFunction(() => window.__pyrpApp.useApp.getState().viewer.activeQuantity === "physical_dose_sequenced");
  await shot("sequenced");
  expect((await tid("dot-result").getAttribute("data-status")) === "done", "result stays up to date after sequencing");

  // The window belongs to the sequence: a new optimization (a result without one) closes it.
  await tid("btn-sequence-view").click();
  await sequenceLoaded();
  await page.evaluate(async () => (await import("/src/theme.ts")).setDarkMode(true));
  await shot("sequence-window-dark", tid("sequence-window"));
  await page.evaluate(async () => (await import("/src/theme.ts")).setDarkMode(false));
  await dragSequenceWindow();
  await tid("btn-optimize").click();
  await waitIdle();
  await page.waitForFunction(() => document.querySelector('[data-testid="dot-result"]')?.getAttribute("data-status") === "done");
  await tid("opt-close").click();
  expect((await tid("sequence-window").count()) === 0 && (await tid("btn-sequence-view").isDisabled()), "a result without a sequence closes the window and disables Show Sequence");

  if (problems.length) throw new Error(`browser reported problems:\n${problems.join("\n")}`);
  console.log("plan e2e passed");
} catch (err) {
  await page.screenshot({ path: join(shots, "plan-failure.png") }).catch(() => undefined);
  console.error(err);
  if (problems.length) console.error(problems.join("\n"));
  process.exitCode = 1;
} finally {
  await browser.close();
  await server.close();
}
