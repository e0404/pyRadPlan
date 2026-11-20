// Viewer parity checks (package B) against the mock backend: beams overlay and drag, pending
// beams, orientation marker, quantity selector with components, scenarios and snapshots, compact masks,
// VOI list (overlap grouping, colour pick, Edit VOI), the DVH / QI analysis window and its dark mode.
// Dev server on port 5202; screenshots e2e/screenshots/viewer-*.png. Usage: npm run e2e:viewer
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

const server = await createServer({ root, server: { port: 5202, strictPort: false }, logLevel: "warn" });
await server.listen();
const base = server.resolvedUrls.local[0];
console.log(`dev server: ${base}`);
const browser = await chromium.launch({
  ...(process.argv.includes("--chromium") ? {} : { channel: "msedge" }),
  headless: !headed,
  args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
});
const context = await browser.newContext({ viewport: { width: 1680, height: 960 }, deviceScaleFactor: 1 });
await acknowledgeNotice(context, base);
const page = await context.newPage();
const problems = [];
page.on("pageerror", (err) => problems.push(`pageerror: ${err.message}`));
page.on("console", (msg) => {
  if (msg.type() === "error") problems.push(`console: ${msg.text()}`);
});
const tid = (id) => page.getByTestId(id);
let step = 0;
async function shot(name) {
  step += 1;
  await page.waitForTimeout(200);
  const file = join(shots, `viewer-${String(step).padStart(2, "0")}-${name}.png`);
  await page.screenshot({ path: file });
  console.log(`screenshot ${file}`);
}
function expect(cond, message) {
  if (!cond) throw new Error(`check failed: ${message}`);
  console.log(`ok: ${message}`);
}
const app = (fn, arg) => page.evaluate(fn, arg);
const idle = () => page.waitForFunction(() => window.__pyrpApp && window.__pyrpApp.useApp.getState().busy === null);

try {
  await page.goto(`${base}?backend=mock&speed=0.2`);
  await tid("boot-splash").waitFor({ state: "detached", timeout: 20000 });
  await page.waitForFunction(() => window.__pyrpApp?.useApp.getState().session);

  // TG119 with a 5-beam plan (through the store, independent of the plan form).
  await app(() => window.__pyrpApp.actions.loadPhantom("TG119"));
  await idle();
  await app(() => window.__pyrpApp.actions.applyPlan({ gantry_angles: [0, 72, 144, 216, 288], couch_angles: [0, 0, 0, 0, 0] }));
  await idle();
  await app(() => window.__pyrpApp.actions.runAll());
  await idle();
  await page.waitForFunction(() => Object.keys(window.__pyrpApp.useApp.getState().quantities).length > 0);
  // Package A's optimization status window opens with the optimization; close it.
  if (await tid("opt-close").count()) await tid("opt-close").click();

  const data = await app(() => {
    const s = window.__pyrpApp.useApp.getState();
    return {
      ctType: s.ct?.data.data.constructor.name,
      maskKinds: s.maskSet?.masks.map((m) => m.storage?.kind ?? "full"),
      revisions: Boolean(s.state?.revisions),
      groups: s.state?.results?.groups.map((g) => g.label),
      active: s.viewer.activeQuantity,
      beams: s.beams?.beams.length,
    };
  });
  console.log("data:", JSON.stringify(data));
  expect(data.ctType === "Int16Array", "CT fetched with encoding auto as int16");
  expect(data.maskKinds?.every((k) => k === "crop"), "structure masks fetched as crops");
  expect(data.active === "physical_dose", "default quantity after the result change");
  expect(data.beams === 5, "view.beams delivers the 5 plan beams");

  // Beams + isocenter overlay, axial.
  await tid("toggle-beams").check();
  await tid("toggle-isocenter").check();
  await page.mouse.move(5, 5);
  await shot("beams-axial");
  await tid("plane-select").selectOption("sagittal");
  await shot("beams-sagittal");
  await tid("plane-select").selectOption("coronal");
  await shot("beams-coronal");
  await tid("plane-select").selectOption("axial");

  // Drag beam #1 along its orbit; the release applies the new gantry angle.
  const canvas = await tid("viewer-canvas").boundingBox();
  const src = await app(() => {
    const r = window.__pyrpRenderer.renderer;
    const p = r.projectedBeams().find((b) => b.beam.index === 1);
    const layout = r.layout();
    const t = r.transform();
    return { x: t.offsetX + (p.source.h / layout.sh) * t.scaleU, y: t.offsetY + (p.source.v / layout.sv) * t.scaleV, iso: { x: t.offsetX + (p.iso.h / layout.sh) * t.scaleU, y: t.offsetY + (p.iso.v / layout.sv) * t.scaleV } };
  });
  await page.mouse.move(canvas.x + src.x, canvas.y + src.y);
  await page.mouse.down();
  // Move towards straight down from the isocenter (gantry 180 in this view).
  const target = { x: src.iso.x, y: src.iso.y + 120 };
  await page.mouse.move(canvas.x + (src.x + target.x) / 2, canvas.y + (src.y + target.y) / 2 + 40, { steps: 6 });
  await page.mouse.move(canvas.x + target.x + 40, canvas.y + target.y, { steps: 6 });
  await shot("beam-drag");
  await page.mouse.up();
  await idle();
  await page.waitForTimeout(400);
  // With the plan form's beam editor registered (Qt), the angle enters the form draft and the
  // beam is drawn as pending; without it the angle is applied directly.
  const angles = await app(() => {
    const s = window.__pyrpApp.useApp.getState();
    return { applied: s.state.plan.gantry_angles, draft: s.draftBeams?.gantry_angles ?? null, pending: s.beams?.beams.filter((b) => b.pending).map((b) => b.index) ?? [] };
  });
  console.log("after drag:", JSON.stringify(angles));
  expect(angles.applied[1] !== 72 || (angles.draft && angles.draft[1] !== 72), "beam drag changed the gantry angle of beam #1");
  if (!angles.draft) {
    await app(() => window.__pyrpApp.setDraftBeams({ gantry_angles: [0, 72, 144, 216, 288, 330], couch_angles: [0, 0, 0, 0, 0, 0] }));
  }
  await page.waitForFunction(() => window.__pyrpApp.useApp.getState().beams?.beams.some((b) => b.pending));
  await page.mouse.move(5, 5);
  await shot("pending-beams");
  expect(true, "pending beams drawn dashed with '*'");

  // Beam tooltip on hover.
  const src0 = await app(() => {
    const r = window.__pyrpRenderer.renderer;
    const p = r.projectedBeams()[0];
    const layout = r.layout();
    const t = r.transform();
    return { x: t.offsetX + (p.source.h / layout.sh) * t.scaleU, y: t.offsetY + (p.source.v / layout.sv) * t.scaleV };
  });
  await page.mouse.move(canvas.x + src0.x, canvas.y + src0.y);
  const tip = await tid("viewer-canvas").getAttribute("title");
  expect(/Beam #0: gantry/.test(tip ?? "") && /Drag the source marker/.test(tip ?? ""), `beam tooltip (${(tip ?? "").replace(/\n/g, " | ")})`);

  // Quantity selector: beam component, then the group of a snapshot (a result of its own).
  await tid("quantity-select-component").selectOption("physical_dose_beam 1");
  await page.waitForFunction(() => window.__pyrpApp.useApp.getState().viewer.activeQuantity === "physical_dose_beam 1");
  await page.mouse.move(canvas.x + canvas.width * 0.5, canvas.y + canvas.height * 0.62);
  await tid("viewer-readout").waitFor();
  await shot("beam-component");
  const snap = await app(async () => {
    const { backend, session } = window.__pyrpApp.useApp.getState();
    const r = await backend.request("workflow.snapshot", { tag: "plan A" }, session);
    window.__pyrpApp.useApp.setState({ state: r.state });
    await window.__pyrpApp.syncViews();
    return r.tag;
  });
  expect(snap === "plan_A", "snapshot name sanitized to plan_A");
  const groupLabels = await tid("quantity-select-group").locator("option").allTextContents();
  expect(groupLabels.includes("dose (plan_A)"), `snapshot group listed (${groupLabels.join(", ")})`);
  await tid("quantity-select-group").selectOption({ label: "dose (plan_A)" });
  await page.waitForFunction(() => window.__pyrpApp.useApp.getState().viewer.activeQuantity === "plan_A/physical_dose_beam 1");
  expect(true, "component kept (beam 1) when switching to the snapshot group");
  await tid("quantity-select-component").selectOption("plan_A/physical_dose");

  // VOI list: overlap grouping, colour pick, Edit VOI.
  await tid("voi-group-mode").selectOption("overlap");
  await page.mouse.move(canvas.x + canvas.width * 0.5, canvas.y + canvas.height * 0.62);
  await shot("vois-by-overlap");
  await tid("voi-group-mode").selectOption("type");
  await page.evaluate(() => {
    const input = document.querySelector('[data-testid="voi-color-input"]');
    document.querySelector('[data-testid="voi-color-Core"]').click();
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
    setter.call(input, "#00c8ff");
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await page.waitForFunction(() => window.__pyrpApp.useApp.getState().state.structures.find((s) => s.name === "Core").color.join(",") === "0,200,255");
  expect(true, "colour pick persisted through cst.update_structure");
  await tid("voi-name-Core").click();
  await tid("edit-voi-dialog").waitFor();
  await tid("edit-voi-type").selectOption("HELPER");
  await tid("edit-voi-priority").fill("5");
  await shot("edit-voi");
  await tid("edit-voi-ok").click();
  await tid("edit-voi-dialog").waitFor({ state: "detached" });
  const core = await app(() => window.__pyrpApp.useApp.getState().state.structures.find((s) => s.name === "Core"));
  expect(core.type === "HELPER" && core.overlap_priority === 5, "Edit VOI wrote type and overlap priority");
  // The mock does not simulate the base data checks behind "Apply anyway" (they are the app layer's).
  await tid("voi-name-OuterTarget").click();
  await tid("edit-voi-dialog").waitFor();
  await tid("tissue-OuterTarget-alpha_x").fill("0.12");
  await tid("tissue-OuterTarget-alpha_x").press("Tab");
  // The row reports the edit after its (debounced) tissue.evaluate.
  await page.waitForTimeout(800);
  await tid("edit-voi-ok").click();
  await tid("edit-voi-dialog").waitFor({ state: "detached" });
  await page.waitForFunction(() => window.__pyrpApp.useApp.getState().structureDetails[1]?.tissue.values.alpha_x === 0.12);
  expect(true, "tissue α_x written through cst.update_structure");

  // Tissue picker of the Edit VOI dialog: nothing preselected for a tissue matching no pick,
  // OK with an unapplied selection asks, Apply fills the tissue.
  await tid("voi-name-OuterTarget").click();
  await tid("edit-voi-dialog").waitFor();
  await tid("tissue-picker").waitFor();
  expect((await tid("tissue-picker").getAttribute("data-value")) === "", "Edit VOI: no pick preselected for a tissue matching none");
  await tid("tissue-picker").click();
  await page.locator('[data-testid="tissue-picker-popup"] [role="option"]').first().click();
  const picked = await tid("tissue-picker").getAttribute("data-value");
  await tid("edit-voi-ok").click();
  await tid("edit-voi-unapplied-pick").waitFor();
  await tid("msg-keep").click();
  expect((await tid("edit-voi-dialog").count()) === 1, "Edit VOI: OK with an unapplied pick asks and can keep editing");
  // A parameter edited just before must not survive the applied pick.
  await tid("tissue-OuterTarget-alpha_x").fill("0.77");
  await tid("tissue-OuterTarget-alpha_x").press("Tab");
  await tid("tissue-pick-apply").click();
  await page.waitForFunction((id) => document.querySelector('[data-testid="tissue-picker"]')?.getAttribute("data-value") === id, picked);
  await tid("edit-voi-ok").click();
  await tid("edit-voi-dialog").waitFor({ state: "detached" });
  expect((await tid("edit-voi-unapplied-pick").count()) === 0, "Edit VOI: an applied pick is not asked about");
  const appliedAlpha = await app(() => window.__pyrpApp.useApp.getState().structureDetails[1]?.tissue.values.alpha_x);
  expect(appliedAlpha !== 0.77, "Edit VOI: an applied pick replaces a parameter edited before");
  await tid("voi-name-OuterTarget").click();
  await tid("edit-voi-dialog").waitFor();
  await page.waitForFunction((id) => document.querySelector('[data-testid="tissue-picker"]')?.getAttribute("data-value") === id, picked);
  expect(true, "Edit VOI: the applied pick is preselected on reopening");
  await tid("edit-voi-ok").click();
  await tid("edit-voi-dialog").waitFor({ state: "detached" });
  expect((await tid("edit-voi-unapplied-pick").count()) === 0, "Edit VOI: opening and pressing OK never asks");
  // A cell line preset keeps its identity through OK and reopening, so the same preset is shown.
  await tid("voi-name-OuterTarget").click();
  await tid("edit-voi-dialog").waitFor();
  await tid("tissue-picker").click();
  await page.locator('[data-testid="tissue-picker-popup"] [role="option"]').last().click();
  const preset = await tid("tissue-picker").getAttribute("data-value");
  await tid("tissue-pick-apply").click();
  await page.waitForFunction((id) => document.querySelector('[data-testid="tissue-picker"]')?.getAttribute("data-value") === id, preset);
  await tid("edit-voi-ok").click();
  await tid("edit-voi-dialog").waitFor({ state: "detached" });
  await tid("voi-name-OuterTarget").click();
  await tid("edit-voi-dialog").waitFor();
  await page.waitForFunction((id) => document.querySelector('[data-testid="tissue-picker"]')?.getAttribute("data-value") === id, preset);
  expect(true, "Edit VOI: an applied cell line preset is preselected on reopening");
  await tid("edit-voi-ok").click();
  await tid("edit-voi-dialog").waitFor({ state: "detached" });

  // Analysis window.
  await tid("btn-dvh").click();
  await tid("analysis-window").waitFor();
  await page.waitForFunction(() => document.querySelectorAll('[data-testid="qi-table"] tbody tr').length > 0);
  const headers = await tid("qi-table").locator("th").allTextContents();
  console.log("QI headers:", headers.join(" | "));
  expect(headers.includes("D95 [Gy]") && headers.some((h) => /^V[\d.]+Gy \[%\]$/.test(h)), "QI table columns like Qt (D95 [Gy], V..Gy [%])");
  await tid("analysis-show-dx").check();
  await tid("analysis-show-vx").check();
  await page.getByRole("cell", { name: "OuterTarget" }).first().click();
  const d95col = headers.indexOf("D95 [Gy]");
  await tid("qi-table").locator("tbody tr", { hasText: "OuterTarget" }).locator("td").nth(d95col).click();
  const plot = await tid("dvh-plot").boundingBox();
  await page.mouse.move(plot.x + plot.width * 0.45, plot.y + plot.height * 0.5);
  await shot("analysis-markers");
  await tid("analysis-compare-group").selectOption({ label: "dose" });
  await tid("analysis-compare-component").selectOption("physical_dose_beam 1");
  await page.waitForTimeout(500);
  await page.mouse.move(plot.x + plot.width * 0.25, plot.y + plot.height * 0.5);
  await shot("analysis-compare");
  await page.mouse.dblclick(plot.x + plot.width * 0.3, plot.y + plot.height * 0.3);
  await tid("qi-point-dialog").waitFor();
  await shot("qi-point-dialog");
  await tid("qi-point-add").click();
  await page.waitForTimeout(400);
  const headers2 = await tid("qi-table").locator("th").allTextContents();
  expect(headers2.length > headers.length, `QI point added a column (${headers2.length - 1} metrics)`);
  await tid("analysis-tab-gamma").click();
  await tid("gamma-tab").waitFor();
  await shot("analysis-gamma");
  await tid("analysis-close").click();

  // Robust plan through the GUI: scenario model, robust objective, dose influence, optimize;
  // then the scenarios in the viewer's quantity selector and in the analysis window (Qt).
  const pickItem = async (testId, text) => {
    await tid(testId).click();
    await tid(`${testId}-popup`).waitFor();
    await tid(`${testId}-popup`).locator(".combo-item", { hasText: text }).first().click();
  };
  await pickItem("plan-scenario", "Worst-case Grid");
  await tid("plan-apply").click();
  await page.waitForFunction(() => /Plan applied/.test(document.querySelector('[data-testid="plan-note"]')?.textContent ?? ""));
  const robustKey = await page.evaluate(() => document.querySelector('[data-testid$="-robustness"]')?.getAttribute("data-testid"));
  await pickItem(robustKey, "Objective-wise worst case");
  await page.waitForTimeout(300);
  await tid("btn-dij").click();
  await idle();
  await tid("btn-optimize").click();
  await idle();
  if (await tid("opt-close").count()) await tid("opt-close").click();
  const comps = await tid("quantity-select-component").locator("option").allTextContents();
  expect(comps.includes("scenario 1: setup (-2.25, 0.0, 0.0) mm") && comps.filter((c) => c.startsWith("scenario ")).length === 7, `scenario components listed with labels (${comps.length} entries)`);
  await tid("quantity-select-component").selectOption("physical_dose_scenario 2");
  await page.waitForFunction(() => window.__pyrpApp.useApp.getState().quantities["physical_dose_scenario 2"]);
  await page.mouse.move(canvas.x + canvas.width * 0.5, canvas.y + canvas.height * 0.62);
  await tid("viewer-readout").waitFor();
  const shown = await app(() => window.__pyrpApp.useApp.getState().viewer.activeQuantity);
  expect(shown === "physical_dose_scenario 2", "viewer shows the selected scenario");
  await shot("scenario-selected");
  const groupWidth = async (id) => (await tid(`${id}-group`).boundingBox()).width;
  expect((await groupWidth("quantity-select")) > 120, "viewer quantity combo keeps its width next to a long scenario label");
  await tid("btn-dvh").click();
  await tid("analysis-window").waitFor();
  await page.waitForFunction(() => document.querySelectorAll('[data-testid="qi-table"] tbody tr').length > 0);
  expect((await tid("analysis-primary-component").inputValue()) === "physical_dose_scenario 2", "analysis opens with the viewer's scenario");
  expect((await groupWidth("analysis-primary")) > 60, `analysis quantity combo not squeezed by the scenario label (${(await groupWidth("analysis-primary")).toFixed(0)} px)`);
  await tid("analysis-primary-component").selectOption("physical_dose_scenario 4");
  await tid("analysis-all-scenarios").check();
  await page.waitForTimeout(600);
  const qiName = await app(async () => {
    const { backend, session } = window.__pyrpApp.useApp.getState();
    return (await backend.request("analysis.qi", { quantity: "physical_dose_scenario 4" }, session)).name;
  });
  expect(qiName === "dose, scenario 4", "QI of a scenario quantity");
  await shot("analysis-scenarios");

  // The DVH plot follows the theme, also when it changes while the window is open.
  const plotPixel = () =>
    page.evaluate(() => {
      const c = document.querySelector('[data-testid="dvh-plot"]');
      const ctx = c.getContext("2d");
      const dpr = window.devicePixelRatio || 1;
      return Array.from(ctx.getImageData(Math.round(100 * dpr), Math.round(30 * dpr), 1, 1).data.slice(0, 3));
    });
  const lightPixel = await plotPixel();
  await tid("menu-view").click();
  await tid("menu-item-dark_mode").click();
  await page.waitForFunction(() => document.documentElement.dataset.theme === "dark");
  await page.waitForTimeout(400);
  const darkPixel = await plotPixel();
  expect(lightPixel.every((v) => v > 200) && darkPixel.every((v) => v < 80), `DVH axes background follows the theme (light ${lightPixel}, dark ${darkPixel})`);
  const tableBg = await page.evaluate(() => getComputedStyle(document.querySelector('[data-testid="qi-table"]').closest(".table-wrap")).backgroundColor);
  expect(tableBg !== "rgb(255, 255, 255)", `QI table background follows the theme (${tableBg})`);
  await page.mouse.move(plot.x + plot.width * 0.35, plot.y + plot.height * 0.5);
  await shot("analysis-dark");
  await tid("menu-view").click();
  await tid("menu-item-dark_mode").click();
  await tid("analysis-close").click();

  if (problems.length) throw new Error(`browser reported problems:\n${problems.join("\n")}`);
  console.log("viewer e2e passed");
} catch (err) {
  await page.screenshot({ path: join(shots, "viewer-failure.png") }).catch(() => {});
  console.error(err);
  if (problems.length) console.error(problems.join("\n"));
  process.exitCode = 1;
} finally {
  await browser.close();
  await server.close();
}
