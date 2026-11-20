// End-to-end check of the web UI against the mock backend: starts the Vite dev server,
// drives the workflow in headless Edge (Playwright, SwiftShader WebGL) and saves
// screenshots to e2e/screenshots/. Usage: npm run e2e [-- --headed] [-- --chromium]
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

const server = await createServer({ root, server: { port: 5199, strictPort: false }, logLevel: "warn" });
await server.listen();
const base = server.resolvedUrls.local[0];
console.log(`dev server: ${base}`);

const browser = await chromium.launch({
  ...(useBundledChromium ? {} : { channel: "msedge" }),
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

let step = 0;
async function shot(name) {
  step += 1;
  const file = join(shots, `${String(step).padStart(2, "0")}-${name}.png`);
  await page.waitForTimeout(150);
  await page.screenshot({ path: file });
  console.log(`screenshot ${file}`);
}

async function waitIdle() {
  await page.waitForFunction(() => {
    const btn = document.querySelector('[data-testid="btn-stop"]');
    return btn && btn.disabled;
  });
}

const tid = (id) => page.getByTestId(id);
// Package A's optimization status window opens with every optimization and covers the viewer.
const closeStatus = async () => {
  if (await tid("opt-close").count()) await tid("opt-close").click();
};

// Phantoms are opened from File > Open Example Phantom, as in the Qt GUI.
async function openPhantom(name) {
  await tid("menu-file").click();
  await tid("menu-item-phantoms").click();
  await tid(`menu-item-phantom:${name}`).click();
}

// Dose influence, then optimization (the Qt workflow buttons).
async function computeResult() {
  await tid("btn-dij").click();
  await waitIdle();
  await tid("btn-optimize").click();
  await waitIdle();
}

function expect(cond, message) {
  if (!cond) throw new Error(`check failed: ${message}`);
  console.log(`ok: ${message}`);
}

try {
  await page.goto(`${base}?backend=mock&speed=0.5`);
  await tid("boot-splash").waitFor({ state: "detached", timeout: 15000 });
  expect(await page.evaluate(() => globalThis.crossOriginIsolated), "page is cross-origin isolated (COOP/COEP)");
  await page.waitForFunction(() => window.__pyrpApp?.useApp.getState().session);
  await shot("start");
  expect(!(await tid("btn-run-all").count()) && !(await tid("phantom-select").count()), "workflow widget has the Qt controls only (no Run All, no phantom selector)");

  // Load the TG119 phantom.
  await openPhantom("TG119");
  await page.waitForFunction(() => document.querySelector('[data-testid="dot-load"]')?.getAttribute("data-status") === "done");
  await waitIdle();
  await page.waitForFunction(() => document.querySelector('[data-testid="slice-label"]')?.textContent?.includes("/ 128"));
  const info = await page.evaluate(() => window.__pyrpRenderer?.info);
  console.log("renderer info:", JSON.stringify(info));
  expect(info && !info.ctSliceMode, "CT resident as a 3-D texture");

  // Hover the centre of the image for the readout + crosshair.
  const box = await tid("viewer-canvas").boundingBox();
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.58);
  await tid("viewer-readout").waitFor();
  const readout = await tid("viewer-readout").textContent();
  expect(readout.includes("HU") && readout.includes("Pos:"), `hover readout: ${readout.replace(/\n/g, " | ")}`);
  await shot("tg119-loaded-hover");

  // A 5-beam plan through the plan form (the default plan has one beam).
  await tid("plan-gantry").fill("0 72 144 216 288");
  expect((await tid("plan-note").textContent()).includes("Modified"), "plan form flags unapplied edits");
  await tid("plan-apply").click();
  await page.waitForFunction(() => !/Modified/.test(document.querySelector('[data-testid="plan-note"]')?.textContent ?? ''));

  // Dose influence with progress.
  await tid("btn-dij").click();
  await tid("workflow-progress").waitFor();
  await page.waitForFunction(() => /beam/.test(document.querySelector('[data-testid="workflow-status"]')?.textContent ?? ""));
  await shot("dij-progress");
  await waitIdle();
  expect(
    (await tid("dot-dij").getAttribute("data-status")) === "done",
    "dose influence stage done",
  );

  // Optimization -> dose overlay.
  await tid("btn-optimize").click();
  await waitIdle();
  await closeStatus();
  expect((await tid("dot-result").getAttribute("data-status")) === "done", "result stage done");
  await tid("colorbar").waitFor();
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.62);
  await tid("viewer-readout").waitFor();
  const doseReadout = await tid("viewer-readout").textContent();
  expect(/dose: [\d.]+ Gy/.test(doseReadout), `dose readout: ${doseReadout.replace(/\n/g, " | ")}`);
  await shot("dose-axial");
  await tid("btn-recalc").click();
  await waitIdle();
  expect((await tid("dot-result").getAttribute("data-status")) === "done", "recalculate dose keeps the result fresh");

  // Isolines + isocenter, sagittal and coronal planes.
  await tid("toggle-isolines").check();
  await tid("toggle-isocenter").check();
  await page.mouse.move(box.x + 5, box.y + 5);
  await page.mouse.move(box.x - 50, box.y - 50);
  await shot("dose-axial-isolines");
  await tid("plane-select").selectOption("sagittal");
  await shot("dose-sagittal");
  await tid("plane-select").selectOption("coronal");
  await shot("dose-coronal");
  await tid("plane-select").selectOption("axial");

  // Wheel scrolls slices; ctrl+wheel zooms; drag pans.
  const before = await tid("slice-spin").inputValue();
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
  for (let i = 0; i < 5; i++) await page.mouse.wheel(0, 100);
  const after = await tid("slice-spin").inputValue();
  expect(before !== after, `wheel scrolls slices (${before} -> ${after})`);
  for (let i = 0; i < 5; i++) await page.mouse.wheel(0, -100);
  await page.keyboard.down("Control");
  for (let i = 0; i < 4; i++) await page.mouse.wheel(0, -150);
  await page.keyboard.up("Control");
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.5 + 60, box.y + box.height * 0.5 + 30, { steps: 5 });
  await page.mouse.up();
  await tid("mode-toggle").click();
  await tid("ct-preset").selectOption("Lung");
  await page.getByTestId("opacity-slider").fill("70");
  await page.mouse.move(box.x - 50, box.y - 50);
  await shot("zoom-pan-lung-window");
  await page.locator(".viewer-canvas").dblclick();
  await tid("mode-toggle").click();

  // Plan change -> downstream stages stale.
  await tid("plan-gantry").fill("0 60 120 180 240 300");
  await tid("plan-apply").click();
  await page.waitForFunction(() => document.querySelector('[data-testid="dot-result"]')?.getAttribute("data-status") === "stale");
  expect((await tid("dot-dij").getAttribute("data-status")) === "stale", "plan change marks the dose influence stale");
  await shot("plan-changed-stale");

  // Dose influence + optimization -> fresh result with 6 beams.
  await computeResult();
  await closeStatus();
  expect((await tid("dot-result").getAttribute("data-status")) === "done", "dose influence + optimize refresh the result");
  await page.mouse.move(box.x - 50, box.y - 50);
  await shot("result-6-beams");

  // DVH / QI analysis window.
  await tid("btn-dvh").click();
  await tid("analysis-window").locator("canvas").waitFor({ state: "visible" });
  await page.waitForFunction(() => (document.querySelectorAll('[data-testid="qi-table"] tbody tr').length ?? 0) > 0);
  await shot("dvh");
  await tid("analysis-close").click();

  // Stop cancels a running computation (mock: at the next simulated step).
  await tid("plan-gantry").fill("0 90 180 270");
  await tid("plan-apply").click();
  await page.waitForFunction(() => !/Modified/.test(document.querySelector('[data-testid="plan-note"]')?.textContent ?? ''));
  await tid("btn-dij").click();
  await tid("workflow-progress").waitFor();
  await tid("btn-stop").click();
  await page.waitForFunction(() => /stopped/.test(document.querySelector('[data-testid="statusbar"]')?.textContent ?? ""));
  await waitIdle();
  expect((await tid("error-dialog").count()) === 0, "stop ends the computation without an error dialog");

  // Errors surface in a dialog and the log.
  await tid("file-input").setInputFiles({ name: "notes.txt", mimeType: "text/plain", buffer: Buffer.from("hello") });
  await tid("error-dialog").waitFor();
  const logText = await tid("log").textContent();
  expect(logText.includes("Unsupported file type"), "error is logged");
  await shot("error-dialog");
  await page.keyboard.press("Escape");

  // Slice-mode fallback renders the same image (texture-size fallback path).
  await page.goto(`${base}?backend=mock&speed=0&sliceMode`);
  await tid("boot-splash").waitFor({ state: "detached", timeout: 15000 });
  await page.waitForFunction(() => window.__pyrpApp?.useApp.getState().session);
  await openPhantom("TG119");
  await page.waitForFunction(() => document.querySelector('[data-testid="dot-load"]')?.getAttribute("data-status") === "done");
  await waitIdle();
  await computeResult();
  await closeStatus();
  const info2 = await page.evaluate(() => window.__pyrpRenderer?.info);
  expect(info2?.ctSliceMode && info2?.quantitySliceMode, "slice-mode fallback active with ?sliceMode");
  await page.mouse.move(0, 0);
  await shot("slice-mode-fallback");

  if (problems.length) throw new Error(`browser reported problems:\n${problems.join("\n")}`);
  console.log("e2e passed");
} catch (err) {
  await page.screenshot({ path: join(shots, "failure.png") }).catch(() => {});
  console.error(err);
  if (problems.length) console.error(problems.join("\n"));
  process.exitCode = 1;
} finally {
  await browser.close();
  await server.close();
}
