// End-to-end check of the package-C GUI (menus, dialogs, file save / load, base data, settings,
// help, log console, dark mode, UI mode, splash) against the mock backend. Starts its own Vite dev
// server on port 5203; screenshots go to e2e/screenshots/menus-*.png.
// Usage: npm run e2e:menus [-- --headed] [-- --chromium]
import { mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { createServer } from "vite";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const shots = join(here, "screenshots");
mkdirSync(shots, { recursive: true });

const headed = process.argv.includes("--headed");
const useBundledChromium = process.argv.includes("--chromium");

const server = await createServer({ root, server: { port: 5203, strictPort: false }, logLevel: "warn" });
await server.listen();
const base = server.resolvedUrls.local[0];
console.log(`dev server: ${base}`);

const browser = await chromium.launch({
  ...(useBundledChromium ? {} : { channel: "msedge" }),
  headless: !headed,
  args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
});
const context = await browser.newContext({ viewport: { width: 1680, height: 960 }, deviceScaleFactor: 1, acceptDownloads: true, colorScheme: "light" });
const page = await context.newPage();
const problems = [];
page.on("pageerror", (err) => problems.push(`pageerror: ${err.message}`));
page.on("console", (msg) => {
  if (msg.type() === "error") problems.push(`console: ${msg.text()}`);
});

let step = 0;
async function shot(name) {
  step += 1;
  const file = join(shots, `menus-${String(step).padStart(2, "0")}-${name}.png`);
  await page.waitForTimeout(150);
  await page.screenshot({ path: file });
  console.log(`screenshot ${file}`);
}

const tid = (id) => page.getByTestId(id);

function expect(cond, message) {
  if (!cond) throw new Error(`check failed: ${message}`);
  console.log(`ok: ${message}`);
}

async function waitIdle() {
  await page.waitForTimeout(80);
  await page.waitForFunction(() => {
    const btn = document.querySelector('[data-testid="btn-stop"]');
    return btn && btn.disabled;
  });
}

/** Package A's optimization status window opens with every optimization; close it. */
async function closeOptimizationStatus() {
  const close = tid("opt-close");
  if (await close.count()) await close.click();
}

async function openMenu(id) {
  await tid(`menu-${id}`).click();
  await tid(`popup-${id}`).waitFor();
}

async function clickItem(menu, item) {
  await openMenu(menu);
  await tid(`menu-item-${item}`).click();
}

const disabled = async (id) => (await tid(id).getAttribute("aria-disabled")) === "true";
const closeTopDialog = async () => {
  await page.keyboard.press("Escape");
  await page.waitForTimeout(80);
};

/** Phantoms are opened from File > Open Example Phantom, as in the Qt GUI. */
async function openPhantom(name) {
  await clickItem("file", "phantoms");
  await tid(`menu-item-phantom:${name}`).click();
}

/** Dose influence, then optimization (the Qt workflow buttons). */
async function computeResult() {
  await tid("btn-dij").click();
  await waitIdle();
  await tid("btn-optimize").click();
  await waitIdle();
}

async function fresh(query = "?backend=mock&speed=0") {
  await page.goto(`${base}${query}`);
  await tid("boot-splash").waitFor({ state: "detached", timeout: 15000 });
  await page.waitForFunction(() => window.__pyrpApp?.useApp.getState().session);
}

/** Click `trigger` and answer the file chooser it opens. */
async function chooseFile(trigger, file) {
  const [chooser] = await Promise.all([page.waitForEvent("filechooser"), trigger()]);
  await chooser.setFiles(file);
}

const bytes = (text) => Buffer.from(text);

/** Click `trigger` and capture the browser download it starts. */
async function download(trigger) {
  const [dl] = await Promise.all([page.waitForEvent("download"), trigger()]);
  const stream = await dl.createReadStream();
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  return { name: dl.suggestedFilename(), data: Buffer.concat(chunks) };
}

try {
  // ------------------------------------------------------------------ first-visit notice (fresh profile)
  await page.goto(`${base}?backend=mock&speed=0`);
  await tid("notice-dialog").waitFor();
  const noticeText = (await tid("notice-dialog").textContent()) ?? "";
  expect(noticeText.includes("NOT FOR CLINICAL USE") && noticeText.includes("not a medical device"), "a fresh profile is shown the disclaimer");
  expect(noticeText.includes("cdn.jsdelivr.net") && noticeText.includes("no telemetry") && noticeText.includes(new URL(base).origin), "the notice states where data and downloads go");
  expect((await tid("notice-dialog").getByLabel("Close").count()) === 0, "the first-visit notice has no close button");
  await page.keyboard.press("Escape");
  await page.waitForTimeout(100);
  expect(await tid("notice-dialog").isVisible(), "Escape does not dismiss the first-visit notice");
  expect((await page.evaluate(() => window.__pyrpApp.useApp.getState().backend)) === null, "the backend does not start before the acknowledgement");
  expect(((await tid("splash-notice").textContent()) ?? "").includes("not for clinical use"), "the splash carries the disclaimer line");
  await page.screenshot({ path: join(shots, "menus-notice-light.png") });
  await tid("notice-accept").click();
  await tid("notice-dialog").waitFor({ state: "detached" });
  await page.waitForFunction(() => window.__pyrpApp?.useApp.getState().session);
  await page.reload();
  await page.waitForFunction(() => window.__pyrpApp?.useApp.getState().session);
  expect((await tid("notice-dialog").count()) === 0, "the notice stays away after the acknowledgement (reload)");
  {
    const narrow = await browser.newContext({ viewport: { width: 390, height: 780 }, deviceScaleFactor: 1, colorScheme: "dark" });
    const other = await narrow.newPage();
    await other.goto(`${base}?backend=mock&speed=0`);
    await other.getByTestId("notice-dialog").waitFor();
    const box = await other.getByTestId("notice-dialog").boundingBox();
    expect(box !== null && box.x >= 0 && box.x + box.width <= 390 && box.y >= 0 && box.y + box.height <= 780, `the notice fits a 390 x 780 window (${box?.width} x ${box?.height})`);
    expect((await other.evaluate(() => document.documentElement.dataset.theme)) === "dark", "dark scheme on a dark platform");
    await other.screenshot({ path: join(shots, "menus-notice-dark-narrow.png") });
    await other.setViewportSize({ width: 1680, height: 960 });
    await other.screenshot({ path: join(shots, "menus-notice-dark.png") });
    await narrow.close();
  }

  // ------------------------------------------------------------------ splash
  await page.goto(`${base}?backend=mock&speed=8`, { waitUntil: "commit" });
  await tid("boot-splash").waitFor();
  const splashText = await tid("boot-splash").textContent();
  expect(/\d\/\d/.test(splashText ?? ""), `splash shows the n/N counter: ${splashText}`);
  await shot("splash");
  await tid("boot-splash").waitFor({ state: "detached", timeout: 15000 });
  await page.waitForFunction(() => window.__pyrpApp?.useApp.getState().session);
  await shot("start");

  // ------------------------------------------------------------------ File menu, empty session
  await openMenu("file");
  expect(await disabled("menu-item-save_pln"), "Save Plan disabled without a plan");
  expect(await disabled("menu-item-save_workspace"), "Save Workspace disabled without a CT");
  expect(await disabled("menu-item-load_folder"), "Load Folder is a stub");
  const folderTip = (await tid("menu-item-load_folder").getAttribute("title")) ?? "";
  expect(folderTip.includes("SimpleITK"), `Load Folder tooltip carries the capability reason: ${folderTip.split("\n").pop()}`);
  await tid("menu-item-phantoms").hover();
  await tid("submenu-phantoms").waitFor();
  await tid("menu-item-phantoms").click();
  expect((await tid("submenu-phantoms").count()) === 1, "clicking a submenu opened by hovering keeps it open");
  await shot("file-menu-empty");
  await page.keyboard.press("Escape");

  // ------------------------------------------------------------------ phantom + result
  expect(!(await tid("btn-run-all").count()) && !(await tid("phantom-select").count()), "workflow widget has the Qt controls only (no Run All, no phantom selector)");
  await openPhantom("TG119");
  await page.waitForFunction(() => document.querySelector('[data-testid="dot-load"]')?.getAttribute("data-status") === "done");
  await waitIdle();
  await computeResult();
  await page.waitForFunction(() => document.querySelector('[data-testid="dot-result"]')?.getAttribute("data-status") === "done");
  await closeOptimizationStatus();

  await openMenu("file");
  expect(!(await disabled("menu-item-save_result")), "Save Result enabled with a result");
  expect(!(await disabled("menu-item-save_workspace")), "Save Workspace enabled with a CT");
  await tid("menu-item-ui_mode").hover();
  await tid("submenu-ui_mode").waitFor();
  expect((await tid("menu-item-ui_mode:default").getAttribute("aria-checked")) === "true", "UI Mode default is checked");
  await shot("file-menu-ui-mode");
  await page.keyboard.press("Escape");

  // ------------------------------------------------------------------ saves
  await clickItem("file", "save_pln");
  await tid("save-dialog").waitFor();
  await shot("save-plan-dialog");
  await tid("save-name").fill("myplan");
  let file = await download(() => tid("save-ok").click());
  expect(file.name === "myplan.mat", `Save Plan downloads ${file.name}`);
  expect(file.data.subarray(0, 15).toString() === "MATLAB 5.0 MAT-", "Save Plan payload is a MAT file");
  await tid("saved-dialog").waitFor();
  await shot("saved-message");
  await tid("msg-ok").click();

  // Unapplied plan edits are offered before Save Plan (Qt `_resolve_pending_changes`).
  await tid("plan-gantry").fill("0 90");
  await clickItem("file", "save_pln");
  await tid("pending-changes-dialog").waitFor();
  expect(((await tid("pending-changes-dialog").textContent()) ?? "").includes("has changes that are not applied yet"), "Save Plan asks about unapplied plan edits");
  await shot("pending-changes-dialog");
  await tid("msg-cancel").click();
  expect((await tid("save-dialog").count()) === 0, "Cancel in the pending-edit question stops the save");
  await clickItem("file", "save_pln");
  await tid("pending-changes-dialog").waitFor();
  await tid("msg-apply").click();
  await tid("save-dialog").waitFor();
  await page.waitForFunction(() => !/Modified/.test(document.querySelector('[data-testid="plan-note"]')?.textContent ?? ""));
  expect(true, "Apply and continue applies the plan edits");
  await tid("save-cancel").click();
  await tid("plan-gantry").fill("0 90 180");
  await tid("btn-optimize").click();
  await tid("pending-changes-dialog").waitFor();
  expect(((await tid("pending-changes-dialog").textContent()) ?? "").includes("outdates the dose influence"), "Optimize explains that applying outdates the dose influence");
  await shot("pending-changes-optimize");
  await tid("msg-continue").click();
  await waitIdle();
  await closeOptimizationStatus();
  await tid("plan-apply").click();
  await page.waitForFunction(() => !/Modified/.test(document.querySelector('[data-testid="plan-note"]')?.textContent ?? ""));
  await computeResult();
  await closeOptimizationStatus();

  await clickItem("file", "save_dij");
  await tid("save-format").selectOption({ label: "Pickle (*.pkl)" });
  file = await download(() => tid("save-ok").click());
  expect(file.name === "dij.pkl" && file.data[0] === 0x80, `Save Dij as pickle downloads ${file.name}`);
  await tid("msg-ok").click();

  await clickItem("file", "save_cst");
  await tid("save-dialog").waitFor();
  const cstOptions = await tid("save-format").locator("option").allTextContents();
  expect(cstOptions.some((o) => o.includes("DICOM RTSTRUCT")) && cstOptions.some((o) => o.includes("NIfTI") && o.includes("not available")), `Save CST lists DICOM and the unavailable label maps: ${cstOptions.join(" | ")}`);
  await shot("save-cst-dialog");
  await tid("save-format").selectOption({ label: "DICOM RTSTRUCT" });
  await tid("save-ok").click();
  await tid("objectives-warning").waitFor();
  expect(((await tid("objectives-warning").textContent()) ?? "").includes("cannot store objectives"), "DICOM export warns that objectives are dropped");
  await shot("save-cst-objectives-warning");
  file = await download(() => tid("msg-ok").click());
  expect(file.name === "cst_rtstruct.zip" && file.data.subarray(0, 2).toString() === "PK", `Save CST as DICOM RTSTRUCT zip: ${file.name}`);
  await tid("msg-ok").click();

  await clickItem("file", "save_result");
  await tid("save-result-dialog").waitFor();
  await tid("save-result-ok").click();
  expect(((await tid("save-result-status").textContent()) ?? "").includes("Select at least one quantity"), "Save Result asks for a quantity");
  await tid("save-result-all").click();
  await shot("save-result-dialog");
  await tid("save-result-ok").click();
  await tid("save-dialog").waitFor();
  await tid("save-format").selectOption({ label: "NumPy (*.npz)" });
  file = await download(() => tid("save-ok").click());
  expect(file.name === "physical_dose.npz" && file.data.subarray(0, 2).toString() === "PK", `Save Result as npz: ${file.name}`);
  await tid("msg-ok").click();

  await clickItem("file", "save_workspace");
  file = await download(() => tid("save-ok").click());
  expect(file.name === "workspace.pkl", `Save Workspace downloads ${file.name}`);
  await tid("msg-ok").click();

  // ------------------------------------------------------------------ loading
  await chooseFile(() => clickItem("file", "load_file"), { name: "patient.mat", mimeType: "application/octet-stream", buffer: bytes("MATLAB") });
  await tid("run-warnings").waitFor();
  const warning = (await tid("run-warnings").textContent()) ?? "";
  expect(warning.includes("replaced by the TG119 phantom"), "warnings of a run are shown once afterwards");
  await shot("run-warnings-dialog");
  await tid("msg-ok").click();
  await waitIdle();

  await tid("file-input").setInputFiles({ name: "scan.nii.gz", mimeType: "application/gzip", buffer: bytes("x") });
  await tid("message-box").waitFor();
  expect(((await tid("message-box").textContent()) ?? "").includes("SimpleITK"), "an image file explains why it cannot be loaded here");
  await shot("image-load-refused");
  await tid("msg-ok").click();

  await chooseFile(() => tid("btn-load-dij").click(), { name: "nodij.mat", mimeType: "application/octet-stream", buffer: bytes("MATLAB") });
  await tid("message-box").waitFor();
  expect(((await tid("message-box").textContent()) ?? "").includes("No dose-influence matrix found"), "Load Dij warns when the file holds no dij");
  await tid("msg-ok").click();
  await chooseFile(() => tid("btn-load-dij").click(), { name: "dij.mat", mimeType: "application/octet-stream", buffer: bytes("MATLAB") });
  await page.waitForFunction(() => document.querySelector('[data-testid="dot-dij"]')?.getAttribute("data-status") === "done");
  expect(true, "Load Dij loads stf + dij");
  await waitIdle();

  // ------------------------------------------------------------------ snapshot
  await tid("btn-optimize").click();
  await waitIdle();
  await closeOptimizationStatus();
  await page.waitForFunction(() => document.querySelector('[data-testid="btn-snapshot"]')?.disabled === false);
  await tid("btn-snapshot").click();
  await tid("prompt-dialog").waitFor();
  expect((await tid("prompt-input").inputValue()) === "photons_30fx", "Snapshot name defaults to <mode>_<n>fx");
  await shot("snapshot-prompt");
  await tid("prompt-input").fill("beam");
  await tid("prompt-ok").click();
  await tid("message-box").waitFor();
  expect(((await tid("message-box").textContent()) ?? "").includes("cannot be used as a snapshot name"), "reserved snapshot names are refused");
  await tid("msg-ok").click();
  await tid("btn-snapshot").click();
  await tid("prompt-input").fill("my snap");
  await tid("prompt-ok").click();
  await page.waitForFunction(() => /snapshot 'my_snap'/.test(document.querySelector('[data-testid="log"]')?.textContent ?? ""));
  expect(true, "snapshot name is sanitized to my_snap");

  // ------------------------------------------------------------------ base data
  await clickItem("base_data", "show_base_data");
  await tid("base-data-dialog").waitFor();
  await tid("base-data-row-photons_Generic.mat").waitFor();
  await shot("base-data-machines");
  expect(((await tid("base-data-folders-machines").textContent()) ?? "").includes("until the page is closed"), "base data tab notes that added files last until the page is closed");
  await tid("base-data-tab-btn-rbe_tables").click();
  await tid("base-data-row-legacy_table.mat").waitFor();
  await shot("base-data-rbe");
  await tid("base-data-tab-btn-machines").click();
  await chooseFile(() => tid("base-data-add-machines").click(), { name: "photons_Generic.mat", mimeType: "application/octet-stream", buffer: bytes("MATLAB") });
  await tid("prompt-dialog").waitFor();
  expect(((await tid("prompt-dialog").textContent()) ?? "").includes("ships with pyRadPlan"), "name clash with a shipped machine asks for another name");
  await shot("base-data-name-clash");
  await tid("prompt-input").fill("MyLinac");
  await tid("prompt-ok").click();
  await tid("base-data-row-photons_MyLinac.mat").waitFor();
  await tid("base-data-row-photons_MyLinac.mat").click();
  await shot("base-data-added");
  expect(!(await tid("base-data-remove-machines").isDisabled()), "Remove is enabled for a user file");
  await tid("base-data-remove-machines").click();
  await tid("remove-dialog").waitFor();
  await tid("msg-yes").click();
  await tid("base-data-row-photons_MyLinac.mat").waitFor({ state: "detached" });
  expect(await tid("base-data-open-machines").isDisabled(), "Open folder is a stub in the browser");
  await closeTopDialog();
  await chooseFile(() => clickItem("base_data", "add_rbe"), { name: "my_rbe.mat", mimeType: "application/octet-stream", buffer: bytes("MATLAB") });
  await tid("base-data-added").waitFor();
  expect(((await tid("base-data-added").textContent()) ?? "").includes("my_rbe.mat"), "Add RBE Table confirms the added file");
  await tid("msg-ok").click();
  await openMenu("base_data");
  expect(await disabled("menu-item-open_data_folder"), "Open Data Folder is a stub in the browser");
  await shot("base-data-menu");
  await page.keyboard.press("Escape");

  // ------------------------------------------------------------------ settings
  await clickItem("settings", "preferences");
  await tid("preferences-dialog").waitFor();
  await shot("preferences-general");
  await tid("preferences-tab-xp").click();
  await shot("preferences-xp");
  await tid("preferences-tab-ai").click();
  await tid("preferences-tab-gui").click();
  await tid("settings-gui-ui_mode").click();
  await page.getByRole("option", { name: "educational" }).click();
  await shot("preferences-gui");
  await tid("preferences-ok").click();
  await tid("preferences-dialog").waitFor({ state: "detached" });
  await openMenu("file");
  await tid("menu-item-ui_mode").hover();
  expect((await tid("menu-item-ui_mode:educational").getAttribute("aria-checked")) === "true", "Preferences GUI tab switches the UI mode");
  await tid("menu-item-ui_mode:default").click();
  await clickItem("settings", "settings_xp");
  await tid("preferences-dialog").waitFor();
  expect(((await tid("preferences-dialog").locator(".title > span").first().textContent()) ?? "") === "XP (Backend) Settings", "XP quick link opens the single-section dialog");
  expect(await tid("settings-xp-prefer_gpu").isDisabled(), "XP settings are read-only in the browser");
  await shot("settings-xp-section");
  await tid("preferences-cancel").click();

  // ------------------------------------------------------------------ help
  await clickItem("help", "licenses");
  await tid("licenses-dialog").waitFor();
  await tid("license-row-numpy").waitFor();
  await tid("license-row-numpy").click();
  await page.waitForFunction(() => document.querySelector('[data-testid="license-text"]')?.value?.includes("BSD"));
  await tid("license-row-tqdm").click();
  await page.waitForFunction(() => document.querySelector('[data-testid="license-text"]')?.value === "No license text found for this distribution.");
  await tid("licenses-filter").fill("py");
  await shot("licenses");
  expect((await tid("licenses-table").locator("tbody tr").count()) >= 3, "license filter narrows the table");
  expect((await tid("licenses-notices-link").getAttribute("href")) === "/THIRD_PARTY_NOTICES.txt", "the licenses view links the JavaScript notices");
  const notices = await page.request.get(`${base}THIRD_PARTY_NOTICES.txt`);
  expect(notices.ok() && (await notices.text()).includes("production build"), "the dev server explains where the notices come from");
  await closeTopDialog();
  await clickItem("help", "about");
  await tid("about-dialog").waitFor();
  const aboutText = (await tid("about-dialog").textContent()) ?? "";
  expect(aboutText.includes("NOT FOR CLINICAL USE") && aboutText.includes("must not be used for clinical treatment planning"), "About shows the disclaimer");
  expect(((await tid("about-build").textContent()) ?? "").startsWith("Web build:"), "About shows the web build");
  await shot("about");
  await tid("about-licenses").click();
  await tid("licenses-dialog").waitFor();
  await closeTopDialog();
  expect((await tid("about-dialog").count()) === 1, "Escape closes only the topmost dialog");
  await tid("about-notice").click();
  await tid("notice-dialog").waitFor();
  await closeTopDialog();
  expect((await tid("notice-dialog").count()) === 0 && (await tid("about-dialog").count()) === 1, "About opens the notice, Escape closes it");
  await tid("about-close").click();
  await tid("status-notice").click();
  await tid("notice-close").waitFor();
  expect((await tid("notice-accept").count()) === 0, "the status bar reopens the notice with a Close button");
  await tid("notice-close").click();

  // ------------------------------------------------------------------ log console
  await tid("log-level").selectOption("warning");
  const infoLines = await tid("log").locator(".info").count();
  expect(infoLines === 0, "the level combo filters the log");
  await shot("log-warning-level");
  await tid("log-level").selectOption("info");
  expect((await tid("log").locator(".info").count()) > 0, "Info level shows the records again");
  await tid("log-clear").click();
  expect((await tid("log").locator("div").count()) === 0, "Clear empties the log");

  // ------------------------------------------------------------------ view menu + dark mode
  await openMenu("view");
  expect(await disabled("menu-item-view:Screenshot"), "View stubs are disabled");
  await shot("view-menu");
  await tid("menu-item-dark_mode").click();
  expect((await page.evaluate(() => document.documentElement.dataset.theme)) === "dark", "Dark Mode switches the page scheme");
  await page.waitForTimeout(200);
  await shot("dark-mode");
  await clickItem("file", "phantoms");
  await shot("dark-file-menu");
  await page.keyboard.press("Escape");
  await clickItem("settings", "preferences");
  await shot("dark-preferences");
  await tid("preferences-cancel").click();
  await clickItem("file", "save_pln");
  await shot("dark-save-dialog");
  await tid("save-cancel").click();
  await clickItem("view", "dark_mode");
  expect((await page.evaluate(() => document.documentElement.dataset.theme)) === "light", "Dark Mode toggles back");
  await clickItem("view", "log_panel");
  expect((await tid("log").count()) === 0, "Log Panel hides the log");
  await clickItem("view", "log_panel");
  expect((await tid("log").count()) === 1, "Log Panel shows the log again");

  // ------------------------------------------------------------------ close while busy
  await page.goto(`${base}?backend=mock&speed=3`);
  await tid("boot-splash").waitFor({ state: "detached", timeout: 15000 });
  await page.waitForFunction(() => window.__pyrpApp?.useApp.getState().session);
  await openPhantom("TG119");
  await page.waitForFunction(() => document.querySelector('[data-testid="btn-stop"]')?.disabled === false);
  const prevented = await page.evaluate(() => {
    const e = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(e);
    return e.defaultPrevented;
  });
  expect(prevented, "closing the page while busy is intercepted");
  await tid("btn-stop").click();
  await waitIdle();
  const idlePrevented = await page.evaluate(() => {
    const e = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(e);
    return e.defaultPrevented;
  });
  expect(!idlePrevented, "closing an idle page is not intercepted");
  await openMenu("file");
  await shot("file-menu-after-stop");
  await page.keyboard.press("Escape");

  // ------------------------------------------------------------------ patient on the command line
  await page.goto(`${base}?backend=mock&speed=0&phantom=TG119`);
  await tid("boot-splash").waitFor({ state: "detached", timeout: 15000 });
  await page.waitForFunction(() => document.querySelector('[data-testid="dot-load"]')?.getAttribute("data-status") === "done");
  expect(true, "?phantom=TG119 loads the patient behind the splash");

  // ------------------------------------------------------------------ native runtime: image import, no stubs
  await fresh("?backend=mock&speed=0&runtime=native");
  await openPhantom("TG119");
  await waitIdle();
  await tid("file-input").setInputFiles({ name: "ct_dose.nii.gz", mimeType: "application/gzip", buffer: bytes("x") });
  await tid("import-image-dialog").waitFor();
  await shot("native-import-image");
  await tid("import-dose").check();
  await tid("import-dose-name").fill("my dose");
  await tid("import-ok").click();
  await tid("error-dialog").waitFor();
  expect(((await tid("error-dialog").textContent()) ?? "").includes("not available through the protocol yet"), "image import kinds are refused by the v1 protocol and reported");
  await shot("native-import-image-refused");
  await page.keyboard.press("Escape");
  await waitIdle();
  await openMenu("file");
  await tid("menu-item-phantoms").hover();
  expect(!(await disabled("menu-item-open_phantom_folder")), "Open Phantom Folder is available on the native runtime");
  await shot("native-file-menu");
  await page.keyboard.press("Escape");
  await clickItem("settings", "settings_xp");
  expect(!(await tid("settings-xp-prefer_gpu").isDisabled()), "XP settings are editable on the native runtime");
  await shot("native-settings-xp");
  await tid("preferences-cancel").click();

  if (problems.length) throw new Error(`browser reported problems:\n${problems.join("\n")}`);
  console.log("menus e2e passed");
} catch (err) {
  console.error(err);
  await page.screenshot({ path: join(shots, "menus-failure.png") }).catch(() => {});
  process.exitCode = 1;
} finally {
  await browser.close();
  await server.close();
}
