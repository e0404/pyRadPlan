// Smoke test of a static (GitHub-Pages-like) deployment against the real Pyodide backend:
// boot, TG119, Calc. Dose Influence + Optimize, optional Stop; checks cross-origin isolation and
// that the app only talked to its own origin and the Pyodide package CDN.
// Usage: node e2e/pages.mjs <url> [--stop] [--no-isolation] [--headed] [--chromium]
//   --no-isolation  expect the page NOT to be isolated (a build without the service worker)
// Run against `node e2e/pages-server.mjs <dist> --prefix=/pyRadPlan/ --port=8080`.
import { chromium } from "playwright";

const args = process.argv.slice(2);
const url = args.find((a) => !a.startsWith("--"));
if (!url) throw new Error("usage: node e2e/pages.mjs <url> [--stop] [--no-isolation]");
const expectIsolated = !args.includes("--no-isolation");
const ALLOWED_HOSTS = new Set([new URL(url).host, "cdn.jsdelivr.net"]);

const browser = await chromium.launch({
  ...(args.includes("--chromium") ? {} : { channel: "msedge" }),
  headless: !args.includes("--headed"),
  args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
});
const context = await browser.newContext({ viewport: { width: 1680, height: 960 } });
const page = await context.newPage();
const hosts = new Map();
const problems = [];
const failures = [];
context.on("request", (r) => {
  const host = new URL(r.url()).host;
  hosts.set(host, (hosts.get(host) ?? 0) + 1);
});
context.on("requestfailed", (r) => {
  // The first visit with the service worker reloads once; the aborted module request is expected.
  if (r.failure()?.errorText !== "net::ERR_ABORTED") problems.push(`requestfailed: ${r.url()} ${r.failure()?.errorText}`);
});
page.on("pageerror", (e) => problems.push(`pageerror: ${e.message}`));
page.on("console", (m) => {
  if (m.type() === "error") problems.push(`console: ${m.text()}`);
});
const tid = (id) => page.getByTestId(id);
const check = (ok, message) => {
  console.log(`${ok ? "ok" : "FAIL"}: ${message}`);
  if (!ok) failures.push(message);
};
const idle = () => page.waitForFunction(() => window.__pyrpApp.useApp.getState().busy === null, null, { timeout: 30 * 60 * 1000 });

async function step(id) {
  await tid(id).click();
  await page.waitForTimeout(300);
  if (await tid("pending-changes-dialog").count()) await tid("msg-apply").click();
  await page.waitForFunction(() => window.__pyrpApp.useApp.getState().busy !== null, null, { timeout: 20000 }).catch(() => undefined);
  await idle();
  await page.waitForTimeout(300);
  if (await tid("run-warnings").count()) await tid("msg-ok").click();
  if (await tid("error-dialog").count()) throw new Error(`${id} failed: ${await tid("error-dialog").textContent()}`);
}

try {
  const t0 = Date.now();
  const response = await page.goto(url);
  if (!response?.ok()) throw new Error(`${url} answered ${response?.status()}`);
  // A fresh profile: the disclaimer comes first, and nothing is loaded from the CDN before it is acknowledged.
  await tid("notice-dialog").waitFor({ timeout: 60000 });
  await page.waitForTimeout(1000);
  check(!hosts.has("cdn.jsdelivr.net"), "the first-visit notice shows before anything is loaded from cdn.jsdelivr.net");
  await tid("notice-accept").click();
  await page.waitForFunction(() => window.__pyrpApp?.useApp.getState().session, null, { timeout: 5 * 60 * 1000 });
  console.log(`boot ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  const isolated = await page.evaluate(() => crossOriginIsolated);
  check(isolated === expectIsolated, `crossOriginIsolated is ${isolated}`);
  const info = await page.evaluate(() => window.__pyrpApp.useApp.getState().info);
  check(info.runtime === "pyodide", `runtime ${info.runtime}, pyRadPlan ${info.version}, Python ${info.python}`);
  for (const [file, needle] of [["THIRD_PARTY_NOTICES.txt", "react-dom"], ["LICENSE.txt", "BSD 3-Clause"]]) {
    const res = await context.request.get(new URL(file, url).href);
    check(res.ok() && (await res.text()).includes(needle), `${file} is deployed`);
  }
  check((await tid("btn-stop").isDisabled()) === true, "Stop is disabled while idle");
  if (!isolated) check(((await tid("btn-stop").getAttribute("title")) ?? "").includes("SharedArrayBuffer"), "Stop explains why it is unavailable");

  await tid("menu-file").click();
  await tid("popup-file").waitFor();
  await tid("menu-item-phantoms").click();
  await tid("menu-item-phantom:TG119").click();
  await page.waitForFunction(() => document.querySelector('[data-testid="dot-load"]')?.getAttribute("data-status") === "done", null, { timeout: 120000 });
  await idle();
  const t1 = Date.now();
  await step("btn-dij");
  await step("btn-optimize");
  check((await tid("dot-result").getAttribute("data-status")) === "done", `dose influence + optimization in ${((Date.now() - t1) / 1000).toFixed(1)} s`);
  if (await tid("opt-close").count()) await tid("opt-close").click();

  if (args.includes("--stop") && isolated) {
    await tid("plan-gantry").fill("0 40 80 120 160 200 240 280 320");
    await tid("plan-gantry").press("Tab");
    await tid("plan-apply").click();
    await page.waitForFunction(() => window.__pyrpApp.useApp.getState().state?.plan?.gantry_angles?.length === 9, null, { timeout: 60000 });
    await idle();
    await tid("btn-dij").click();
    await page.waitForFunction(() => window.__pyrpApp.useApp.getState().busy !== null, null, { timeout: 20000 });
    await page.waitForTimeout(2500);
    check((await tid("btn-stop").isEnabled()) === true, "Stop is enabled while a calculation runs");
    const t2 = Date.now();
    await tid("btn-stop").click();
    await idle();
    check((await tid("dot-dij").getAttribute("data-status")) !== "done", `Stop interrupted the dose calculation (${Date.now() - t2} ms)`);
  }
} catch (err) {
  failures.push(String(err.message).split("\n")[0]);
  console.error(err);
}
console.log(`hosts contacted: ${JSON.stringify(Object.fromEntries(hosts))}`);
const foreign = [...hosts.keys()].filter((h) => !ALLOWED_HOSTS.has(h));
check(foreign.length === 0, `no host besides the app and cdn.jsdelivr.net was contacted${foreign.length ? `: ${foreign.join(", ")}` : ""}`);
check(problems.length === 0, `no browser errors${problems.length ? `: ${[...new Set(problems)].join(" | ")}` : ""}`);
await browser.close();
if (failures.length) {
  console.log(`FAILED: ${failures.join("; ")}`);
  process.exitCode = 1;
}
