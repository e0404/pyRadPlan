// Layout check of the main window at desktop, tablet and phone widths against the mock backend:
// layout mode, default collapsed panels, no horizontal overflow, a usable viewer, dialogs and menus
// inside the viewport. Screenshots go to e2e/screenshots/responsive-*.png.
// Usage: npm run e2e:responsive [-- --headed] [-- --chromium]
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

const VIEWPORTS = [
  { name: "desktop", layout: "desktop", viewport: { width: 1680, height: 960 } },
  { name: "tablet-landscape", layout: "tablet", viewport: { width: 1024, height: 768 }, touch: true },
  { name: "tablet-portrait", layout: "tablet", viewport: { width: 820, height: 1180 }, touch: true },
  { name: "phone", layout: "phone", viewport: { width: 390, height: 844 }, touch: true },
];

const server = await createServer({ root, server: { port: 5199, strictPort: false }, logLevel: "warn" });
await server.listen();
const base = server.resolvedUrls.local[0];
console.log(`dev server: ${base}`);

const browser = await chromium.launch({
  ...(useBundledChromium ? {} : { channel: "msedge" }),
  headless: !headed,
  args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
});

function expect(cond, message) {
  if (!cond) throw new Error(`check failed: ${message}`);
  console.log(`ok: ${message}`);
}

async function checkViewport({ name, layout, viewport, touch }) {
  console.log(`--- ${name} (${viewport.width}x${viewport.height})`);
  const context = await browser.newContext({ viewport, deviceScaleFactor: 1, hasTouch: !!touch, isMobile: false });
  await acknowledgeNotice(context, base);
  const page = await context.newPage();
  const problems = [];
  page.on("pageerror", (err) => problems.push(`pageerror: ${err.message}`));
  page.on("console", (msg) => {
    if (msg.type() === "error") problems.push(`console: ${msg.text()}`);
  });
  const tid = (id) => page.getByTestId(id);
  const shot = async (suffix) => {
    const file = join(shots, `responsive-${name}${suffix ? `-${suffix}` : ""}.png`);
    await page.waitForTimeout(150);
    await page.screenshot({ path: file });
    console.log(`screenshot ${file}`);
  };
  const collapsed = (id) => tid(id).evaluate((el) => el.classList.contains("collapsed"));
  const noHorizontalOverflow = async () => {
    const offenders = await page.evaluate(() => {
      const cols = document.querySelector('[data-testid="main-columns"]');
      const right = cols.getBoundingClientRect().right + 1;
      // Content inside its own scroll or clip box (e.g. the objectives table) is allowed to be wider.
      const clipped = (el) => {
        for (let p = el.parentElement; p && p !== cols; p = p.parentElement) {
          if (getComputedStyle(p).overflowX !== "visible") return true;
        }
        return false;
      };
      const out = [...cols.querySelectorAll(".groupbox > .gb-body *")]
        .filter((el) => el.getClientRects().length && el.getBoundingClientRect().right > right && !el.closest(".menu-popup") && !clipped(el))
        .slice(0, 5)
        .map((el) => `${el.tagName.toLowerCase()}.${el.className} "${(el.textContent ?? "").trim().slice(0, 30)}"`);
      if (document.documentElement.scrollWidth > window.innerWidth) out.push("document");
      if (cols.scrollWidth > cols.clientWidth + 1) out.push(".columns");
      return out;
    });
    if (offenders.length) console.log(`overflowing: ${offenders.join(", ")}`);
    return offenders.length === 0;
  };
  const insideViewport = (locator) =>
    locator.evaluate((el) => {
      const r = el.getBoundingClientRect();
      return r.left >= 0 && r.top >= 0 && r.right <= window.innerWidth + 0.5 && r.bottom <= window.innerHeight + 0.5;
    });

  try {
    await page.goto(`${base}?backend=mock&speed=0`);
    await tid("boot-splash").waitFor({ state: "detached", timeout: 15000 });
    await page.waitForFunction(() => window.__pyrpApp?.useApp.getState().session);
    expect((await page.locator(".main-window").getAttribute("data-layout")) === layout, `layout mode is ${layout}`);

    await tid("menu-file").click();
    await tid("menu-item-phantoms").click();
    const sub = tid("submenu-phantoms");
    expect(await insideViewport(sub), "example phantom submenu inside the viewport");
    await tid("menu-item-phantom:TG119").click();
    await page.waitForFunction(() => document.querySelector('[data-testid="dot-load"]')?.getAttribute("data-status") === "done");
    await page.waitForFunction(() => document.querySelector('[data-testid="btn-stop"]')?.disabled);

    expect(await noHorizontalOverflow(), "no horizontal overflow");
    const canvas = await tid("viewer-canvas").boundingBox();
    expect(canvas && canvas.width >= Math.min(300, viewport.width - 40) && canvas.height >= 250, `viewer canvas is usable (${Math.round(canvas?.width)}x${Math.round(canvas?.height)})`);

    if (layout === "desktop") {
      expect((await tid("panel-plan-toggle").count()) === 0 && (await tid("config-toggle").count()) === 0, "desktop panels are not collapsible");
    } else if (layout === "tablet") {
      expect(!(await collapsed("panel-plan")) && !(await collapsed("panel-workflow")), "tablet opens with all panels expanded");
      await shot();
      await tid("config-toggle").click();
      expect(!(await tid("panel-workflow").isVisible()) && !(await tid("panel-plan").isVisible()), "configuration row collapses");
      const top = (await tid("viewer-canvas").boundingBox()).y;
      expect(top < canvas.y, "viewer moves up when the configuration row is collapsed");
      await shot("config-collapsed");
      await tid("config-toggle").click();
      expect(await tid("panel-plan").isVisible(), "configuration row expands again");
    } else {
      expect(!(await collapsed("panel-workflow")), "phone opens with the workflow expanded");
      for (const id of ["panel-plan", "panel-objectives", "panel-visualization", "panel-viewer-options", "panel-structures", "panel-info"]) {
        expect(await collapsed(id), `phone opens with ${id} collapsed`);
      }
      const order = await page.evaluate(() =>
        ["panel-workflow", "panel-viewer", "panel-plan"].map((id) => document.querySelector(`[data-testid="${id}"]`).getBoundingClientRect().top),
      );
      expect(order[0] < order[1] && order[1] < order[2], "phone stacks workflow, viewer, then plan");
      await shot();
      await tid("panel-plan-toggle").click();
      expect(!(await collapsed("panel-plan")) && (await tid("plan-apply").isVisible()), "plan panel expands on tap");
      expect(await noHorizontalOverflow(), "no horizontal overflow with the plan panel open");
      await tid("panel-plan").scrollIntoViewIfNeeded();
      await shot("plan-open");
      for (const id of ["panel-objectives", "panel-visualization", "panel-viewer-options", "panel-structures", "panel-info"]) {
        await tid(`${id}-toggle`).click();
      }
      expect(await noHorizontalOverflow(), "no horizontal overflow with every panel open");
      await tid("panel-viewer-options").scrollIntoViewIfNeeded();
      await shot("all-open");
    }

    await tid("menu-help").click();
    expect(await insideViewport(tid("popup-help")), "help menu inside the viewport");
    await tid("menu-item-about").click();
    await tid("about-dialog").waitFor();
    expect(await insideViewport(tid("about-dialog")), "about dialog inside the viewport");
    if (layout === "phone") await shot("about");
    await page.keyboard.press("Escape");

    if (layout === "desktop") await shot();
    if (problems.length) throw new Error(`browser reported problems:\n${problems.join("\n")}`);
  } catch (err) {
    await shot("failure").catch(() => {});
    throw err;
  } finally {
    await context.close();
  }
}

try {
  for (const vp of VIEWPORTS) await checkViewport(vp);
  console.log("responsive e2e passed");
} finally {
  await browser.close();
  await server.close();
}
