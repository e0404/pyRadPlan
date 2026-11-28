// Assemble the GitHub Pages site: one directory per channel (e.g. latest/ and develop/), each
// built from its own checkout, plus a landing page at the root. A Pages deployment replaces the
// whole site, so every deployment assembles all channels. A checkout without web/ (releases
// before the web version) gets a static placeholder instead of the app. The landing page opens the
// first channel that contains the app (latest, or develop while the newest release has none)
// after a short countdown, unless the visitor stays or the URL has ?stay.
//
//   node web/scripts/assemble_pages.mjs --out _site --channel latest=<checkout> --channel develop=<checkout>
//        [--build] [--install] [--skip-wheels] [--coi]
//
//   --channel name=dir  a git checkout of the repository; the first channel is the default
//   --build             build each checkout: wheels (web/pyodide, needs PYTHON with pip) and the
//                       frontend with the relative base and the channel, version and commit in
//                       VITE_PYRADPLAN_*; without it the existing web/dist is copied
//   --install           npm ci in web/pyodide and web before building
//   --skip-wheels       keep the wheels already in web/public/pyrp
//   --coi               include the cross-origin isolation service worker (PYRADPLAN_COI=1)
//
// Usage is documented in docs/development/web_publishing.md.
import { spawnSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const here = dirname(fileURLToPath(import.meta.url));
const { values: args } = parseArgs({
  options: {
    out: { type: "string", default: "_site" },
    channel: { type: "string", multiple: true, default: [] },
    build: { type: "boolean", default: false },
    install: { type: "boolean", default: false },
    "skip-wheels": { type: "boolean", default: false },
    coi: { type: "boolean", default: false },
  },
});

const CHANNEL_NAME = /^[a-z0-9][a-z0-9._-]*$/;
const FALLBACK_ABOUT = {
  disclaimer_title: "NOT FOR CLINICAL USE",
  disclaimer:
    "pyRadPlan is intended for research and education. It is not a medical device and must not be used for clinical treatment planning.",
  imprint_url: "https://docs.pyradplan.org/en/latest/legal/imprint.html",
  privacy_url: "https://docs.pyradplan.org/en/latest/legal/privacy.html",
};
const REDIRECT_SECONDS = 5;

function run(cmd, cmdArgs, cwd, env = {}) {
  console.log(`$ (${cwd}) ${cmd} ${cmdArgs.join(" ")}`);
  // npm is npm.cmd on Windows, which only starts through a shell; the arguments contain no spaces.
  const windows = process.platform === "win32";
  const result = spawnSync(windows ? [cmd, ...cmdArgs].join(" ") : cmd, windows ? [] : cmdArgs, {
    cwd,
    stdio: "inherit",
    env: { ...process.env, ...env },
    shell: windows,
  });
  if (result.status !== 0) throw new Error(`${cmd} ${cmdArgs.join(" ")} failed in ${cwd} (exit ${result.status})`);
}

function git(dir, ...gitArgs) {
  const result = spawnSync("git", ["-C", dir, ...gitArgs], { encoding: "utf8" });
  return result.status === 0 ? result.stdout.trim() : "";
}

const escapeHtml = (text) =>
  String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const PAGE_STYLE = `
  :root { color-scheme: light dark; --bg: #f0f0f0; --card: #fff; --text: #1a1a1a; --dim: #5a5a5a; --border: #c8c8c8; --link: #005fb8; --danger: #c0392b; }
  @media (prefers-color-scheme: dark) { :root { --bg: #2a2a2a; --card: #353535; --text: #f0f0f0; --dim: #a0a0a0; --border: #555; --link: #56a0e8; --danger: #ff6b5a; } }
  body { margin: 0; background: var(--bg); color: var(--text); font: 15px/1.5 system-ui, sans-serif; }
  main { max-width: 640px; margin: 0 auto; padding: 32px 16px; }
  h1 { font-size: 22px; margin: 0 0 16px; }
  a { color: var(--link); }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin: 12px 0; }
  .card h2 { font-size: 17px; margin: 0 0 4px; }
  .dim { color: var(--dim); font-size: 13px; }
  .disclaimer { color: var(--danger); border: 1px solid var(--danger); border-radius: 4px; padding: 8px 12px; margin: 16px 0; }
  code { font-size: 13px; overflow-wrap: anywhere; }
  .redirect { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; border-color: var(--link); }
  .redirect[hidden] { display: none; }
  .redirect .text { flex: 1 1 260px; }
  .redirect button { font: inherit; padding: 4px 12px; border: 1px solid var(--border); border-radius: 4px; background: var(--bg); color: var(--text); cursor: pointer; }
  footer { margin-top: 24px; }
`;

function page(title, body) {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeHtml(title)}</title>
<style>${PAGE_STYLE}</style>
</head>
<body>
<main>
${body}
</main>
</body>
</html>
`;
}

const disclaimerHtml = (about) =>
  `<p class="disclaimer"><b>${escapeHtml(about.disclaimer_title)}</b>. ${escapeHtml(about.disclaimer)}</p>`;

const footerHtml = (about, allChannels) =>
  `<footer class="dim">${allChannels ? `<a href="../?stay">All channels</a> · ` : ""}<a href="https://github.com/e0404/pyRadPlan">Source code</a> ·
<a href="https://pyradplan.readthedocs.io">Documentation</a> ·
<a href="${escapeHtml(about.imprint_url)}">Imprint</a> ·
<a href="${escapeHtml(about.privacy_url)}">Privacy policy</a></footer>`;

// Counts down and opens the target with location.replace, so Back does not return to the landing
// page and redirect again. "Stay on this page", Escape or ?stay in the URL cancel it.
const REDIRECT_SCRIPT = `
(() => {
  const box = document.getElementById("redirect");
  if (!box || new URLSearchParams(location.search).has("stay")) return;
  const counter = document.getElementById("redirect-seconds");
  let left = Number(box.dataset.seconds);
  box.hidden = false;
  const timer = setInterval(() => {
    left -= 1;
    counter.textContent = String(Math.max(left, 0));
    if (left <= 0) {
      clearInterval(timer);
      location.replace(box.dataset.target);
    }
  }, 1000);
  const stay = () => {
    clearInterval(timer);
    box.hidden = true;
  };
  document.getElementById("redirect-stay").addEventListener("click", stay);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") stay(); });
})();
`;

function redirectHtml(target, channels) {
  if (!target) return "";
  const latest = channels.find((c) => c.name === "latest");
  const why = latest && !latest.app ? "The newest release does not contain the browser version yet. " : "";
  return `<div class="card redirect" id="redirect" data-target="./${target.name}/" data-seconds="${REDIRECT_SECONDS}" role="status" aria-live="polite" hidden>
<div class="text">${why}Opening <b>${escapeHtml(target.name)}</b> (${escapeHtml(target.version)}) in <span id="redirect-seconds">${REDIRECT_SECONDS}</span>&nbsp;s…</div>
<a href="./${target.name}/">Open now</a>
<button type="button" id="redirect-stay">Stay on this page</button>
</div>
<script>${REDIRECT_SCRIPT}</script>`;
}

const describe = (c) => [c.version, c.commit && `commit ${c.commit}`].filter(Boolean).join(", ");

function placeholder(channel, apps, about) {
  const others = apps.map((c) => `<a href="../${c.name}/">${escapeHtml(c.name)}</a> (${escapeHtml(describe(c))})`).join(", ");
  return page(
    `pyRadPlan ${channel.version} (NOT FOR CLINICAL USE)`,
    `<h1>pyRadPlan in the browser: ${escapeHtml(channel.name)}</h1>
<p>The ${escapeHtml(channel.name)} channel is pyRadPlan ${escapeHtml(channel.version)}, which does not contain the browser
version yet. It is first included in a later release.</p>
${others ? `<p>Until then, try the development version: ${others}.</p>` : ""}
${disclaimerHtml(about)}
${footerHtml(about, true)}`,
  );
}

function landing(channels, about) {
  const target = channels.find((c) => c.app);
  const cards = channels
    .map(
      (c) => `<div class="card">
<h2><a href="./${c.name}/">${escapeHtml(c.name)}</a></h2>
<div>${escapeHtml(c.summary)}${c.app ? "" : " It does not contain the browser version yet."}</div>
<div class="dim">${escapeHtml(describe(c))}</div>
</div>`,
    )
    .join("\n");
  return page(
    "pyRadPlan in the browser (NOT FOR CLINICAL USE)",
    `<h1>pyRadPlan in the browser</h1>
<p>pyRadPlan, the Python radiotherapy treatment planning toolkit, running in the browser with
<a href="https://pyodide.org">Pyodide</a>. All computation and data stay in your browser.</p>
${redirectHtml(target, channels)}
${cards}
${disclaimerHtml(about)}
${footerHtml(about, false)}`,
  );
}

const SUMMARIES = { latest: "The newest release.", develop: "The development version (develop branch): newer, less tested." };

const channels = args.channel.map((spec) => {
  const at = spec.indexOf("=");
  const name = spec.slice(0, at);
  const dir = resolve(spec.slice(at + 1));
  if (at < 1 || !CHANNEL_NAME.test(name)) throw new Error(`Invalid --channel ${spec} (expected name=dir)`);
  if (!existsSync(dir)) throw new Error(`Channel ${name}: ${dir} does not exist`);
  return {
    name,
    dir,
    app: existsSync(join(dir, "web", "package.json")),
    version: git(dir, "describe", "--tags", "--always"),
    commit: git(dir, "rev-parse", "--short", "HEAD"),
    summary: SUMMARIES[name] ?? `The ${name} channel.`,
  };
});
if (!channels.length) throw new Error("Give at least one --channel name=dir");

const aboutFile = [join(here, "..", "..", "pyRadPlan", "app", "about.json"), ...channels.map((c) => join(c.dir, "pyRadPlan", "app", "about.json"))].find((f) => existsSync(f));
const about = { ...FALLBACK_ABOUT, ...(aboutFile ? JSON.parse(readFileSync(aboutFile, "utf8")) : {}) };

const out = resolve(args.out);
rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

for (const channel of channels) {
  const target = join(out, channel.name);
  if (!channel.app) {
    console.log(`${channel.name}: ${channel.version} has no web/, writing a placeholder`);
    mkdirSync(target, { recursive: true });
    writeFileSync(join(target, "index.html"), placeholder(channel, channels.filter((c) => c.app), about));
    continue;
  }
  const web = join(channel.dir, "web");
  if (args.build) {
    if (args.install) run("npm", ["ci"], join(web, "pyodide"));
    if (!args["skip-wheels"]) run("npm", ["run", "build"], join(web, "pyodide"));
    if (args.install) run("npm", ["ci"], web);
    run("npm", ["run", "build", "--", "--base=./"], web, {
      ...(args.coi ? { PYRADPLAN_COI: "1" } : {}),
      VITE_PYRADPLAN_CHANNEL: channel.name,
      VITE_PYRADPLAN_VERSION: channel.version,
      VITE_PYRADPLAN_COMMIT: channel.commit,
    });
  }
  const dist = join(web, "dist");
  if (!existsSync(join(dist, "index.html"))) throw new Error(`${channel.name}: ${dist} has no build (use --build)`);
  for (const required of ["THIRD_PARTY_NOTICES.txt", "LICENSE.txt", "pyrp/manifest.json"]) {
    if (!existsSync(join(dist, required))) throw new Error(`${channel.name}: the build has no ${required}`);
  }
  cpSync(dist, target, { recursive: true });
  console.log(`${channel.name}: ${channel.version} (${channel.commit}) from ${dist}`);
}

writeFileSync(join(out, "index.html"), landing(channels, about));
writeFileSync(
  join(out, "channels.json"),
  `${JSON.stringify(channels.map(({ name, app, version, commit }) => ({ name, app, version, commit })), null, 2)}\n`,
);
console.log(`assembled ${channels.map((c) => c.name).join(", ")} into ${out}`);
