// Build and test the GitHub Pages site in one command, as the CI does: the newest release tag as
// /latest/ (a placeholder while releases have no web/), this checkout as /develop/, assembled into
// <repo>/_site, served the way Pages does below a sub-path, then the landing page, the placeholder
// and the notice file are fetched and web/e2e/pages.mjs runs against /develop/ (fresh profile,
// real Pyodide backend, TG119, Stop, allowed hosts).
//
//   node web/scripts/test_pages.mjs [--skip-wheels] [--install] [--latest <dir>] [--develop <dir>]
//        [--out <dir>] [--prefix /pyRadPlan/] [--port 8080] [--no-smoke]
//
//   --latest <dir>  checkout of the release; default: a shallow clone of the newest vX.Y.Z tag
//                   of this repository into <repo>/_pages/latest (needs the tags locally)
//   --skip-wheels   keep the wheels in web/public/pyrp (otherwise rebuilt, needs PYTHON with pip)
//   --install       npm ci before building (CI)
//   --no-smoke      only assemble and fetch, skip the browser test
import { spawn, spawnSync } from "node:child_process";
import { existsSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

const here = dirname(fileURLToPath(import.meta.url));
const { values: args } = parseArgs({
  options: {
    latest: { type: "string" },
    develop: { type: "string", default: resolve(here, "..", "..") },
    out: { type: "string" },
    prefix: { type: "string", default: "/pyRadPlan/" },
    port: { type: "string", default: "8080" },
    "skip-wheels": { type: "boolean", default: false },
    install: { type: "boolean", default: false },
    "no-smoke": { type: "boolean", default: false },
  },
});

const develop = resolve(args.develop);
const out = resolve(args.out ?? join(develop, "_site"));
const web = join(develop, "web");

function run(cmd, cmdArgs, cwd = develop) {
  console.log(`$ ${cmd} ${cmdArgs.join(" ")}`);
  const result = spawnSync(cmd, cmdArgs, { cwd, stdio: "inherit" });
  if (result.status !== 0) throw new Error(`${cmd} failed (exit ${result.status})`);
}

let latest = args.latest && resolve(args.latest);
if (!latest) {
  const tags = spawnSync("git", ["-C", develop, "tag", "--list", "v*", "--sort=-v:refname"], { encoding: "utf8" });
  const tag = tags.stdout.split(/\r?\n/).find((t) => /^v\d+\.\d+\.\d+$/.test(t));
  if (!tag) throw new Error("No release tag vX.Y.Z in this repository (git fetch --tags)");
  latest = join(develop, "_pages", "latest");
  rmSync(latest, { recursive: true, force: true });
  run("git", ["-c", "advice.detachedHead=false", "clone", "-q", "--depth", "1", "--branch", tag, pathToFileURL(develop).href, latest]);
}

run(process.execPath, [
  join(here, "assemble_pages.mjs"),
  "--out",
  out,
  "--build",
  "--coi",
  ...(args.install ? ["--install"] : []),
  ...(args["skip-wheels"] ? ["--skip-wheels"] : []),
  "--channel",
  `latest=${latest}`,
  "--channel",
  `develop=${develop}`,
]);

const prefix = args.prefix.endsWith("/") ? args.prefix : `${args.prefix}/`;
const base = `http://localhost:${args.port}${prefix}`;
const server = spawn(process.execPath, [join(web, "e2e", "pages-server.mjs"), out, `--prefix=${prefix}`, `--port=${args.port}`], { stdio: "inherit" });
let failed = false;
try {
  for (let i = 0; ; i++) {
    try {
      await fetch(base);
      break;
    } catch (err) {
      if (i > 50) throw err;
      await new Promise((r) => setTimeout(r, 200));
    }
  }
  for (const path of ["", "latest/", "develop/", "develop/THIRD_PARTY_NOTICES.txt", "develop/LICENSE.txt"]) {
    const res = await fetch(base + path);
    console.log(`${res.ok ? "ok" : "FAIL"}: ${res.status} ${prefix}${path}`);
    failed ||= !res.ok;
  }
  if (!args["no-smoke"] && existsSync(join(out, "develop", "index.html"))) {
    const smoke = spawnSync(process.execPath, [join(web, "e2e", "pages.mjs"), `${base}develop/`, "--stop"], { cwd: web, stdio: "inherit" });
    failed ||= smoke.status !== 0;
  }
} finally {
  server.kill();
}
console.log(failed ? "Pages test FAILED" : `Pages test passed; the site is in ${out}`);
process.exitCode = failed ? 1 : 0;
