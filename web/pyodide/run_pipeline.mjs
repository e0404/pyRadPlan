// Headless Node harness: boot Pyodide, install pyRadPlan via bootstrap.js, run pipeline.py.
//
//   node run_pipeline.mjs [--mode photons|protons] [--preset default|demo|fast]
//                         [--solver scipy|ipax] [--max-iter N] [--no-dij-sampling]
//                         [--plotting] [--install-only] [--cache-dir DIR] [--tag SUFFIX]
//
// Writes results/pyodide_<mode>_<preset><tag>.npz and .json (timings, sizes, memory).

import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { loadPyodide } from "pyodide";

import { createServer } from "node:http";

import {
  installPyRadPlan,
  PYODIDE_VERSION,
  PYRADPLAN_ENV,
  wheelsFromManifest,
} from "./bootstrap.js";

const here = dirname(fileURLToPath(import.meta.url));
const { values: args } = parseArgs({
  options: {
    mode: { type: "string", default: "photons" },
    preset: { type: "string", default: "fast" },
    solver: { type: "string", default: "scipy" },
    "max-iter": { type: "string", default: "200" },
    plotting: { type: "boolean", default: false },
    verbose: { type: "boolean", default: false },
    "install-only": { type: "boolean", default: false },
    "no-dij-sampling": { type: "boolean", default: false },
    tag: { type: "string", default: "" },
    "cache-dir": { type: "string" },
    "via-http": { type: "boolean", default: false },
  },
});

const artifactDir = join(here, "..", "public", "pyrp");
const manifestPath = join(artifactDir, "manifest.json");
if (!existsSync(manifestPath)) {
  console.error(`Missing ${manifestPath}; run \`npm run build\` first.`);
  process.exit(1);
}
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
const source = (file) => ({
  name: file.split("/").pop(),
  bytes: readFileSync(join(artifactDir, file)),
});

const cacheDir = args["cache-dir"] ?? join(here, ".cache", "packages");
mkdirSync(cacheDir, { recursive: true });
const cachedBefore = new Set(readdirSync(cacheDir));

const report = { pyodide_version: PYODIDE_VERSION, args, node: process.version };
const t0 = performance.now();
const pyodide = await loadPyodide({
  packageCacheDir: cacheDir,
  env: { ...PYRADPLAN_ENV, HOME: "/home/pyodide" },
  stdout: (line) => console.log(line),
  stderr: (line) => {
    if (args.verbose || !/(\d+%\||it\/s|r\/s|b\/s)/.test(line)) console.error(line);
  },
});
report.boot_s = (performance.now() - t0) / 1000;
console.log(`Pyodide ${pyodide.version} booted in ${report.boot_s.toFixed(2)} s`);

let sources = {
  wheel: source(manifest.pyradplan.file),
  shimWheel: source(manifest.sitk_shim.file),
  pureWheels: manifest.pure_wheels.map((w) => source(w.file)),
};
let server;
if (args["via-http"]) {
  // Serve the artifacts like Vite does, to exercise the URL path the browser worker uses.
  server = createServer((req, res) => {
    const path = join(artifactDir, decodeURIComponent(new URL(req.url, "http://x").pathname));
    if (!path.startsWith(artifactDir) || !existsSync(path)) {
      res.writeHead(404).end();
      return;
    }
    res.writeHead(200, { "Content-Type": "application/octet-stream" }).end(readFileSync(path));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const base = `http://127.0.0.1:${server.address().port}/`;
  const served = JSON.parse(await (await fetch(`${base}manifest.json`)).text());
  sources = wheelsFromManifest(served, base);
  console.log(`installing from ${base}`);
}
const install = await installPyRadPlan(pyodide, {
  ...sources,
  plotting: args.plotting,
  onProgress: ({ stage }) => args.verbose && console.log(`[install] ${stage}`),
});
server?.close();
report.install = install;
report.install_s = (performance.now() - t0) / 1000 - report.boot_s;
console.log(
  `pyRadPlan ${install.version} installed in ${report.install_s.toFixed(2)} s`,
  install.timings,
);

const lock = JSON.parse(
  readFileSync(join(here, "node_modules", "pyodide", "pyodide-lock.json"), "utf8"),
);
const fileOf = new Map(Object.values(lock.packages).map((p) => [p.name, p.file_name]));
const distFiles = Object.keys(pyodide.loadedPackages)
  .map((name) => [name, fileOf.get(name)])
  .filter(([, file]) => file && existsSync(join(cacheDir, file)));
const packagesMb = Object.fromEntries(
  distFiles
    .map(([name, file]) => [name, statSync(join(cacheDir, file)).size / 1e6])
    .sort((a, b) => b[1] - a[1]),
);
const mb = (bytes) => bytes / 1e6;
report.download = {
  cache_was_warm: distFiles.every(([, file]) => cachedBefore.has(file)),
  pyodide_runtime_mb: mb(
    ["pyodide.asm.wasm", "python_stdlib.zip", "pyodide.asm.mjs", "pyodide-lock.json"]
      .map((f) => statSync(join(here, "node_modules", "pyodide", f)).size)
      .reduce((a, b) => a + b, 0),
  ),
  distribution_packages_mb: Object.values(packagesMb).reduce((a, b) => a + b, 0),
  distribution_packages: packagesMb,
  pyradplan_wheel_mb: mb(manifest.pyradplan.size),
  sitk_shim_wheel_mb: mb(manifest.sitk_shim.size),
  pure_wheels_mb: mb(manifest.pure_wheels.reduce((a, w) => a + w.size, 0)),
  pure_wheels: Object.fromEntries(manifest.pure_wheels.map((w) => [w.requirement, mb(w.size)])),
};

if (args["install-only"]) {
  console.log(JSON.stringify(report, null, 2));
  process.exit(0);
}

pyodide.FS.mkdirTree("/home/pyodide/harness");
pyodide.FS.writeFile("/home/pyodide/harness/pipeline.py", readFileSync(join(here, "pipeline.py")));
pyodide.globals.set(
  "_args",
  pyodide.toPy({
    mode: args.mode,
    preset: args.preset,
    solver: args.solver,
    max_iter: Number(args["max-iter"]),
    dij_sampling: !args["no-dij-sampling"],
  }),
);
const t1 = performance.now();
const infoJson = pyodide.runPython(`
import json, sys
sys.path.insert(0, "/home/pyodide/harness")
import pipeline
_result = pipeline.run_pipeline(**_args)
pipeline.save_result(_result, "/home/pyodide/harness/result.npz")
json.dumps(_result["info"])
`);
report.pipeline_wall_s = (performance.now() - t1) / 1000;
report.info = JSON.parse(infoJson);
report.wasm_memory_mb = pyodide._module.HEAPU8.buffer.byteLength / 2 ** 20;
report.node_rss_mb = process.memoryUsage().rss / 2 ** 20;

const outDir = join(here, "results");
mkdirSync(outDir, { recursive: true });
const stem = join(outDir, `pyodide_${args.mode}_${args.preset}${args.tag}`);
writeFileSync(`${stem}.npz`, pyodide.FS.readFile("/home/pyodide/harness/result.npz"));
writeFileSync(`${stem}.json`, JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
console.log(`wrote ${stem}.npz`);
