import { createReadStream, existsSync, mkdirSync, copyFileSync, readFileSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vitest/config";
import { thirdPartyNotices } from "./scripts/thirdPartyNotices";

const here = dirname(fileURLToPath(import.meta.url));

/** About texts shared with the Qt GUI; the disclaimer must show before the backend has started. */
const about = JSON.parse(readFileSync(resolve(here, "../pyRadPlan/app/about.json"), "utf8"));

const pyodideDir = resolve(here, "node_modules/pyodide");
const notices = thirdPartyNotices({
  overridesDir: resolve(here, "licenses"),
  projectLicense: resolve(here, "../LICENSE"),
  extras: () => {
    const lock = JSON.parse(readFileSync(join(pyodideDir, "pyodide-lock.json"), "utf8")).info ?? {};
    const runtime = "the Pyodide runtime in pyodide-runtime/";
    return [
      { name: "pyodide", packageDir: pyodideDir, usedIn: `pyodide-runtime/ (${PYODIDE_FILES.join(", ")})` },
      {
        name: "CPython",
        version: lock.python,
        license: "PSF-2.0",
        homepage: "https://github.com/python/cpython",
        usedIn: `${runtime} (interpreter in pyodide.asm.wasm, standard library in python_stdlib.zip)`,
        textFile: resolve(here, "licenses/cpython.txt"),
      },
      {
        name: "Emscripten",
        version: String(lock.platform ?? "").replace(/^emscripten_/, "").replaceAll("_", "."),
        license: "MIT OR NCSA",
        homepage: "https://github.com/emscripten-core/emscripten",
        usedIn: `${runtime} (JavaScript runtime and system libraries in pyodide.asm.mjs / .wasm)`,
        textFile: resolve(here, "licenses/emscripten.txt"),
      },
      ...(process.env.PYRADPLAN_COI === "1"
        ? [
            {
              name: "coi-serviceworker",
              license: "MIT",
              homepage: "https://github.com/gzuidhof/coi-serviceworker",
              usedIn: "coi-serviceworker.js (pyRadPlan's rewrite, following its idea and structure)",
              textFile: resolve(here, "licenses/coi-serviceworker.txt"),
            },
          ]
        : []),
    ];
  },
});

/** Headers needed for SharedArrayBuffer (Pyodide interrupt buffer) in dev and preview. */
const isolationHeaders = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Resource-Policy": "same-origin",
};

/** Pyodide runtime files self-hosted from the npm package under `/pyodide-runtime/`. */
const PYODIDE_FILES = [
  "pyodide.mjs",
  "pyodide.asm.mjs",
  "pyodide.asm.wasm",
  "python_stdlib.zip",
  "pyodide-lock.json",
];

const MIME: Record<string, string> = {
  ".mjs": "text/javascript",
  ".js": "text/javascript",
  ".wasm": "application/wasm",
  ".zip": "application/zip",
  ".json": "application/json",
};

function pyodideAssets(): Plugin {
  const pkgDir = pyodideDir;
  return {
    name: "pyradplan-pyodide-assets",
    configureServer(server) {
      server.middlewares.use("/pyodide-runtime/", (req, res, next) => {
        const name = (req.url ?? "").split("?")[0].replace(/^\//, "");
        if (!PYODIDE_FILES.includes(name)) return next();
        const file = join(pkgDir, name);
        const ext = name.slice(name.lastIndexOf("."));
        res.setHeader("Content-Type", MIME[ext] ?? "application/octet-stream");
        res.setHeader("Content-Length", String(statSync(file).size));
        for (const [k, v] of Object.entries(isolationHeaders)) res.setHeader(k, v);
        createReadStream(file).pipe(res);
      });
    },
    writeBundle(options) {
      const outDir = options.dir ?? resolve(here, "dist");
      const target = join(outDir, "pyodide-runtime");
      mkdirSync(target, { recursive: true });
      for (const name of PYODIDE_FILES) {
        const src = join(pkgDir, name);
        if (existsSync(src)) copyFileSync(src, join(target, name));
      }
    },
  };
}

/**
 * Static hosts that cannot set response headers (GitHub Pages): with `PYRADPLAN_COI=1` the build
 * ships `pages/coi-serviceworker.js` and loads it first in `index.html`, which makes the page
 * cross-origin isolated (SharedArrayBuffer, Stop button) through a service worker.
 */
function coiServiceWorker(): Plugin {
  let base = "/";
  return {
    name: "pyradplan-coi-serviceworker",
    apply: "build",
    configResolved(config) {
      base = config.base;
    },
    transformIndexHtml() {
      if (process.env.PYRADPLAN_COI !== "1") return;
      return [{ tag: "script", attrs: { src: `${base}coi-serviceworker.js` }, injectTo: "head-prepend" }];
    },
    generateBundle() {
      if (process.env.PYRADPLAN_COI !== "1") return;
      const source = readFileSync(resolve(here, "pages/coi-serviceworker.js"), "utf8");
      this.emitFile({ type: "asset", fileName: "coi-serviceworker.js", source });
    },
  };
}

export default defineConfig({
  plugins: [react(), pyodideAssets(), coiServiceWorker(), notices.main],
  define: { __PYRADPLAN_ABOUT__: JSON.stringify(about) },
  server: { headers: isolationHeaders, port: 5173 },
  preview: { headers: isolationHeaders, port: 4173 },
  worker: { format: "es", plugins: () => [notices.worker] },
  build: { target: "es2022", chunkSizeWarningLimit: 1500 },
  test: {
    include: ["test/**/*.test.ts"],
    environment: "node",
  },
});
