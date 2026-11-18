import { createReadStream, existsSync, mkdirSync, copyFileSync, readFileSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vitest/config";

const here = dirname(fileURLToPath(import.meta.url));

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
  const pkgDir = resolve(here, "node_modules/pyodide");
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
  plugins: [react(), pyodideAssets(), coiServiceWorker()],
  server: { headers: isolationHeaders, port: 5173 },
  preview: { headers: isolationHeaders, port: 4173 },
  worker: { format: "es" },
  build: { target: "es2022", chunkSizeWarningLimit: 1500 },
  test: {
    include: ["test/**/*.test.ts"],
    environment: "node",
  },
});
