// Headless smoke test of pyRadPlan.app.dispatch under Pyodide in Node.
//
//   node app_smoke.mjs [--strict] [--only v0|v1] [--table app_smoke.table.js] [--verbose]
//
// --only runs the entries of one protocol version plus the `setup` entries they build on.
//
// Installs the wheels of web/public/pyrp (npm run build) and runs the method table in order,
// printing ok/error and timing per entry. Exits 1 if a v0 entry (any entry with --strict) fails.

import { existsSync, readFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

import { loadPyodide } from "pyodide";

import { installPyRadPlan } from "./bootstrap.js";

const here = dirname(fileURLToPath(import.meta.url));
const { values: args } = parseArgs({
  options: {
    strict: { type: "boolean", default: false },
    only: { type: "string" },
    table: { type: "string", default: "app_smoke.table.js" },
    verbose: { type: "boolean", default: false },
  },
});

const artifactDir = join(here, "..", "public", "pyrp");
const manifest = JSON.parse(readFileSync(join(artifactDir, "manifest.json"), "utf8"));
const source = (file) => ({ name: basename(file), bytes: readFileSync(join(artifactDir, file)) });
const table = (await import(pathToFileURL(resolve(here, args.table)).href)).default;

const t0 = performance.now();
const pyodide = await loadPyodide({
  packageCacheDir: join(here, ".cache", "packages"),
  stdout: (line) => args.verbose && console.log(line),
  stderr: (line) => args.verbose && console.error(line),
});
const install = await installPyRadPlan(pyodide, {
  wheel: source(manifest.pyradplan.file),
  shimWheel: source(manifest.sitk_shim.file),
  pureWheels: manifest.pure_wheels.map((w) => source(w.file)),
});
console.log(
  `pyRadPlan ${install.version} (wheel sha256 ${manifest.pyradplan.sha256.slice(0, 12)}) ` +
    `ready in ${((performance.now() - t0) / 1000).toFixed(1)} s`,
);

pyodide.runPython(String.raw`
import json, time
from collections import Counter
import numpy as np
from pyRadPlan.app import dispatch

def _smoke_params(obj, buffers):
    if isinstance(obj, dict):
        if set(obj) == {"$fsfile"}:
            with open(obj["$fsfile"], "rb") as fh:
                data = np.frombuffer(fh.read(), dtype=np.uint8)
            buffers.append(memoryview(data))
            return {"$array": len(buffers) - 1, "dtype": "uint8", "shape": [int(data.size)]}
        return {k: _smoke_params(v, buffers) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_smoke_params(v, buffers) for v in obj]
    return obj

def _smoke_call(request_json):
    request = json.loads(request_json)
    buffers = []
    request["params"] = _smoke_params(request.get("params") or {}, buffers)
    request["buffers"] = buffers
    events = Counter()
    t = time.perf_counter()
    response = dispatch(request, emit=lambda e: events.update([e.get("event")]))
    response["seconds"] = time.perf_counter() - t
    response["buffers"] = [{"nbytes": memoryview(b).nbytes} for b in response.get("buffers") or []]
    response["events"] = dict(events)
    return json.dumps(response, default=repr)

def _smoke_python(code):
    t = time.perf_counter()
    try:
        scope = {}
        *body, last = code.strip().splitlines()
        exec("\n".join(body), scope)
        response = {"ok": True, "result": eval(last, scope)}
    except Exception as exc:
        response = {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
    response["seconds"] = time.perf_counter() - t
    return json.dumps(response, default=repr)
`);
const call = pyodide.globals.get("_smoke_call");
const runPython = pyodide.globals.get("_smoke_python");

pyodide.FS.mkdirTree("/tmp/smoke");
const stage = (path) => {
  const target = `/tmp/smoke/${basename(path)}`;
  pyodide.FS.writeFile(target, readFileSync(join(here, path)));
  return target;
};
const resolveParams = (value) => {
  if (Array.isArray(value)) return value.map(resolveParams);
  if (value && typeof value === "object") {
    if (Object.keys(value).length === 1 && "$file" in value) return { $fsfile: stage(value.$file) };
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, resolveParams(v)]));
  }
  return value;
};

const ctx = { session: null };
const rows = [];
let id = 0;
for (const entry of table) {
  if (args.only && `v${entry.v}` !== args.only && !entry.setup) continue;
  const row = { v: entry.v, label: entry.method ?? "python", note: entry.note ?? "" };
  rows.push(row);
  if (entry.needs && !existsSync(join(here, entry.needs))) {
    row.status = "skip";
    row.detail = `needs ${entry.needs}`;
    continue;
  }
  let response;
  if (entry.python) {
    (entry.files ?? []).forEach(stage);
    response = JSON.parse(runPython(typeof entry.python === "function" ? entry.python(ctx) : entry.python));
  } else {
    const params = typeof entry.params === "function" ? entry.params(ctx) : (entry.params ?? {});
    const request = { id: ++id, method: entry.method, params: resolveParams(params) };
    if (entry.session !== false) request.session = ctx.session;
    response = JSON.parse(call(JSON.stringify(request)));
  }
  row.seconds = response.seconds;
  if (entry.expectError) {
    // The entry must fail with this error type (and code).
    const { type, code } = entry.expectError;
    const matches = !response.ok && response.error?.type === type && (!code || response.error?.code === code);
    row.status = matches ? "ok" : "error";
    row.detail = response.ok ? `expected ${type}, got a result` : `${response.error?.type}${response.error?.code ? `/${response.error.code}` : ""}: ${response.error?.message}`.slice(0, 160);
    continue;
  }
  if (!response.ok) {
    row.status = "error";
    row.detail = `${response.error?.type}: ${response.error?.message}`.slice(0, 160);
    continue;
  }
  let verdict;
  try {
    verdict = entry.expect ? entry.expect(response.result, response, ctx) : true;
  } catch (err) {
    verdict = `expectation threw ${err}`;
  }
  if (verdict === true) {
    row.status = "ok";
    entry.after?.(response.result, ctx);
    const events = Object.entries(response.events ?? {})
      .map(([k, n]) => `${k}:${n}`)
      .join(" ");
    row.detail = events ? `events ${events}` : "";
  } else {
    row.status = "fail";
    row.detail = String(typeof verdict === "object" ? JSON.stringify(verdict) : verdict).slice(
      0,
      160,
    );
  }
}

const required = (row) => args.strict || row.v === 0;
const bad = (row) => row.status === "error" || row.status === "fail";
for (const row of rows) {
  const time = row.seconds === undefined ? "       " : `${row.seconds.toFixed(2).padStart(6)}s`;
  const flag = bad(row) && !required(row) ? " (pending)" : "";
  const note = row.note ? `  [${row.note}]` : "";
  console.log(`${row.status.padEnd(5)} v${row.v} ${row.label.padEnd(24)} ${time}  ${row.detail}${flag}${note}`);
}
const failed = rows.filter((r) => bad(r) && required(r));
const pending = rows.filter((r) => bad(r) && !required(r));
const count = (s) => rows.filter((r) => r.status === s).length;
console.log(
  `\n${count("ok")} ok, ${failed.length} failed, ${pending.length} pending (v1), ` +
    `${count("skip")} skipped`,
);
process.exit(failed.length ? 1 : 0);
