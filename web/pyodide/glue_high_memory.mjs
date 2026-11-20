// Regression test of the worker bridge (web/src/api/pyodide/glue.py) for wasm memory above
// 2 GiB, where Pyodide's own buffer conversions copy from a wrong address or raise. Runs the
// bridge against a stub `pyRadPlan.app.dispatch` that echoes the request buffer, once with a
// small heap and once after 2.4 GB of ballast, and compares the bytes that come back.
// Usage: npm run test:glue   (needs numpy in .cache/packages or network access)
import { mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadPyodide } from "pyodide";

const here = dirname(fileURLToPath(import.meta.url));
const cacheDir = join(here, ".cache", "packages");
mkdirSync(cacheDir, { recursive: true });
const glueSource = readFileSync(join(here, "..", "src", "api", "pyodide", "glue.py"), "utf8");

const pyodide = await loadPyodide({ packageCacheDir: cacheDir });
await pyodide.loadPackage("numpy");
pyodide.runPython(`
import sys, types
import numpy as np

_keep = []

def _dispatch(request, emit=None):
    # Echo through fresh allocations, so both directions cross the current top of the heap.
    data = np.frombuffer(request["buffers"][0], dtype=np.float32)
    out = data.copy()
    _keep.append(out)
    address = out.__array_interface__["data"][0]
    return {"id": request["id"], "ok": True, "result": {"address": address}, "buffers": [memoryview(out).cast("B")]}

app = types.ModuleType("pyRadPlan.app")
app.dispatch = _dispatch
core = types.ModuleType("pyRadPlan.core")
core.ProgressReporter = type("ProgressReporter", (), {})
package = types.ModuleType("pyRadPlan")
package.app, package.core = app, core
sys.modules.update({"pyRadPlan": package, "pyRadPlan.app": app, "pyRadPlan.core": core})
`);
const glue = pyodide.toPy({});
pyodide.runPython(glueSource, { globals: glue });
const dispatch = glue.get("web_dispatch");

const COUNT = 2_000_000;
let failures = 0;
let highest = 0;

async function roundTrips(label, rounds) {
  for (let i = 0; i < rounds; i++) {
    const values = new Float32Array(COUNT);
    for (let k = 0; k < COUNT; k++) values[k] = i + 1 + (k % 97) / 128;
    const parts = await dispatch(JSON.stringify({ id: i, method: "echo" }), [new Uint8Array(values.buffer)], () => {});
    const [json, bytes] = Array.isArray(parts) ? parts : Array.from(parts);
    const address = JSON.parse(json).result.address;
    highest = Math.max(highest, address);
    const back = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    let same = back.length === COUNT;
    for (let k = 0; same && k < COUNT; k++) same = back[k] === values[k];
    if (!same) {
      failures += 1;
      console.log(`FAIL ${label} round ${i}: buffer at ${(address / 2 ** 30).toFixed(2)} GiB came back changed`);
    }
  }
}

await roundTrips("low memory", 3);
pyodide.runPython("_ballast = [np.ones(50_000_000) for _ in range(6)]");
await roundTrips("above 2 GiB", 6);

const empty = await dispatch(JSON.stringify({ id: 99, method: "echo" }), [new Uint8Array(0)], () => {});
if ((Array.isArray(empty) ? empty : Array.from(empty))[1].byteLength !== 0) {
  failures += 1;
  console.log("FAIL empty buffer");
}

if (highest < 2 ** 31) {
  failures += 1;
  console.log(`FAIL the test never reached memory above 2 GiB (highest ${(highest / 2 ** 30).toFixed(2)} GiB)`);
}
console.log(failures ? `${failures} failure(s)` : `ok: buffers round-trip up to ${(highest / 2 ** 30).toFixed(2)} GiB`);
process.exit(failures ? 1 : 0);
