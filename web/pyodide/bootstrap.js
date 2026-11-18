// Environment-agnostic Pyodide bootstrap for pyRadPlan (browser worker and Node harness).
// Plain ESM without Node-only imports; see README.md for the API.

export const PYODIDE_VERSION = "314.0.8";

export const PYODIDE_CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

// Packages taken from the Pyodide distribution (their dependencies load automatically).
export const PYODIDE_PACKAGES = [
  "micropip",
  "numpy",
  "scipy",
  "h5py",
  "pydantic",
  "typing-extensions",
  "typing-inspection",
  "jinja2",
  "regex",
  "tqdm",
  "platformdirs",
  "contourpy", // DICOM RTSTRUCT export (pyRadPlan.io.dicom); 0.12 MB
];

// Optional Pyodide packages, only needed for plotting (pyRadPlan.visualization, DVH plots).
export const PYODIDE_PLOT_PACKAGES = ["matplotlib"];

// Pure-Python dependencies that are not part of the Pyodide distribution. Installed with
// micropip without dependency resolution; build.py vendors exactly these wheels next to the
// pyRadPlan wheel. Keep in sync with pyproject.toml's bounds.
export const PURE_WHEELS = [
  "array-api-compat==1.15.0",
  "array-api-extra==0.11.2",
  "array-api-strict==2.6.1",
  "flexcache==0.3",
  "flexparser==0.4",
  "ipax==0.13.0",
  "numpydantic==1.10.0",
  "pint==0.25.3",
  "pydantic-settings==2.15.0",
  "pydicom==3.0.2",
  "pymatreader==1.2.3",
  "python-dotenv==1.2.3",
  "types-array-api==1.1.4",
  "xmltodict==1.0.4",
];

// Environment for a single-threaded CPU session: numpy backend, no GPU probing, no numba.
export const PYRADPLAN_ENV = {
  PYRADPLAN_XP_PREFER_GPU: "false",
  PYRADPLAN_XP_PREFERRED_CPU_ARRAY_BACKEND: "numpy",
  PYRADPLAN_XP_JIT_BACKENDS: "",
  MPLBACKEND: "Agg",
};

const WHEEL_DIR = "/tmp/pyrp-wheels";

function joinUrl(base, name) {
  return base.endsWith("/") ? base + name : `${base}/${name}`;
}

// A wheel source is a URL string (http(s) or emfs:) or {name, bytes} with the file name and a
// Uint8Array / ArrayBuffer of its content; byte sources are written into the Emscripten FS.
function stageWheel(pyodide, source) {
  if (typeof source === "string") return source;
  const { name, bytes } = source;
  if (!name || !bytes) throw new Error("A wheel source needs `name` and `bytes`.");
  pyodide.FS.mkdirTree(WHEEL_DIR);
  const path = `${WHEEL_DIR}/${name}`;
  pyodide.FS.writeFile(path, bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes));
  return `emfs:${path}`;
}

/**
 * Install pyRadPlan and its dependencies into a booted Pyodide instance and import it.
 *
 * @param {object} pyodide  The object returned by `loadPyodide()`.
 * @param {object} options
 * @param {string|{name: string, bytes: Uint8Array|ArrayBuffer}} options.wheel  pyRadPlan wheel.
 * @param {string|{name: string, bytes: Uint8Array|ArrayBuffer}} options.shimWheel  SimpleITK shim
 *   wheel.
 * @param {Array<string|{name: string, bytes: Uint8Array|ArrayBuffer}>} [options.pureWheels]
 *   Vendored pure-Python wheels (see manifest.json). Defaults to resolving PURE_WHEELS from PyPI.
 * @param {boolean} [options.plotting=false]  Also load matplotlib.
 * @param {(event: {stage: string, fraction: number, detail?: string}) => void} [options.onProgress]
 * @returns {Promise<{version: string, timings: Record<string, number>}>}
 */
export async function installPyRadPlan(pyodide, options) {
  const { wheel, shimWheel, pureWheels, plotting = false, onProgress = () => {} } = options;
  if (!wheel || !shimWheel) throw new Error("installPyRadPlan needs `wheel` and `shimWheel`.");
  const timings = {};
  const now = () => (globalThis.performance ? performance.now() : Date.now());
  let t = now();
  const lap = (name) => {
    const t1 = now();
    timings[name] = (t1 - t) / 1000;
    t = t1;
  };

  onProgress({ stage: "pyodide-packages", fraction: 0.0 });
  const packages = plotting ? [...PYODIDE_PACKAGES, ...PYODIDE_PLOT_PACKAGES] : PYODIDE_PACKAGES;
  await pyodide.loadPackage(packages, { messageCallback: () => {} });
  lap("pyodide_packages");

  onProgress({ stage: "pure-wheels", fraction: 0.5 });
  const micropip = pyodide.pyimport("micropip");
  const pure = pureWheels ? pureWheels.map((w) => stageWheel(pyodide, w)) : PURE_WHEELS;
  await micropip.install.callKwargs(pure, { deps: false });
  lap("pure_wheels");

  onProgress({ stage: "pyradplan", fraction: 0.8 });
  await micropip.install.callKwargs(
    [stageWheel(pyodide, shimWheel), stageWheel(pyodide, wheel)],
    { deps: false },
  );
  micropip.destroy();
  lap("pyradplan_wheel");

  onProgress({ stage: "import", fraction: 0.9 });
  const env = pyodide.toPy(PYRADPLAN_ENV);
  pyodide.globals.set("_pyrp_env", env);
  const version = pyodide.runPython(`
import os
os.environ.update(_pyrp_env)
del _pyrp_env
import tqdm
tqdm.tqdm.monitor_interval = 0  # its monitor thread cannot start under Pyodide
import SimpleITK
assert getattr(SimpleITK, "IS_PYRADPLAN_SHIM", False)
import pyRadPlan
pyRadPlan.settings.xp.prefer_gpu = False
pyRadPlan.settings.xp.preferred_cpu_array_backend = "numpy"
pyRadPlan.settings.xp.jit_backends = ""
getattr(pyRadPlan, "__version__", "unknown")
`);
  env.destroy();
  lap("import");
  onProgress({ stage: "ready", fraction: 1.0 });
  return { version, timings };
}

/**
 * Resolve the wheel sources listed in `manifest.json` (written by build.py) against its URL.
 *
 * @param {object} manifest  Parsed manifest.json.
 * @param {string} baseUrl  URL of the directory holding manifest.json.
 * @returns {{wheel: string, shimWheel: string, pureWheels: string[]}}
 */
export function wheelsFromManifest(manifest, baseUrl) {
  return {
    wheel: joinUrl(baseUrl, manifest.pyradplan.file),
    shimWheel: joinUrl(baseUrl, manifest.sitk_shim.file),
    pureWheels: manifest.pure_wheels.map((w) => joinUrl(baseUrl, w.file)),
  };
}
