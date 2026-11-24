// Runs gen_mock_fixtures.py, which writes src/api/mock/fixtures/*.json (not committed).
// PYTHON selects the interpreter; its environment needs pyRadPlan's dependencies.
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const python = process.env.PYTHON ?? (process.platform === "win32" ? "python" : "python3");
const script = fileURLToPath(new URL("./gen_mock_fixtures.py", import.meta.url));

const result = spawnSync(python, [script], { stdio: "inherit" });
if (result.status !== 0) {
  const reason = result.error ? `could not start it (${result.error.message})` : `it exited with ${result.status}`;
  console.error(
    `\nGenerating the mock fixtures failed: \`${python}\` ${reason}.\n` +
      "The mock backend's fixtures are generated from pyRadPlan.app, so the Python environment needs\n" +
      "pyRadPlan's dependencies (for example `pip install -e .` in the repository root).\n" +
      "Set the PYTHON environment variable to the interpreter to use, e.g. PYTHON=/path/to/venv/bin/python.",
  );
  process.exit(result.status || 1);
}
