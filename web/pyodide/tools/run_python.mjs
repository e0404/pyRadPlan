// Run the native Python ($PYTHON, default "python") with the repository root on PYTHONPATH,
// so that the checkout's pyRadPlan is imported rather than an installed one.
import { spawnSync } from "node:child_process";
import { delimiter, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repo = join(here, "..", "..", "..");
const env = { ...process.env };
env.PYTHONPATH = env.PYTHONPATH ? `${repo}${delimiter}${env.PYTHONPATH}` : repo;
const python = env.PYTHON || "python";
const result = spawnSync(python, process.argv.slice(2), {
  stdio: "inherit",
  env,
  cwd: join(here, ".."),
});
if (result.error) {
  console.error(`Could not run ${python}: ${result.error.message} (set PYTHON)`);
  process.exit(1);
}
process.exit(result.status ?? 1);
