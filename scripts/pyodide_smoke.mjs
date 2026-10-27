// Load a pyayay Pyodide wheel in Node and run smoke_render_frame.py.
// Usage: node scripts/pyodide_smoke.mjs dist-pyodide/pyayay-*.whl
// Needs `npm install pyodide@0.29.3` in the current directory.
import { loadPyodide } from "pyodide";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const wheel = resolve(process.argv[2]);
const here = dirname(fileURLToPath(import.meta.url));
const pyodide = await loadPyodide({ stdout: console.log, stderr: console.error });
console.error(`Pyodide ${pyodide.version}, wheel ${wheel}`);
await pyodide.loadPackage(wheel, { messageCallback: console.error });
pyodide.runPython(readFileSync(resolve(here, "smoke_render_frame.py"), "utf8"));
