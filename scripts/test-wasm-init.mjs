// Smoke-test the final wasm-bindgen package, including wasm-opt's output.
// No model weights or HF credentials are needed.
// Usage: node scripts/test-wasm-init.mjs [path/to/pkg]
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";

const pkg = process.argv[2] ?? "crates/pocket-tts/pkg";
const js = await readFile(path.join(pkg, "pocket_tts.js"));
const bytes = await readFile(path.join(pkg, "pocket_tts_bg.wasm"));
// Load the web-target ES module without requiring a package.json in pkg.
const { default: init, WasmTTSModel } = await import(
  `data:text/javascript;base64,${js.toString("base64")}`
);

await init({ module_or_path: bytes });
const model = new WasmTTSModel();
try {
  assert.equal(model.is_ready(), false);
  assert.equal(model.sample_rate, 24000);
  // Exercise a Rust -> JS reference, not just numeric exports. Old Binaryen
  // versions export the function table as __wbindgen_externrefs by mistake.
  assert.throws(
    () => model.start_stream("Smoke test"),
    (error) => error === "Model not loaded. Call load_from_buffer first.",
  );
} finally {
  model.free();
}
console.log("WASM initialization smoke test passed.");
