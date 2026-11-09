import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const output = process.argv[2];
if (!output || process.argv.length !== 3) {
  throw new Error("Usage: bun scripts/generate-wasm-metadata.ts OUTPUT_PATH");
}
const manifest = Bun.TOML.parse(readFileSync(resolve(import.meta.dir, "../Cargo.toml"), "utf8")) as {
  workspace: { package: { version: string } };
};
writeFileSync(output, JSON.stringify({
  name: "pocket-tts",
  version: manifest.workspace.package.version,
  type: "module",
  main: "pocket_tts.js",
  types: "pocket_tts.d.ts",
  files: ["pocket_tts.js", "pocket_tts_bg.wasm", "pocket_tts.d.ts", "pocket_tts_bg.wasm.d.ts", "models.json", "THIRD_PARTY_NOTICES.md"],
}, null, 2) + "\n");
