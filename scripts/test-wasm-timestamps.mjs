// Model-backed WASM acceptance, using local files described by an upstream
// fixture manifest. Missing assets are fatal. No credentials/network required.
// node scripts/test-wasm-timestamps.mjs <pkg-dir> <manifest.json>
// Provenance only: node scripts/test-wasm-timestamps.mjs --validate-fixture <manifest.json>
// Schema 2 YAML parsing needs PyYAML; POCKET_TTS_REFERENCE_PYTHON selects its interpreter.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";

const validationOnly = process.argv[2] === "--validate-fixture";
const [pkg, manifestPath] = process.argv.slice(2);
assert(manifestPath && (validationOnly || pkg), "Pass pkg-dir and model manifest.json, or --validate-fixture manifest.json");
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
const fixturePath = (value) => path.resolve(path.dirname(manifestPath), value);
if (manifest.upstream_commit) {
  assert(["3dbee45d343d7dddd0d105468d17f8dcba14db3e", "9a1b48922deb44cd4a805fde279eb33b71c1a473", "c91f50f38c6b914d49dbdff51b8d322edf843db2"].includes(manifest.upstream_commit), "Unknown reference revision");
}
const readAsset = async (key) => {
  const asset = manifest[key];
  const bytes = await readFile(fixturePath(asset.path));
  if (manifest.schema_version >= 2) assert(asset.sha256, `Missing ${key} SHA-256`);
  if (asset.sha256) assert.equal(createHash("sha256").update(bytes).digest("hex"), asset.sha256, `${key} asset hash mismatch`);
  return bytes;
};
if (manifest.schema_version >= 2) {
  assert([manifest.config.weights_path, manifest.config.weights_path_without_voice_cloning].includes(manifest.weights.uri), "config/asset mismatch: weights");
  assert.equal(manifest.tokenizer.uri, manifest.config.flow_lm.lookup_table.tokenizer_path, "config/asset mismatch: tokenizer");
  for (const [name, digest] of Object.entries(manifest.files)) {
    assert.equal(createHash("sha256").update(await readFile(fixturePath(name))).digest("hex"), digest, name);
  }
}
const configPath = manifest.fixture_config_path ?? manifest.config_path;
const selectedConfigPath = configPath ? fixturePath(configPath) : path.resolve(`crates/pocket-tts/config/${manifest.language}.yaml`);
let yaml = await readFile(selectedConfigPath, "utf8");
if (manifest.schema_version >= 2) {
  const selectedDigest = Object.entries(manifest.files).find(([name]) => fixturePath(name) === selectedConfigPath)?.[1];
  assert(typeof selectedDigest === "string" && /^[0-9a-f]{64}$/.test(selectedDigest), "Selected config requires its own manifest SHA-256");
  assert.equal(createHash("sha256").update(yaml).digest("hex"), selectedDigest, "Selected config hash mismatch");
  let normative;
  try {
    normative = JSON.parse(execFileSync(process.env.POCKET_TTS_REFERENCE_PYTHON ?? "python3", ["-c",
      "import json, sys; import yaml; json.dump(yaml.safe_load(sys.stdin), sys.stdout)"], { input: yaml, encoding: "utf8" }));
  } catch (error) {
    throw new Error(`Selected config YAML parsing requires Python with PyYAML. Set POCKET_TTS_REFERENCE_PYTHON to the pinned reference interpreter. ${error.stderr?.toString() ?? error.message}`, { cause: error });
  }
  assert(normative && typeof normative === "object" && !Array.isArray(normative), "Selected config must be a YAML mapping");
  // The only documented Rust-only config metadata in these fixtures.
  delete normative.timestamp_heads;
  assert.deepEqual(normative, manifest.config, "Selected config/asset identity mismatch");
}
if ((manifest.schema_version ?? 1) < 2 && !/^timestamp_heads:/m.test(yaml)
    && ["english", "english_2m_2026-10", "english_7m_2026-10", "english_18m_2026-10"].includes(manifest.language)) {
  const archived = path.join(path.dirname(fileURLToPath(import.meta.url)), "../assets/upstream-2026-11-09/baseline", manifest.language);
  const source = await readFile(path.join(archived, "config.yaml"), "utf8");
  assert.equal(createHash("sha256").update(yaml).digest("hex"), createHash("sha256").update(source).digest("hex"), "legacy config/calibration mismatch");
  yaml = await readFile(path.join(archived, "rust-config.yaml"), "utf8");
}
if (validationOnly) {
  await Promise.all(["weights", "tokenizer", "voice_state"].map(readAsset));
  console.log("Fixture config and assets validated.");
} else {
const js = await readFile(path.join(pkg, "pocket_tts.js"));
const { default: init, WasmTTSModel } = await import(`data:text/javascript;base64,${js.toString("base64")}`);
await init({ module_or_path: await readFile(path.join(pkg, "pocket_tts_bg.wasm")) });
yaml = yaml.replace(/^default_temperature:.*$/m, "default_temperature: 0.0");
const model = new WasmTTSModel();
const text = "The quick brown fox jumps over the lazy dog.";
const expected = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"];
try {
  model.load_from_buffer(new TextEncoder().encode(yaml),
    await readAsset("weights"), await readAsset("tokenizer"), false);
  model.load_voice_from_safetensors(await readAsset("voice_state"));
  const ordinary = model.start_stream(text);
  const audio = [];
  try {
    for (;;) {
      const chunk = ordinary.next_chunk_min_samples(4000);
      if (chunk == null) break;
      audio.push(...chunk);
    }
  } finally { ordinary.free(); }
  const stream = model.start_stream_with_timestamps(text);
  let offset = 0;
  let open = null;
  let lastEnd = 0;
  const words = [];
  let batchCount = 0;
  try {
    for (;;) {
      const batch = stream.next_batch(4000);
      if (batch == null) break;
      batchCount++;
      assert(batch.audio instanceof Float32Array);
      assert(Number.isFinite(batch.compute_ms));
      if (batch.audio.length) {
        assert.equal(batch.start_time, offset / model.sample_rate);
        assert.equal(batch.end_time, (offset + batch.audio.length) / model.sample_rate);
        assert(batch.chunks_merged > 0);
      } else {
        assert.equal(batch.start_time, null);
        assert.equal(batch.end_time, null);
        assert(batch.events.length > 0);
      }
      for (const sample of batch.audio) {
        assert.equal(sample, audio[offset++], `PCM sample ${offset}`);
      }
      for (const event of batch.events) {
        assert.equal(event.word, expected[event.word_index]);
        if (event.kind === "word_start") {
          assert.equal(open, null);
          assert(event.start_time >= lastEnd);
          open = event;
        } else {
          assert.equal(event.kind, "word_end");
          assert.equal(event.word_index, open.word_index);
          assert.equal(event.start_time, open.start_time);
          assert(event.end_time > event.start_time);
          assert(event.end_time <= offset / model.sample_rate);
          words.push(event.word);
          lastEnd = event.end_time;
          open = null;
        }
      }
    }
    assert.equal(stream.next_batch(1), undefined);
  } finally { stream.free(); }
  assert.equal(offset, audio.length);
  assert.equal(open, null);
  assert.deepEqual(words, expected);
  const cancelled = model.start_stream_with_timestamps(text);
  cancelled.next_batch(1);
  cancelled.free();
  const restarted = model.start_stream_with_timestamps(text);
  try {
    assert.equal(restarted.next_batch(1).audio[0], audio[0]);
  } finally { restarted.free(); }
  console.log(JSON.stringify({ samples: offset, batchCount, words, audioParity: "exact" }));
} finally { model.free(); }
}
