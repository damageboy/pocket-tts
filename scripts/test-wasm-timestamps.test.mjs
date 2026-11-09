// Focused provenance checks: real YAML parsing and filesystem hashes, no models.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const script = fileURLToPath(new URL("./test-wasm-timestamps.mjs", import.meta.url));
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
const source = "weights_path: hf://repo/weights@fixed\nweights_path_without_voice_cloning: hf://repo/open@fixed\nflow_lm:\n  lookup_table:\n    tokenizer_path: hf://repo/tokenizer@fixed\ndefault_temperature: 0.3\n";

async function fixture(callback) {
  const directory = await mkdtemp(path.join(os.tmpdir(), "pocket-fixture-config-"));
  const config = {
    weights_path: "hf://repo/weights@fixed",
    weights_path_without_voice_cloning: "hf://repo/open@fixed",
    flow_lm: { lookup_table: { tokenizer_path: "hf://repo/tokenizer@fixed" } },
    default_temperature: 0.3,
  };
  const manifest = {
    schema_version: 2,
    upstream_commit: "c91f50f38c6b914d49dbdff51b8d322edf843db2",
    config,
    config_path: "config.yaml",
    fixture_config_path: "selected.yaml",
    files: { "config.yaml": hash(source) },
  };
  await writeFile(path.join(directory, "config.yaml"), source);
  for (const [key, uri] of [["weights", config.weights_path], ["tokenizer", config.flow_lm.lookup_table.tokenizer_path], ["voice_state", "hf://repo/voice@fixed"]]) {
    const bytes = Buffer.from(key);
    await writeFile(path.join(directory, key), bytes);
    manifest[key] = { path: key, uri, sha256: hash(bytes) };
  }
  const run = async (environment = {}) => {
    await writeFile(path.join(directory, "manifest.json"), JSON.stringify(manifest));
    return spawnSync(process.execPath, [script, "--validate-fixture", path.join(directory, "manifest.json")], { encoding: "utf8", env: { ...process.env, ...environment } });
  };
  const selected = async (yaml, authenticate = true) => {
    await writeFile(path.join(directory, "selected.yaml"), yaml);
    if (authenticate) manifest.files["selected.yaml"] = hash(yaml);
  };
  try { await callback({ directory, manifest, run, selected }); }
  finally { await rm(directory, { recursive: true, force: true }); }
}

test("rejects an unauthenticated selected config while all existing hashes remain valid", async () => {
  await fixture(async ({ manifest, directory, run, selected }) => {
    await selected(source.replaceAll("@fixed", "@other"), false);
    for (const [name, digest] of Object.entries(manifest.files)) assert.equal(hash(await readFile(path.join(directory, name))), digest);
    for (const key of ["weights", "tokenizer", "voice_state"]) assert.equal(hash(await readFile(path.join(directory, manifest[key].path))), manifest[key].sha256);
    const result = await run();
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /selected config.*SHA-256/i);
  });
});

test("rejects a hashed selected config whose declared weights disagree", async () => {
  await fixture(async ({ run, selected }) => {
    await selected(source.replaceAll("weights@fixed", "weights@other").replaceAll("open@fixed", "open@other"));
    const result = await run();
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /selected config.*mismatch/i);
  });
});

test("rejects selected temperature drift before the zero-temperature override", async () => {
  await fixture(async ({ run, selected }) => {
    await selected(source.replace("0.3", "0.8"));
    const result = await run();
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /selected config.*mismatch/i);
  });
});

test("accepts authenticated normative config with documented Rust calibration metadata", async () => {
  await fixture(async ({ run, selected }) => {
    await selected(source + "timestamp_heads: [[3, 8], [2, 0]]\n");
    const result = await run();
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /Fixture config and assets validated/);
  });
});

test("rejects calibration changes against the selected config digest", async () => {
  await fixture(async ({ directory, run, selected }) => {
    await selected(source + "timestamp_heads: [[3, 8]]\n");
    await writeFile(path.join(directory, "selected.yaml"), source + "timestamp_heads: [[0, 0]]\n");
    const result = await run();
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /selected.yaml|selected config.*hash mismatch/i);
  });
});


test("reports PyYAML absence with the reference interpreter setting", async () => {
  await fixture(async ({ directory, run, selected }) => {
    await selected(source);
    const interpreter = path.join(directory, "python-without-site-packages");
    await writeFile(interpreter, "#!/usr/bin/env python3\nimport os, sys\nos.execv(sys.executable, [sys.executable, '-S', *sys.argv[1:]])\n", { mode: 0o755 });
    const result = await run({ POCKET_TTS_REFERENCE_PYTHON: interpreter });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /PyYAML/);
    assert.match(result.stderr, /POCKET_TTS_REFERENCE_PYTHON/);
    assert.match(result.stderr, /No module named 'yaml'/);
  });
});

test("rejects undocumented metadata instead of excluding arbitrary config keys", async () => {
  await fixture(async ({ run, selected }) => {
    await selected(source + "timestamp_untrusted: true\n");
    const result = await run();
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /selected config.*mismatch/i);
  });
});
