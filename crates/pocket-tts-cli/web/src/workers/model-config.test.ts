/// <reference types="node" />

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { configAssets, selectConfig, hfDownloadUrl, presetVoiceUrl } from "./model-config";

const root = new URL("../../../../pocket-tts/config/", import.meta.url);
const configs = Object.fromEntries(readdirSync(root).filter(name => name.endsWith(".yaml"))
  .map(name => [name.slice(0, -5), readFileSync(new URL(name, root), "utf8")]));

describe("canonical model configs", () => {
  it("selects all 18 released configs without changing their YAML", () => {
    assert.equal(Object.keys(configs).length, 18);
    for (const [name, yaml] of Object.entries(configs)) {
      assert.equal(selectConfig(configs, name).yaml, yaml);
      const assets = configAssets(yaml);
      assert.match(assets.weightsUrl, /\/resolve\/[a-f0-9]{40}\//);
      assert.match(assets.tokenizerUrl, /\/resolve\/[a-f0-9]{40}\/.*tokenizer\.json$/);
    }
  });
  it("preserves hyphenated legacy names and accepts old underscore aliases", () => {
    for (const date of ["01", "04", "09"]) {
      assert.equal(selectConfig(configs, `english_2026_${date}`).name, `english_2026-${date}`);
      assert.equal(selectConfig(configs, `english_2026-${date}`).yaml, configs[`english_2026-${date}`]);
    }
    assert.match(selectConfig(configs, "english_2026-01").yaml, /insert_bos_before_voice: false/);
    assert.match(selectConfig(configs, "english_2026-01").yaml, /inner_dim: 512/);
    assert.match(selectConfig(configs, "english").yaml, /default_temperature: 0.3/);
  });
  it("rejects unknown configs instead of falling back to English", () => {
    assert.throws(() => selectConfig(configs, "not-released"), /Unknown model config/);
    assert.throws(() => selectConfig(configs, "toString"), /Unknown model config/);
  });
  it("extracts only declared download scalars and requires valid pinned HF URIs", () => {
    assert.equal(hfDownloadUrl("hf://owner/repo/path/file.json@abc123"),
      "https://huggingface.co/owner/repo/resolve/abc123/path/file.json");
    assert.throws(() => hfDownloadUrl("hf://owner/repo/file.json"), /Invalid pinned/);
    assert.throws(() => configAssets("# tokenizer_path: hf://o/r/f@pin\n"), /Missing/);
    assert.deepEqual(configAssets("weights_path_without_voice_cloning: 'hf://o/r/model@pin'\n    tokenizer_path: \"hf://o/r/tokenizer@pin\" # note"), {
      weightsUrl: "https://huggingface.co/o/r/resolve/pin/model",
      tokenizerUrl: "https://huggingface.co/o/r/resolve/pin/tokenizer",
    });
  });
  it("pins every preset including daan and uses canonical legacy language names", () => {
    assert.equal(presetVoiceUrl("english_2026-01", "daan"),
      "https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/4e1e0a3e611c51c0b4ed8174fc10f32a54644303/languages/english_2026-01/embeddings/daan.safetensors");
  });
});
