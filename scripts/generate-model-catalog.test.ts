import { expect, test } from "bun:test";
import { execFileSync, spawnSync } from "node:child_process";
import { mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { generateModelCatalog } from "./generate-model-catalog";
import { FALLBACK_LANGUAGES, FALLBACK_VOICES } from "../crates/pocket-tts-cli/web/src/lib/model-catalog";
import { configAssets, presetVoiceUrl } from "../crates/pocket-tts-cli/web/src/workers/model-config";

const configDir = resolve(import.meta.dir, "../crates/pocket-tts/config");
const configs = Object.fromEntries(readdirSync(configDir).map(file => [file.replace(/\.yaml$/, ""), readFileSync(resolve(configDir, file), "utf8")]));

test("real configs publish all metadata, verbatim YAML and existing public asset URLs", () => {
  const catalog = generateModelCatalog(configs, "revision");
  expect(catalog.schemaVersion).toBe(1);
  expect(catalog.sourceRevision).toBe("revision");
  expect(catalog.defaultModel).toBe("english");
  expect(catalog.models).toHaveLength(18);
  expect(catalog.voices).toHaveLength(27);
  expect(catalog.voices).toEqual(FALLBACK_VOICES.map(({ name, ...voice }) => ({ id: name, ...voice })));
  for (const [i, model] of catalog.models.entries()) {
    const entry = FALLBACK_LANGUAGES[i];
    expect(model.id).toBe(entry.name);
    expect(model.description).toBe(entry.description);
    expect(model.defaultVoice).toBe(entry.default_voice);
    expect(model.layers).toBe(entry.layers);
    expect(model.status).toBe(entry.status);
    expect(model.configYaml).toBe(configs[model.id]);
    expect(model).toMatchObject(configAssets(configs[model.id]));
    expect(model.weightsUrl).toStartWith("https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/");
    expect(model.language).toBe(model.id.split("_")[0]);
    const voices = FALLBACK_VOICES;
    expect(Object.keys(model.voices)).toEqual(voices.map(voice => voice.name));
    for (const voice of voices) expect(model.voices[voice.name]).toBe(presetVoiceUrl(model.id, voice.name));
    expect(model.voices[model.defaultVoice]).toBeDefined();
  }
  const german = catalog.models.find(model => model.id === "german")!;
  expect(german.voices.alba).toBe("https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/4e1e0a3e611c51c0b4ed8174fc10f32a54644303/languages/german/embeddings/alba.safetensors");
});

test("asymmetric fixtures preserve distinct pins, architectures and model-scoped voices", () => {
  const languages = [FALLBACK_LANGUAGES[0], FALLBACK_LANGUAGES[2]];
  const yaml = (layers: number, pin: string) => `weights_path: hf://private/repo/gated@ignored\nweights_path_without_voice_cloning: hf://kyutai/pocket-tts-without-voice-cloning/${pin}.safetensors@${pin}\nflow_lm:\n  transformer:\n    num_layers: ${layers}\n  lookup_table:\n    tokenizer_path: hf://kyutai/pocket-tts-without-voice-cloning/${pin}.json@token-${pin}\nmimi:\n  transformer:\n    num_layers: 2\n`;
  const fixture = { english: yaml(6, "one"), "english_2026-09_24l": yaml(24, "two") };
  const catalog = generateModelCatalog(fixture, "fixture", languages, [FALLBACK_VOICES[0]]);
  expect(catalog.models[0].weightsUrl).toContain("/resolve/one/one.safetensors");
  expect(catalog.models[1].tokenizerUrl).toContain("/resolve/token-two/two.json");
  expect(catalog.models[0].voices.alba).not.toBe(catalog.models[1].voices.alba);
  expect(catalog.models[1].layers).toBe(24);
});

test("fails on config/catalog drift, missing defaults, duplicate ids and architecture drift", () => {
  const { english, ...missing } = configs;
  expect(() => generateModelCatalog(missing, "rev")).toThrow("coverage");
  expect(() => generateModelCatalog({ ...configs, extra: english }, "rev")).toThrow("coverage");
  expect(() => generateModelCatalog(configs, "rev", FALLBACK_LANGUAGES, FALLBACK_VOICES.slice(1))).toThrow("default voice");
  expect(() => generateModelCatalog(configs, "rev", [...FALLBACK_LANGUAGES, FALLBACK_LANGUAGES[0]])).toThrow("Duplicate model");
  expect(() => generateModelCatalog(configs, "rev", FALLBACK_LANGUAGES, [...FALLBACK_VOICES, FALLBACK_VOICES[0]])).toThrow("Duplicate voice");
  expect(() => generateModelCatalog({ ...configs, english: english.replace("num_layers: 6", "num_layers: 24") }, "rev")).toThrow("layers");
  expect(() => generateModelCatalog({ ...configs, english: english.replace("weights_path_without_voice_cloning:", "removed:") }, "rev")).toThrow("scalar");
  expect(() => generateModelCatalog({ ...configs, english: english.replace("weights_path_without_voice_cloning: hf://kyutai/pocket-tts-without-voice-cloning/", "weights_path_without_voice_cloning: hf://kyutai/pocket-tts/") }, "rev")).toThrow("public");
});

test("CLI writes deterministic JSON outside the repo and requires an output path", () => {
  const dir = mkdtempSync(resolve(tmpdir(), "pocket-model-catalog-"));
  try {
    const script = resolve(import.meta.dir, "generate-model-catalog.ts");
    const output = resolve(dir, "nested/models.json");
    execFileSync(process.execPath, [script, output], { cwd: dir });
    const first = readFileSync(output, "utf8");
    const revision = execFileSync("git", ["rev-parse", "HEAD"], { cwd: import.meta.dir, encoding: "utf8" }).trim();
    expect(JSON.parse(first)).toEqual(generateModelCatalog(configs, revision));
    execFileSync(process.execPath, [script, output], { cwd: dir });
    expect(readFileSync(output, "utf8")).toBe(first);
    const missing = spawnSync(process.execPath, [script], { cwd: dir, encoding: "utf8" });
    expect(missing.status).not.toBe(0);
    expect(missing.stderr).toContain("Usage:");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
