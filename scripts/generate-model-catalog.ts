import { execFileSync } from "node:child_process";
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { FALLBACK_LANGUAGES, FALLBACK_VOICES, type LanguageEntry, type VoiceEntry } from "../crates/pocket-tts-cli/web/src/lib/model-catalog";
import { configAssets, presetVoiceUrl } from "../crates/pocket-tts-cli/web/src/workers/model-config";

export const SCHEMA_VERSION = 1;

export function generateModelCatalog(
  configs: Record<string, string>,
  sourceRevision: string,
  languages: LanguageEntry[] = FALLBACK_LANGUAGES,
  voices: VoiceEntry[] = FALLBACK_VOICES,
) {
  const unique = (ids: string[], kind: string) => {
    if (new Set(ids).size !== ids.length) throw new Error(`Duplicate ${kind} id`);
  };
  unique(languages.map(model => model.name), "model");
  unique(voices.map(voice => voice.name), "voice");
  const modelIds = languages.map(model => model.name).sort();
  if (JSON.stringify(modelIds) !== JSON.stringify(Object.keys(configs).sort())) {
    throw new Error("Config/catalog coverage mismatch");
  }
  if (!modelIds.includes("english")) throw new Error("Missing default model: english");
  if (!sourceRevision.trim()) throw new Error("Missing source revision");

  const models = languages.map(model => {
    const configYaml = configs[model.name];
    const language = model.name.split("_")[0];
    // Read only FlowLM's depth (not Mimi's). Never re-serialize the YAML:
    // WASM receives the complete original config, including comments/whitespace.
    const parsed = Bun.YAML.parse(configYaml) as {
      flow_lm?: { transformer?: { num_layers?: number } };
    };
    if (parsed?.flow_lm?.transformer?.num_layers !== model.layers) {
      throw new Error(`Config/catalog layers mismatch: ${model.name}`);
    }
    const assets = configAssets(configYaml);
    if (!assets.weightsUrl.startsWith("https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/")) {
      throw new Error(`Expected public non-cloning weights: ${model.name}`);
    }
    const modelVoices = Object.fromEntries(voices
      .map(voice => [voice.name, presetVoiceUrl(model.name, voice.name)]));
    if (!Object.hasOwn(modelVoices, model.default_voice)) {
      throw new Error(`Missing default voice for ${model.name}: ${model.default_voice}`);
    }
    return {
      id: model.name,
      language,
      description: model.description,
      defaultVoice: model.default_voice,
      layers: model.layers,
      status: model.status,
      configYaml,
      ...assets,
      voices: modelVoices,
    };
  });
  for (const voice of voices) {
    if (!models.some(model => model.language === voice.language)) {
      throw new Error(`Voice language has no model: ${voice.name}`);
    }
  }
  return {
    schemaVersion: SCHEMA_VERSION,
    sourceRevision,
    defaultModel: "english",
    voices: voices.map(({ name, ...voice }) => ({ id: name, ...voice })),
    models,
  };
}

if (import.meta.main) {
  const output = process.argv[2];
  if (!output || process.argv.length !== 3) throw new Error("Usage: bun scripts/generate-model-catalog.ts OUTPUT_PATH");
  const root = resolve(import.meta.dir, "..");
  const configDir = resolve(root, "crates/pocket-tts/config");
  const configs = Object.fromEntries(readdirSync(configDir).filter(file => file.endsWith(".yaml"))
    .map(file => [file.slice(0, -5), readFileSync(resolve(configDir, file), "utf8")]));
  const revision = execFileSync("git", ["rev-parse", "HEAD"], { cwd: root, encoding: "utf8" }).trim();
  const catalog = generateModelCatalog(configs, revision);
  mkdirSync(dirname(resolve(output)), { recursive: true });
  writeFileSync(output, JSON.stringify(catalog, null, 2) + "\n");
  console.log(`Model catalog: ${output} (${catalog.models.length} models, ${catalog.voices.length} voices)`);
}
