// The YAML remains the model schema. Only its two download scalars are read here.
export function hfDownloadUrl(uri: string): string {
  const match = /^hf:\/\/([^/\s]+\/[^/\s]+)\/([^@\s]+)@([^@\s]+)$/.exec(uri);
  if (!match) throw new Error(`Invalid pinned HuggingFace URI: ${uri}`);
  const [, repo, path, revision] = match;
  return `https://huggingface.co/${repo}/resolve/${encodeURIComponent(revision)}/${path}`;
}

export function selectConfig(configs: Record<string, string>, language: string) {
  // Older browser clients used underscores in the release date.
  const name = language.replace(/^english_2026_(01|04|09)(?=_24l$|$)/, "english_2026-$1");
  if (!Object.hasOwn(configs, name)) throw new Error(`Unknown model config: ${language}`);
  return { name, yaml: configs[name] };
}

export function configAssets(yaml: string) {
  const scalar = (key: string) => {
    const matches = [...yaml.matchAll(new RegExp(`^ *${key}: *([^\\r\\n]+)$`, "gm"))];
    if (matches.length !== 1) throw new Error(`Missing or ambiguous config download scalar: ${key}`);
    const value = matches[0][1].replace(/\s+#.*$/, "").trim().replace(/^(['"])(.*)\1$/, "$2");
    return hfDownloadUrl(value);
  };
  return {
    weightsUrl: scalar("weights_path_without_voice_cloning"),
    tokenizerUrl: scalar("tokenizer_path"),
  };
}

export function presetVoiceUrl(language: string, voice: string): string {
  return hfDownloadUrl(`hf://kyutai/pocket-tts-without-voice-cloning/languages/${language}/embeddings/${voice}.safetensors@4e1e0a3e611c51c0b4ed8174fc10f32a54644303`);
}
