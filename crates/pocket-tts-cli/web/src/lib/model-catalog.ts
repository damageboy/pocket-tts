// Shared by the API fallback selectors and the published WASM model catalog.
// Keep labels and ordering stable. Download pins live only in YAML/model-config.ts.
export interface LanguageEntry {
  name: string;
  description: string;
  default_voice: string;
  layers: number;
  status: string;
}

export interface VoiceEntry {
  name: string;
  gender: string;
  style: string;
  language: string;
}

export const FALLBACK_LANGUAGES: LanguageEntry[] = [
  { name: "english", description: "English (latest)", default_voice: "alba", layers: 6, status: "production" },
  { name: "english_2026-09", description: "English (September 2026)", default_voice: "alba", layers: 6, status: "production" },
  { name: "english_2026-09_24l", description: "English (September 2026, 24-layer)", default_voice: "alba", layers: 24, status: "preview" },
  { name: "english_2026-04", description: "English (April 2026)", default_voice: "alba", layers: 6, status: "legacy" },
  { name: "english_2026-04_24l", description: "English (April 2026, 24-layer)", default_voice: "alba", layers: 24, status: "legacy" },
  { name: "english_2026-01", description: "English (January 2026)", default_voice: "alba", layers: 6, status: "legacy" },
  { name: "dutch", description: "Dutch", default_voice: "daan", layers: 6, status: "production" },
  { name: "dutch_24l", description: "Dutch (24-layer)", default_voice: "daan", layers: 24, status: "preview" },
  { name: "french", description: "French", default_voice: "estelle", layers: 6, status: "production" },
  { name: "german", description: "German", default_voice: "juergen", layers: 6, status: "production" },
  { name: "italian", description: "Italian", default_voice: "giovanni", layers: 6, status: "production" },
  { name: "portuguese", description: "Portuguese", default_voice: "rafael", layers: 6, status: "production" },
  { name: "spanish", description: "Spanish", default_voice: "lola", layers: 6, status: "production" },
  { name: "french_24l", description: "French (24-layer)", default_voice: "estelle", layers: 24, status: "preview" },
  { name: "german_24l", description: "German (24-layer)", default_voice: "juergen", layers: 24, status: "preview" },
  { name: "italian_24l", description: "Italian (24-layer)", default_voice: "giovanni", layers: 24, status: "preview" },
  { name: "portuguese_24l", description: "Portuguese (24-layer)", default_voice: "rafael", layers: 24, status: "preview" },
  { name: "spanish_24l", description: "Spanish (24-layer)", default_voice: "lola", layers: 24, status: "preview" },
];

// Fallback used when API is unavailable (e.g. WASM-only mode).
export const FALLBACK_VOICES: VoiceEntry[] = [
  { name: "alba", gender: "m", style: "reading", language: "english" },
  { name: "anna", gender: "f", style: "conversation", language: "english" },
  { name: "azelma", gender: "f", style: "reading", language: "english" },
  { name: "bill_boerst", gender: "m", style: "reading", language: "english" },
  { name: "caro_davy", gender: "f", style: "reading", language: "english" },
  { name: "charles", gender: "m", style: "conversation", language: "english" },
  { name: "cosette", gender: "f", style: "expressive", language: "english" },
  { name: "daan", gender: "m", style: "reading", language: "dutch" },
  { name: "eponine", gender: "f", style: "reading", language: "english" },
  { name: "estelle", gender: "f", style: "conversation", language: "french" },
  { name: "eve", gender: "f", style: "conversation", language: "english" },
  { name: "fantine", gender: "f", style: "reading", language: "english" },
  { name: "george", gender: "m", style: "conversation", language: "english" },
  { name: "giovanni", gender: "m", style: "conversation", language: "italian" },
  { name: "jane", gender: "f", style: "conversation", language: "english" },
  { name: "javert", gender: "m", style: "conversation", language: "english" },
  { name: "jean", gender: "m", style: "conversation", language: "english" },
  { name: "juergen", gender: "m", style: "conversation", language: "german" },
  { name: "lola", gender: "f", style: "conversation", language: "spanish" },
  { name: "marius", gender: "m", style: "conversation", language: "english" },
  { name: "mary", gender: "f", style: "conversation", language: "english" },
  { name: "michael", gender: "m", style: "conversation", language: "english" },
  { name: "paul", gender: "m", style: "conversation", language: "english" },
  { name: "peter_yearsley", gender: "m", style: "reading", language: "english" },
  { name: "rafael", gender: "m", style: "conversation", language: "portuguese" },
  { name: "stuart_bell", gender: "m", style: "reading", language: "english" },
  { name: "vera", gender: "f", style: "conversation", language: "english" },
];
