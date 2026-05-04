//! List command implementation
//!
//! Provides `pocket-tts list` for listing available languages and voices.

use owo_colors::OwoColorize;

use crate::voice::PREDEFINED_VOICES;

/// Voice metadata sourced from upstream and the official Kyutai demo site.
pub struct VoiceCatalogEntry {
    pub name: &'static str,
    pub gender: &'static str,   // "m" or "f"
    pub style: &'static str,    // "conversation", "reading", "expressive"
    pub language: &'static str, // native language
}

/// Get the full voice catalog.
pub fn voice_catalog() -> &'static [VoiceCatalogEntry] {
    VOICE_CATALOG
}

const VOICE_CATALOG: &[VoiceCatalogEntry] = &[
    VoiceCatalogEntry {
        name: "alba",
        gender: "m",
        style: "reading",
        language: "english",
    },
    VoiceCatalogEntry {
        name: "anna",
        gender: "f",
        style: "conversation",
        language: "english",
    },
    VoiceCatalogEntry {
        name: "azelma",
        gender: "f",
        style: "reading",
        language: "english",
    },
    VoiceCatalogEntry {
        name: "bill_boerst",
        gender: "m",
        style: "reading",
        language: "english",
    },
    VoiceCatalogEntry {
        name: "caro_davy",
        gender: "f",
        style: "reading",
        language: "english",
    },
    VoiceCatalogEntry {
        name: "charles",
        gender: "m",
        style: "conversation",
        language: "english",
    },
    VoiceCatalogEntry {
        name: "cosette",
        gender: "f",
        style: "expressive",
        language: "english",
    },
    VoiceCatalogEntry {
        name: "eponine",
        gender: "f",
        style: "reading",
        language: "english",
    },
    VoiceCatalogEntry {
        name: "estelle",
        gender: "f",
        style: "conversation",
        language: "french",
    },
    VoiceCatalogEntry {
        name: "eve",
        gender: "f",
        style: "conversation",
        language: "english",
    },
    VoiceCatalogEntry {
        name: "fantine",
        gender: "f",
        style: "reading",
        language: "english",
    },
    VoiceCatalogEntry {
        name: "george",
        gender: "m",
        style: "conversation",
        language: "english",
    },
    VoiceCatalogEntry {
        name: "giovanni",
        gender: "m",
        style: "conversation",
        language: "italian",
    },
    VoiceCatalogEntry {
        name: "jane",
        gender: "f",
        style: "conversation",
        language: "english",
    },
    VoiceCatalogEntry {
        name: "javert",
        gender: "m",
        style: "conversation",
        language: "english",
    },
    VoiceCatalogEntry {
        name: "jean",
        gender: "m",
        style: "conversation",
        language: "english",
    },
    VoiceCatalogEntry {
        name: "juergen",
        gender: "m",
        style: "conversation",
        language: "german",
    },
    VoiceCatalogEntry {
        name: "lola",
        gender: "f",
        style: "conversation",
        language: "spanish",
    },
    VoiceCatalogEntry {
        name: "marius",
        gender: "m",
        style: "conversation",
        language: "english",
    },
    VoiceCatalogEntry {
        name: "mary",
        gender: "f",
        style: "conversation",
        language: "english",
    },
    VoiceCatalogEntry {
        name: "michael",
        gender: "m",
        style: "conversation",
        language: "english",
    },
    VoiceCatalogEntry {
        name: "paul",
        gender: "m",
        style: "conversation",
        language: "english",
    },
    VoiceCatalogEntry {
        name: "peter_yearsley",
        gender: "m",
        style: "reading",
        language: "english",
    },
    VoiceCatalogEntry {
        name: "rafael",
        gender: "m",
        style: "conversation",
        language: "portuguese",
    },
    VoiceCatalogEntry {
        name: "stuart_bell",
        gender: "m",
        style: "reading",
        language: "english",
    },
    VoiceCatalogEntry {
        name: "vera",
        gender: "f",
        style: "conversation",
        language: "english",
    },
];

/// Language metadata.
pub struct LanguageCatalogEntry {
    pub name: &'static str,
    pub description: &'static str,
    pub default_voice: &'static str,
    pub layers: usize,
    pub status: &'static str, // "production", "preview", "legacy"
}

/// Get the full language catalog.
pub fn language_catalog() -> &'static [LanguageCatalogEntry] {
    LANGUAGE_CATALOG
}

const LANGUAGE_CATALOG: &[LanguageCatalogEntry] = &[
    // ── Production models (distilled 6-layer, fast, real-time on CPU) ──
    LanguageCatalogEntry {
        name: "english",
        description: "English (latest)",
        default_voice: "alba",
        layers: 6,
        status: "production",
    },
    LanguageCatalogEntry {
        name: "german",
        description: "German",
        default_voice: "juergen",
        layers: 6,
        status: "production",
    },
    LanguageCatalogEntry {
        name: "italian",
        description: "Italian",
        default_voice: "giovanni",
        layers: 6,
        status: "production",
    },
    LanguageCatalogEntry {
        name: "portuguese",
        description: "Portuguese",
        default_voice: "rafael",
        layers: 6,
        status: "production",
    },
    LanguageCatalogEntry {
        name: "spanish",
        description: "Spanish",
        default_voice: "lola",
        layers: 6,
        status: "production",
    },
    // ── Preview models (undistilled 24-layer, slower, higher quality) ──
    LanguageCatalogEntry {
        name: "french_24l",
        description: "French (no distilled model available yet)",
        default_voice: "estelle",
        layers: 24,
        status: "preview",
    },
    LanguageCatalogEntry {
        name: "german_24l",
        description: "German (undistilled, slower but higher quality)",
        default_voice: "juergen",
        layers: 24,
        status: "preview",
    },
    LanguageCatalogEntry {
        name: "italian_24l",
        description: "Italian (undistilled, slower but higher quality)",
        default_voice: "giovanni",
        layers: 24,
        status: "preview",
    },
    LanguageCatalogEntry {
        name: "portuguese_24l",
        description: "Portuguese (undistilled, slower but higher quality)",
        default_voice: "rafael",
        layers: 24,
        status: "preview",
    },
    LanguageCatalogEntry {
        name: "spanish_24l",
        description: "Spanish (undistilled, slower but higher quality)",
        default_voice: "lola",
        layers: 24,
        status: "preview",
    },
    // ── Legacy / alternate English models ──
    LanguageCatalogEntry {
        name: "english_2026-04",
        description: "English April 2026 (same weights as 'english')",
        default_voice: "alba",
        layers: 6,
        status: "legacy",
    },
    LanguageCatalogEntry {
        name: "english_2026-01",
        description: "English January 2026 (v1 compatible)",
        default_voice: "alba",
        layers: 6,
        status: "legacy",
    },
];

pub fn list_languages() {
    println!();
    println!("  {}  {}", "🌍".bold(), "Available Languages".bold().cyan());
    println!();
    println!(
        "    {:<22} {:<50} {}",
        "NAME".dimmed(),
        "DESCRIPTION".dimmed(),
        "DEFAULT VOICE".dimmed(),
    );
    println!("  {}", "─".repeat(80).dimmed());
    let mut last_status = "";
    for lang in LANGUAGE_CATALOG {
        if lang.status != last_status {
            if !last_status.is_empty() {
                println!();
            }
            let header = match lang.status {
                "production" => "Production (distilled 6-layer, real-time on CPU)",
                "preview" => "Preview (undistilled 24-layer, slower)",
                "legacy" => "Legacy / alternate",
                _ => lang.status,
            };
            println!("  {}", header.dimmed().italic());
            last_status = lang.status;
        }
        let status_badge = match lang.status {
            "production" => "✓".green().to_string(),
            "preview" => "β".yellow().to_string(),
            "legacy" => "·".dimmed().to_string(),
            _ => " ".to_string(),
        };
        println!(
            "  {} {:<22} {:<50} {}",
            status_badge,
            lang.name.green(),
            lang.description,
            lang.default_voice.cyan(),
        );
    }
    println!();
    println!(
        "  {} Use with: {} or {}",
        "💡".dimmed(),
        "--language english".cyan(),
        "--language french_24l".cyan(),
    );
    println!();
}

pub fn list_voices() {
    println!();
    println!("  {}  {}", "🗣️".bold(), "Available Voices".bold().cyan());
    println!();
    println!(
        "  {:<18} {:<8} {:<16} {}",
        "NAME".dimmed(),
        "GENDER".dimmed(),
        "STYLE".dimmed(),
        "LANGUAGE".dimmed(),
    );
    println!("  {}", "─".repeat(60).dimmed());
    for voice in VOICE_CATALOG {
        let lang_display = match voice.language {
            "english" => voice.language.to_string(),
            other => other.yellow().to_string(),
        };
        println!(
            "  {:<18} {:<8} {:<16} {}",
            voice.name.green(),
            voice.gender,
            voice.style,
            lang_display,
        );
    }
    println!();
    println!(
        "  {} All voices work with all languages via cross-lingual embeddings.",
        "ℹ️".dimmed(),
    );
    println!(
        "  {} Native language voices sound best for that language.",
        "💡".dimmed(),
    );
    println!(
        "  {} Use with: {} or {}",
        "💡".dimmed(),
        "--voice alba".cyan(),
        "--voice giovanni".cyan(),
    );
    println!();

    // Sanity check
    debug_assert_eq!(
        VOICE_CATALOG.len(),
        PREDEFINED_VOICES.len(),
        "VOICE_CATALOG and PREDEFINED_VOICES are out of sync"
    );
}
