//! Text preparation and token-based chunking, matching pinned upstream pocket-tts.
//!
//! Pause markers belong to orchestration; this module does not consume them.

use std::collections::HashMap;
use std::sync::LazyLock;

use anyhow::{Result, anyhow, bail};
use regex::Regex;
use serde::Deserialize;
use tokenizers::Tokenizer;

/// The upstream text options, with the defaults from Python's Config.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct TextOptions {
    pub pad_with_spaces_for_short_inputs: bool,
    pub remove_semicolons: bool,
    pub append_terminal_punctuation: bool,
    pub capitalize_first_letter: bool,
    /// Simultaneous per-character translation. An empty value deletes a character.
    pub replace_characters: HashMap<char, String>,
}

impl Default for TextOptions {
    fn default() -> Self {
        Self {
            pad_with_spaces_for_short_inputs: false,
            remove_semicolons: false,
            append_terminal_punctuation: true,
            capitalize_first_letter: true,
            replace_characters: HashMap::new(),
        }
    }
}

impl TextOptions {
    /// Prepare a prompt and return the upstream tail guess (3 for at most four
    /// words, otherwise 1). This is the guess **before** orchestration adds 2.
    pub fn prepare(&self, text: &str) -> Result<(String, usize)> {
        let mut text = text.trim_matches(is_whitespace).to_owned();
        if !self.replace_characters.is_empty() {
            let mut translated = String::new();
            for c in text.chars() {
                if let Some(replacement) = self.replace_characters.get(&c) {
                    translated.push_str(replacement);
                } else {
                    translated.push(c);
                }
            }
            text = translated
                .split(is_whitespace)
                .filter(|word| !word.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            static STRAY_PUNCTUATION: LazyLock<Regex> =
                LazyLock::new(|| Regex::new(r"([.!?…])[\s\x1c-\x1f]*[,;:]").unwrap());
            text = STRAY_PUNCTUATION.replace_all(&text, "$1").into_owned();
        }
        if text.is_empty() {
            bail!("Text prompt cannot be empty");
        }
        // Python does one non-overlapping replacement pass, not whitespace
        // normalization, unless a nonempty translation table was supplied.
        text = text.replace(['\n', '\r'], " ").replace("  ", " ");
        if self.remove_semicolons {
            text = text.replace(';', ",");
        }
        let tail_guess = if word_count(&text) <= 4 { 3 } else { 1 };
        let first = text.chars().next().unwrap();
        if self.capitalize_first_letter && !first.is_uppercase() {
            text = first.to_uppercase().collect::<String>() + &text[first.len_utf8()..];
        }
        if self.append_terminal_punctuation {
            text = ensure_terminal_punctuation(text);
        }
        if self.pad_with_spaces_for_short_inputs && word_count(&text) < 5 {
            text = " ".repeat(8) + &text;
        }
        Ok((text, tail_guess))
    }

    /// Prepare, tokenize, split after sentence-boundary token runs, refine only
    /// oversized sentences at comma/semicolon/colon token runs, then greedily
    /// regroup decoded segments. `max_tokens` is a soft limit, even when zero:
    /// indivisible oversized segments are retained, never split by characters.
    /// Encoding includes special tokens; decoding skips them, as in Python.
    pub fn split(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        max_tokens: usize,
    ) -> Result<Vec<String>> {
        let (text, _) = self.prepare(text)?;
        let tokens = encode(tokenizer, text.trim_matches(is_whitespace))?;
        // Upstream intentionally drops the first probe token (the leading
        // SentencePiece metaspace), rather than looking up punctuation strings.
        let sentence_tokens = encode(tokenizer, ".!...?")?;
        let boundaries = find_boundaries(
            &tokens,
            sentence_tokens.get(1..).unwrap_or_default(),
            tokenizer,
        )?;
        let segments = decode_segments(tokenizer, &tokens, &boundaries)?;
        let fallback_tokens = encode(tokenizer, ",;:")?;
        let mut refined = Vec::new();
        for (count, sentence) in segments {
            if count <= max_tokens {
                refined.push((count, sentence));
            } else {
                let sub_tokens = encode(tokenizer, sentence.trim_matches(is_whitespace))?;
                let boundaries = find_boundaries(
                    &sub_tokens,
                    fallback_tokens.get(1..).unwrap_or_default(),
                    tokenizer,
                )?;
                let sub_segments = decode_segments(tokenizer, &sub_tokens, &boundaries)?;
                if sub_segments.len() > 1 {
                    refined.extend(sub_segments);
                } else {
                    refined.push((count, sentence));
                }
            }
        }

        // Decimal-ended segments remain with the next clause, even beyond
        // the soft budget: a prepared trailing comma would otherwise become a
        // sentence end and the model can stop before speaking the fraction.
        static ENDS_WITH_DECIMAL: LazyLock<Regex> = LazyLock::new(|| {
            // Python's \w is letters/numbers/underscore, excluding combining
            // marks included by Rust regex's Unicode \w.
            Regex::new(r"\d[.,]\d+[^\p{L}\p{N}_]*$").unwrap()
        });
        let mut glued: Vec<(usize, String)> = Vec::new();
        for (count, text) in refined {
            if let Some((previous_count, previous)) = glued.last_mut()
                && ENDS_WITH_DECIMAL.is_match(previous)
            {
                *previous_count += count;
                *previous = previous.trim_end_matches(is_whitespace).to_owned()
                    + " "
                    + text.trim_matches(is_whitespace);
            } else {
                glued.push((count, text));
            }
        }

        let mut chunks = Vec::new();
        let mut current = String::new();
        let mut current_count = 0usize;
        for (count, sentence) in glued {
            if current.is_empty() {
                current = sentence;
                current_count = count;
                continue;
            }
            if current_count.saturating_add(count) > max_tokens {
                chunks.push(current.trim_matches(is_whitespace).to_owned());
                current = sentence;
                current_count = count;
            } else {
                current.push(' ');
                current.push_str(&sentence);
                current_count += count;
            }
        }
        if !current.is_empty() {
            chunks.push(current.trim_matches(is_whitespace).to_owned());
        }
        for chunk in &chunks {
            let count = encode(tokenizer, chunk.trim_matches(is_whitespace))?.len();
            if count > max_tokens {
                tracing::warn!(
                    count,
                    max_tokens,
                    "Chunk exceeds token limit; generation may skip words"
                );
            }
        }
        Ok(chunks)
    }
}

const TERMINAL: &str = ".!?…";
const WEAK: &str = ",;:-–—";
const CLOSERS: &str = "\"'”’)]»";

fn ensure_terminal_punctuation(text: String) -> String {
    let core = text.trim_end_matches(|c| c == ' ' || CLOSERS.contains(c));
    let closers = text[core.len()..].trim_matches(is_whitespace);
    let Some(last) = core.chars().last() else {
        return text;
    };
    if TERMINAL.contains(last) {
        text
    } else if WEAK.contains(last) {
        core.trim_end_matches(|c| c == ' ' || WEAK.contains(c))
            .to_owned()
            + "."
            + closers
    } else {
        text + "."
    }
}

// Python's str.isspace includes these four information separators in addition
// to Unicode White_Space, which Rust's char::is_whitespace implements.
fn is_whitespace(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}

fn word_count(text: &str) -> usize {
    text.split(is_whitespace).filter(|s| !s.is_empty()).count()
}

fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    tokenizer
        .encode(text, true)
        .map(|e| e.get_ids().to_vec())
        .map_err(|e| anyhow!("Text tokenization failed: {e}"))
}

fn decode(tokenizer: &Tokenizer, tokens: &[u32]) -> Result<String> {
    #[cfg(test)]
    tests::DECODED_TOKENS.with(|count| count.set(count.get() + tokens.len()));
    tokenizer
        .decode(tokens, true)
        .map_err(|e| anyhow!("Text decoding failed: {e}"))
}

// Match the source's bounded six-token title window and eight-token closer
// lookahead. Never decode the complete prefix/suffix at every boundary.
fn is_inner_punctuation(tokens: &[u32], start: usize, tokenizer: &Tokenizer) -> Result<bool> {
    let after = decode(tokenizer, &tokens[start - 1..start + 1])?;
    static ALNUM: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[\p{L}\p{N}]$").unwrap());
    if after
        .chars()
        .nth(1)
        .is_some_and(|c| ALNUM.is_match(c.encode_utf8(&mut [0; 4])))
    {
        return Ok(true);
    }
    let window_start = start.saturating_sub(6);
    let before = decode(tokenizer, &tokens[window_start..start])?;
    let words = before
        .split(is_whitespace)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    const TITLES: &[&str] = &[
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Rev.", "vs.", "e.g.", "i.e.",
    ];
    Ok(words.last().is_some_and(|last| TITLES.contains(last))
        && (words.len() > 1 || window_start == 0))
}

fn skip_closers(tokens: &[u32], start: usize, tokenizer: &Tokenizer) -> Result<usize> {
    let punctuation = decode(tokenizer, &tokens[start - 1..start])?;
    let mut complete = start;
    for end in start + 1..=tokens.len().min(start.saturating_add(8)) {
        let window = decode(tokenizer, &tokens[start - 1..end])?;
        let after = window.strip_prefix(&punctuation).unwrap_or_default();
        if after.is_empty()
            || after
                .chars()
                .any(|c| c != '\u{fffd}' && !CLOSERS.contains(c))
        {
            break;
        }
        // Byte fallback decodes an incomplete scalar as U+FFFD. Only move
        // past complete characters, never through the middle of a quote.
        if !after.contains('\u{fffd}') {
            complete = end;
        }
    }
    Ok(complete)
}

fn find_boundaries(
    tokens: &[u32],
    boundary_tokens: &[u32],
    tokenizer: &Tokenizer,
) -> Result<Vec<usize>> {
    let mut indices = vec![0];
    let mut previous_was_boundary = false;
    for (index, token) in tokens.iter().enumerate() {
        if boundary_tokens.contains(token) {
            previous_was_boundary = true;
        } else {
            if previous_was_boundary && !is_inner_punctuation(tokens, index, tokenizer)? {
                let start = skip_closers(tokens, index, tokenizer)?;
                if start < tokens.len() {
                    indices.push(start);
                }
            }
            previous_was_boundary = false;
        }
    }
    indices.push(tokens.len());
    Ok(indices)
}

fn decode_segments(
    tokenizer: &Tokenizer,
    tokens: &[u32],
    boundaries: &[usize],
) -> Result<Vec<(usize, String)>> {
    boundaries
        .windows(2)
        .map(|pair| {
            Ok((
                pair[1] - pair[0],
                decode(tokenizer, &tokens[pair[0]..pair[1]])?,
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    thread_local! {
        pub(super) static DECODED_TOKENS: Cell<usize> = const { Cell::new(0) };
    }

    #[test]
    fn decoded_token_volume_scales_linearly() {
        let tokenizer = Tokenizer::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/upstream-v3.3/english.tokenizer.json"
        ))
        .unwrap();
        let mut volumes = Vec::new();
        for repeats in [64, 128] {
            DECODED_TOKENS.with(|count| count.set(0));
            let text = "Dr. Miller paid $822.95, which was due. He said “Stop.” Then left. "
                .repeat(repeats);
            TextOptions::default().split(&tokenizer, &text, 50).unwrap();
            volumes.push(DECODED_TOKENS.with(Cell::get));
        }
        eprintln!("decoded token volume for 64/128 repetitions: {volumes:?}");
        assert!(
            volumes[1] <= 3 * volumes[0],
            "doubling input must keep decode work linear: {volumes:?}"
        );
    }

    #[test]
    fn closer_lookahead_never_consumes_partial_multibyte_character() {
        let tokenizer = Tokenizer::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/upstream-v3.3/english.tokenizer.json"
        ))
        .unwrap();
        let tokens = encode(&tokenizer, "He left.”””Then.").unwrap();
        let period = *encode(&tokenizer, ".").unwrap().last().unwrap();
        let start = tokens.iter().position(|id| *id == period).unwrap() + 1;
        let boundary = skip_closers(&tokens, start, &tokenizer).unwrap();
        assert_eq!(
            decode(&tokenizer, &tokens[..boundary]).unwrap(),
            "He left.””"
        );
        assert_eq!(decode(&tokenizer, &tokens[boundary..]).unwrap(), "”Then.");
    }

    #[test]
    #[ignore = "requires external pinned SentencePiece asset; no model inference"]
    fn actual_sentencepiece_matches_target_text_fixtures() -> Result<()> {
        use crate::conditioners::text::LUTConditioner;
        use candle_core::{DType, Device};
        use candle_nn::VarBuilder;
        use sha2::{Digest, Sha256};
        let directory = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/upstream-2026-11-09/target/english_2m_2026-10");
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("manifest.json"))?)?;
        assert_eq!(
            manifest["upstream_commit"],
            "c91f50f38c6b914d49dbdff51b8d322edf843db2"
        );
        let asset = &manifest["tokenizer"];
        let bytes = std::fs::read(asset["path"].as_str().unwrap())?;
        assert_eq!(
            Sha256::digest(&bytes)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>(),
            asset["sha256"].as_str().unwrap()
        );
        let fixture = std::fs::read(directory.join("text.json"))?;
        assert_eq!(
            Sha256::digest(&fixture)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>(),
            manifest["files"]["text.json"].as_str().unwrap()
        );
        let bins = manifest["config"]["flow_lm"]["lookup_table"]["n_bins"]
            .as_u64()
            .unwrap() as usize;
        let from_bytes = LUTConditioner::new_from_bytes(
            bins,
            &bytes,
            1,
            1,
            VarBuilder::zeros(DType::F32, &Device::Cpu),
        )?;
        let from_file = LUTConditioner::new(
            bins,
            std::path::Path::new(asset["path"].as_str().unwrap()),
            1,
            1,
            VarBuilder::zeros(DType::F32, &Device::Cpu),
        )?;
        let cases: Vec<serde_json::Value> = serde_json::from_slice(&fixture)?;
        for conditioner in [from_bytes, from_file] {
            let tokenizer = conditioner.tokenizer();
            for case in &cases {
                let options: TextOptions = serde_json::from_value(case["options"].clone())?;
                let input = case["input"].as_str().unwrap();
                if let Some(error) = case["error"].as_str() {
                    assert_eq!(options.prepare(input).unwrap_err().to_string(), error);
                    continue;
                }
                let (prepared, tail) = options.prepare(input)?;
                assert_eq!(prepared, case["prepared"]);
                assert_eq!(
                    tail as u64,
                    case["frames_after_eos_guess"].as_u64().unwrap()
                );
                assert_eq!(
                    serde_json::to_value(encode(tokenizer, &prepared)?)?,
                    case["token_ids"],
                    "{input}"
                );
                for (limit, expected) in case["chunks"].as_object().unwrap() {
                    assert_eq!(
                        serde_json::to_value(options.split(tokenizer, input, limit.parse()?)?)?,
                        *expected,
                        "{input} limit={limit}"
                    );
                }
            }
        }
        Ok(())
    }
}
