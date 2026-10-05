//! Text preparation and token-based chunking, matching pocket-tts v3.3.0.
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
            Some(tokenizer),
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
                    None,
                )?;
                let sub_segments = decode_segments(tokenizer, &sub_tokens, &boundaries)?;
                if sub_segments.len() > 1 {
                    refined.extend(sub_segments);
                } else {
                    refined.push((count, sentence));
                }
            }
        }

        let mut chunks = Vec::new();
        let mut current = String::new();
        let mut current_count = 0usize;
        for (count, sentence) in refined {
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
    tokenizer
        .decode(tokens, true)
        .map_err(|e| anyhow!("Text decoding failed: {e}"))
}

fn find_boundaries(
    tokens: &[u32],
    boundary_tokens: &[u32],
    decimals: Option<&Tokenizer>,
) -> Result<Vec<usize>> {
    let mut indices = vec![0];
    let mut previous_was_boundary = false;
    for (index, token) in tokens.iter().enumerate() {
        if boundary_tokens.contains(token) {
            previous_was_boundary = true;
        } else {
            if previous_was_boundary {
                let decimal = if let Some(tokenizer) = decimals {
                    let prefix = decode(tokenizer, &tokens[..index])?;
                    let suffix = decode(tokenizer, &tokens[index..])?;
                    let mut end = prefix.chars().rev();
                    end.next() == Some('.')
                        && end.next().is_some_and(is_digit)
                        && suffix.chars().next().is_some_and(is_digit)
                } else {
                    false
                };
                if !decimal {
                    indices.push(index);
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

// Python str.isdigit uses Unicode Numeric_Type=Decimal or Digit, not the
// broader Numeric property (fractions and Roman numerals are not digits).
fn is_digit(c: char) -> bool {
    static DECIMAL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\p{Nd}$").unwrap());
    let mut buffer = [0; 4];
    DECIMAL.is_match(c.encode_utf8(&mut buffer))
        || matches!(c,
            '\u{b2}'..='\u{b3}' | '\u{b9}' | '\u{1369}'..='\u{1371}' |
            '\u{19da}' | '\u{2070}' | '\u{2074}'..='\u{2079}' |
            '\u{2080}'..='\u{2089}' | '\u{2460}'..='\u{2468}' |
            '\u{2474}'..='\u{247c}' | '\u{2488}'..='\u{2490}' | '\u{24ea}' |
            '\u{24f5}'..='\u{24fd}' | '\u{24ff}' | '\u{2776}'..='\u{277e}' |
            '\u{2780}'..='\u{2788}' | '\u{278a}'..='\u{2792}' |
            '\u{10a40}'..='\u{10a43}' | '\u{10e60}'..='\u{10e68}' |
            '\u{11052}'..='\u{1105a}' | '\u{1f100}'..='\u{1f10a}'
        )
}
