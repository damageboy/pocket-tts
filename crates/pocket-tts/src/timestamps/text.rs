//! Source-word mapping ported from pocket-tts-timestamped `timestamps/text.py`,
//! commit 36e72b29d346c427c415acda99ce7118f37a541e (MIT; see THIRD_PARTY_NOTICES.md).
//! Mapping normalizes comparisons, never the actual generation prompt.

use std::{collections::HashMap, sync::LazyLock};

use anyhow::{Result, anyhow, bail};
use regex::Regex;
use tokenizers::{NormalizedString, Tokenizer};
use unicode_casefold::UnicodeCaseFold;

#[derive(Clone, Debug)]
pub(crate) struct SourceWord {
    pub text: String,
    pub word_index: usize,
}

#[derive(Debug)]
pub(crate) struct TextUnit {
    pub word: Option<SourceWord>,
    pub synthetic: bool,
}

#[derive(Debug)]
pub(crate) struct TimestampTextChunk {
    pub text: String,
    pub tail_guess: usize,
    pub units: Vec<TextUnit>,
    pub token_to_unit: Vec<Vec<f32>>,
}

pub(crate) fn prepare_timestamp_chunks(
    text: &str,
    options: &crate::text::TextOptions,
    tokenizer: &Tokenizer,
) -> Result<Vec<TimestampTextChunk>> {
    let chunks = options.split(tokenizer, text, 50)?;
    let prepared = chunks
        .iter()
        .map(|s| options.prepare(s))
        .collect::<Result<Vec<_>>>()?;
    let (texts, tails): (Vec<_>, Vec<_>) = prepared.into_iter().unzip();
    let mut chunks = build_chunks(text, texts, tokenizer)?;
    for (chunk, tail) in chunks.iter_mut().zip(tails) {
        chunk.tail_guess = tail;
    }
    Ok(chunks)
}

// All spans are UTF-8 byte offsets, as returned by Rust tokenizers::encode.
#[derive(Clone, Copy, Debug)]
struct Span {
    begin: usize,
    end: usize,
}

#[derive(Clone)]
struct LocatedWord {
    word: SourceWord,
    source: Span,
    chunk: usize,
    span: Span,
}

fn category(c: char, pattern: &LazyLock<Regex>) -> bool {
    pattern.is_match(c.encode_utf8(&mut [0; 4]))
}

fn alnum(c: char) -> bool {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[\p{L}\p{N}]$").unwrap());
    category(c, &RE)
}

fn mark(c: char) -> bool {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\p{M}$").unwrap());
    category(c, &RE)
}

fn punctuation(c: char) -> bool {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\p{P}$").unwrap());
    category(c, &RE)
}

fn nfkc(text: &str) -> String {
    let mut normalized = NormalizedString::from(text);
    normalized.nfkc();
    normalized.get().to_owned()
}

fn fold(c: char) -> impl Iterator<Item = char> {
    // Lowercase first also handles characters added after the casefold table's
    // Unicode version; the full fold is essential for sharp S and final sigma.
    c.to_lowercase().case_fold().map(|c| match c {
        '’' => '\'',
        '‐' | '‑' => '-',
        c => c,
    })
}

fn canonical_char(c: char) -> bool {
    alnum(c) || matches!(c, '\'' | '-')
}

fn key(text: &str) -> String {
    nfkc(text)
        .chars()
        .flat_map(fold)
        .filter(|c| canonical_char(*c))
        .collect()
}

fn variants(text: &str) -> Vec<String> {
    let mut variants = vec![key(text)];
    if let Some(first) = text.chars().next() {
        let capitalized = first.to_uppercase().collect::<String>() + &text[first.len_utf8()..];
        let capitalized = key(&capitalized);
        if !variants.contains(&capitalized) {
            variants.push(capitalized);
        }
    }
    variants
}

fn lexical_spans(text: &str) -> Vec<Span> {
    let chars = text.char_indices().collect::<Vec<_>>();
    let mut spans = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if !alnum(chars[i].1) {
            i += 1;
            continue;
        }
        let begin = chars[i].0;
        loop {
            while i < chars.len() && (alnum(chars[i].1) || mark(chars[i].1)) {
                i += 1;
            }
            if i + 1 < chars.len() && "-‐‑'’".contains(chars[i].1) && alnum(chars[i + 1].1) {
                i += 1;
            } else {
                break;
            }
        }
        let end = chars.get(i).map_or(text.len(), |c| c.0);
        if !key(&text[begin..end]).is_empty() {
            spans.push(Span { begin, end });
        }
    }
    spans
}

fn projection(text: &str) -> (Vec<char>, Vec<Span>) {
    // Prefix normalization faithfully tracks compositions, reordered combining
    // marks and compatibility expansions, including several scalars per origin.
    // Only the robust path needs this more expensive projection.
    let mut normalized = Vec::new();
    let mut origins: Vec<Span> = Vec::new();
    for (begin, c) in text.char_indices() {
        let end = begin + c.len_utf8();
        let updated = nfkc(&text[..end]).chars().collect::<Vec<_>>();
        let common = normalized
            .iter()
            .zip(&updated)
            .take_while(|(a, b)| a == b)
            .count();
        let origin = Span {
            begin: origins[common..]
                .iter()
                .map(|s| s.begin)
                .min()
                .unwrap_or(begin)
                .min(begin),
            end: origins[common..]
                .iter()
                .map(|s| s.end)
                .max()
                .unwrap_or(end)
                .max(end),
        };
        origins.truncate(common);
        origins.resize(updated.len(), origin);
        normalized = updated;
    }
    let mut canonical = Vec::new();
    let mut canonical_origins = Vec::new();
    for (c, origin) in normalized.into_iter().zip(origins) {
        for c in fold(c).filter(|c| canonical_char(*c)) {
            canonical.push(c);
            canonical_origins.push(origin);
        }
    }
    (canonical, canonical_origins)
}

fn robust_map(source: &str, chunks: &[String], words: &[LocatedWord]) -> Result<Vec<LocatedWord>> {
    let mut canonical = Vec::new();
    let mut origins = Vec::new();
    for (chunk, text) in chunks.iter().enumerate() {
        let (chars, spans) = projection(text);
        canonical.extend(chars);
        origins.extend(spans.into_iter().map(|span| (chunk, span)));
    }
    if !variants(source.trim()).contains(&canonical.iter().collect::<String>()) {
        bail!("Prepared timestamp text does not match the original input");
    }
    let mut mapped = Vec::new();
    let mut minimum = 0;
    for word in words {
        let candidates = if word.word.word_index == 0 {
            variants(&word.word.text)
        } else {
            vec![key(&word.word.text)]
        };
        let found = candidates
            .iter()
            .filter_map(|candidate| {
                let candidate = candidate.chars().collect::<Vec<_>>();
                canonical[minimum..]
                    .windows(candidate.len())
                    .position(|s| s == candidate)
                    .map(|p| (minimum + p, candidate.len()))
            })
            .min_by_key(|(begin, _)| *begin)
            .ok_or_else(|| anyhow!("Prepared text lacks an input word"))?;
        let (begin, length) = found;
        let (_, source_origins) = projection(&word.word.text);
        if source_origins.len() != length {
            bail!("Cannot project input word onto prepared text");
        }
        let mut groups = Vec::new();
        let mut p = begin;
        while p < begin + length {
            let chunk = origins[p].0;
            let mut end = p + 1;
            while end < begin + length && origins[end].0 == chunk {
                end += 1;
            }
            let source_begin = source_origins[p - begin..end - begin]
                .iter()
                .map(|s| s.begin)
                .min()
                .unwrap();
            groups.push((chunk, p, end, source_begin));
            p = end;
        }
        if groups.windows(2).any(|g| g[1].3 <= g[0].3) {
            bail!("Prepared chunks split an input character");
        }
        for (i, &(chunk, p, end, relative_begin)) in groups.iter().enumerate() {
            let relative_begin = if i == 0 { 0 } else { relative_begin };
            let relative_end = groups.get(i + 1).map_or(word.word.text.len(), |g| g.3);
            let part = &word.word.text[relative_begin..relative_end];
            let mut span = Span {
                begin: origins[p].1.begin,
                end: origins[end - 1].1.end,
            };
            // Include compatibility punctuation and combining marks that belong
            // to the source word's surface rather than separate punctuation.
            let mut surfaces = vec![part.to_owned(), nfkc(part)];
            if word.word.word_index == 0
                && i == 0
                && let Some(first) = part.chars().next()
            {
                let capitalized =
                    first.to_uppercase().collect::<String>() + &part[first.len_utf8()..];
                surfaces.push(capitalized.clone());
                surfaces.push(nfkc(&capitalized));
            }
            if let Some(surface) = surfaces
                .iter()
                .flat_map(|s| {
                    chunks[chunk].match_indices(s).map(|(begin, s)| Span {
                        begin,
                        end: begin + s.len(),
                    })
                })
                .filter(|s| s.begin <= span.begin && s.end >= span.end)
                .min_by_key(|s| (span.begin - s.begin + s.end - span.end, s.begin))
            {
                span = surface;
            }
            for c in chunks[chunk][span.end..].chars() {
                if !mark(c) {
                    break;
                }
                span.end += c.len_utf8();
            }
            mapped.push(LocatedWord {
                word: SourceWord {
                    text: part.to_owned(),
                    word_index: mapped.len(),
                },
                source: Span {
                    begin: word.source.begin + relative_begin,
                    end: word.source.begin + relative_end,
                },
                chunk,
                span,
            });
        }
        minimum = begin + length;
    }
    Ok(mapped)
}

fn best_effort(
    words: &[LocatedWord],
    prepared: &[(usize, Span, String)],
) -> (Vec<LocatedWord>, Vec<(usize, Span)>) {
    let mut source_counts = HashMap::new();
    let mut prepared_counts = HashMap::new();
    for word in words {
        *source_counts.entry(key(&word.word.text)).or_insert(0) += 1;
    }
    for (_, _, k) in prepared {
        *prepared_counts.entry(k.clone()).or_insert(0) += 1;
    }
    let positions = prepared
        .iter()
        .enumerate()
        .map(|(i, (_, _, k))| (k.as_str(), i))
        .collect::<HashMap<_, _>>();
    let mut mapped = Vec::new();
    let mut paired = vec![false; prepared.len()];
    let mut previous = None;
    for word in words {
        let k = key(&word.word.text);
        if source_counts[&k] != 1 || prepared_counts.get(&k) != Some(&1) {
            continue;
        }
        let i = positions[k.as_str()];
        if previous.is_some_and(|p| i <= p) {
            continue;
        }
        previous = Some(i);
        paired[i] = true;
        let mut word = word.clone();
        word.chunk = prepared[i].0;
        word.span = prepared[i].1;
        mapped.push(word);
    }
    let ignored = prepared
        .iter()
        .enumerate()
        .filter(|(i, _)| !paired[*i])
        .map(|(_, (chunk, span, _))| (*chunk, *span))
        .collect();
    (mapped, ignored)
}

fn expanded_offsets(offsets: &[(usize, usize)], pieces: &[String]) -> Vec<Span> {
    fn byte(piece: &str) -> Option<u8> {
        if piece.len() != 6 || !piece.starts_with("<0x") || !piece.ends_with('>') {
            return None;
        }
        u8::from_str_radix(&piece[3..5], 16).ok()
    }
    let mut spans = offsets
        .iter()
        .map(|&(begin, end)| Span { begin, end })
        .collect::<Vec<_>>();
    let mut i = 0;
    while i < pieces.len() {
        let Some(first) = byte(&pieces[i]) else {
            i += 1;
            continue;
        };
        let length = match first {
            0xc2..=0xdf => 2,
            0xe0..=0xef => 3,
            0xf0..=0xf4 => 4,
            _ => 1,
        };
        let end = if i + length <= pieces.len()
            && pieces[i + 1..i + length]
                .iter()
                .all(|p| byte(p).is_some_and(|b| (0x80..=0xbf).contains(&b)))
        {
            i + length
        } else {
            i + 1
        };
        let span = Span {
            begin: spans[i..end].iter().map(|s| s.begin).min().unwrap(),
            end: spans[i..end].iter().map(|s| s.end).max().unwrap(),
        };
        if span.end > span.begin {
            spans[i..end].fill(span);
        }
        i = end;
    }
    spans
}

fn overlaps(tokens: &[Span], units: &[Span]) -> Vec<Vec<f32>> {
    tokens
        .iter()
        .map(|token| {
            let mut row = units
                .iter()
                .map(|unit| {
                    token
                        .end
                        .min(unit.end)
                        .saturating_sub(token.begin.max(unit.begin)) as f32
                })
                .collect::<Vec<_>>();
            let total: f32 = row.iter().sum();
            if total > 0.0 {
                for value in &mut row {
                    *value /= total;
                }
            }
            row
        })
        .collect()
}

fn build_chunks(
    source: &str,
    chunks: Vec<String>,
    tokenizer: &Tokenizer,
) -> Result<Vec<TimestampTextChunk>> {
    let words = lexical_spans(source)
        .into_iter()
        .enumerate()
        .map(|(i, span)| LocatedWord {
            word: SourceWord {
                text: source[span.begin..span.end].to_owned(),
                word_index: i,
            },
            source: span,
            chunk: 0,
            span,
        })
        .collect::<Vec<_>>();
    let prepared = chunks
        .iter()
        .enumerate()
        .flat_map(|(i, text)| {
            lexical_spans(text)
                .into_iter()
                .map(move |span| (i, span, key(&text[span.begin..span.end])))
        })
        .collect::<Vec<_>>();
    let (mapped, ignored) = if words.len() == prepared.len()
        && words
            .iter()
            .zip(&prepared)
            .all(|(word, (_, _, k))| key(&word.word.text) == *k)
    {
        (
            words
                .iter()
                .zip(&prepared)
                .map(|(word, (chunk, span, _))| {
                    let mut word = word.clone();
                    word.chunk = *chunk;
                    word.span = *span;
                    word
                })
                .collect::<Vec<_>>(),
            Vec::new(),
        )
    } else {
        match robust_map(source, &chunks, &words) {
            Ok(mapped) => (mapped, Vec::new()),
            Err(_) => {
                let (mapped, ignored) = best_effort(&words, &prepared);
                tracing::warn!(
                    omitted = words.len() - mapped.len(),
                    "Timestamp text mapping omitted source words; audio generation continues"
                );
                (mapped, ignored)
            }
        }
    };
    let mut result = Vec::new();
    for (chunk, text) in chunks.into_iter().enumerate() {
        let chunk_words = mapped
            .iter()
            .filter(|w| w.chunk == chunk)
            .collect::<Vec<_>>();
        let mut units = Vec::new();
        let mut covered = vec![false; text.len()];
        for word in &chunk_words {
            covered[word.span.begin..word.span.end].fill(true);
            units.push((
                word.span,
                TextUnit {
                    word: Some(word.word.clone()),
                    synthetic: false,
                },
            ));
        }
        for (_, span) in ignored.iter().filter(|(i, _)| *i == chunk) {
            covered[span.begin..span.end].fill(true);
            units.push((
                *span,
                TextUnit {
                    word: None,
                    synthetic: true,
                },
            ));
        }
        let mut punctuation_spans: Vec<Span> = Vec::new();
        for (begin, c) in text.char_indices() {
            if !covered[begin] && punctuation(c) {
                let end = begin + c.len_utf8();
                if let Some(last) = punctuation_spans.last_mut().filter(|s| s.end == begin) {
                    last.end = end;
                } else {
                    punctuation_spans.push(Span { begin, end });
                }
            }
        }
        for span in punctuation_spans {
            let synthetic = chunk_words.last().is_some_and(|last| {
                if span.begin < last.span.end {
                    return false;
                }
                // The robust path can split words and renumber parts. Its source
                // boundaries therefore come from the mapped sequence, not the
                // original lexical indices. Gapped fallback retains originals.
                let boundary = if ignored.is_empty() {
                    mapped
                        .iter()
                        .find(|w| w.word.word_index == last.word.word_index + 1)
                        .map(|w| w.source.begin)
                } else {
                    words.get(last.word.word_index + 1).map(|w| w.source.begin)
                }
                .unwrap_or(source.len());
                !source[last.source.end..boundary].chars().any(punctuation)
            });
            units.push((
                span,
                TextUnit {
                    word: None,
                    synthetic,
                },
            ));
        }
        units.sort_by_key(|(span, _)| (span.begin, span.end));
        let encoding = tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| anyhow!("Timestamp text tokenization failed: {e}"))?;
        // The fork's tokenizers adapter measures fractional overlap in Unicode
        // codepoints. Rust encode offsets are bytes, so convert both sides after
        // expanding byte-fallback groups to their complete scalar spans.
        let mut codepoints = vec![0; text.len() + 1];
        for (index, (begin, c)) in text.char_indices().enumerate() {
            codepoints[begin..begin + c.len_utf8()].fill(index);
            codepoints[begin + c.len_utf8()] = index + 1;
        }
        let codepoint_span = |span: Span| Span {
            begin: codepoints[span.begin],
            end: codepoints[span.end],
        };
        let token_spans = expanded_offsets(encoding.get_offsets(), encoding.get_tokens());
        let token_spans = token_spans
            .into_iter()
            .map(codepoint_span)
            .collect::<Vec<_>>();
        let unit_spans = units
            .iter()
            .map(|(span, _)| codepoint_span(*span))
            .collect::<Vec<_>>();
        result.push(TimestampTextChunk {
            text,
            tail_guess: 0, // Set from TextOptions::prepare by prepare_timestamp_chunks.
            token_to_unit: overlaps(&token_spans, &unit_spans),
            units: units.into_iter().map(|(_, unit)| unit).collect(),
        });
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::TextOptions;

    fn tokenizer() -> Tokenizer {
        Tokenizer::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/upstream-v3.3/english.tokenizer.json"
        ))
        .unwrap()
    }

    fn words(chunks: &[TimestampTextChunk]) -> Vec<(usize, &str)> {
        chunks
            .iter()
            .flat_map(|c| &c.units)
            .filter_map(|u| u.word.as_ref())
            .map(|w| (w.word_index, w.text.as_str()))
            .collect()
    }

    #[test]
    fn curated_source_words_and_exact_generation_prompts() {
        let tokenizer = tokenizer();
        for (text, expected) in [
            ("  hello\n\r world  ", vec!["hello", "world"]),
            (
                "Cafe\u{301} nai\u{308}ve.",
                vec!["Cafe\u{301}", "nai\u{308}ve"],
            ),
            ("A ﬁne result.", vec!["A", "ﬁne", "result"]),
            ("Ｆｕｌｌ ① test.", vec!["Ｆｕｌｌ", "①", "test"]),
            ("Roman Ⅷ test.", vec!["Roman", "Ⅷ", "test"]),
            ("5㈠ test.", vec!["5㈠", "test"]),
            ("It is 20℃.", vec!["It", "is", "20"]),
            ("Room №5.", vec!["Room", "5"]),
            ("Meet at ㏂ 10.", vec!["Meet", "at", "10"]),
            ("ıstanbul is here.", vec!["ıstanbul", "is", "here"]),
            ("ßtraße ǆungla σς.", vec!["ßtraße", "ǆungla", "σς"]),
            (
                "a\u{327}\u{301} la carte.",
                vec!["a\u{327}\u{301}", "la", "carte"],
            ),
            (
                "l’esprit isn't blue‑green.",
                vec!["l’esprit", "isn't", "blue‑green"],
            ),
            (
                "Price_1 is 3.14—okay...",
                vec!["Price", "1", "is", "3", "14", "okay"],
            ),
            ("Pay €5 😀 now。", vec!["Pay", "5", "now"]),
            ("zero\u{200b}width text.", vec!["zero", "width", "text"]),
            ("😀 €", vec![]),
        ] {
            let options = TextOptions::default();
            let chunks = prepare_timestamp_chunks(text, &options, &tokenizer).unwrap();
            assert_eq!(
                words(&chunks),
                expected
                    .iter()
                    .enumerate()
                    .map(|(i, w)| (i, *w))
                    .collect::<Vec<_>>(),
                "{text}"
            );
            let ordinary = options
                .split(&tokenizer, text, 50)
                .unwrap()
                .iter()
                .map(|c| options.prepare(c).unwrap().0)
                .collect::<Vec<_>>();
            assert_eq!(
                chunks.iter().map(|c| &c.text).collect::<Vec<_>>(),
                ordinary.iter().collect::<Vec<_>>()
            );
            for chunk in chunks {
                assert_eq!(
                    chunk.token_to_unit.len(),
                    tokenizer.encode(chunk.text.as_str(), true).unwrap().len()
                );
                for row in chunk.token_to_unit {
                    assert_eq!(row.len(), chunk.units.len());
                    assert!(row.iter().all(|v| v.is_finite() && *v >= 0.0));
                    let sum: f32 = row.iter().sum();
                    assert!(sum == 0.0 || (sum - 1.0).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn terminal_punctuation_and_ambiguous_omissions() {
        let t = tokenizer();
        for (source, synthetic) in [("hello world", true), ("hello world!", false)] {
            let chunks = prepare_timestamp_chunks(source, &TextOptions::default(), &t).unwrap();
            assert_eq!(
                chunks[0]
                    .units
                    .iter()
                    .filter(|u| u.word.is_none())
                    .map(|u| u.synthetic)
                    .collect::<Vec<_>>(),
                vec![synthetic]
            );
        }
        let chunks = build_chunks(
            "one missing three one",
            vec!["One different three one.".into()],
            &t,
        )
        .unwrap();
        assert_eq!(words(&chunks), vec![(2, "three")]);
        // Three ignored lexical units plus preparation-added terminal period.
        assert_eq!(chunks[0].units.iter().filter(|u| u.synthetic).count(), 4);
    }

    #[test]
    fn compatibility_word_split_and_scalar_split_rejection() {
        let t = tokenizer();
        let chunks = build_chunks("A⒚B 10.", vec!["A19.".into(), "B 10.".into()], &t).unwrap();
        assert_eq!(words(&chunks), vec![(0, "A⒚"), (1, "B"), (2, "10")]);
        let chunks = build_chunks("ﬁ next", vec!["F".into(), "i next.".into()], &t).unwrap();
        assert_eq!(words(&chunks), vec![(1, "next")]);
    }

    #[test]
    fn empty_inputs_reject_like_ordinary_text() {
        for text in ["", " ", "\n\r"] {
            assert!(prepare_timestamp_chunks(text, &TextOptions::default(), &tokenizer()).is_err());
        }
    }

    #[test]
    fn fractional_overlap_and_zero_length_special_tokens() {
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                [("[UNK]".into(), 0), ("Hi!".into(), 1), ("é!".into(), 2)]
                    .into_iter()
                    .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let t = Tokenizer::new(model);
        let chunks = build_chunks("Hi!", vec!["Hi!".into()], &t).unwrap();
        assert_eq!(chunks[0].token_to_unit, vec![vec![2.0 / 3.0, 1.0 / 3.0]]);
        // Rust offsets are bytes, but the fork's tokenizers layout uses Unicode
        // codepoints: a multibyte letter must not outweigh one punctuation mark.
        let chunks = build_chunks("é!", vec!["é!".into()], &t).unwrap();
        assert_eq!(chunks[0].token_to_unit, vec![vec![0.5, 0.5]]);
        assert_eq!(
            overlaps(&[Span { begin: 0, end: 0 }], &[Span { begin: 0, end: 2 }]),
            vec![vec![0.0]]
        );
    }

    #[test]
    fn every_byte_fallback_piece_in_unicode_words_is_mapped() {
        let t = tokenizer();
        for text in ["A中B", "A𐐀B", "a\u{301}\u{300}"] {
            let chunks = prepare_timestamp_chunks(text, &TextOptions::default(), &t).unwrap();
            assert_eq!(words(&chunks), vec![(0, text)]);
            let chunk = &chunks[0];
            let encoding = t.encode(chunk.text.as_str(), true).unwrap();
            let column = chunk.units.iter().position(|u| u.word.is_some()).unwrap();
            let mut count = 0;
            for (i, piece) in encoding.get_tokens().iter().enumerate() {
                if piece.starts_with("<0x") {
                    count += 1;
                    assert_eq!(chunk.token_to_unit[i][column], 1.0, "{piece}");
                }
            }
            assert!(count > 0);
        }
        let pieces = ["<0xE4>", "<0xB8>", "<0xAD>"].map(String::from);
        let expanded = expanded_offsets(&[(0, 0), (0, 0), (0, 3)], &pieces);
        assert!(expanded.iter().all(|s| s.begin == 0 && s.end == 3));
    }

    #[test]
    fn chunk_boundaries_and_text_options_preserve_source_indices() {
        let t = tokenizer();
        let source = "Room №5. Meet at ㏂ 10. Café naïve; blue-green world!";
        let expected = vec![
            (0, "Room"),
            (1, "5"),
            (2, "Meet"),
            (3, "at"),
            (4, "10"),
            (5, "Café"),
            (6, "naïve"),
            (7, "blue-green"),
            (8, "world"),
        ];
        let options = TextOptions {
            remove_semicolons: true,
            pad_with_spaces_for_short_inputs: true,
            ..TextOptions::default()
        };
        for limit in [8, 20, 100] {
            let chunks = options
                .split(&t, source, limit)
                .unwrap()
                .iter()
                .map(|c| options.prepare(c).unwrap().0)
                .collect();
            let mapped = build_chunks(source, chunks, &t).unwrap();
            assert_eq!(words(&mapped), expected);
        }
        let chunks = prepare_timestamp_chunks("X ﾞ Y.", &options, &t).unwrap();
        assert_eq!(words(&chunks), vec![(0, "X"), (1, "Y")]);
        let options = TextOptions {
            replace_characters: [('x', "changed".into())].into(),
            ..TextOptions::default()
        };
        let chunks = prepare_timestamp_chunks("one x three", &options, &t).unwrap();
        assert_eq!(words(&chunks), vec![(0, "one"), (2, "three")]);
    }
}
