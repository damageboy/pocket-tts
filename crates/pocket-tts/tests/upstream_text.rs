use pocket_tts::text::TextOptions;
use tokenizers::Tokenizer;
use tokenizers::models::unigram::Unigram;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};

// Offline, real SentencePiece-style tokenizer: one token per character, plus
// the leading metaspace token which upstream discards from boundary probes.
fn tokenizer(merged_period: bool) -> Tokenizer {
    let mut vocab = vec![("<unk>".to_owned(), -100.0), ("▁".to_owned(), -1.0)];
    vocab.extend(('!'..='~').map(|c| (c.to_string(), -1.0)));
    vocab.extend(['²', '½', '٣', '٤'].map(|c| (c.to_string(), -1.0)));
    if merged_period {
        vocab.push(("A.".to_owned(), 10.0));
    }
    let mut tokenizer = Tokenizer::new(Unigram::from(vocab, Some(0), false).unwrap());
    tokenizer.with_pre_tokenizer(Some(Metaspace::new('▁', PrependScheme::Always, true)));
    tokenizer.with_decoder(Some(Metaspace::new('▁', PrependScheme::Always, true)));
    tokenizer
}

#[test]
fn default_preparation_and_tail_guess() {
    let options: TextOptions = serde_json::from_str("{}").unwrap();
    assert_eq!(
        options.prepare(" \nhello; world \r").unwrap(),
        ("Hello; world.".into(), 3)
    );
    assert_eq!(
        options.prepare("one two three four five").unwrap(),
        ("One two three four five.".into(), 1)
    );
    assert_eq!(options.prepare("ßeta").unwrap(), ("SSeta.".into(), 3));
    assert_eq!(
        options.prepare("a     b\t c\nd").unwrap(),
        ("A   b\t c d.".into(), 3)
    );
    assert!(
        options
            .prepare(" \n\t")
            .unwrap_err()
            .to_string()
            .contains("empty")
    );
}

#[test]
fn terminal_punctuation_matches_closers_and_weak_marks() {
    for (input, expected) in [
        ("hello world", "Hello world."),
        ("hello world!", "Hello world!"),
        ("wait for it...", "Wait for it..."),
        ("hello world -", "Hello world."),
        ("hello world,;:—", "Hello world."),
        ("he said \"go home\"", "He said \"go home\"."),
        ("he said \"go home.\"", "He said \"go home.\""),
        ("he said \"go home,\"", "He said \"go home.\""),
        ("see the note (below)", "See the note (below)."),
        ("yes…»", "Yes…»"),
        ("\"'", "\"'"),
    ] {
        assert_eq!(
            TextOptions::default().prepare(input).unwrap().0,
            expected,
            "{input}"
        );
    }
}

#[test]
fn options_control_preparation_without_consuming_pause_markers() {
    let options: TextOptions = serde_json::from_str(
        r#"{
        "pad_with_spaces_for_short_inputs": true,
        "remove_semicolons": true,
        "append_terminal_punctuation": false,
        "capitalize_first_letter": false
    }"#,
    )
    .unwrap();
    assert_eq!(
        options.prepare("salAm; hAle SomA").unwrap(),
        ("        salAm, hAle SomA".into(), 3)
    );
    assert_eq!(
        options.prepare("one two three four five").unwrap().0,
        "one two three four five"
    );
    assert_eq!(
        options.prepare("hi [pause:500ms]").unwrap().0,
        "        hi [pause:500ms]"
    );
}

#[test]
fn translation_is_simultaneous_and_retidies_before_capitalization() {
    let options: TextOptions = serde_json::from_str(r#"{
        "replace_characters": {"\"": "", "¡": "", "¿": "", "«": "", "»": "", "’": "'", "x": "y", "y": "z"}
    }"#).unwrap();
    for (input, expected) in [
        (
            "\"¡Venid a mí, hombres!\" Alzó la voz.",
            "Venid a mí, hombres! Alzó la voz.",
        ),
        ("il a dit « l’homme »", "Il a dit l'homme."),
        ("\"Vieni stasera?\", chiese.", "Vieni stasera? chiese."),
        ("a! \t ,; b", "A!; b."),
        ("xy   x", "Yz y."),
    ] {
        assert_eq!(options.prepare(input).unwrap().0, expected);
    }
    assert!(options.prepare("\"  \"").is_err());
    assert_eq!(
        TextOptions::default().prepare("a? , b").unwrap().0,
        "A? , b."
    );
}

#[test]
fn splitting_uses_token_counts_and_keeps_boundary_runs() {
    let tokenizer = tokenizer(false);
    let options = TextOptions::default();
    assert_eq!(options.split(&tokenizer, "A. B.", 6).unwrap(), ["A. B."]);
    assert_eq!(options.split(&tokenizer, "A. B.", 5).unwrap(), ["A.", "B."]);
    assert_eq!(
        options.split(&tokenizer, "A...?! B.", 3).unwrap(),
        ["A...?!", "B."]
    );
    assert_eq!(options.split(&tokenizer, ". A.", 2).unwrap(), [".", "A."]);
    assert!(options.split(&tokenizer, "", 5).is_err());
}

#[test]
fn fallback_only_refines_oversized_sentences_and_limits_stay_soft() {
    let tokenizer = tokenizer(false);
    let options = TextOptions::default();
    assert_eq!(options.split(&tokenizer, "A,B.", 5).unwrap(), ["A,B."]);
    assert_eq!(options.split(&tokenizer, "A,B.", 3).unwrap(), ["A,B."]);
    assert_eq!(
        options.split(&tokenizer, "A; B: C.", 3).unwrap(),
        ["A;", "B:", "C."]
    );
    assert_eq!(
        options.split(&tokenizer, "Abcdefghijk", 2).unwrap(),
        ["Abcdefghijk."]
    );
    assert_eq!(options.split(&tokenizer, "A, B.", 0).unwrap(), ["A,", "B."]);
    // A leading metaspace is added again during fallback re-encoding, so A,
    // costs three tokens rather than two even though it was an interior segment.
    assert_eq!(
        options.split(&tokenizer, "Z. A, B.", 4).unwrap(),
        ["Z.", "A,", "B."]
    );
}

#[test]
fn neighboring_punctuation_preserves_python_alphanumeric_characters() {
    let tokenizer = tokenizer(false);
    let options = TextOptions::default();
    assert_eq!(
        options.split(&tokenizer, "Pi 3.14. E 2.718.", 1).unwrap(),
        ["Pi 3.14. E 2.718."]
    );
    assert_eq!(options.split(&tokenizer, "A ٣.٤.", 1).unwrap(), ["A ٣.٤."]);
    assert_eq!(options.split(&tokenizer, "A ².4.", 1).unwrap(), ["A ².4."]);
    assert_eq!(options.split(&tokenizer, "A ½.4.", 1).unwrap(), ["A ½.4."]);
    // Decode across the boundary to preserve the intervening space.
    assert_eq!(
        options.split(&tokenizer, "A 3. 4.", 1).unwrap(),
        ["A 3.", "4."]
    );
}

#[test]
fn boundaries_are_token_ids_not_raw_punctuation_and_padding_is_stripped() {
    let tokenizer = tokenizer(true);
    let options = TextOptions {
        pad_with_spaces_for_short_inputs: true,
        ..Default::default()
    };
    assert_eq!(options.split(&tokenizer, "A. B.", 1).unwrap(), ["A. B."]);
    let disabled = TextOptions {
        append_terminal_punctuation: false,
        capitalize_first_letter: false,
        ..Default::default()
    };
    assert_eq!(disabled.split(&tokenizer, "abc", 1).unwrap(), ["abc"]);
}

fn target_tokenizer() -> Tokenizer {
    Tokenizer::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/upstream-v3.3/english.tokenizer.json"
    ))
    .unwrap()
}

// These expectations come from the pinned Python source using the identical
// checked-in JSON tokenizer, not from another Rust splitting path.
fn assert_target_cases(inputs: &[&str]) {
    let cases: Vec<serde_json::Value> = serde_json::from_str(include_str!(
        "../../../assets/upstream-2026-11-09/target/english/text.json"
    ))
    .unwrap();
    let tokenizer = target_tokenizer();
    for input in inputs {
        let case = cases.iter().find(|c| c["input"] == *input).unwrap();
        let options: TextOptions = serde_json::from_value(case["options"].clone()).unwrap();
        for (limit, expected) in case["chunks"].as_object().unwrap() {
            let expected: Vec<String> = serde_json::from_value(expected.clone()).unwrap();
            let chunks = options
                .split(&tokenizer, input, limit.parse().unwrap())
                .unwrap();
            assert_eq!(chunks, expected, "{input}, limit={limit}");
            // Splitting may insert spaces at boundaries, but must retain every
            // non-whitespace scalar, including multibyte closing quotes.
            let joined: String = chunks
                .join("")
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect();
            let prepared: String = case["prepared"]
                .as_str()
                .unwrap()
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect();
            assert_eq!(joined, prepared, "corrupted source: {input}");
        }
    }
}

#[test]
fn decimal_boundaries_preserve_numbers_and_spaces() {
    assert_target_cases(&[
        "Pi is 3.14. Next comes a sentence.",
        "I counted to 3. 14 people were there.",
        "La somme est de 822,95 €, ensuite partez. Puis revenez.",
        "The fee is 1,994.42, which must be paid. Then leave.",
    ]);
}

#[test]
fn titles_urls_and_emails_stay_intact() {
    assert_target_cases(&[
        "Dr. Miller met Prof. Smith. Next!",
        "Mr. Mrs. Ms. Rev. vs. e.g. i.e. Smith.",
        "Visit https://www.kyutai.org. Email jo@icloud.com. Then leave.",
        "Meet at 4 p.m. Next comes a sentence.",
        "We live in the U.S.",
        "St. John left. Next came Jane.",
    ]);
}

#[test]
fn closers_stay_with_their_sentence() {
    assert_target_cases(&[
        "He said “Stop.” Then left.",
        "He left. “Hi.”",
        "He said \"Stop.\")] Then left.",
        "“Stop.” She left. (Quietly.) Then returned.",
    ]);
}

#[test]
fn decimal_segments_keep_the_following_clause() {
    assert_target_cases(&[
        "The fee is $822.95, which must be paid, before leaving. Then leave.",
        "The fee is $822.95.",
    ]);
}

#[path = "common/fixture.rs"]
mod fixture;

#[test]
#[ignore = "requires external pinned tokenizer/model fixture assets; no model inference"]
fn actual_json_tokenizer_matches_all_target_text_fixtures() -> anyhow::Result<()> {
    let directory = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../assets/upstream-2026-11-09/target/english");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("manifest.json"))?)?;
    fixture::config(&directory, &manifest)?;
    assert_eq!(
        manifest["upstream_commit"],
        "c91f50f38c6b914d49dbdff51b8d322edf843db2"
    );
    let tokenizer = Tokenizer::from_file(fixture::path(
        &directory,
        manifest["tokenizer"]["path"].as_str().unwrap(),
    ))
    .map_err(anyhow::Error::msg)?;
    let cases: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(directory.join("text.json"))?)?;
    for case in cases {
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
            serde_json::to_value(
                tokenizer
                    .encode(prepared.as_str(), true)
                    .map_err(anyhow::Error::msg)?
                    .get_ids()
            )?,
            case["token_ids"]
        );
        for (limit, expected) in case["chunks"].as_object().unwrap() {
            assert_eq!(
                serde_json::to_value(options.split(&tokenizer, input, limit.parse()?)?)?,
                *expected,
                "{input} limit={limit}"
            );
        }
    }
    Ok(())
}
