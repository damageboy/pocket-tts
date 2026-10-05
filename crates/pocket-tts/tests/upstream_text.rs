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
    assert_eq!(options.split(&tokenizer, "A,B.", 3).unwrap(), ["A,", "B."]);
    assert_eq!(
        options.split(&tokenizer, "A;B:C.", 3).unwrap(),
        ["A;", "B:", "C."]
    );
    assert_eq!(
        options.split(&tokenizer, "Abcdefghijk", 2).unwrap(),
        ["Abcdefghijk."]
    );
    assert_eq!(options.split(&tokenizer, "A,B.", 0).unwrap(), ["A,", "B."]);
    // A leading metaspace is added again during fallback re-encoding, so A,
    // costs three tokens rather than two even though it was an interior segment.
    assert_eq!(
        options.split(&tokenizer, "Z. A,B.", 4).unwrap(),
        ["Z.", "A,", "B."]
    );
}

#[test]
fn decimal_detection_decodes_prefix_and_suffix_and_uses_python_digits() {
    let tokenizer = tokenizer(false);
    let options = TextOptions::default();
    assert_eq!(
        options.split(&tokenizer, "Pi 3.14. E 2.718.", 1).unwrap(),
        ["Pi 3.14.", "E 2.718."]
    );
    assert_eq!(options.split(&tokenizer, "A ٣.٤.", 1).unwrap(), ["A ٣.٤."]);
    assert_eq!(options.split(&tokenizer, "A ².4.", 1).unwrap(), ["A ².4."]);
    assert_eq!(
        options.split(&tokenizer, "A ½.4.", 1).unwrap(),
        ["A ½.", "4."]
    );
    // Slice decoding removes the suffix's leading metaspace, as Python does:
    // even this space-separated pair is considered decimal at the boundary.
    assert_eq!(
        options.split(&tokenizer, "A 3. 4.", 1).unwrap(),
        ["A 3. 4."]
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
