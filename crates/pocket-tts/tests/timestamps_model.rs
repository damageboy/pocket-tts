//! Model-backed acceptance; missing assets are fatal, never silently skipped.
//! POCKET_TTS_TIMESTAMP_MODEL=<upstream fixture manifest.json>
//! cargo test --release -p pocket-tts --test timestamps_model -- --ignored --nocapture
use pocket_tts::{TTSModel, anyhow::Result, candle_core::Tensor, timestamps::TimestampEvent};
use std::{path::PathBuf, time::Instant};

#[test]
#[ignore = "requires a released English model, tokenizer and matching voice manifest"]
fn timestamps_preserve_audio_and_identify_original_words() -> Result<()> {
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(std::env::var(
        "POCKET_TTS_TIMESTAMP_MODEL",
    )?)?)?;
    let read = |key: &str| std::fs::read(manifest[key]["path"].as_str().unwrap());
    let config = std::fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("config")
            .join(format!("{}.yaml", manifest["language"].as_str().unwrap())),
    )?;
    let mut model = TTSModel::load_from_bytes(&config, &read("weights")?, &read("tokenizer")?)?;
    model.temp = 0.0;
    let voice = model.get_voice_state_from_prompt_bytes(&read("voice_state")?)?;
    let text = "The quick brown fox jumps over the lazy dog.";
    let expected = [
        "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog",
    ];
    let start = Instant::now();
    let ordinary = model
        .generate_stream_owned(text, &voice)
        .collect::<Result<Vec<_>>>()?;
    let ordinary_ms = start.elapsed().as_millis();
    let start = Instant::now();
    let mut stream = model.generate_audio_with_timestamps_stream(text, &voice)?;
    let mut frames = Vec::new();
    let mut words = Vec::new();
    let mut open = None;
    let mut samples = 0;
    let mut previous_end = 0.0;
    for event in &mut stream {
        match event? {
            TimestampEvent::AudioChunk(chunk) => {
                assert_eq!(chunk.start_time, samples as f64 / model.sample_rate as f64);
                samples += chunk.audio.dim(2)?;
                assert_eq!(chunk.end_time, samples as f64 / model.sample_rate as f64);
                frames.push(chunk.audio);
            }
            TimestampEvent::WordStart(word) => {
                assert!(open.is_none());
                assert_eq!(word.word, expected[word.word_index]);
                assert!(word.start_time >= previous_end);
                open = Some(word);
            }
            TimestampEvent::WordEnd(word) => {
                let start = open.take().expect("end must pair with a start");
                assert_eq!(word.word_index, start.word_index);
                assert_eq!(word.start_time, start.start_time);
                assert!(word.end_time > word.start_time);
                assert!(word.end_time <= samples as f64 / model.sample_rate as f64);
                previous_end = word.end_time;
                words.push(word);
            }
        }
    }
    let timestamp_ms = start.elapsed().as_millis();
    assert!(stream.next().is_none());
    assert!(open.is_none());
    assert_eq!(
        words.iter().map(|w| w.word.as_str()).collect::<Vec<_>>(),
        expected
    );
    assert_eq!(frames.len(), ordinary.len());
    for (actual, expected) in frames.iter().zip(&ordinary) {
        assert_eq!(
            (actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()?,
            0.0
        );
    }
    assert!(
        Tensor::cat(&frames, 2)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?
            > 0.01
    );
    // Cancelling a timestamped stream must not modify the reusable voice state.
    let mut cancelled = model.generate_audio_with_timestamps_stream(text, &voice)?;
    cancelled.next().unwrap()?;
    drop(cancelled);
    let restarted = model.generate_stream_owned(text, &voice).next().unwrap()?;
    assert_eq!(
        (&restarted - &ordinary[0])?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?,
        0.0
    );
    eprintln!("ordinary_ms={ordinary_ms}, timestamp_ms={timestamp_ms}, samples={samples}");
    eprintln!("{}", serde_json::to_string_pretty(&words)?);
    Ok(())
}
