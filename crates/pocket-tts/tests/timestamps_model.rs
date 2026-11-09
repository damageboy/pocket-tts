//! Model-backed acceptance; missing assets are fatal, never silently skipped.
//! POCKET_TTS_TIMESTAMP_MODEL=<upstream fixture manifest.json>
//! cargo test --release -p pocket-tts --test timestamps_model -- --ignored --nocapture
#[path = "common/fixture.rs"]
mod fixture;
use pocket_tts::{TTSModel, anyhow::Result, candle_core::Tensor, timestamps::TimestampEvent};
use std::{path::PathBuf, time::Instant};

#[test]
#[ignore = "requires a released English model, tokenizer and matching voice manifest"]
fn timestamps_preserve_audio_and_identify_original_words() -> Result<()> {
    let manifest_path = PathBuf::from(std::env::var("POCKET_TTS_TIMESTAMP_MODEL")?);
    let directory = manifest_path.parent().unwrap();
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
    let read = |key: &str| {
        std::fs::read(fixture::path(
            directory,
            manifest[key]["path"].as_str().unwrap(),
        ))
    };
    let config = fixture::config(directory, &manifest)?;
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
    let mut event_order = Vec::new();
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
                event_order.push(serde_json::json!({"kind": "audio", "samples": samples}));
                frames.push(chunk.audio);
            }
            TimestampEvent::WordStart(word) => {
                event_order.push(serde_json::json!({"kind": "word_start", "word_index": word.word_index, "word": word.word, "start_time": word.start_time}));
                assert!(open.is_none());
                assert_eq!(word.word, expected[word.word_index]);
                assert!(word.start_time >= previous_end);
                open = Some(word);
            }
            TimestampEvent::WordEnd(word) => {
                event_order.push(serde_json::json!({"kind": "word_end", "word_index": word.word_index, "word": word.word, "start_time": word.start_time, "end_time": word.end_time}));
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
    if let Some(preservation) = manifest["preservation"].as_str() {
        let saved = fixture::path(directory, preservation);
        let expected_pcm = candle_core::safetensors::load(
            saved.join("independent.safetensors"),
            &pocket_tts::candle_core::Device::Cpu,
        )?;
        // Archived PCM predates the fresh-decoder onset fade. Transform only
        // each segment onset; never replace the immutable archived fixture.
        let mut expected_samples = expected_pcm["pcm"].flatten_all()?.to_vec1::<f32>()?;
        let fade = model.sample_rate / 200;
        for (i, sample) in expected_samples.iter_mut().take(fade).enumerate() {
            *sample *= i as f32 / (fade - 1) as f32;
        }
        let faded_expected = Tensor::from_vec(
            expected_samples,
            expected_pcm["pcm"].shape(),
            &pocket_tts::candle_core::Device::Cpu,
        )?;
        let actual_pcm = Tensor::cat(&frames, 2)?;
        assert_eq!(actual_pcm.dims(), expected_pcm["pcm"].dims());
        assert_eq!(
            (&actual_pcm - &faded_expected)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?,
            0.0,
            "preserved independent PCM"
        );
        let expected: serde_json::Value =
            serde_json::from_slice(&std::fs::read(saved.join("events.json"))?)?;
        let mut expected_events = expected["events"].clone();
        if manifest["language"] == "english_2m_2026-10" {
            // This archived onset burst was barely voiced. Its 5 ms fade
            // crosses the existing gate, so the first real voiced frame now
            // opens "The" at 0.24 s. No PCM samples or clock time are removed.
            let raw = expected_pcm["pcm"]
                .narrow(2, 0, frames[0].dim(2)?)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let faded = frames[0].flatten_all()?.to_vec1::<f32>()?;
            let rms = |samples: &[f32]| {
                (samples.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / samples.len() as f64)
                    .sqrt()
            };
            let threshold = f64::from(1e-3f32);
            assert!(rms(&raw) > threshold);
            assert!(rms(&faded) <= threshold);
            eprintln!(
                "2M onset voiced gate: raw_rms={}, faded_rms={}",
                rms(&raw),
                rms(&faded)
            );
            let events = expected_events.as_array_mut().unwrap();
            let old_prefix = serde_json::json!([
                {"kind":"word_start","word_index":0,"word":"The","start_time":0.0},
                {"kind":"audio","samples":1920},
                {"kind":"audio","samples":3840},
                {"kind":"word_end","word_index":0,"word":"The","start_time":0.0,"end_time":0.16},
                {"kind":"audio","samples":5760},
                {"kind":"word_start","word_index":1,"word":"quick","start_time":0.24},
                {"kind":"audio","samples":7680},
                {"kind":"audio","samples":9600},
                {"kind":"audio","samples":11520},
                {"kind":"word_end","word_index":1,"word":"quick","start_time":0.24,"end_time":0.48}
            ]);
            assert_eq!(&events[..10], old_prefix.as_array().unwrap());
            let faded_prefix = serde_json::json!([
                {"kind":"audio","samples":1920},
                {"kind":"audio","samples":3840},
                {"kind":"audio","samples":5760},
                {"kind":"word_start","word_index":0,"word":"The","start_time":0.24},
                {"kind":"audio","samples":7680},
                {"kind":"word_end","word_index":0,"word":"The","start_time":0.24,"end_time":0.32},
                {"kind":"word_start","word_index":1,"word":"quick","start_time":0.32},
                {"kind":"audio","samples":9600},
                {"kind":"audio","samples":11520},
                {"kind":"word_end","word_index":1,"word":"quick","start_time":0.32,"end_time":0.48}
            ]);
            events.splice(..10, faded_prefix.as_array().unwrap().iter().cloned());
        }
        assert_eq!(
            serde_json::to_value(&event_order)?,
            expected_events,
            "preserved event order and original word indices"
        );
    }
    if let Ok(output) = std::env::var("POCKET_TTS_PRESERVATION_OUTPUT") {
        let output = PathBuf::from(output);
        std::fs::create_dir_all(&output)?;
        let audio = Tensor::cat(&frames, 2)?;
        candle_core::safetensors::save(
            &std::collections::HashMap::from([("pcm".to_owned(), audio)]),
            output.join("independent.safetensors"),
        )?;
        std::fs::write(
            output.join("events.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "language": manifest["language"], "text": text, "temperature": 0.0,
                "sampler_decode_steps": model.lsd_decode_steps, "samples": samples,
                "timestamp_heads": model.timestamp_heads, "events": event_order,
            }))?,
        )?;
    }
    eprintln!("ordinary_ms={ordinary_ms}, timestamp_ms={timestamp_ms}, samples={samples}");
    eprintln!("{}", serde_json::to_string_pretty(&words)?);
    Ok(())
}

#[test]
#[ignore = "requires an explicit released model and matching voice manifest"]
fn fresh_decoders_fade_once_per_text_segment() -> Result<()> {
    let manifest_path = PathBuf::from(std::env::var("POCKET_TTS_TIMESTAMP_MODEL")?);
    let directory = manifest_path.parent().unwrap();
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
    let read = |key: &str| {
        std::fs::read(fixture::path(
            directory,
            manifest[key]["path"].as_str().unwrap(),
        ))
    };
    let mut model = TTSModel::load_from_bytes(
        &fixture::config(directory, &manifest)?,
        &read("weights")?,
        &read("tokenizer")?,
    )?;
    model.temp = 0.0;
    let voice = model.get_voice_state_from_prompt_bytes(&read("voice_state")?)?;
    let text = "Hello world. ".repeat(30);
    let chunks = model.try_split_into_best_sentences(&text)?;
    assert!(chunks.len() > 1);
    let mut expected = Vec::new();
    for chunk in &chunks {
        let frames = model
            .generate_stream_owned(chunk, &voice)
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(
            frames[0].flatten_all()?.to_vec1::<f32>()?[0],
            0.0,
            "fresh segment starts at zero"
        );
        expected.extend(frames);
    }
    let actual = model
        .generate_stream_owned(&text, &voice)
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(
            (actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()?,
            0.0
        );
    }
    eprintln!(
        "{} segments, {} frames: exact independent-segment PCM",
        chunks.len(),
        actual.len()
    );
    Ok(())
}
