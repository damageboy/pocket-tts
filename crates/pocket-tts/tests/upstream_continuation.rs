use pocket_tts::GenerationOptions;

#[test]
fn continuation_off_preserves_independent_segments() {
    let defaults = GenerationOptions::default();
    assert!(!defaults.teacher_force_previous_chunk);
    assert_eq!(defaults.resolved_max_tokens(), 50);
    assert_eq!(
        GenerationOptions {
            teacher_force_previous_chunk: true,
            ..defaults
        }
        .resolved_max_tokens(),
        25
    );
    for continuation in [false, true] {
        for limit in [0, 1, 12, 80] {
            assert_eq!(
                GenerationOptions {
                    max_tokens: Some(limit),
                    teacher_force_previous_chunk: continuation
                }
                .resolved_max_tokens(),
                limit
            );
        }
    }
}

#[path = "common/fixture.rs"]
mod fixture;
use pocket_tts::{
    ModelState, TTSModel,
    anyhow::Result,
    candle_core::{DType, Device, Tensor},
    timestamps::TimestampEvent,
};
use std::path::PathBuf;

fn load(
    reference: &str,
    shipped: bool,
) -> Result<(TTSModel, ModelState, serde_json::Value, PathBuf)> {
    let directory = PathBuf::from(std::env::var(reference)?);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("manifest.json"))?)?;
    let read = |key: &str| {
        std::fs::read(fixture::path(
            &directory,
            manifest[key]["path"].as_str().unwrap(),
        ))
    };
    let config = if shipped {
        std::fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("config")
                .join(format!("{}.yaml", manifest["language"].as_str().unwrap())),
        )?
    } else {
        fixture::config(&directory, &manifest)?
    };
    let mut model = TTSModel::load_from_bytes(&config, &read("weights")?, &read("tokenizer")?)?;
    model.temp = 0.0;
    let voice = model.get_voice_state_from_prompt_bytes(&read("voice_state")?)?;
    Ok((model, voice, manifest, directory))
}

fn close(actual: &Tensor, expected: &Tensor, tolerance: f32, label: &str) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims(), "{label}");
    let error = (actual.to_dtype(DType::F32)? - expected.to_dtype(DType::F32)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(error <= tolerance, "{label}: {error} > {tolerance}");
    eprintln!("{label}: max_abs={error}");
    Ok(())
}

#[test]
#[ignore = "requires explicit target fixture directory in POCKET_TTS_REFERENCE"]
fn continuation_replays_only_previous_emitted_latents() -> Result<()> {
    let (model, mut state, manifest, directory) = load("POCKET_TTS_REFERENCE", false)?;
    let tensors =
        candle_core::safetensors::load(directory.join("model.safetensors"), &Device::Cpu)?;
    let tokens = model.conditioner.prepare(
        manifest["continuation"]["prepared"].as_str().unwrap(),
        &Device::Cpu,
    )?;
    close(
        &tokens,
        &tensors["continuation.tokens"],
        0.0,
        "combined tokens",
    )?;
    let forced = &tensors["continuation.forced_latents"];
    let count = forced.dim(1)?;
    let bos = model.flow_lm.bos_emb.reshape((1, 1, model.ldim))?;
    let prefix = Tensor::cat(&[&bos, &forced.narrow(1, 0, count - 1)?], 1)?;
    let reference_prefix = &tensors["continuation.forced_prefix"];
    assert!(
        reference_prefix
            .narrow(1, 0, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|v| v.is_nan())
    );
    let reference_prefix = Tensor::cat(&[&bos, &reference_prefix.narrow(1, 1, count - 1)?], 1)?;
    close(
        &prefix,
        &reference_prefix,
        0.0,
        "BOS plus all but last normalized latent",
    )?;
    let mut input = forced.narrow(1, count - 1, 1)?;
    close(
        &input,
        &tensors["continuation.first_input"],
        0.0,
        "last emitted latent starts generation",
    )?;
    model
        .flow_lm
        .prefill(&prefix, &model.conditioner.forward(&tokens)?, &mut state)?;
    for (name, expected) in &tensors {
        if let Some(module) = name
            .strip_prefix("continuation.prefill.")
            .and_then(|s| s.strip_suffix(".cache"))
        {
            let cache = &state[&format!("flow_lm.{module}")];
            let length = expected.dim(2)?;
            let start = manifest["prefill_cache_start"].as_u64().unwrap_or(0) as usize;
            close(
                &cache["k_buf"].narrow(2, start, length)?,
                &expected.get(0)?.transpose(1, 2)?,
                5e-4,
                "prefill keys",
            )?;
            close(
                &cache["v_buf"].narrow(2, start, length)?,
                &expected.get(1)?.transpose(1, 2)?,
                5e-4,
                "prefill values",
            )?;
        }
    }
    let warmup = count.min(12);
    assert_eq!(
        warmup,
        manifest["continuation"]["warmup_frames"].as_u64().unwrap() as usize
    );
    let warm = forced
        .narrow(1, count - warmup, warmup)?
        .broadcast_mul(&model.flow_lm.emb_std)?
        .broadcast_add(&model.flow_lm.emb_mean)?
        .transpose(1, 2)?;
    let mut decoder = ModelState::new();
    let discarded = model
        .mimi
        .decode_from_latent(&model.mimi.quantize(&warm)?, &mut decoder, 0)?;
    close(
        &discarded,
        &tensors["continuation.discarded_warmup_pcm"],
        2e-3,
        "discarded warmup PCM",
    )?;
    let mut decoder_caches = 0;
    for (name, expected) in &tensors {
        if let Some(module) = name
            .strip_prefix("continuation.decoder_warmup.")
            .and_then(|s| s.strip_suffix(".cache"))
        {
            if !module.starts_with("decoder_transformer.") {
                continue;
            }
            let cache = &decoder[&format!("mimi.{module}")];
            let cursor = pocket_tts::voice_state::read_attention_cursor(cache);
            let length = expected.dim(2)?;
            assert_eq!(cursor.pos, length);
            assert_eq!(cursor.head, 0); // single fresh warmup batch
            let retained = cursor.len;
            for (key, index) in [("k_buf", 0), ("v_buf", 1)] {
                close(
                    &cache[key].narrow(2, 0, retained)?,
                    &expected.get(index)?.transpose(1, 2)?.narrow(
                        2,
                        length - retained,
                        retained,
                    )?,
                    5e-4,
                    "warm decoder retained KV",
                )?;
            }
            decoder_caches += 1;
        }
    }
    assert!(decoder_caches > 0);
    let empty = Tensor::zeros((1, 0, model.dim), DType::F32, &Device::Cpu)?;
    let times = model.flow_lm.flow_net.compute_time_embeddings(
        manifest["sampler_decode_steps"].as_u64().unwrap() as usize,
        &Device::Cpu,
        DType::F32,
    )?;
    for step in 0..manifest["continuation"]["emitted_frames"].as_u64().unwrap() as usize {
        let key = format!("continuation.step_{step}");
        let (latent, eos) = model.flow_lm.forward_with_noise(
            &input,
            &empty,
            &mut state,
            &times,
            &tensors[&format!("{key}.noise")],
            model.eos_threshold,
            step,
        )?;
        let expected = tensors[&format!("{key}.latent")].squeeze(1)?;
        close(&latent, &expected, 5e-4, "new normalized latent")?;
        assert_eq!(
            eos,
            tensors[&format!("{key}.eos_logit")]
                .flatten_all()?
                .to_vec1::<f32>()?[0]
                > model.eos_threshold
        );
        let denorm = expected
            .broadcast_mul(&model.flow_lm.emb_std)?
            .broadcast_add(&model.flow_lm.emb_mean)?
            .unsqueeze(2)?;
        let pcm = model.mimi.decode_from_latent(
            &model.mimi.quantize(&denorm)?,
            &mut decoder,
            warmup + step,
        )?;
        close(
            &pcm,
            &tensors[&format!("{key}.pcm")],
            2e-3,
            "emitted continuation PCM",
        )?;
        input = expected.unsqueeze(1)?;
    }
    Ok(())
}

#[test]
#[ignore = "requires shipped identity manifest directory in POCKET_TTS_SHIPPED_REFERENCE"]
fn warmup_is_excluded_from_sample_clock() -> Result<()> {
    let (model, voice, _, _) = load("POCKET_TTS_SHIPPED_REFERENCE", true)?;
    let text = "Ready ready. Ready again. Go slowly. Stop now.";
    let options = GenerationOptions {
        max_tokens: Some(1),
        teacher_force_previous_chunk: true,
    };
    let ordinary = model
        .generate_stream_owned_with_options(text, &voice, options)
        .collect::<Result<Vec<_>>>()?;
    let mut stream =
        model.generate_audio_with_timestamps_stream_with_options(text, &voice, options)?;
    let mut pcm = Vec::new();
    let mut samples = 0;
    let mut open = None;
    let mut previous = None;
    let mut ends = 0;
    for event in &mut stream {
        match event? {
            TimestampEvent::AudioChunk(chunk) => {
                assert_eq!(chunk.start_time, samples as f64 / model.sample_rate as f64);
                samples += chunk.audio.dim(2)?;
                assert_eq!(chunk.end_time, samples as f64 / model.sample_rate as f64);
                pcm.push(chunk.audio);
            }
            TimestampEvent::WordStart(word) => {
                assert!(open.is_none());
                assert!(previous.is_none_or(|index| word.word_index > index));
                assert!(word.start_time <= samples as f64 / model.sample_rate as f64);
                open = Some(word);
            }
            TimestampEvent::WordEnd(word) => {
                let start = open.take().expect("paired start");
                assert_eq!(word.word_index, start.word_index);
                assert_eq!(word.start_time, start.start_time);
                assert!(word.end_time >= word.start_time);
                assert!(word.end_time <= samples as f64 / model.sample_rate as f64);
                previous = Some(word.word_index);
                ends += 1;
            }
        }
    }
    assert!(open.is_none());
    assert!(ends > 0);
    assert!(stream.next().is_none());
    assert_eq!(ordinary.len(), pcm.len());
    close(
        &Tensor::cat(&ordinary, 2)?,
        &Tensor::cat(&pcm, 2)?,
        0.0,
        "timestamp/plain PCM",
    )?;
    eprintln!(
        "4 chunks: {} emitted frames, {ends} paired unique words, {} samples",
        pcm.len(),
        samples
    );
    Ok(())
}

#[test]
#[ignore = "requires shipped identity manifest directory in POCKET_TTS_SHIPPED_REFERENCE"]
fn continuation_cancel_preserves_voice() -> Result<()> {
    let (model, voice, _, _) = load("POCKET_TTS_SHIPPED_REFERENCE", true)?;
    let snapshots = voice
        .iter()
        .flat_map(|(module, values)| {
            values
                .iter()
                .map(move |(key, t)| Ok((module.clone(), key.clone(), t.copy()?)))
        })
        .collect::<Result<Vec<_>>>()?;
    let options = GenerationOptions {
        max_tokens: Some(1),
        teacher_force_previous_chunk: true,
    };
    let text = "Ready now. Go again. Stop there. Come back.";
    let first_frames = model
        .generate_stream_owned_with_options("Ready now.", &voice, options)
        .count();
    for count in [0, 1, first_frames, first_frames + 1] {
        let mut stream = model.generate_stream_owned_with_options(text, &voice, options);
        for _ in 0..count {
            stream.next().expect("frame")?;
        }
        drop(stream);
        for (module, key, expected) in &snapshots {
            close(
                &voice[module][key],
                expected,
                0.0,
                "pristine voice after cancel",
            )?;
        }
    }
    Ok(())
}
