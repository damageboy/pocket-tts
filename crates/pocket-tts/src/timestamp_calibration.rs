//! Test-only all-query-head export. No calibration hooks enter release libraries.
use super::*;
use crate::timestamps::{
    TimestampEvent, attention::AttentionCapture, text::prepare_timestamp_chunks,
};
use serde_json::{Value, json};
use std::path::PathBuf;

#[test]
#[ignore = "offline calibration: requires explicit job manifest and pinned local model assets"]
fn export_timestamp_calibration() -> Result<()> {
    let job: Value = serde_json::from_slice(&std::fs::read(std::env::var(
        "POCKET_TTS_CALIBRATION_JOB",
    )?)?)?;
    let output = PathBuf::from(job["output"].as_str().unwrap());
    std::fs::create_dir_all(&output)?;
    let config_bytes = std::fs::read(job["config"].as_str().unwrap())?;
    let config: crate::config::Config = serde_yaml::from_slice(&config_bytes)?;
    let mut model = TTSModel::load_from_bytes(
        &config_bytes,
        &std::fs::read(job["weights"].as_str().unwrap())?,
        &std::fs::read(job["tokenizer"].as_str().unwrap())?,
    )?;
    model.temp = job["temperature"].as_f64().unwrap() as f32;
    model.lsd_decode_steps = job["decode_steps"].as_u64().unwrap() as usize;
    let heads = (0..config.flow_lm.transformer.num_layers)
        .flat_map(|l| (0..config.flow_lm.transformer.num_heads).map(move |h| (l, h)))
        .collect::<Vec<_>>();
    for voice_spec in job["voices"].as_array().unwrap() {
        let voice_name = voice_spec["name"].as_str().unwrap();
        let voice = model.get_voice_state_from_prompt_bytes(&std::fs::read(
            voice_spec["path"].as_str().unwrap(),
        )?)?;
        for case in job["cases"].as_array().unwrap() {
            let id = case["id"].as_str().unwrap();
            let file = output.join(format!("{id}-{voice_name}.json"));
            anyhow::ensure!(
                !file.exists(),
                "Refusing to overwrite captured calibration: {}",
                file.display()
            );
            let text = case["text"].as_str().unwrap();
            let seed = case["seeds"][voice_name].as_u64().unwrap();
            let _capture_seed = crate::models::flow_lm::CalibrationNoiseSeed::new(seed);
            let chunks = prepare_timestamp_chunks(
                text,
                &model.text_options(),
                model.conditioner.tokenizer(),
                50,
            )?;
            let mut segments = Vec::new();
            let mut pcm = Vec::new();
            for chunk in chunks {
                let units = chunk.units.iter().map(|u| json!({"word": u.word.as_ref().map(|w| json!({"text":w.text,"index":w.word_index})), "synthetic":u.synthetic})).collect::<Vec<_>>();
                let cursor = crate::voice_state::get_attention_cursor(
                    &voice,
                    "flow_lm.transformer.layers.0.self_attn",
                )
                .len;
                let capture = AttentionCapture::new(heads.clone(), cursor, chunk.token_to_unit)?;
                let mut frames = Vec::new();
                for frame in model.generate_segment_frames(
                    chunk.text.clone(),
                    chunk.tail_guess,
                    &voice,
                    Some(capture),
                    None,
                    false,
                ) {
                    let frame = frame?;
                    let samples = frame.audio.flatten_all()?.to_vec1::<f32>()?;
                    frames.push(json!({"start":pcm.len() as f64/model.sample_rate as f64, "samples":samples.len(), "voiced":crate::timestamps::alignment::is_voiced(&samples), "scores":frame.head_scores.unwrap()}));
                    pcm.extend(samples);
                }
                segments.push(json!({"text":chunk.text,"units":units,"frames":frames,"end":pcm.len() as f64/model.sample_rate as f64}));
            }
            // Independent stream with historical/candidate pair verifies ordinary PCM
            // and provides exact Rust alignment replay expectations.
            let _native_seed = crate::models::flow_lm::CalibrationNoiseSeed::new(seed);
            let mut events = Vec::new();
            let mut native_pcm = Vec::new();
            for event in model.generate_audio_with_timestamps_stream(text, &voice)? {
                match event? {
                    TimestampEvent::AudioChunk(c) => native_pcm.extend(c.audio.flatten_all()?.to_vec1::<f32>()?),
                    TimestampEvent::WordEnd(w) => events.push(json!({"word":w.word,"index":w.word_index,"start":w.start_time,"end":w.end_time})),
                    _ => {}
                }
            }
            anyhow::ensure!(pcm == native_pcm, "all-head capture changed emitted PCM");
            let _ordinary_seed = crate::models::flow_lm::CalibrationNoiseSeed::new(seed);
            let ordinary = model
                .generate_stream_owned(text, &voice)
                .collect::<Result<Vec<_>>>()?;
            let ordinary_pcm = Tensor::cat(&ordinary, 2)?.flatten_all()?.to_vec1::<f32>()?;
            anyhow::ensure!(
                pcm == ordinary_pcm,
                "timestamp capture changed ordinary PCM"
            );
            let wav = output.join(format!("{id}-{voice_name}.wav"));
            let mut writer = hound::WavWriter::create(
                wav,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: model.sample_rate as u32,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )?;
            for sample in &pcm {
                writer.write_sample(*sample)?;
            }
            writer.finalize()?;
            std::fs::write(
                &file,
                serde_json::to_vec(
                    &json!({"schema_version":1,"temperature":model.temp,"decode_steps":model.lsd_decode_steps,"seed":seed,"text":text,"voice":voice_name,"sample_rate":model.sample_rate,"samples":pcm.len(),"heads":heads,"native_heads":model.timestamp_heads,"native_events":events,"segments":segments}),
                )?,
            )?;
            eprintln!(
                "captured {id} {voice_name}: {} segments, {} samples",
                segments.len(),
                pcm.len()
            );
        }
    }
    Ok(())
}
