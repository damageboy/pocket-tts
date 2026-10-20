//! Explicit model acceptance: run with POCKET_TTS_REFERENCE=<fixture directory>
//! cargo test --release -p pocket-tts --test upstream_model -- --ignored --nocapture
//! Unlike historical smoke tests, a missing model or fixture is an error.
use candle_core::{DType, Device, Tensor};
use pocket_tts::{TTSModel, anyhow::Result};
use std::{collections::HashMap, path::PathBuf};

fn close(actual: &Tensor, expected: &Tensor, tolerance: f32, label: &str) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims(), "{label}");
    let error = (actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
    eprintln!("{label}: max absolute error {error}");
    assert!(error <= tolerance, "{label}: {error} > {tolerance}");
    Ok(())
}

#[test]
#[ignore = "requires explicitly generated upstream model fixtures"]
fn teacher_forced_noise_matches_python_prefill_latents_and_decoder() -> Result<()> {
    let directory = PathBuf::from(std::env::var("POCKET_TTS_REFERENCE")?);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("manifest.json"))?)?;
    assert!(
        [
            "3dbee45d343d7dddd0d105468d17f8dcba14db3e", // v3.3
            "9a1b48922deb44cd4a805fde279eb33b71c1a473", // v4 small models
        ]
        .contains(&manifest["upstream_commit"].as_str().unwrap())
    );
    let path = |key: &str| -> &str { manifest[key]["path"].as_str().unwrap() };
    let reference =
        candle_core::safetensors::load(directory.join("model.safetensors"), &Device::Cpu)?;
    let config = std::fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("config")
            .join(format!("{}.yaml", manifest["language"].as_str().unwrap())),
    )?;
    let mut actual_config: serde_yaml::Value = serde_yaml::from_slice(&config)?;
    actual_config
        .as_mapping_mut()
        .unwrap()
        .remove("timestamp_heads");
    let expected_config: serde_yaml::Value =
        serde_yaml::from_slice(&std::fs::read(manifest["config_path"].as_str().unwrap())?)?;
    assert_eq!(actual_config, expected_config, "released config drift");
    let model = TTSModel::load_from_bytes(
        &config,
        &std::fs::read(path("weights"))?,
        &std::fs::read(path("tokenizer"))?,
    )?;
    let eos_threshold = manifest["eos_threshold"].as_f64().unwrap_or(-4.0) as f32;
    assert_eq!(model.eos_threshold, eos_threshold);
    let mut state = model.get_voice_state_from_prompt_file(path("voice_state"))?;
    let tokens = model
        .conditioner
        .prepare(manifest["prepared"].as_str().unwrap(), &Device::Cpu)?;
    assert_eq!(
        tokens.to_dtype(DType::I64)?.to_vec2::<i64>()?,
        reference["tokens"].to_vec2::<i64>()?
    );
    let embeddings = model.conditioner.forward(&tokens)?;
    close(
        &embeddings,
        &reference["text_embeddings"],
        2e-6, // Low-rank lookup includes a float32 projection.
        "text embeddings",
    )?;
    model
        .flow_lm
        .transformer
        .forward(&embeddings, &mut state, 0)?;
    for (name, expected) in &reference {
        if let Some(module) = name
            .strip_prefix("prefill.")
            .and_then(|s| s.strip_suffix(".cache"))
        {
            let name = if module.starts_with("flow_lm.") {
                module.to_owned()
            } else {
                format!("flow_lm.{module}")
            };
            let cache = &state[&name];
            let length = expected.dim(2)?;
            close(
                &cache["k_buf"].narrow(2, 0, length)?,
                &expected.get(0)?.transpose(1, 2)?,
                5e-4,
                &format!("{name} keys"),
            )?;
            close(
                &cache["v_buf"].narrow(2, 0, length)?,
                &expected.get(1)?.transpose(1, 2)?,
                5e-4,
                &format!("{name} values"),
            )?;
        }
    }
    let mut sequence = model.flow_lm.bos_emb.reshape((1, 1, model.ldim))?;
    let empty = Tensor::zeros((1, 0, model.dim), DType::F32, &Device::Cpu)?;
    let times = model.flow_lm.flow_net.compute_time_embeddings(
        manifest["sampler_decode_steps"].as_u64().unwrap() as usize,
        &Device::Cpu,
        DType::F32,
    )?;
    let mut mimi_state = HashMap::new();
    for step in 0..manifest["steps"].as_u64().unwrap() as usize {
        let prefix = format!("step_{step}");
        let (latent, eos) = model.flow_lm.forward_with_noise(
            &sequence,
            &empty,
            &mut state,
            &times,
            &reference[&format!("{prefix}.noise")],
            eos_threshold,
            step,
        )?;
        close(
            &latent,
            &reference[&format!("{prefix}.latent")].squeeze(1)?,
            5e-4,
            &format!("{prefix} latent"),
        )?;
        let logit = reference[&format!("{prefix}.eos_logit")]
            .flatten_all()?
            .to_vec1::<f32>()?[0];
        assert_eq!(eos, logit > eos_threshold, "{prefix} EOS");
        let expected_latent = reference[&format!("{prefix}.latent")].squeeze(1)?;
        let denorm = expected_latent
            .broadcast_mul(&model.flow_lm.emb_std)?
            .broadcast_add(&model.flow_lm.emb_mean)?;
        let encoded = model.mimi.quantize(&denorm.unsqueeze(2)?)?;
        let pcm = model
            .mimi
            .decode_from_latent(&encoded, &mut mimi_state, step)?;
        close(
            &pcm,
            &reference[&format!("{prefix}.pcm")],
            2e-3,
            &format!("{prefix} PCM"),
        )?;
        // Teacher forcing isolates the implementation from amplification of
        // floating-point differences through autoregressive latent feedback.
        // Attention and decoder states still advance across every frame.
        sequence = expected_latent.unsqueeze(1)?;
    }
    // Decode the same Python latents in unequal partitions. This catches codec
    // buffering/window bugs hidden by always decoding one frame at a time.
    let steps = manifest["steps"].as_u64().unwrap() as usize;
    let latents = (0..steps)
        .map(|step| reference[&format!("step_{step}.latent")].clone())
        .collect::<Vec<_>>();
    let expected_pcm = (0..steps)
        .map(|step| reference[&format!("step_{step}.pcm")].clone())
        .collect::<Vec<_>>();
    let latents = Tensor::cat(&latents, 1)?;
    let mut partition_state = HashMap::new();
    let mut audio = Vec::new();
    let mut offset = 0;
    for size in [1, 3, 7, steps] {
        let count = size.min(steps - offset);
        if count == 0 {
            break;
        }
        let denorm = latents
            .narrow(1, offset, count)?
            .broadcast_mul(&model.flow_lm.emb_std)?
            .broadcast_add(&model.flow_lm.emb_mean)?
            .transpose(1, 2)?;
        audio.push(model.mimi.decode_from_latent(
            &model.mimi.quantize(&denorm)?,
            &mut partition_state,
            offset,
        )?);
        offset += count;
    }
    close(
        &Tensor::cat(&audio, 2)?,
        &Tensor::cat(&expected_pcm, 2)?,
        2e-3,
        "partitioned PCM",
    )?;
    Ok(())
}
