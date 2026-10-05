//! Compare native and byte loading with the same released German bundle.
//! Run: cargo test --release -p pocket-tts --test wasm_path_parity -- --ignored
use pocket_tts::TTSModel;
use pocket_tts::anyhow::Result;
use pocket_tts::candle_core::{Device, Tensor};
use pocket_tts::weights::download_if_necessary;

fn equal(actual: &Tensor, expected: &Tensor) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims());
    assert_eq!(
        (actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()?,
        0.0
    );
    Ok(())
}

#[test]
#[ignore = "requires released German weights and preset; missing assets are fatal"]
fn test_german_wasm_vs_cli_path() -> Result<()> {
    let mut cli_model = TTSModel::load("german")?;
    // Like the worker, consume canonical YAML instead of constructing a second
    // schema or pairing today's native model with yesterday's byte bundle.
    let config_yaml = include_bytes!("../config/german.yaml");
    let config: pocket_tts::config::Config = serde_yaml::from_slice(config_yaml)?;
    let weights_uri = if cli_model.has_voice_cloning {
        config.weights_path.as_ref().unwrap()
    } else {
        config.weights_path_without_voice_cloning.as_ref().unwrap()
    };
    let weights = std::fs::read(download_if_necessary(weights_uri)?)?;
    let tokenizer = std::fs::read(download_if_necessary(
        &config.flow_lm.lookup_table.tokenizer_path,
    )?)?;
    let mut byte_model = TTSModel::load_from_bytes(config_yaml, &weights, &tokenizer)?;
    byte_model.has_voice_cloning = cli_model.has_voice_cloning;

    let text = "Es ist klein genug, um in Ihre Tasche zu passen.";
    let native_tokens = cli_model.conditioner.prepare(text, &Device::Cpu)?;
    let byte_tokens = byte_model.conditioner.prepare(text, &Device::Cpu)?;
    assert_eq!(
        native_tokens.to_vec2::<u32>()?,
        byte_tokens.to_vec2::<u32>()?
    );
    equal(
        &cli_model.conditioner.forward(&native_tokens)?,
        &byte_model.conditioner.forward(&byte_tokens)?,
    )?;

    let voice = download_if_necessary(
        "hf://kyutai/pocket-tts-without-voice-cloning/languages/german/embeddings/juergen.safetensors@4e1e0a3e611c51c0b4ed8174fc10f32a54644303",
    )?;
    let native_state = cli_model.get_voice_state_from_prompt_file(&voice)?;
    let byte_state = byte_model.get_voice_state_from_prompt_file(&voice)?;
    // Zero temperature eliminates independent random draws. Unlike a comparison
    // of means or a filter_map, this checks every sample and propagates errors.
    cli_model.temp = 0.0;
    byte_model.temp = 0.0;
    let native_audio = cli_model.generate(text, &native_state)?;
    let byte_audio = byte_model.generate(text, &byte_state)?;
    assert!(native_audio.elem_count() > 24_000);
    equal(&native_audio, &byte_audio)?;
    Ok(())
}
