//! Test that simulates the exact WASM loading path for German
//! and compares to the normal CLI loading path.
//!
//! Run: cargo test -p pocket-tts --release --test wasm_path_parity -- --nocapture

use anyhow::Result;
use candle_core::DType;
use pocket_tts::TTSModel;

#[test]
fn test_german_wasm_vs_cli_path() -> Result<()> {
    if TTSModel::load("german").is_err() {
        eprintln!("Skipping: german model not available");
        return Ok(());
    }

    use std::io::Write;
    let tmp = std::env::temp_dir();
    let mut f = std::fs::File::create(tmp.join("wasm_path_parity.txt"))?;

    // === Path 1: Normal CLI load (known working) ===
    writeln!(f, "=== CLI path: TTSModel::load(\"german\") ===")?;
    let cli_model = TTSModel::load("german")?;
    writeln!(f, "  remove_semicolons={}", cli_model.remove_semicolons)?;
    writeln!(
        f,
        "  pad_with_spaces={}",
        cli_model.pad_with_spaces_for_short_inputs
    )?;
    writeln!(f, "  dim={}, ldim={}", cli_model.dim, cli_model.ldim)?;
    writeln!(
        f,
        "  insert_bos={}",
        cli_model.flow_lm.insert_bos_before_voice
    )?;
    writeln!(
        f,
        "  bos_before_voice={}",
        cli_model.flow_lm.bos_before_voice.is_some()
    )?;

    // === Path 2: WASM-style load_from_bytes ===
    writeln!(f, "\n=== WASM path: TTSModel::load_from_bytes ===")?;

    // Build the same config YAML that the WASM worker generates for German
    let config_yaml = r#"remove_semicolons: true
flow_lm:
  insert_bos_before_voice: true
  dtype: float32
  flow:
    depth: 6
    dim: 512
  transformer:
    d_model: 1024
    hidden_scale: 4
    max_period: 10000
    num_heads: 16
    num_layers: 6
  lookup_table:
    dim: 1024
    n_bins: 4000
    tokenizer: sentencepiece
    tokenizer_path: dummy
mimi:
  dtype: float32
  sample_rate: 24000
  inner_dim: 32
  outer_dim: 512
  channels: 1
  frame_rate: 12.5
  seanet:
    dimension: 512
    channels: 1
    n_filters: 64
    n_residual_layers: 1
    ratios: [6, 5, 4]
    kernel_size: 7
    residual_kernel_size: 3
    last_kernel_size: 3
    dilation_base: 2
    pad_mode: constant
    compress: 2
  transformer:
    d_model: 512
    num_heads: 8
    num_layers: 2
    layer_scale: 0.01
    context: 250
    dim_feedforward: 2048
    input_dimension: 512
    output_dimensions: [512]
  quantizer:
    dimension: 32
    output_dimension: 512
"#;

    // Load the same files the WASM worker fetches
    let weights_path = pocket_tts::weights::download_if_necessary(
        "hf://kyutai/pocket-tts-without-voice-cloning/languages/german/model.safetensors@d29db7978e464fb90cb3359ee0c69a273b9142cc",
    )?;
    let weights_bytes = std::fs::read(&weights_path)?;
    writeln!(f, "  weights: {} bytes", weights_bytes.len())?;

    let tokenizer_path = pocket_tts::weights::download_if_necessary(
        "hf://kyutai/pocket-tts-without-voice-cloning/languages/german/tokenizer.model@d29db7978e464fb90cb3359ee0c69a273b9142cc",
    )?;
    let tokenizer_bytes = std::fs::read(&tokenizer_path)?;
    writeln!(f, "  tokenizer: {} bytes", tokenizer_bytes.len())?;

    let wasm_model =
        TTSModel::load_from_bytes(config_yaml.as_bytes(), &weights_bytes, &tokenizer_bytes)?;
    writeln!(f, "  remove_semicolons={}", wasm_model.remove_semicolons)?;
    writeln!(
        f,
        "  pad_with_spaces={}",
        wasm_model.pad_with_spaces_for_short_inputs
    )?;
    writeln!(f, "  dim={}, ldim={}", wasm_model.dim, wasm_model.ldim)?;
    writeln!(
        f,
        "  insert_bos={}",
        wasm_model.flow_lm.insert_bos_before_voice
    )?;
    writeln!(
        f,
        "  bos_before_voice={}",
        wasm_model.flow_lm.bos_before_voice.is_some()
    )?;

    // === Compare tokenization ===
    let text = "Es ist klein genug, um in Ihre Tasche zu passen.";

    let cli_tokens = cli_model
        .conditioner
        .prepare(text, &candle_core::Device::Cpu)?;
    let wasm_tokens = wasm_model
        .conditioner
        .prepare(text, &candle_core::Device::Cpu)?;

    let cli_ids: Vec<Vec<i64>> = cli_tokens.to_dtype(DType::I64)?.to_vec2()?;
    let wasm_ids: Vec<Vec<i64>> = wasm_tokens.to_dtype(DType::I64)?.to_vec2()?;

    writeln!(f, "\n=== Tokenization ===")?;
    writeln!(f, "  CLI tokens:  {:?}", cli_ids[0])?;
    writeln!(f, "  WASM tokens: {:?}", wasm_ids[0])?;
    let tokens_match = cli_ids[0] == wasm_ids[0];
    writeln!(f, "  Match: {}", tokens_match)?;
    assert!(
        tokens_match,
        "TOKEN MISMATCH!\n  CLI:  {:?}\n  WASM: {:?}",
        cli_ids[0], wasm_ids[0]
    );

    // === Compare text embeddings ===
    let cli_emb = cli_model.conditioner.forward(&cli_tokens)?;
    let wasm_emb = wasm_model.conditioner.forward(&wasm_tokens)?;

    let cli_emb_mean = cli_emb.mean_all()?.to_scalar::<f32>()?;
    let wasm_emb_mean = wasm_emb.mean_all()?.to_scalar::<f32>()?;

    writeln!(f, "\n=== Text Embeddings ===")?;
    writeln!(f, "  CLI  mean: {}", cli_emb_mean)?;
    writeln!(f, "  WASM mean: {}", wasm_emb_mean)?;
    let emb_match = (cli_emb_mean - wasm_emb_mean).abs() < 1e-6;
    writeln!(f, "  Match: {}", emb_match)?;
    assert!(
        emb_match,
        "EMBEDDING MISMATCH: CLI={} WASM={}",
        cli_emb_mean, wasm_emb_mean
    );

    // === Load voice and compare ===
    let voice_path = pocket_tts::weights::download_if_necessary(
        "hf://kyutai/pocket-tts-without-voice-cloning/languages/german/embeddings/juergen.safetensors@e041936c75475d350b405bc870bcf7c22da4e9e6",
    )?;

    let cli_voice = cli_model.get_voice_state_from_prompt_file(&voice_path)?;
    let wasm_voice = wasm_model.get_voice_state_from_prompt_file(&voice_path)?;

    writeln!(f, "\n=== Voice State ===")?;
    let cli_cursor = pocket_tts::voice_state::get_attention_cursor(
        &cli_voice,
        "flow_lm.transformer.layers.0.self_attn",
    );
    let wasm_cursor = pocket_tts::voice_state::get_attention_cursor(
        &wasm_voice,
        "flow_lm.transformer.layers.0.self_attn",
    );
    writeln!(f, "  CLI  pos={}, len={}", cli_cursor.pos, cli_cursor.len)?;
    writeln!(f, "  WASM pos={}, len={}", wasm_cursor.pos, wasm_cursor.len)?;
    assert_eq!(cli_cursor.pos, wasm_cursor.pos, "Voice state pos mismatch");
    assert_eq!(cli_cursor.len, wasm_cursor.len, "Voice state len mismatch");

    // Compare k_buf values
    let cli_k = &cli_voice["flow_lm.transformer.layers.0.self_attn"]["k_buf"];
    let wasm_k = &wasm_voice["flow_lm.transformer.layers.0.self_attn"]["k_buf"];
    let cli_k_mean = cli_k.mean_all()?.to_scalar::<f32>()?;
    let wasm_k_mean = wasm_k.mean_all()?.to_scalar::<f32>()?;
    writeln!(f, "  CLI  k_buf mean: {}", cli_k_mean)?;
    writeln!(f, "  WASM k_buf mean: {}", wasm_k_mean)?;
    assert!(
        (cli_k_mean - wasm_k_mean).abs() < 1e-5,
        "k_buf mean mismatch: CLI={} WASM={}",
        cli_k_mean,
        wasm_k_mean
    );

    // === Generate and save WAV for both ===
    writeln!(f, "\n=== Full Generation ===")?;

    let cli_chunks: Vec<_> = cli_model
        .generate_stream(text, &cli_voice)
        .filter_map(|r| r.ok())
        .collect();
    let wasm_chunks: Vec<_> = wasm_model
        .generate_stream(text, &wasm_voice)
        .filter_map(|r| r.ok())
        .collect();

    let cli_total: usize = cli_chunks
        .iter()
        .map(|t| *t.dims().last().unwrap_or(&0))
        .sum();
    let wasm_total: usize = wasm_chunks
        .iter()
        .map(|t| *t.dims().last().unwrap_or(&0))
        .sum();
    writeln!(
        f,
        "  CLI:  {} frames, {} samples ({:.2}s)",
        cli_chunks.len(),
        cli_total,
        cli_total as f64 / 24000.0
    )?;
    writeln!(
        f,
        "  WASM: {} frames, {} samples ({:.2}s)",
        wasm_chunks.len(),
        wasm_total,
        wasm_total as f64 / 24000.0
    )?;

    // Save both as WAV
    let cli_audio = candle_core::Tensor::cat(&cli_chunks, 2)?.squeeze(0)?;
    let wasm_audio = candle_core::Tensor::cat(&wasm_chunks, 2)?.squeeze(0)?;

    let cli_wav = tmp.join("parity_cli_german.wav");
    let wasm_wav = tmp.join("parity_wasm_german.wav");
    pocket_tts::audio::write_wav(&cli_wav, &cli_audio, cli_model.sample_rate as u32)?;
    pocket_tts::audio::write_wav(&wasm_wav, &wasm_audio, wasm_model.sample_rate as u32)?;
    writeln!(f, "  Saved {:?}", cli_wav)?;
    writeln!(f, "  Saved {:?}", wasm_wav)?;

    writeln!(f, "\n✓ All checks passed")?;
    Ok(())
}
