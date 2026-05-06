//! Parity tests comparing Rust v2 generation against Python reference.
//!
//! Run with: cargo test -p pocket-tts --release --test v2_parity_test
//!
//! Prerequisites: v2 english model must be downloaded (run `pocket-tts generate --language english` first)

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use pocket_tts::TTSModel;

fn has_v2_model() -> bool {
    TTSModel::load("english").is_ok()
}

#[test]
fn test_v2_voice_state_import() -> Result<()> {
    if !has_v2_model() {
        eprintln!("Skipping: v2 english model not available");
        return Ok(());
    }

    let model = TTSModel::load("english")?;

    // Load voice state for alba
    let alba_path = pocket_tts::weights::download_if_necessary(
        "hf://kyutai/pocket-tts-without-voice-cloning/languages/english/embeddings/alba.safetensors@e041936c75475d350b405bc870bcf7c22da4e9e6",
    )?;
    let voice_state = model.get_voice_state_from_prompt_file(&alba_path)?;

    // Check that we have 6 layers of FlowLM transformer state
    let mut layer_count = 0;
    for (name, state) in &voice_state {
        if name.contains("self_attn") {
            layer_count += 1;
            // Check k_buf and v_buf exist
            assert!(state.contains_key("k_buf"), "Missing k_buf in {}", name);
            assert!(state.contains_key("v_buf"), "Missing v_buf in {}", name);
            assert!(state.contains_key("pos"), "Missing pos in {}", name);

            let k_buf = &state["k_buf"];
            let v_buf = &state["v_buf"];
            println!(
                "{}: k_buf={:?}, v_buf={:?}",
                name,
                k_buf.dims(),
                v_buf.dims()
            );

            // Should be (1, 16, 126, 64) for 16 heads, 64 dim, 126 positions
            assert_eq!(
                k_buf.dims(),
                &[1, 16, 126, 64],
                "k_buf shape wrong for {}",
                name
            );
            assert_eq!(
                v_buf.dims(),
                &[1, 16, 126, 64],
                "v_buf shape wrong for {}",
                name
            );

            // Check offset = 126
            let pos = state["pos"].to_scalar::<u32>()? as usize;
            assert_eq!(pos, 126, "offset wrong for {}", name);
        }
    }
    assert_eq!(layer_count, 6, "Expected 6 transformer layers");

    // Verify k_buf mean matches Python reference
    // Python: transformer.layers.0.self_attn/k_buf mean=-0.058732
    let layer0_name = "flow_lm.transformer.layers.0.self_attn";
    let k_buf = &voice_state[layer0_name]["k_buf"];
    let k_mean = k_buf.mean_all()?.to_scalar::<f32>()?;
    println!("Layer 0 k_buf mean: {} (Python: -0.058732)", k_mean);
    assert!(
        (k_mean - (-0.058732)).abs() < 0.001,
        "k_buf mean mismatch: got {}",
        k_mean
    );

    println!("✓ Voice state import parity check passed");
    Ok(())
}

#[test]
fn test_v2_tokenization() -> Result<()> {
    if !has_v2_model() {
        eprintln!("Skipping: v2 english model not available");
        return Ok(());
    }

    let model = TTSModel::load("english")?;

    // Python: token_ids = [2994, 578, 263] for "Hello world."
    let tokens = model.conditioner.prepare("Hello world.", &Device::Cpu)?;
    // tokens is [1, N] (batched)
    let tokens_i64 = tokens.to_dtype(DType::I64)?;
    let token_vec: Vec<Vec<i64>> = tokens_i64.to_vec2()?;
    println!("Rust tokens: {:?}", token_vec[0]);
    assert_eq!(token_vec[0], vec![2994, 578, 263], "Token mismatch");

    // Text embeddings
    let text_emb = model.conditioner.forward(&tokens)?;
    let emb_mean = text_emb.mean_all()?.to_scalar::<f32>()?;
    println!("Text emb mean: {} (Python: 0.000163)", emb_mean);
    assert!(
        (emb_mean - 0.000163).abs() < 0.001,
        "Text emb mean mismatch: got {}",
        emb_mean
    );

    println!("✓ Tokenization parity check passed");
    Ok(())
}

#[test]
fn test_v2_first_generation_step() -> Result<()> {
    if !has_v2_model() {
        eprintln!("Skipping: v2 english model not available");
        return Ok(());
    }

    let model = TTSModel::load("english")?;

    // Load voice state
    let alba_path = pocket_tts::weights::download_if_necessary(
        "hf://kyutai/pocket-tts-without-voice-cloning/languages/english/embeddings/alba.safetensors@e041936c75475d350b405bc870bcf7c22da4e9e6",
    )?;
    let voice_state = model.get_voice_state_from_prompt_file(&alba_path)?;

    // Check state positions before text
    let cursor_before = pocket_tts::voice_state::get_attention_cursor(
        &voice_state,
        "flow_lm.transformer.layers.0.self_attn",
    );
    println!(
        "State before text: pos={}, len={}",
        cursor_before.pos, cursor_before.len
    );
    assert_eq!(cursor_before.pos, 126, "Initial pos should be 126");
    assert_eq!(cursor_before.len, 126, "Initial len should be 126");

    // Process text through transformer (like generate_stream_segment does)
    let mut state = voice_state.clone();
    let text = "Hello world.";
    let tokens = model.conditioner.prepare(text, &Device::Cpu)?;
    let text_emb = model.conditioner.forward(&tokens)?;
    println!("Text emb shape: {:?}", text_emb.dims());

    // Process text through transformer
    model
        .flow_lm
        .transformer
        .forward(&text_emb, &mut state, 0)?;

    let cursor_after_text = pocket_tts::voice_state::get_attention_cursor(
        &state,
        "flow_lm.transformer.layers.0.self_attn",
    );
    println!(
        "State after text: pos={}, len={}",
        cursor_after_text.pos, cursor_after_text.len
    );
    // Should be 126 + 3 = 129
    assert_eq!(cursor_after_text.pos, 129, "Pos after text should be 129");

    // Now run first FlowLM forward (BOS as input)
    let bos = model.flow_lm.bos_emb.clone().reshape((1, 1, model.ldim))?;
    let empty_text = Tensor::zeros((1, 0, model.dim), DType::F32, &model.device)?;

    // Pre-compute time embeddings
    let time_embeddings = model.flow_lm.flow_net.compute_time_embeddings(
        model.lsd_decode_steps,
        &model.device,
        DType::F32,
    )?;

    let (latent, is_eos) = model.flow_lm.forward(
        &bos,
        &empty_text,
        &mut state,
        &time_embeddings,
        model.temp,
        model.eos_threshold,
        0, // step (ignored by transformer, reads from state)
    )?;

    use std::io::Write;
    let mut f = std::fs::File::create(std::env::temp_dir().join("rust_parity_output.txt"))?;

    writeln!(f, "Latent step 1: shape={:?}", latent.dims())?;
    let latent_flat = latent.flatten_all()?;
    let latent_vals: Vec<f32> = latent_flat.to_vec1()?;
    let latent_mean = latent_flat.mean_all()?.to_scalar::<f32>()?;
    writeln!(f, "Latent step 1 mean: {} (Python: 0.000268)", latent_mean)?;
    writeln!(f, "Latent first 5: {:?}", &latent_vals[..5])?;
    writeln!(f, "Is EOS: {}", is_eos)?;

    // Denormalize
    let latent_squeezed = latent.squeeze(0)?;
    let denorm = latent_squeezed
        .broadcast_mul(&model.flow_lm.emb_std)?
        .broadcast_add(&model.flow_lm.emb_mean)?;
    let denorm_vals: Vec<f32> = denorm.to_vec1()?;
    let denorm_mean = denorm.mean_all()?.to_scalar::<f32>()?;
    writeln!(f, "Denormalized mean: {} (Python: -0.160415)", denorm_mean)?;
    writeln!(f, "Denormalized first 5: {:?}", &denorm_vals[..5])?;
    // Python: [-0.8214, 0.5737, -0.4604, 0.0763, 0.3748]

    writeln!(f, "✓ First generation step completed")?;
    Ok(())
}

#[test]
fn test_v2_german_parity() -> Result<()> {
    if TTSModel::load("german").is_err() {
        eprintln!("Skipping: v2 german model not available");
        return Ok(());
    }

    use std::io::Write;
    let mut f = std::fs::File::create(std::env::temp_dir().join("rust_german_parity.txt"))?;

    let model = TTSModel::load("german")?;
    writeln!(f, "remove_semicolons={}", model.remove_semicolons)?;
    writeln!(
        f,
        "pad_with_spaces={}",
        model.pad_with_spaces_for_short_inputs
    )?;
    assert!(
        model.remove_semicolons,
        "German should have remove_semicolons=true"
    );
    assert!(!model.pad_with_spaces_for_short_inputs);

    // Load voice state
    let juergen_path = pocket_tts::weights::download_if_necessary(
        "hf://kyutai/pocket-tts-without-voice-cloning/languages/german/embeddings/juergen.safetensors@e041936c75475d350b405bc870bcf7c22da4e9e6",
    )?;
    let voice_state = model.get_voice_state_from_prompt_file(&juergen_path)?;

    // German voice state offset should be 127 (not 126 like English)
    let cursor = pocket_tts::voice_state::get_attention_cursor(
        &voice_state,
        "flow_lm.transformer.layers.0.self_attn",
    );
    writeln!(f, "Voice state pos={}, len={}", cursor.pos, cursor.len)?;
    assert_eq!(cursor.pos, 127, "German voice state pos should be 127");
    assert_eq!(cursor.len, 127, "German voice state len should be 127");

    // Check k_buf mean for layer 0
    let k_buf = &voice_state["flow_lm.transformer.layers.0.self_attn"]["k_buf"];
    let k_mean = k_buf.mean_all()?.to_scalar::<f32>()?;
    writeln!(f, "Layer 0 k_buf mean: {} (Python: 0.00168773)", k_mean)?;
    assert!(
        (k_mean - 0.001688).abs() < 0.001,
        "k_buf mean mismatch: {}",
        k_mean
    );

    // Prepare text (remove_semicolons should NOT change this text since no semicolons)
    let text = "Es ist klein genug, um in Ihre Tasche zu passen.";
    let tokens = model.conditioner.prepare(text, &Device::Cpu)?;
    let tokens_i64 = tokens.to_dtype(DType::I64)?;
    let token_vec: Vec<Vec<i64>> = tokens_i64.to_vec2()?;
    writeln!(f, "Token IDs: {:?}", &token_vec[0])?;
    // Python: [392, 270, 621, 1384, 261, 357, 277, 1103, 264, 1452, 435, 264, 281, 545, 263, 262]
    assert_eq!(
        token_vec[0].len(),
        16,
        "Expected 16 tokens, got {}",
        token_vec[0].len()
    );
    assert_eq!(
        token_vec[0],
        vec![392, 270, 621, 1384, 261, 357, 277, 1103, 264, 1452, 435, 264, 281, 545, 263, 262]
    );

    // Text embeddings
    let text_emb = model.conditioner.forward(&tokens)?;
    let emb_mean = text_emb.mean_all()?.to_scalar::<f32>()?;
    writeln!(f, "Text emb mean: {} (Python: -0.00004402)", emb_mean)?;
    assert!(
        (emb_mean - (-0.00004402)).abs() < 0.001,
        "Text emb mean mismatch: {}",
        emb_mean
    );

    // Process text through transformer
    let mut state = voice_state.clone();
    model
        .flow_lm
        .transformer
        .forward(&text_emb, &mut state, 0)?;

    let cursor_after = pocket_tts::voice_state::get_attention_cursor(
        &state,
        "flow_lm.transformer.layers.0.self_attn",
    );
    writeln!(
        f,
        "After text: pos={}, len={}",
        cursor_after.pos, cursor_after.len
    )?;
    // 127 + 16 = 143
    assert_eq!(
        cursor_after.pos, 143,
        "After text pos should be 143, got {}",
        cursor_after.pos
    );

    // First FlowLM step (BOS)
    let bos = model.flow_lm.bos_emb.clone().reshape((1, 1, model.ldim))?;
    let empty_text = Tensor::zeros((1, 0, model.dim), DType::F32, &model.device)?;
    let time_embeddings = model.flow_lm.flow_net.compute_time_embeddings(
        model.lsd_decode_steps,
        &model.device,
        DType::F32,
    )?;

    let (latent, _is_eos) = model.flow_lm.forward(
        &bos,
        &empty_text,
        &mut state,
        &time_embeddings,
        model.temp,
        model.eos_threshold,
        0,
    )?;

    let latent_flat = latent.flatten_all()?;
    let latent_mean = latent_flat.mean_all()?.to_scalar::<f32>()?;
    writeln!(
        f,
        "Step 1 latent mean: {} (Python: 0.39381978)",
        latent_mean
    )?;
    // Due to random noise, means won't match exactly, but should be in same ballpark
    // The key check is that the structure is correct (offset, token count, etc.)

    let latent_vals: Vec<f32> = latent_flat.to_vec1()?;
    writeln!(f, "Step 1 latent first 5: {:?}", &latent_vals[..5])?;
    // Python: [-0.5754, 0.1008, -0.3085, 0.0823, 0.4127]

    writeln!(f, "✓ German parity check passed")?;
    Ok(())
}
