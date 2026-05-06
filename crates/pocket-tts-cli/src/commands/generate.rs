//! Generate command implementation
//!
//! Provides `pocket-tts generate` for text-to-speech synthesis.

use anyhow::Result;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use owo_colors::OwoColorize;
use pocket_tts::TTSModel;
use std::path::PathBuf;

use crate::voice::{PREDEFINED_VOICES, resolve_voice};
use pocket_tts::config::defaults;

#[derive(Parser, Debug)]
pub struct GenerateArgs {
    /// Text to synthesize (auto-selected per language if not specified)
    #[arg(short, long)]
    pub text: Option<String>,

    /// Voice for synthesis. Can be:
    /// - Predefined name: alba, marius, javert, jean, fantine, cosette, eponine, azelma,
    ///   giovanni, lola, juergen, rafael, estelle
    /// - Path to .wav file for voice cloning
    /// - Path to .safetensors embeddings file
    /// - HuggingFace URL: hf://owner/repo/file.wav
    #[arg(short, long)]
    pub voice: Option<String>,

    /// Output audio file path
    #[arg(short, long, default_value = "output.wav")]
    pub output: PathBuf,

    /// Language for the TTS model (e.g., "english", "french_24l", "german").
    /// Incompatible with --config. Default is "english".
    #[arg(long)]
    pub language: Option<String>,

    /// Path to a custom YAML config file, or legacy variant name (e.g., "b6369a24").
    /// Incompatible with --language.
    #[arg(long)]
    pub config: Option<String>,

    /// Model variant (deprecated, use --language or --config instead)
    #[arg(long, hide = true)]
    pub variant: Option<String>,

    /// Sampling temperature (higher = more variation)
    #[arg(long, default_value = "0.7")]
    pub temperature: f32,

    /// LSD decode steps (more steps = better quality, slower)
    #[arg(long, default_value = "1")]
    pub lsd_decode_steps: usize,

    /// EOS threshold (more negative = longer audio)
    #[arg(long, default_value = "-4.0")]
    pub eos_threshold: f32,

    /// Noise clamp value (optional)
    #[arg(long)]
    pub noise_clamp: Option<f32>,

    /// Frames to generate after EOS detection (optional, auto-estimated if not set)
    #[arg(long)]
    pub frames_after_eos: Option<usize>,

    /// Stream raw PCM audio to stdout (for piping to audio players)
    #[arg(long)]
    pub stream: bool,

    /// Use simulated int8 quantization for inference
    #[arg(long)]
    pub quantized: bool,

    /// Use Metal acceleration (macOS only)
    #[arg(long)]
    pub use_metal: bool,

    /// Suppress all output except errors
    #[arg(short, long)]
    pub quiet: bool,
}

/// Print styled message (respects quiet mode)
macro_rules! info {
    ($quiet:expr, $($arg:tt)*) => {
        if !$quiet {
            println!($($arg)*);
        }
    };
}

pub fn run(args: GenerateArgs) -> Result<()> {
    let quiet = args.quiet || args.stream;

    // Print banner
    if !quiet {
        print_banner();
    }

    // Set up device
    let device = if args.use_metal {
        #[cfg(feature = "metal")]
        {
            candle_core::Device::new_metal(0)?
        }
        #[cfg(not(feature = "metal"))]
        {
            anyhow::bail!("Metal feature not enabled. Rebuild with --features metal");
        }
    } else {
        candle_core::Device::Cpu
    };

    if !quiet {
        println!("  {} Using device: {:?}", "▶".cyan(), device);
    }

    // Resolve model identifier: --language, --config, or --variant (deprecated)
    let model_id = resolve_model_id(
        args.language.as_deref(),
        args.config.as_deref(),
        args.variant.as_deref(),
    )?;

    // Load model
    info!(quiet, "{} Loading model ({})...", "▶".cyan(), model_id);

    let quantized = args.quantized;

    let model = if quantized {
        #[cfg(feature = "quantized")]
        {
            TTSModel::load_quantized_with_params_device(
                &model_id,
                args.temperature,
                args.lsd_decode_steps,
                args.eos_threshold,
                args.noise_clamp,
                &device,
            )?
        }
        #[cfg(not(feature = "quantized"))]
        {
            anyhow::bail!("Quantization feature not enabled. Rebuild with --features quantized");
        }
    } else {
        TTSModel::load_with_params_device(
            &model_id,
            args.temperature,
            args.lsd_decode_steps,
            args.eos_threshold,
            args.noise_clamp,
            &device,
        )?
    };

    info!(
        quiet,
        "  {} Model loaded (sample rate: {}Hz)",
        "✓".green(),
        model.sample_rate
    );

    // Resolve text and voice (use language-appropriate defaults if not specified)
    let language = model.language().unwrap_or("english");
    let text = args
        .text
        .clone()
        .unwrap_or_else(|| defaults::default_text_for_language(language).to_string());

    let voice_display = args
        .voice
        .as_deref()
        .unwrap_or_else(|| defaults::default_voice_for_language(language));
    info!(
        quiet,
        "{} Using voice: {}",
        "▶".cyan(),
        voice_display.yellow()
    );

    let voice_state = resolve_voice(&model, args.voice.as_deref())?;

    info!(quiet, "  {} Voice ready", "✓".green());

    // Generate
    if args.stream {
        run_streaming(&model, &text, &voice_state)
    } else {
        run_to_file(&model, &args, &text, &voice_state, quiet)
    }
}

/// Run streaming generation to stdout
fn run_streaming(model: &TTSModel, text: &str, voice_state: &pocket_tts::ModelState) -> Result<()> {
    use std::io::Write;
    let mut stdout = std::io::stdout();

    for chunk_res in model.generate_stream_long(text, voice_state) {
        let chunk = chunk_res?;
        // Convert tensor to 16-bit PCM
        let chunk = chunk.squeeze(0)?;
        let bytes = pocket_tts::audio::pcm_i16_le_bytes(&chunk)?;
        stdout.write_all(&bytes)?;
        stdout.flush()?;
    }

    Ok(())
}

/// Run generation to file with progress bar
fn run_to_file(
    model: &TTSModel,
    args: &GenerateArgs,
    text: &str,
    voice_state: &pocket_tts::ModelState,
    quiet: bool,
) -> Result<()> {
    use candle_core::Tensor;

    info!(
        quiet,
        "{} Generating: \"{}\"",
        "▶".cyan(),
        truncate_text(text, 60).italic()
    );

    let total_steps = model.estimate_generation_steps(text) as u64;

    let pb = if quiet {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new(total_steps);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.cyan} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}",
                )
                .unwrap()
                .progress_chars("█▓░"),
        );
        pb.set_message("generating...");
        pb
    };

    let mut audio_chunks = Vec::new();
    let mut total_samples = 0;

    for chunk_res in model.generate_stream_long(text, voice_state) {
        let chunk = chunk_res?;
        let dims = chunk.dims();
        let samples = if dims.len() == 2 { dims[1] } else { dims[0] };
        total_samples += samples;

        audio_chunks.push(chunk);
        pb.inc(1);
        pb.set_message(format!(
            "{:.2}s generated",
            total_samples as f32 / model.sample_rate as f32
        ));
    }

    pb.finish_and_clear();

    // Concatenate all audio chunks
    if audio_chunks.is_empty() {
        anyhow::bail!("No audio generated - text may be too short or invalid");
    }
    let audio = Tensor::cat(&audio_chunks, 2)?;
    let audio = audio.squeeze(0)?; // Remove batch dimension

    let dims = audio.dims();
    let num_samples = if dims.len() == 2 { dims[1] } else { dims[0] };
    let duration_sec = num_samples as f32 / model.sample_rate as f32;

    // Save to file
    info!(
        quiet,
        "{} Saving to: {}",
        "▶".cyan(),
        args.output.display().yellow()
    );
    pocket_tts::audio::write_wav(&args.output, &audio, model.sample_rate as u32)?;

    // Success message
    if !quiet {
        println!();
        println!(
            "  {} {}",
            "✓".green().bold(),
            "Audio generated successfully!".green().bold()
        );
        println!(
            "    Duration: {:.2}s ({} samples @ {}Hz)",
            duration_sec, num_samples, model.sample_rate
        );
        println!("    Output:   {}", args.output.display().cyan());
        println!();
        println!(
            "  {} {}",
            "💡".dimmed(),
            format!("Play with: ffplay -autoexit {:?}", args.output).dimmed()
        );
    }

    Ok(())
}

/// Print startup banner
fn print_banner() {
    println!();
    println!("  {}  {}", "🗣️".bold(), "Pocket TTS".bold().cyan());
    println!(
        "      {} {}",
        "Rust/Candle port".dimmed(),
        format!("v{}", env!("CARGO_PKG_VERSION")).dimmed()
    );
    println!();
}

/// Truncate text for display
fn truncate_text(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        format!("{}...", &text[..max_len - 3])
    }
}

/// Print available voices (for help text)
pub fn available_voices_help() -> String {
    format!("Predefined voices: {}", PREDEFINED_VOICES.join(", "))
}

/// Resolve the model identifier from --language, --config, or --variant (deprecated)
pub fn resolve_model_id(
    language: Option<&str>,
    config: Option<&str>,
    variant: Option<&str>,
) -> Result<String> {
    // Check for conflicting options
    let specified_count = [language.is_some(), config.is_some(), variant.is_some()]
        .iter()
        .filter(|&&x| x)
        .count();
    if specified_count > 1 {
        anyhow::bail!(
            "Cannot specify multiple of --language, --config, and --variant. Choose one."
        );
    }

    if let Some(lang) = language {
        if lang == "french" {
            anyhow::bail!(
                "For technical reasons, only a larger 24-layer model is available for French. \
                 Please use --language french_24l instead."
            );
        }
        return Ok(lang.to_string());
    }

    if let Some(cfg) = config {
        return Ok(cfg.to_string());
    }

    if let Some(var) = variant {
        return Ok(var.to_string());
    }

    // Default to the configured default language
    Ok(defaults::DEFAULT_LANGUAGE.to_string())
}
