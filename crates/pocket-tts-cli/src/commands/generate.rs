//! Generate command implementation
//!
//! Provides `pocket-tts generate` for text-to-speech synthesis.

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use owo_colors::OwoColorize;
use pocket_tts::TTSModel;
use pocket_tts::anyhow::Result;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::voice::resolve_voice;
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

    /// Path to a custom YAML config file, or config name (e.g., "english_2026-01").
    /// Incompatible with --language.
    #[arg(long)]
    pub config: Option<String>,

    /// Sampling temperature. Defaults to the model's recommended value
    /// (0.3 for English, 0.7 otherwise).
    #[arg(long)]
    pub temperature: Option<f32>,

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

    /// Print internal timing measurements to stderr
    #[arg(long)]
    pub timings: bool,
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
    let total_start = Instant::now();
    let quiet = args.quiet || args.stream;

    // Print banner
    if !quiet {
        print_banner();
    }

    // Set up device
    let device = if args.use_metal {
        #[cfg(feature = "metal")]
        {
            pocket_tts::candle_core::Device::new_metal(0)?
        }
        #[cfg(not(feature = "metal"))]
        {
            pocket_tts::anyhow::bail!("Metal feature not enabled. Rebuild with --features metal");
        }
    } else {
        pocket_tts::candle_core::Device::Cpu
    };

    if !quiet {
        println!("  {} Using device: {:?}", "▶".cyan(), device);
    }

    // Resolve model identifier: --language, --config, or --variant (deprecated)
    let model_id = resolve_model_id(args.language.as_deref(), args.config.as_deref())?;

    // Load model
    info!(quiet, "{} Loading model ({})...", "▶".cyan(), model_id);

    let quantized = args.quantized;

    let load_start = Instant::now();
    let model = if quantized {
        #[cfg(feature = "quantized")]
        {
            TTSModel::load_quantized_with_optional_params_device(
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
            pocket_tts::anyhow::bail!(
                "Quantization feature not enabled. Rebuild with --features quantized"
            );
        }
    } else {
        TTSModel::load_with_optional_params_device(
            &model_id,
            args.temperature,
            args.lsd_decode_steps,
            args.eos_threshold,
            args.noise_clamp,
            &device,
        )?
    };
    let model_load = load_start.elapsed();

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

    let voice_start = Instant::now();
    let voice_state = resolve_voice(&model, args.voice.as_deref())?;
    let voice_resolve = voice_start.elapsed();

    info!(quiet, "  {} Voice ready", "✓".green());

    // Generate
    let generation = if args.stream {
        run_streaming(&model, &text, &voice_state)
    } else {
        run_to_file(&model, &args, &text, &voice_state, quiet)
    }?;

    if args.timings {
        print_timings(model_load, voice_resolve, generation, total_start.elapsed());
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct GenerationTimings {
    inference: Duration,
    concat: Duration,
    wav_write: Duration,
    audio_duration_sec: f32,
    chunks: usize,
}

/// Run streaming generation to stdout
fn run_streaming(
    model: &TTSModel,
    text: &str,
    voice_state: &pocket_tts::ModelState,
) -> Result<GenerationTimings> {
    use std::io::Write;
    let mut stdout = std::io::stdout();
    let mut chunks = 0usize;
    let mut total_samples = 0usize;

    let inference_start = Instant::now();
    for chunk_res in model.generate_stream_long(text, voice_state) {
        let chunk = chunk_res?;
        let dims = chunk.dims();
        total_samples += if dims.len() == 2 { dims[1] } else { dims[0] };
        chunks += 1;

        // Convert tensor to 16-bit PCM
        let chunk = chunk.squeeze(0)?;
        let bytes = pocket_tts::audio::pcm_i16_le_bytes(&chunk)?;
        stdout.write_all(&bytes)?;
        stdout.flush()?;
    }
    model.device.synchronize()?;
    let inference = inference_start.elapsed();

    Ok(GenerationTimings {
        inference,
        concat: Duration::ZERO,
        wav_write: Duration::ZERO,
        audio_duration_sec: total_samples as f32 / model.sample_rate as f32,
        chunks,
    })
}

/// Run generation to file with progress bar
fn run_to_file(
    model: &TTSModel,
    args: &GenerateArgs,
    text: &str,
    voice_state: &pocket_tts::ModelState,
    quiet: bool,
) -> Result<GenerationTimings> {
    use pocket_tts::candle_core::Tensor;

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

    let inference_start = Instant::now();
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
    model.device.synchronize()?;
    let inference = inference_start.elapsed();

    pb.finish_and_clear();

    // Concatenate all audio chunks
    if audio_chunks.is_empty() {
        pocket_tts::anyhow::bail!("No audio generated - text may be too short or invalid");
    }
    let concat_start = Instant::now();
    let audio = Tensor::cat(&audio_chunks, 2)?;
    let audio = audio.squeeze(0)?; // Remove batch dimension
    model.device.synchronize()?;
    let concat = concat_start.elapsed();

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
    let wav_write_start = Instant::now();
    pocket_tts::audio::write_wav(&args.output, &audio, model.sample_rate as u32)?;
    let wav_write = wav_write_start.elapsed();

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

    Ok(GenerationTimings {
        inference,
        concat,
        wav_write,
        audio_duration_sec: duration_sec,
        chunks: audio_chunks.len(),
    })
}

fn print_timings(
    model_load: Duration,
    voice_resolve: Duration,
    generation: GenerationTimings,
    total: Duration,
) {
    eprintln!("{}", format_timing_line("model_load", model_load));
    eprintln!("{}", format_timing_line("voice_resolve", voice_resolve));
    eprintln!("{}", format_timing_line("inference", generation.inference));
    eprintln!("{}", format_timing_line("concat", generation.concat));
    eprintln!("{}", format_timing_line("wav_write", generation.wav_write));
    eprintln!("{}", format_timing_line("total", total));
    eprintln!(
        "TIMING audio_duration_s={:.3}",
        generation.audio_duration_sec
    );
    eprintln!("TIMING chunks={}", generation.chunks);
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

/// Resolve the model identifier from --language or --config.
pub fn format_timing_line(name: &str, duration: std::time::Duration) -> String {
    format!("TIMING {name}_ms={:.3}", duration.as_secs_f64() * 1000.0)
}

pub fn resolve_model_id(language: Option<&str>, config: Option<&str>) -> Result<String> {
    if language.is_some() && config.is_some() {
        pocket_tts::anyhow::bail!("Cannot specify both --language and --config. Choose one.");
    }

    if let Some(lang) = language {
        if lang == "french" {
            pocket_tts::anyhow::bail!(
                "For technical reasons, only a larger 24-layer model is available for French. \
                 Please use --language french_24l instead."
            );
        }
        return Ok(lang.to_string());
    }

    if let Some(cfg) = config {
        return Ok(cfg.to_string());
    }

    Ok(defaults::DEFAULT_LANGUAGE.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{CommandFactory, Parser};
    use std::time::Duration;

    #[test]
    fn timing_line_is_machine_readable_milliseconds() {
        assert_eq!(
            format_timing_line("inference", Duration::from_micros(1_234_567)),
            "TIMING inference_ms=1234.567"
        );
    }

    #[test]
    fn generate_help_includes_timings_flag() {
        let mut help = Vec::new();
        GenerateArgs::command().write_long_help(&mut help).unwrap();
        let help = String::from_utf8(help).unwrap();
        assert!(help.contains("--timings"), "help was:\n{help}");
    }

    #[test]
    fn omitted_temperature_uses_model_recommendation() {
        let args = GenerateArgs::try_parse_from(["pocket-tts"]).unwrap();
        assert_eq!(args.temperature, None);

        let mut help = Vec::new();
        GenerateArgs::command().write_long_help(&mut help).unwrap();
        let help = String::from_utf8(help).unwrap();
        assert!(
            help.contains("model's recommended value"),
            "help was:\n{help}"
        );
    }
}
