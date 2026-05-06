//! Configuration types for pocket-tts, matching Python's utils/config.py

use serde::Deserialize;
use std::path::Path;

/// Flow network configuration
#[derive(Debug, Clone, Deserialize)]
pub struct FlowConfig {
    pub dim: usize,
    pub depth: usize,
}

/// Transformer configuration for FlowLM
#[derive(Debug, Clone, Deserialize)]
pub struct FlowLMTransformerConfig {
    pub hidden_scale: usize,
    pub max_period: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
}

/// Lookup table (text conditioner) configuration
#[derive(Debug, Clone, Deserialize)]
pub struct LookupTableConfig {
    pub dim: usize,
    pub n_bins: usize,
    pub tokenizer: String,
    pub tokenizer_path: String,
}

/// FlowLM model configuration
#[derive(Debug, Clone, Deserialize)]
pub struct FlowLMConfig {
    pub dtype: String,
    pub flow: FlowConfig,
    pub transformer: FlowLMTransformerConfig,
    pub lookup_table: LookupTableConfig,
    #[serde(default)]
    pub insert_bos_before_voice: bool,
    #[serde(default)]
    pub weights_path: Option<String>,
}

/// SEANet encoder/decoder configuration
#[derive(Debug, Clone, Deserialize)]
pub struct SEANetConfig {
    pub dimension: usize,
    pub channels: usize,
    pub n_filters: usize,
    pub n_residual_layers: usize,
    pub ratios: Vec<usize>,
    pub kernel_size: usize,
    pub residual_kernel_size: usize,
    pub last_kernel_size: usize,
    pub dilation_base: usize,
    pub pad_mode: String,
    pub compress: usize,
}

/// Transformer configuration for Mimi
#[derive(Debug, Clone, Deserialize)]
pub struct MimiTransformerConfig {
    pub d_model: usize,
    pub input_dimension: usize,
    pub output_dimensions: Vec<usize>,
    pub num_heads: usize,
    pub num_layers: usize,
    pub layer_scale: f64,
    pub context: usize,
    #[serde(default = "default_max_period")]
    pub max_period: f64,
    pub dim_feedforward: usize,
}

fn default_max_period() -> f64 {
    10000.0
}

/// Quantizer configuration
#[derive(Debug, Clone, Deserialize)]
pub struct QuantizerConfig {
    pub dimension: usize,
    pub output_dimension: usize,
}

/// Mimi model configuration
#[derive(Debug, Clone, Deserialize)]
pub struct MimiConfig {
    pub dtype: String,
    pub sample_rate: usize,
    pub channels: usize,
    pub frame_rate: f64,
    #[serde(default)]
    pub inner_dim: Option<usize>,
    #[serde(default)]
    pub outer_dim: Option<usize>,
    pub seanet: SEANetConfig,
    pub transformer: MimiTransformerConfig,
    pub quantizer: QuantizerConfig,
    #[serde(default)]
    pub weights_path: Option<String>,
}

/// Root configuration
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub flow_lm: FlowLMConfig,
    pub mimi: MimiConfig,
    #[serde(default)]
    pub pad_with_spaces_for_short_inputs: bool,
    #[serde(default)]
    pub remove_semicolons: bool,
    #[serde(default)]
    pub model_recommended_frames_after_eos: Option<usize>,
    #[serde(default)]
    pub weights_path: Option<String>,
    #[serde(default)]
    pub weights_path_without_voice_cloning: Option<String>,
}

/// Load configuration from a YAML file
pub fn load_config<P: AsRef<Path>>(path: P) -> anyhow::Result<Config> {
    let contents = std::fs::read_to_string(path)?;
    let config: Config = serde_yaml::from_str(&contents)?;
    Ok(config)
}

/// Default generation parameters (matching Python's default_parameters.py)
pub mod defaults {
    pub const TEMPERATURE: f32 = 0.7;
    pub const LSD_DECODE_STEPS: usize = 1;
    pub const NOISE_CLAMP: Option<f32> = None;
    pub const EOS_THRESHOLD: f32 = -4.0;
    pub const DEFAULT_LANGUAGE: &str = "english";

    pub fn default_text_for_language(language: &str) -> &'static str {
        if language.contains("french") {
            "C'est assez petit pour tenir dans votre poche."
        } else if language.contains("german") {
            "Es ist klein genug, um in Ihre Tasche zu passen."
        } else if language.contains("portuguese") {
            "É pequeno o suficiente para caber no seu bolso."
        } else if language.contains("italian") {
            "È abbastanza piccolo da stare in tasca."
        } else if language.contains("spanish") {
            "Es lo suficientemente pequeño como para caber en tu bolsillo."
        } else {
            "It's small enough to fit in your pocket."
        }
    }

    pub fn default_voice_for_language(language: &str) -> &'static str {
        if language.contains("italian") {
            "giovanni"
        } else if language.contains("spanish") {
            "lola"
        } else if language.contains("german") {
            "juergen"
        } else if language.contains("portuguese") {
            "rafael"
        } else if language.contains("french") {
            "estelle"
        } else {
            "alba"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn get_config_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("config")
            .join("english.yaml")
    }

    #[test]
    fn test_load_config() {
        let path = get_config_path();
        if path.exists() {
            let config = load_config(&path).expect("Failed to load config");

            // Verify FlowLM config
            assert_eq!(config.flow_lm.transformer.d_model, 1024);
            assert_eq!(config.flow_lm.transformer.num_heads, 16);
            assert_eq!(config.flow_lm.transformer.num_layers, 6);
            assert_eq!(config.flow_lm.flow.dim, 512);
            assert_eq!(config.flow_lm.flow.depth, 6);
            assert_eq!(config.flow_lm.lookup_table.n_bins, 4000);

            // Verify Mimi config
            assert_eq!(config.mimi.sample_rate, 24000);
            assert_eq!(config.mimi.channels, 1);
            assert!((config.mimi.frame_rate - 12.5).abs() < 1e-6);
            assert_eq!(config.mimi.seanet.dimension, 512);
            assert_eq!(config.mimi.seanet.ratios, vec![6, 5, 4]);
            assert_eq!(config.mimi.transformer.num_layers, 2);
            assert_eq!(config.mimi.quantizer.dimension, 32);
        } else {
            eprintln!("Config file not found at {:?}, skipping test", path);
        }
    }
}
