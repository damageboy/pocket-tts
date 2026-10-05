//! Python-generated expectations, rather than comparisons against another Rust path.
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use pocket_tts::conditioners::text::LUTConditioner;
use pocket_tts::models::flow_lm::lsd_decode;
use pocket_tts::modules::mlp::{LayerNorm, RMSNorm, SimpleMLPAdaLN};
use pocket_tts::text::TextOptions;
use serde::Deserialize;
use std::{collections::HashMap, path::PathBuf};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/upstream-v3.3")
}

#[derive(Deserialize)]
struct TextCase {
    language: String,
    input: String,
    options: TextOptions,
    prepared: Option<String>,
    frames_after_eos_guess: Option<usize>,
    tokens: Option<Vec<u32>>,
    chunks: Option<HashMap<String, Vec<String>>>,
    error: Option<String>,
}

#[test]
fn text_and_native_byte_tokenizers_match_python_in_all_languages() -> anyhow::Result<()> {
    let cases: Vec<TextCase> =
        serde_json::from_slice(&std::fs::read(fixtures().join("text.json"))?)?;
    for case in cases {
        let prepared = case.options.prepare(&case.input);
        if let Some(error) = case.error {
            assert_eq!(
                prepared.unwrap_err().to_string(),
                error,
                "{}",
                case.language
            );
            continue;
        }
        let (text, tail) = prepared?;
        assert_eq!(
            Some(&text),
            case.prepared.as_ref(),
            "{}: {:?}",
            case.language,
            case.input
        );
        assert_eq!(Some(tail), case.frames_after_eos_guess);
        let path = fixtures().join(format!("{}.tokenizer.json", case.language));
        let bytes = std::fs::read(&path)?;
        let tokenizer = tokenizers::Tokenizer::from_bytes(&bytes).map_err(anyhow::Error::msg)?;
        let bins = tokenizer.get_vocab_size(true);
        for conditioner in [
            LUTConditioner::new(
                bins,
                &path,
                1,
                1,
                VarBuilder::zeros(DType::F32, &Device::Cpu),
            )?,
            LUTConditioner::new_from_bytes(
                bins,
                &bytes,
                1,
                1,
                VarBuilder::zeros(DType::F32, &Device::Cpu),
            )?,
        ] {
            let tokens = conditioner
                .prepare(&text, &Device::Cpu)?
                .flatten_all()?
                .to_vec1::<u32>()?;
            assert_eq!(
                Some(&tokens),
                case.tokens.as_ref(),
                "{}: {:?}",
                case.language,
                case.input
            );
        }
        for (limit, expected) in case.chunks.unwrap() {
            let actual = case
                .options
                .split(&tokenizer, &case.input, limit.parse()?)?;
            assert_eq!(
                actual, expected,
                "{}: {:?}, limit {}",
                case.language, case.input, limit
            );
        }
    }
    Ok(())
}

fn close(actual: &Tensor, expected: &Tensor, label: &str) -> anyhow::Result<()> {
    assert_eq!(actual.dims(), expected.dims(), "{label}");
    let difference = (actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(
        difference <= 2e-6,
        "{label}: max absolute error {difference}"
    );
    Ok(())
}

#[test]
fn normalization_activation_and_samplers_match_python_tensors() -> anyhow::Result<()> {
    let tensors =
        candle_core::safetensors::load(fixtures().join("operators.safetensors"), &Device::Cpu)?;
    let x = &tensors["input"];
    close(&x.gelu()?, &tensors["gelu"], "tanh GELU")?;
    let norm = LayerNorm::new(4, 1e-6, false, VarBuilder::zeros(DType::F32, &Device::Cpu))?;
    close(&norm.forward(x)?, &tensors["layer_norm"], "LayerNorm")?;
    let rms = RMSNorm::new(
        4,
        1e-5,
        VarBuilder::from_tensors(
            HashMap::from([("alpha".into(), Tensor::ones(4, DType::F32, &Device::Cpu)?)]),
            DType::F32,
            &Device::Cpu,
        ),
    )?;
    close(&rms.forward(x)?, &tensors["rms_norm"], "variance RMSNorm")?;
    for (kind, time_conditions) in [("lsd", 2), ("flow_matching", 1)] {
        let vb = VarBuilder::from_tensors(tensors.clone(), DType::F32, &Device::Cpu);
        let net = SimpleMLPAdaLN::new(
            4,
            8,
            4,
            6,
            2,
            time_conditions,
            10000.,
            vb.pp(format!("{kind}.weights")),
        )?;
        let condition = net.embed_condition(&tensors["condition"])?;
        for steps in [1, 3] {
            let times = net.compute_time_embeddings(steps, &Device::Cpu, DType::F32)?;
            let mods = net.precompute_modulations(&condition, &times)?;
            let key = format!("{kind}.steps_{steps}");
            close(&lsd_decode(&net, &mods, x)?, &tensors[&key], &key)?;
        }
    }
    Ok(())
}
