//! Model-free comparisons with the pinned upstream prompt-normalization oracle.
use candle_core::{Device, Tensor};
use pocket_tts::{anyhow::Result, audio::end_on_pause};
use std::path::PathBuf;

#[test]
fn prompt_normalization_matches_python_without_mutating_input() -> Result<()> {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../assets/upstream-2026-11-09/target/english");
    let reference =
        candle_core::safetensors::load(directory.join("prompt.safetensors"), &Device::Cpu)?;
    let cases = [
        "short",
        "all_zero",
        "speech_and_pause",
        "speech_ending",
        "partial_frame",
        "long_prompt",
        "batch_channels",
        "strict_floor",
    ];
    for name in cases {
        let input = &reference[&format!("{name}.input")];
        let before = input.copy()?;
        let output = end_on_pause(input, 24000)?;
        let expected = &reference[&format!("{name}.output")];
        assert_eq!(output.dims(), expected.dims(), "{name} shape");
        let error = (&output - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        eprintln!(
            "{name}: samples={}, max_error={error}",
            output.dim(output.rank() - 1)?
        );
        assert!(error < 2e-7, "{name}: {error}");
        assert_eq!(
            (input - before)?.abs()?.max_all()?.to_scalar::<f32>()?,
            0.0,
            "{name} input changed"
        );
    }
    Ok(())
}

#[test]
fn complete_frames_trim_strictly_and_append_exact_pause() -> Result<()> {
    // 20 ms frames at 100 Hz; final partial frame must never retain the loud spike.
    let input = Tensor::from_vec(vec![1f32, 1., 0., 0., 99.], (1, 1, 5), &Device::Cpu)?;
    let output = end_on_pause(&input, 100)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(output, vec![1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]);
    // All-zero dB is finite because upstream adds 1e-12; every complete frame is active.
    let zeros = Tensor::zeros((2, 3, 5), candle_core::DType::F32, &Device::Cpu)?;
    assert_eq!(end_on_pause(&zeros, 100)?.dims(), &[2, 3, 12]);
    Ok(())
}
