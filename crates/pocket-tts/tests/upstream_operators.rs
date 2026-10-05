//! Controlled nonzero operators matching upstream v3.3.0 (3dbee45d343d7dddd0d105468d17f8dcba14db3e).
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use pocket_tts::models::flow_lm::lsd_decode;
use pocket_tts::modules::mlp::{SimpleMLPAdaLN, TimestepEmbedder};
use std::collections::HashMap;

fn parameters(count: usize) -> Result<HashMap<String, Tensor>> {
    let mut map = HashMap::new();
    let mut linear = |name: &str, input: usize, output: usize, seed: f32| -> Result<()> {
        let weights: Vec<f32> = (0..input * output)
            .map(|i| ((i as f32 * 0.71 + seed).sin()) * 0.13)
            .collect();
        let bias: Vec<f32> = (0..output)
            .map(|i| (i as f32 * 0.3 + seed).cos() * 0.17)
            .collect();
        map.insert(
            format!("{name}.weight"),
            Tensor::from_vec(weights, (output, input), &Device::Cpu)?,
        );
        map.insert(
            format!("{name}.bias"),
            Tensor::from_vec(bias, output, &Device::Cpu)?,
        );
        Ok(())
    };
    for i in 0..count {
        linear(&format!("time_embed.{i}.mlp.0"), 256, 4, 0.4 + i as f32)?;
        linear(&format!("time_embed.{i}.mlp.2"), 4, 4, 1.2 + i as f32)?;
    }
    linear("cond_embed", 4, 4, 0.7)?;
    linear("input_proj", 2, 4, 1.1)?;
    linear("res_blocks.0.mlp.0", 4, 4, 2.1)?;
    linear("res_blocks.0.mlp.2", 4, 4, 3.1)?;
    linear("res_blocks.0.adaLN_modulation.1", 4, 12, 4.1)?;
    linear("final_layer.adaLN_modulation.1", 4, 8, 5.1)?;
    linear("final_layer.linear", 4, 2, 6.1)?;
    for i in 0..count {
        map.insert(
            format!("time_embed.{i}.mlp.3.alpha"),
            Tensor::new(&[0.8f32, 1.1, 0.9, 1.3], &Device::Cpu)?,
        );
    }
    map.insert(
        "res_blocks.0.in_ln.weight".into(),
        Tensor::new(&[1.2f32, 0.8, 1.1, 0.9], &Device::Cpu)?,
    );
    map.insert(
        "res_blocks.0.in_ln.bias".into(),
        Tensor::new(&[0.1f32, -0.2, 0.3, -0.1], &Device::Cpu)?,
    );
    Ok(map)
}

fn network(count: usize) -> Result<SimpleMLPAdaLN> {
    SimpleMLPAdaLN::new(
        2,
        4,
        2,
        4,
        1,
        count,
        10000.,
        VarBuilder::from_tensors(parameters(count)?, DType::F32, &Device::Cpu),
    )
}

fn max_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    (a - b)?.abs()?.max_all()?.to_scalar()
}

fn check_schedule(count: usize) -> Result<()> {
    let net = network(count)?;
    let vb = VarBuilder::from_tensors(parameters(count)?, DType::F32, &Device::Cpu);
    let first = TimestepEmbedder::new(4, 256, 10000., vb.pp("time_embed.0"))?;
    let second = if count == 2 {
        Some(TimestepEmbedder::new(
            4,
            256,
            10000.,
            vb.pp("time_embed.1"),
        )?)
    } else {
        None
    };
    let c = Tensor::new(&[[0.3f32, -0.7, 1.2, 0.4]], &Device::Cpu)?;
    let noise = Tensor::new(&[[0.8f32, -1.3]], &Device::Cpu)?;
    let c_emb = net.embed_condition(&c)?;
    let embeddings = net.compute_time_embeddings(3, &Device::Cpu, DType::F32)?;
    let cached = net.precompute_modulations(&c_emb, &embeddings)?;
    let mut current = noise.clone();
    assert_eq!(cached.len(), 3);
    for (i, modulation) in cached.iter().enumerate() {
        let s = Tensor::new(&[i as f32 / 3.], &Device::Cpu)?;
        let end = Tensor::new(&[(i + 1) as f32 / 3.], &Device::Cpu)?;
        let expected = match &second {
            Some(second) => ((first.forward(&s)? + second.forward(&end)?)? / 2.)?,
            None => first.forward(&s)?,
        };
        assert!(max_diff(&embeddings.narrow(0, i, 1)?, &expected)? < 1e-6);
        // For the one-condition API, t is the Euler start time; s is ignored.
        let t = if count == 1 { &s } else { &end };
        let direction = net.forward(&c, &s, t, &current)?;
        if count == 1 {
            assert!(max_diff(&direction, &net.forward(&c, &end, t, &current)?)? < 1e-7);
        }
        assert!(max_diff(&direction, &net.forward_step_cached(&current, modulation)?)? < 2e-6);
        let swapped = net.forward(&c, t, &end, &current)?;
        assert!(
            max_diff(&direction, &swapped)? > 1e-5,
            "asymmetric times must affect output"
        );
        current = (current + (direction / 3.)?)?;
    }
    assert!(max_diff(&current, &lsd_decode(&net, &cached, &noise)?)? < 2e-6);
    assert!(max_diff(&current, &noise)? > 1e-3);
    Ok(())
}

#[test]
fn one_condition_euler_schedule_and_cache() -> Result<()> {
    check_schedule(1)
}

#[test]
fn two_condition_lsd_schedule_and_cache() -> Result<()> {
    check_schedule(2)
}

#[test]
fn unsupported_time_counts_are_rejected() -> Result<()> {
    assert!(network(0).is_err());
    assert!(network(3).is_err());
    Ok(())
}

#[test]
fn zero_steps_are_rejected() -> Result<()> {
    let net = network(2)?;
    assert!(
        net.compute_time_embeddings(0, &Device::Cpu, DType::F32)
            .is_err()
    );
    let noise = Tensor::new(&[[0.8f32, -1.3]], &Device::Cpu)?;
    assert!(lsd_decode(&net, &[], &noise).is_err());
    Ok(())
}

#[test]
fn explicit_noise_is_used_verbatim_by_both_samplers() -> Result<()> {
    use candle_nn::Module;
    use pocket_tts::models::{flow_lm::FlowLMModel, transformer::StreamingTransformer};
    for count in [1, 2] {
        let mut map = HashMap::new();
        map.insert(
            "input_linear.weight".into(),
            Tensor::new(
                &[[0.3f32, -0.2], [0.5, 0.7], [-0.4, 0.1], [0.8, -0.6]],
                &Device::Cpu,
            )?,
        );
        map.insert(
            "out_norm.weight".into(),
            Tensor::new(&[0.9f32, 1.1, 0.8, 1.2], &Device::Cpu)?,
        );
        map.insert(
            "out_norm.bias".into(),
            Tensor::new(&[0.1f32, -0.2, 0.3, 0.4], &Device::Cpu)?,
        );
        map.insert(
            "out_eos.weight".into(),
            Tensor::new(&[[0.2f32, -0.3, 0.4, 0.1]], &Device::Cpu)?,
        );
        map.insert("out_eos.bias".into(), Tensor::new(&[0.7f32], &Device::Cpu)?);
        for name in ["bos_emb", "emb_mean", "emb_std"] {
            map.insert(name.into(), Tensor::ones(2, DType::F32, &Device::Cpu)?);
        }
        let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
        // No transformer layers: the real projection, norm, EOS and sampler remain active.
        let transformer =
            StreamingTransformer::new(4, 1, 0, None, 8, None, 10000., "flow", "test", vb.clone())?;
        let mut model = FlowLMModel::new(network(count)?, transformer, 2, 4, false, vb)?;
        model.noise_clamp = Some(0.01); // Explicit noise must bypass truncation, not clamp.
        let sequence = Tensor::new(&[[[0.6f32, -0.9]]], &Device::Cpu)?;
        let text = Tensor::zeros((1, 0, 4), DType::F32, &Device::Cpu)?;
        let times = model
            .flow_net
            .compute_time_embeddings(3, &Device::Cpu, DType::F32)?;
        let noise = Tensor::new(&[[0.8f32, -1.3]], &Device::Cpu)?;
        let condition = model
            .out_norm
            .forward(&model.input_linear.forward(&sequence)?)?
            .squeeze(1)?;
        let mods = model
            .flow_net
            .precompute_modulations(&model.flow_net.embed_condition(&condition)?, &times)?;
        let expected = lsd_decode(&model.flow_net, &mods, &noise)?;
        let threshold = model.out_eos.forward(&condition)?.to_vec2::<f32>()?[0][0];
        let (actual, eos) = model.forward_with_noise(
            &sequence,
            &text,
            &mut HashMap::new(),
            &times,
            &noise,
            threshold,
            0,
        )?;
        assert!(max_diff(&actual, &expected)? < 2e-6);
        assert!(
            !eos,
            "EOS comparison is strictly greater than the threshold"
        );
        let (repeated, eos) = model.forward_with_noise(
            &sequence,
            &text,
            &mut HashMap::new(),
            &times,
            &noise,
            threshold - 0.1,
            0,
        )?;
        assert!(max_diff(&actual, &repeated)? < 1e-7);
        assert!(eos);
        let other_noise = Tensor::new(&[[-0.5f32, 0.2]], &Device::Cpu)?;
        let (other, _) = model.forward_with_noise(
            &sequence,
            &text,
            &mut HashMap::new(),
            &times,
            &other_noise,
            threshold,
            0,
        )?;
        assert!(max_diff(&actual, &other)? > 0.1);
        let wrong_shape = Tensor::ones((1, 3), DType::F32, &Device::Cpu)?;
        assert!(
            model
                .forward_with_noise(
                    &sequence,
                    &text,
                    &mut HashMap::new(),
                    &times,
                    &wrong_shape,
                    threshold,
                    0
                )
                .is_err()
        );
        let wrong_dtype = noise.to_dtype(DType::F64)?;
        assert!(
            model
                .forward_with_noise(
                    &sequence,
                    &text,
                    &mut HashMap::new(),
                    &times,
                    &wrong_dtype,
                    threshold,
                    0
                )
                .is_err()
        );
        let empty_times = Tensor::zeros((0, 4), DType::F32, &Device::Cpu)?;
        assert!(
            model
                .forward_with_noise(
                    &sequence,
                    &text,
                    &mut HashMap::new(),
                    &empty_times,
                    &noise,
                    threshold,
                    0
                )
                .is_err()
        );
    }
    Ok(())
}
