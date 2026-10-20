use crate::ModelState;
use crate::modules::attention::{StreamingMultiheadAttention, validate_attention_dims};
use crate::modules::mlp::{LayerNorm, LayerScale};
use crate::modules::rope::RotaryEmbedding;
use crate::voice_state::get_attention_cursor;
use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

#[derive(Clone)]
pub struct StreamingTransformerLayer {
    self_attn: StreamingMultiheadAttention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
    layer_scale_1: Option<LayerScale>,
    layer_scale_2: Option<LayerScale>,
}

impl StreamingTransformerLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_model: usize,
        num_heads: usize,
        dim_feedforward: usize,
        context: Option<usize>,
        rope: RotaryEmbedding,
        layer_scale: Option<f32>,
        _attention_kind: &str,
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new_with_attention_dims(
            d_model,
            num_heads,
            d_model,
            num_heads,
            dim_feedforward,
            context,
            rope,
            layer_scale,
            _attention_kind,
            name,
            vb,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_attention_dims(
        d_model: usize,
        num_heads: usize,
        attn_dim: usize,
        num_kv_heads: usize,
        dim_feedforward: usize,
        context: Option<usize>,
        rope: RotaryEmbedding,
        layer_scale: Option<f32>,
        _attention_kind: &str,
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = StreamingMultiheadAttention::new_with_attention_dims(
            d_model,
            num_heads,
            attn_dim,
            num_kv_heads,
            rope,
            context,
            &format!("{}.self_attn", name),
            vb.pp("self_attn"),
        )?;
        let norm1 = LayerNorm::new(d_model, 1e-5, true, vb.pp("norm1"))?;
        let norm2 = LayerNorm::new(d_model, 1e-5, true, vb.pp("norm2"))?;
        let linear1 = candle_nn::linear_no_bias(d_model, dim_feedforward, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear_no_bias(dim_feedforward, d_model, vb.pp("linear2"))?;

        let (layer_scale_1, layer_scale_2) = if let Some(init) = layer_scale {
            (
                Some(LayerScale::new(d_model, init, vb.pp("layer_scale_1"))?),
                Some(LayerScale::new(d_model, init, vb.pp("layer_scale_2"))?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            self_attn,
            norm1,
            norm2,
            linear1,
            linear2,
            layer_scale_1,
            layer_scale_2,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        model_state: &mut ModelState,
        current_pos: usize,
        current_len: usize,
    ) -> Result<Tensor> {
        self.forward_with_capture(x, model_state, current_pos, current_len, None)
    }

    fn forward_with_capture(
        &self,
        x: &Tensor,
        model_state: &mut ModelState,
        current_pos: usize,
        current_len: usize,
        capture: Option<(usize, &mut crate::timestamps::attention::AttentionCapture)>,
    ) -> Result<Tensor> {
        let x_orig = x.clone();
        let h = self.norm1.forward(x)?;
        let mut update = self.self_attn.forward_with_capture(
            &h,
            model_state,
            current_pos,
            current_len,
            capture,
        )?;
        if let Some(ls) = &self.layer_scale_1 {
            update = ls.forward(&update)?;
        }
        let x = (x_orig + update)?;

        let x_orig = x.clone();
        let h = self.norm2.forward(&x)?;
        let mut update = self.linear2.forward(&self.linear1.forward(&h)?.gelu()?)?;
        if let Some(ls) = &self.layer_scale_2 {
            update = ls.forward(&update)?;
        }
        x_orig + update
    }
}

#[derive(Clone)]
pub struct StreamingTransformer {
    layers: Vec<StreamingTransformerLayer>,
    _rope: RotaryEmbedding,
    name: String,
}

impl StreamingTransformer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        layer_scale: Option<f32>,
        dim_feedforward: usize,
        context: Option<usize>,
        max_period: f32,
        kind: &str,
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new_with_attention_dims(
            d_model,
            num_heads,
            d_model,
            num_heads,
            num_layers,
            layer_scale,
            dim_feedforward,
            context,
            max_period,
            kind,
            name,
            vb,
        )
    }

    /// Construct a streaming transformer with independent model/attention widths
    /// and grouped KV heads. The legacy constructor retains MHA dimensions.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_attention_dims(
        d_model: usize,
        num_heads: usize,
        attn_dim: usize,
        num_kv_heads: usize,
        num_layers: usize,
        layer_scale: Option<f32>,
        dim_feedforward: usize,
        context: Option<usize>,
        max_period: f32,
        kind: &str,
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        validate_attention_dims(attn_dim, num_heads, num_kv_heads)?;
        let rope = RotaryEmbedding::new(max_period, attn_dim / num_heads, vb.device())?;
        let mut layers = Vec::new();
        for i in 0..num_layers {
            layers.push(StreamingTransformerLayer::new_with_attention_dims(
                d_model,
                num_heads,
                attn_dim,
                num_kv_heads,
                dim_feedforward,
                context,
                rope.clone(),
                layer_scale,
                kind,
                &format!("{}.layers.{}", name, i),
                vb.pp(format!("layers.{}", i)),
            )?);
        }
        Ok(Self {
            layers,
            _rope: rope,
            name: name.to_string(),
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        model_state: &mut ModelState,
        _step: usize,
    ) -> Result<Tensor> {
        self.forward_with_capture(x, model_state, None)
    }

    pub(crate) fn forward_with_capture(
        &self,
        x: &Tensor,
        model_state: &mut ModelState,
        mut capture: Option<&mut crate::timestamps::attention::AttentionCapture>,
    ) -> Result<Tensor> {
        let mut x = x.clone();
        // Fetch current_pos once from the first attention layer's state to avoid redundant to_scalar calls.
        let first_layer_name = format!("{}.layers.0.self_attn", self.name);
        let cursor = get_attention_cursor(model_state, &first_layer_name);
        let current_pos = cursor.pos;
        let current_len = cursor.len;

        for (index, layer) in self.layers.iter().enumerate() {
            x = layer.forward_with_capture(
                &x,
                model_state,
                current_pos,
                current_len,
                capture.as_deref_mut().map(|c| (index, c)),
            )?;
        }
        Ok(x)
    }
}

#[derive(Clone)]
pub struct ProjectedTransformer {
    transformer: StreamingTransformer,
    input_proj: Option<Linear>,
    output_projs: Vec<Option<Linear>>,
    _input_dimension: usize,
    _output_dimensions: Vec<usize>,
    _d_model: usize,
}

impl ProjectedTransformer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_dimension: usize,
        output_dimensions: Vec<usize>,
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        layer_scale: f32,
        context: usize,
        max_period: f32,
        dim_feedforward: usize,
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let transformer = StreamingTransformer::new(
            d_model,
            num_heads,
            num_layers,
            Some(layer_scale),
            dim_feedforward,
            Some(context),
            max_period,
            "mimi",
            &format!("{}.transformer", name),
            vb.pp("transformer"),
        )?;

        let input_proj = if d_model != input_dimension {
            Some(candle_nn::linear_no_bias(
                input_dimension,
                d_model,
                vb.pp("input_proj"),
            )?)
        } else {
            None
        };

        let mut output_projs = Vec::new();
        for (i, &output_dim) in output_dimensions.iter().enumerate() {
            if d_model == output_dim {
                output_projs.push(None);
            } else {
                output_projs.push(Some(candle_nn::linear_no_bias(
                    d_model,
                    output_dim,
                    vb.pp(format!("output_projs.{}", i)),
                )?));
            }
        }

        Ok(Self {
            transformer,
            input_proj,
            output_projs,
            _input_dimension: input_dimension,
            _output_dimensions: output_dimensions,
            _d_model: d_model,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        model_state: &mut ModelState,
        step: usize,
    ) -> Result<Vec<Tensor>> {
        // x is [B, C, T]
        let mut x = x.transpose(1, 2)?; // [B, T, C]
        if let Some(proj) = &self.input_proj {
            x = proj.forward(&x)?;
        }
        let z = self.transformer.forward(&x, model_state, step)?;

        let mut ys = Vec::new();
        for output_proj in &self.output_projs {
            let mut y = if let Some(proj) = output_proj {
                proj.forward(&z)?
            } else {
                z.clone()
            };
            y = y.transpose(1, 2)?; // [B, C_out, T]
            ys.append(&mut vec![y]);
        }
        Ok(ys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn gqa_transformer_propagates_dimensions_and_streaming_positions() -> Result<()> {
        let vars = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu);
        // d_model is deliberately not divisible by Q heads. RoPE must use
        // attn_dim / num_heads, and each layer must preserve residual width.
        let transformer = StreamingTransformer::new_with_attention_dims(
            5,
            3,
            12,
            1,
            2,
            None,
            9,
            None,
            10000.,
            "flow_lm",
            "tr",
            vb.clone(),
        )?;
        // VarBuilder's default norm weights are zero. Install deterministic,
        // nonzero weights so an accidental identity transformer cannot pass.
        for (name, var) in vars.data().lock().unwrap().iter() {
            let value = if name.ends_with("norm1.weight") || name.ends_with("norm2.weight") {
                Tensor::ones(var.shape(), DType::F32, &Device::Cpu)?
            } else if name.ends_with("bias") {
                Tensor::zeros(var.shape(), DType::F32, &Device::Cpu)?
            } else {
                let phase = name.bytes().map(|b| b as usize).sum::<usize>() as f32;
                Tensor::from_vec(
                    (0..var.elem_count())
                        .map(|i| ((i as f32 + phase) * 0.31).sin() * 0.2)
                        .collect(),
                    var.shape(),
                    &Device::Cpu,
                )?
            };
            var.set(&value)?;
        }
        let input = Tensor::from_vec(
            (0..30).map(|i| (i as f32 * 0.7).sin()).collect(),
            (1, 6, 5),
            &Device::Cpu,
        )?;
        let full = transformer.forward(&input, &mut ModelState::new(), 0)?;
        assert!((&full - &input)?.abs()?.max_all()?.to_scalar::<f32>()? > 1e-4);
        let mut state = ModelState::new();
        let first = transformer.forward(&input.narrow(1, 0, 4)?, &mut state, 0)?;
        let second = transformer.forward(&input.narrow(1, 4, 2)?, &mut state, 1)?;
        let streamed = Tensor::cat(&[first, second], 1)?;
        assert!((&full - &streamed)?.abs()?.max_all()?.to_scalar::<f32>()? < 1e-5);
        for i in 0..2 {
            let name = format!("tr.layers.{i}.self_attn");
            assert_eq!(get_attention_cursor(&state, &name).pos, 6);
            assert_eq!(
                state[&name][crate::voice_state::ATTN_K_BUF_KEY].dims(),
                &[1, 1, 64, 4]
            );
        }
        for (heads, attn, kv) in [(0, 12, 1), (3, 11, 1), (3, 12, 2), (3, 9, 1)] {
            assert!(
                StreamingTransformer::new_with_attention_dims(
                    5,
                    heads,
                    attn,
                    kv,
                    0,
                    None,
                    9,
                    None,
                    10000.,
                    "flow_lm",
                    "bad",
                    vb.clone()
                )
                .is_err()
            );
        }
        Ok(())
    }
}
