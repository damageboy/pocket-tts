use crate::ModelState;
use crate::modules::rope::RotaryEmbedding;
use crate::voice_state::{
    ATTN_K_BUF_KEY, ATTN_LEN_KEY, ATTN_POS_KEY, ATTN_V_BUF_KEY, AttentionCursor,
    read_attention_cursor, write_attention_cursor,
};
use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use std::collections::HashMap;

pub(crate) fn validate_attention_dims(
    attn_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
) -> Result<()> {
    if num_heads == 0 || attn_dim == 0 || attn_dim % num_heads != 0 {
        candle_core::bail!("Attention width must be a positive multiple of query heads");
    }
    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        candle_core::bail!("Query heads must be a positive multiple of KV heads");
    }
    if (attn_dim / num_heads) % 2 != 0 {
        candle_core::bail!("RoPE requires an even attention head dimension");
    }
    Ok(())
}

fn ring_chunks(buf: &Tensor, head: usize, len: usize) -> Result<Vec<Tensor>> {
    let cap = buf.dim(2)?;
    if cap == 0 {
        return Ok(Vec::new());
    }

    let len = len.min(cap);
    if len == 0 {
        return Ok(Vec::new());
    }

    let head = head % cap;
    let first_len = std::cmp::min(len, cap - head);
    let second_len = len - first_len;
    let mut chunks = Vec::with_capacity(if second_len > 0 { 2 } else { 1 });
    chunks.push(buf.narrow(2, head, first_len)?);
    if second_len > 0 {
        chunks.push(buf.narrow(2, 0, second_len)?);
    }
    Ok(chunks)
}

#[derive(Clone)]
pub struct StreamingMultiheadAttention {
    attn_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    rope: RotaryEmbedding,
    in_proj: Linear,
    out_proj: Linear,
    context: Option<usize>,
    name: String,
}

impl StreamingMultiheadAttention {
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        rope: RotaryEmbedding,
        context: Option<usize>,
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new_with_attention_dims(
            embed_dim, num_heads, embed_dim, num_heads, rope, context, name, vb,
        )
    }

    /// Attention width is independent of model width; KV heads are shared by
    /// consecutive groups of query heads and remain grouped in streaming state.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_attention_dims(
        embed_dim: usize,
        num_heads: usize,
        attn_dim: usize,
        num_kv_heads: usize,
        rope: RotaryEmbedding,
        context: Option<usize>,
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        validate_attention_dims(attn_dim, num_heads, num_kv_heads)?;
        let kv_dim = (attn_dim / num_heads) * num_kv_heads;
        let in_proj =
            candle_nn::linear_no_bias(embed_dim, attn_dim + 2 * kv_dim, vb.pp("in_proj"))?;
        let out_proj = candle_nn::linear_no_bias(attn_dim, embed_dim, vb.pp("out_proj"))?;

        Ok(Self {
            attn_dim,
            num_heads,
            num_kv_heads,
            rope,
            in_proj,
            out_proj,
            context,
            name: name.to_string(),
        })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        _sequence_length: usize,
        device: &candle_core::Device,
    ) -> Result<HashMap<String, Tensor>> {
        let dim_per_head = self.attn_dim / self.num_heads;
        let mut state = HashMap::new();

        // Initial capacity: match context if windowed, otherwise reasonable default
        let cap = self.context.unwrap_or(64);
        state.insert(
            ATTN_K_BUF_KEY.to_string(),
            Tensor::zeros(
                (batch_size, self.num_kv_heads, cap, dim_per_head),
                DType::F32,
                device,
            )?,
        );
        state.insert(
            ATTN_V_BUF_KEY.to_string(),
            Tensor::zeros(
                (batch_size, self.num_kv_heads, cap, dim_per_head),
                DType::F32,
                device,
            )?,
        );
        write_attention_cursor(&mut state, AttentionCursor::default(), device)?;
        Ok(state)
    }

    pub fn forward(
        &self,
        query: &Tensor,
        model_state: &mut ModelState,
        current_pos: usize,
        current_len: usize,
    ) -> Result<Tensor> {
        self.forward_with_capture(query, model_state, current_pos, current_len, None)
    }

    pub(crate) fn forward_with_capture(
        &self,
        query: &Tensor,
        model_state: &mut ModelState,
        current_pos: usize,
        current_len: usize,
        capture: Option<(usize, &mut crate::timestamps::attention::AttentionCapture)>,
    ) -> Result<Tensor> {
        let (b, t, _) = query.dims3()?;
        let d = self.attn_dim / self.num_heads;
        let window_size = self.context;

        // Auto-initialize state if missing
        if !model_state.contains_key(&self.name) {
            model_state.insert(self.name.clone(), self.init_state(b, 0, query.device())?);
        }

        let module_state = model_state.get_mut(&self.name).unwrap();
        let mut cursor = read_attention_cursor(module_state);
        if !module_state.contains_key(ATTN_POS_KEY) {
            cursor.pos = current_pos;
        }
        if !module_state.contains_key(ATTN_LEN_KEY) {
            cursor.len = current_len;
        }

        let projected = self.in_proj.forward(query)?;

        // Packed projection order is Q (attn_dim), K (kv_dim), V (kv_dim).
        let kv_dim = self.num_kv_heads * d;
        let mut q = projected
            .narrow(2, 0, self.attn_dim)?
            .reshape((b, t, self.num_heads, d))?;
        let mut k =
            projected
                .narrow(2, self.attn_dim, kv_dim)?
                .reshape((b, t, self.num_kv_heads, d))?;
        let mut v = projected
            .narrow(2, self.attn_dim + kv_dim, kv_dim)?
            .reshape((b, t, self.num_kv_heads, d))?;

        // current_pos passed as argument

        // Apply RoPE
        // RoPE expects (B, T, H, D)
        (q, k) = self.rope.forward(&q, &k, current_pos)?;

        // Transpose q, k, v to (B, H, T, D) for SDPA and KV cache
        q = q.transpose(1, 2)?;
        k = k.transpose(1, 2)?;
        v = v.transpose(1, 2)?;

        // KV cache management.
        // We take ownership from the state to avoid clones and ensure uniqueness for slice_set.
        let (mut k_buf, mut v_buf) = match (
            module_state.remove(ATTN_K_BUF_KEY),
            module_state.remove(ATTN_V_BUF_KEY),
        ) {
            (Some(kb), Some(vb)) => (kb, vb),
            _ => {
                let initial_cap = window_size.unwrap_or(64);
                let kb = Tensor::zeros(
                    (b, self.num_kv_heads, initial_cap, d),
                    q.dtype(),
                    q.device(),
                )?;
                let vb = Tensor::zeros(
                    (b, self.num_kv_heads, initial_cap, d),
                    q.dtype(),
                    q.device(),
                )?;
                (kb, vb)
            }
        };

        let mut cap = k_buf.dim(2)?; // Current capacity of the buffer
        let mut cache_len = cursor.len.min(cap);
        let mut cache_head = if cap > 0 { cursor.head % cap } else { 0 };

        let x = if let Some(window_size) = self.context {
            // Ensure fixed ring capacity for windowed attention.
            if cap != window_size {
                if cap > window_size {
                    k_buf = k_buf.narrow(2, 0, window_size)?.contiguous()?;
                    v_buf = v_buf.narrow(2, 0, window_size)?.contiguous()?;
                } else {
                    let zeros_shape = (b, self.num_kv_heads, window_size - cap, d);
                    let k_zeros = Tensor::zeros(zeros_shape, q.dtype(), q.device())?;
                    let v_zeros = Tensor::zeros(zeros_shape, q.dtype(), q.device())?;
                    k_buf = Tensor::cat(&[k_buf, k_zeros], 2)?;
                    v_buf = Tensor::cat(&[v_buf, v_zeros], 2)?;
                }
                cap = window_size;
                cache_len = cache_len.min(cap);
                cache_head = 0;
            }

            // Build chronological KV chunks from ring cache + current K/V chunk.
            let mut k_chunks = ring_chunks(&k_buf, cache_head, cache_len)?;
            let mut v_chunks = ring_chunks(&v_buf, cache_head, cache_len)?;
            k_chunks.push(k.clone());
            v_chunks.push(v.clone());

            let scale = 1.0 / (d as f64).sqrt();
            if k_chunks.len() == 1 {
                crate::modules::sdpa::sdpa(
                    &q,
                    &k_chunks[0],
                    &v_chunks[0],
                    scale,
                    true,
                    self.context,
                )?
            } else {
                crate::modules::sdpa::sdpa_chunked(
                    &q,
                    &k_chunks,
                    &v_chunks,
                    scale,
                    true,
                    self.context,
                )?
            }
        } else {
            // Linear attention (FlowLM) with doubling contiguous buffer.
            if cache_len + t > cap {
                let new_cap = (cache_len + t).next_power_of_two();
                let zeros_shape = (b, self.num_kv_heads, new_cap - cap, d);
                let k_zeros = Tensor::zeros(zeros_shape, q.dtype(), q.device())?;
                let v_zeros = Tensor::zeros(zeros_shape, q.dtype(), q.device())?;
                k_buf = Tensor::cat(&[k_buf, k_zeros], 2)?;
                v_buf = Tensor::cat(&[v_buf, v_zeros], 2)?;
            }
            k_buf.slice_set(&k.contiguous()?, 2, cache_len)?;
            v_buf.slice_set(&v.contiguous()?, 2, cache_len)?;
            cache_len += t;
            cache_head = 0;

            // Get current KV for attention
            let kc = k_buf.narrow(2, 0, cache_len)?;
            let vc = v_buf.narrow(2, 0, cache_len)?;
            let scale = 1.0 / (d as f64).sqrt();
            crate::modules::sdpa::sdpa_with_capture(
                &q,
                &kc,
                &vc,
                scale,
                true,
                self.context,
                capture,
            )?
        };

        if let Some(window_size) = self.context {
            if t >= window_size {
                k_buf = k.narrow(2, t - window_size, window_size)?.contiguous()?;
                v_buf = v.narrow(2, t - window_size, window_size)?.contiguous()?;
                cache_head = 0;
                cache_len = window_size;
            } else if window_size > 0 {
                let evict = (cache_len + t).saturating_sub(window_size);
                if evict > 0 {
                    cache_head = (cache_head + evict) % window_size;
                    cache_len -= evict;
                }

                let write_start = (cache_head + cache_len) % window_size;
                let first = std::cmp::min(t, window_size - write_start);
                let second = t - first;

                if first > 0 {
                    let k_first = k.narrow(2, 0, first)?.contiguous()?;
                    let v_first = v.narrow(2, 0, first)?.contiguous()?;
                    k_buf.slice_set(&k_first, 2, write_start)?;
                    v_buf.slice_set(&v_first, 2, write_start)?;
                }
                if second > 0 {
                    let k_second = k.narrow(2, first, second)?.contiguous()?;
                    let v_second = v.narrow(2, first, second)?.contiguous()?;
                    k_buf.slice_set(&k_second, 2, 0)?;
                    v_buf.slice_set(&v_second, 2, 0)?;
                }
                cache_len += t;
            }
        }

        module_state.insert(ATTN_K_BUF_KEY.to_string(), k_buf);
        module_state.insert(ATTN_V_BUF_KEY.to_string(), v_buf);
        write_attention_cursor(
            module_state,
            AttentionCursor {
                pos: current_pos + t,
                len: cache_len,
                head: cache_head,
            },
            q.device(),
        )?;

        // Transpose back to [B, T, H, D] and project out
        let x = x.transpose(1, 2)?.reshape((b, t, self.attn_dim))?;
        let x = self.out_proj.forward(&x)?;

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn weights(
        model: usize,
        attn: usize,
        heads: usize,
        kv_heads: usize,
    ) -> Result<HashMap<String, Tensor>> {
        let width = attn + 2 * kv_heads * (attn / heads);
        let tensor = |rows: usize, cols: usize, phase: f32| {
            Tensor::from_vec(
                (0..rows * cols)
                    .map(|i| ((i as f32 + phase) * 0.17).sin() * 0.08)
                    .collect(),
                (rows, cols),
                &Device::Cpu,
            )
        };
        Ok(HashMap::from([
            ("in_proj.weight".into(), tensor(width, model, 0.3)?),
            ("out_proj.weight".into(), tensor(model, attn, 2.1)?),
        ]))
    }

    #[test]
    fn gqa_projection_streaming_growth_and_ring_match_dense() -> Result<()> {
        // Include the real small-model dimensions, a batched asymmetric grouping,
        // and legacy MHA. A single full dense calculation is the streaming oracle.
        for (model, attn, heads, kv_heads) in [
            (224, 512, 8, 2),
            (112, 192, 3, 1),
            (5, 12, 6, 2),
            (8, 8, 2, 2),
        ] {
            for window in [None, Some(7)] {
                let tensors = weights(model, attn, heads, kv_heads)?;
                let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
                let rope = RotaryEmbedding::new(10000., attn / heads, &Device::Cpu)?;
                let layer = StreamingMultiheadAttention::new_with_attention_dims(
                    model,
                    heads,
                    attn,
                    kv_heads,
                    rope.clone(),
                    window,
                    "attn",
                    vb.clone(),
                )?;
                let b = 2;
                let t = 70;
                let input = Tensor::from_vec(
                    (0..b * t * model)
                        .map(|i| ((i as f32 + 0.7) * 0.13).cos())
                        .collect(),
                    (b, t, model),
                    &Device::Cpu,
                )?;
                let packed = candle_nn::linear_no_bias(
                    model,
                    attn + 2 * kv_heads * (attn / heads),
                    vb.pp("in_proj"),
                )?
                .forward(&input)?;
                let d = attn / heads;
                let kv = kv_heads * d;
                let q = packed.narrow(2, 0, attn)?.reshape((b, t, heads, d))?;
                let k = packed.narrow(2, attn, kv)?.reshape((b, t, kv_heads, d))?;
                let v = packed
                    .narrow(2, attn + kv, kv)?
                    .reshape((b, t, kv_heads, d))?
                    .transpose(1, 2)?;
                let (q, k) = rope.forward(&q, &k, 0)?;
                let dense = crate::modules::sdpa::tests::dense_reference(
                    &q.transpose(1, 2)?,
                    &k.transpose(1, 2)?,
                    &v,
                    window,
                )?;
                let expected = candle_nn::linear_no_bias(attn, model, vb.pp("out_proj"))?
                    .forward(&dense.transpose(1, 2)?.reshape((b, t, attn))?)?;
                let mut state = ModelState::new();
                let mut offset = 0;
                // Prefill, cached single-token decoding, cache growth above 64,
                // ring wraparound, and a chunk larger than the window.
                for len in [4, 1, 3, 1, 56, 1, 4] {
                    let actual = layer.forward(
                        &input.narrow(1, offset, len)?,
                        &mut state,
                        offset,
                        offset,
                    )?;
                    let error = (&actual - expected.narrow(1, offset, len)?)?
                        .abs()?
                        .max_all()?
                        .to_scalar::<f32>()?;
                    assert!(
                        error < 2e-5,
                        "model={model}, window={window:?}, offset={offset}, error={error}"
                    );
                    offset += len;
                    let module_state = &state["attn"];
                    let cursor = read_attention_cursor(module_state);
                    assert_eq!(cursor.pos, offset);
                    assert_eq!(cursor.len, window.map_or(offset, |w| offset.min(w)));
                    for key in [ATTN_K_BUF_KEY, ATTN_V_BUF_KEY] {
                        let shape = module_state[key].dims();
                        assert_eq!(shape[1], kv_heads); // persistent cache must stay grouped
                        assert_eq!(shape[3], d);
                        assert!(shape[2] >= cursor.len);
                    }
                }
                if model == attn && heads == kv_heads {
                    let legacy =
                        StreamingMultiheadAttention::new(model, heads, rope, window, "legacy", vb)?;
                    let actual = legacy.forward(&input, &mut ModelState::new(), 0, 0)?;
                    assert!(
                        (&actual - &expected)?
                            .abs()?
                            .max_all()?
                            .to_scalar::<f32>()?
                            < 2e-5
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn gqa_rejects_invalid_head_dimensions() -> Result<()> {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        for (attn, heads, kv_heads) in [
            (12, 0, 1),
            (12, 3, 0),
            (11, 3, 1),
            (12, 3, 2),
            (0, 3, 1),
            (9, 3, 1),
        ] {
            let rope = RotaryEmbedding::new(10000., 4, &Device::Cpu)?;
            assert!(
                StreamingMultiheadAttention::new_with_attention_dims(
                    5,
                    heads,
                    attn,
                    kv_heads,
                    rope,
                    None,
                    "attn",
                    vb.clone()
                )
                .is_err()
            );
        }
        Ok(())
    }
}
