//! Selected-head reduction adapted from pocket-tts-timestamped (MIT).
//! We reuse SDPA logits instead of recomputing QK. See THIRD_PARTY_NOTICES.md.

use candle_core::{D, DType, Result, Tensor};

pub(crate) struct AttentionCapture {
    heads: Vec<(usize, usize)>,
    text_start: usize,
    token_to_unit: Vec<Vec<f32>>,
    scores: Vec<f32>,
    recorded: usize,
}

impl AttentionCapture {
    pub fn new(
        heads: Vec<(usize, usize)>,
        text_start: usize,
        token_to_unit: Vec<Vec<f32>>,
    ) -> Result<Self> {
        if heads.is_empty() || token_to_unit.is_empty() {
            candle_core::bail!("Timestamp capture requires heads and text tokens");
        }
        let units = token_to_unit[0].len();
        if token_to_unit.iter().any(|row| row.len() != units) {
            candle_core::bail!("Inconsistent timestamp token mapping");
        }
        Ok(Self {
            heads,
            text_start,
            token_to_unit,
            scores: vec![0.0; units],
            recorded: 0,
        })
    }

    pub fn record(&mut self, layer: usize, logits: &Tensor) -> Result<()> {
        let (batch, _, queries, _) = logits.dims4()?;
        if batch != 1 || queries != 1 {
            candle_core::bail!("Timestamp capture requires a single decoding query");
        }
        for &(_, head) in self.heads.iter().filter(|(l, _)| *l == layer) {
            // Softmax MUST be over text only, separately for each head. Normal
            // attention also includes voice and audio keys, which can dominate.
            let text = logits
                .narrow(1, head, 1)?
                .narrow(3, self.text_start, self.token_to_unit.len())?
                .flatten_all()?
                .to_dtype(DType::F32)?;
            let probabilities = candle_nn::ops::softmax(&text, D::Minus1)?.to_vec1::<f32>()?;
            for (p, row) in probabilities.iter().zip(&self.token_to_unit) {
                for (score, weight) in self.scores.iter_mut().zip(row) {
                    *score += p * weight;
                }
            }
            self.recorded += 1;
        }
        Ok(())
    }

    pub fn finish_frame(&mut self) -> Result<Vec<f32>> {
        if self.recorded != self.heads.len() {
            candle_core::bail!("Timestamp attention missing configured heads");
        }
        let result = self
            .scores
            .iter()
            .map(|s| s / self.recorded as f32)
            .collect();
        self.scores.fill(0.0);
        self.recorded = 0;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn normalizes_only_text_keys_and_weights_heads_not_layers() -> candle_core::Result<()> {
        // Voice/audio logits dominate full-context softmax. Only text keys 1..3
        // matter; the second token belongs half to each of two text units.
        let mut capture = AttentionCapture::new(
            vec![(0, 0), (0, 1), (1, 1)],
            1,
            vec![vec![1.0, 0.0], vec![0.5, 0.5]],
        )?;
        let ln3 = 3f32.ln();
        let logits = Tensor::from_vec(
            vec![1000., ln3, 0., 1000., 1000., 0., ln3, 1000.],
            (1, 2, 1, 4),
            &Device::Cpu,
        )?;
        capture.record(0, &logits)?;
        capture.record(1, &logits)?;
        let scores = capture.finish_frame()?;
        // Selected text distributions: (.75,.25), (.25,.75), (.25,.75).
        // Mapped means: (17/24, 7/24), NOT a mean of the two layer means.
        assert!((scores[0] - 17.0 / 24.0).abs() < 1e-6);
        assert!((scores[1] - 7.0 / 24.0).abs() < 1e-6);
        assert!(capture.finish_frame().is_err()); // frame accumulator is reset
        capture.record(0, &logits)?;
        assert!(capture.finish_frame().is_err()); // missing configured layer
        Ok(())
    }
}
