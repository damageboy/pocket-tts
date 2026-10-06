use anyhow::Result;
use serde::Serialize;

use super::{TimestampEvent, WordEnd, WordStart};

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WordEvent {
    WordStart(WordStart),
    WordEnd(WordEnd),
}

/// Transport batch. A metadata-only final batch has empty audio and null times.
#[derive(Debug, Default, Serialize)]
pub struct TimestampBatch {
    /// Mono PCM; skipped in JSON because WASM exports this as a Float32Array.
    #[serde(skip)]
    pub audio: Vec<f32>,
    pub events: Vec<WordEvent>,
    pub start_time: Option<f64>,
    pub end_time: Option<f64>,
    pub chunks_merged: usize,
}

pub(super) fn next_batch(
    events: &mut impl Iterator<Item = Result<TimestampEvent>>,
    min_samples: usize,
) -> Result<Option<TimestampBatch>> {
    let mut batch = TimestampBatch::default();
    while batch.audio.len() < min_samples.max(1) {
        match events.next().transpose()? {
            Some(TimestampEvent::AudioChunk(chunk)) => {
                anyhow::ensure!(
                    chunk.audio.dim(1)? == 1,
                    "Timestamp batching requires mono audio"
                );
                batch
                    .audio
                    .extend(chunk.audio.flatten_all()?.to_vec1::<f32>()?);
                batch.start_time.get_or_insert(chunk.start_time);
                batch.end_time = Some(chunk.end_time);
                batch.chunks_merged += 1;
            }
            Some(TimestampEvent::WordStart(word)) => batch.events.push(WordEvent::WordStart(word)),
            Some(TimestampEvent::WordEnd(word)) => batch.events.push(WordEvent::WordEnd(word)),
            None => break,
        }
    }
    Ok((!batch.audio.is_empty() || !batch.events.is_empty()).then_some(batch))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timestamps::{AudioChunk, WordEnd, WordStart};
    use candle_core::{Device, Tensor};

    #[test]
    fn batching_preserves_word_events_including_final_metadata_only_batch() -> anyhow::Result<()> {
        let mut events = vec![
            TimestampEvent::WordStart(WordStart {
                word: "hi".into(),
                word_index: 2,
                start_time: 0.0,
            }),
            TimestampEvent::AudioChunk(AudioChunk {
                audio: Tensor::new(&[[[0.2f32, -0.3]]], &Device::Cpu)?,
                start_time: 0.0,
                end_time: 0.02,
            }),
            TimestampEvent::AudioChunk(AudioChunk {
                audio: Tensor::new(&[[[0.4f32]]], &Device::Cpu)?,
                start_time: 0.02,
                end_time: 0.03,
            }),
            TimestampEvent::WordEnd(WordEnd {
                word: "hi".into(),
                word_index: 2,
                start_time: 0.0,
                end_time: 0.03,
            }),
        ]
        .into_iter()
        .map(Ok);
        let batch = next_batch(&mut events, 3)?.unwrap();
        assert_eq!(batch.audio, vec![0.2, -0.3, 0.4]);
        assert_eq!(batch.start_time, Some(0.0));
        assert_eq!(batch.end_time, Some(0.03));
        assert_eq!(batch.chunks_merged, 2);
        assert_eq!(
            serde_json::to_value(&batch.events)?[0],
            serde_json::json!({
                "kind": "word_start", "word": "hi", "word_index": 2, "start_time": 0.0
            })
        );
        let batch = next_batch(&mut events, 3)?.unwrap();
        assert!(batch.audio.is_empty());
        assert_eq!(batch.start_time, None);
        assert_eq!(batch.end_time, None);
        assert_eq!(
            serde_json::to_value(&batch.events)?[0],
            serde_json::json!({
                "kind": "word_end", "word": "hi", "word_index": 2, "start_time": 0.0, "end_time": 0.03
            })
        );
        assert!(next_batch(&mut events, 3)?.is_none());
        Ok(())
    }
}
