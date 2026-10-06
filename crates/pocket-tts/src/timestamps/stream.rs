use std::collections::VecDeque;

use anyhow::Result;

use super::{
    AudioChunk, TimestampEvent,
    alignment::{WordAlignment, is_voiced},
    attention::AttentionCapture,
    text::TimestampTextChunk,
};
use crate::{ModelState, TTSModel, tts_model::GeneratedFrame, voice_state::get_attention_cursor};

/// Owned, pull-based audio and word-event stream. Dropping it cancels generation
/// and releases its private decoder/KV state. Errors terminate the stream.
pub struct TimestampStream {
    model: TTSModel,
    voice: ModelState,
    chunks: std::vec::IntoIter<TimestampTextChunk>,
    segment: Option<Segment>,
    pending: VecDeque<TimestampEvent>,
    samples: usize,
    finished: bool,
}

struct Segment {
    frames: Box<dyn Iterator<Item = Result<GeneratedFrame>>>,
    alignment: WordAlignment,
}

impl TimestampStream {
    /// Batch mono audio without losing word events. Continue until `None`, even
    /// when a batch has no audio: it can carry the final word boundary.
    pub fn next_batch(&mut self, min_samples: usize) -> Result<Option<super::TimestampBatch>> {
        super::batch::next_batch(self, min_samples)
    }

    pub(crate) fn new(model: TTSModel, voice: ModelState, chunks: Vec<TimestampTextChunk>) -> Self {
        Self {
            model,
            voice,
            chunks: chunks.into_iter(),
            segment: None,
            pending: VecDeque::new(),
            samples: 0,
            finished: false,
        }
    }

    fn advance(&mut self) -> Result<Option<TimestampEvent>> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Ok(Some(event));
            }
            if let Some(segment) = &mut self.segment {
                let start = self.samples as f64 / self.model.sample_rate as f64;
                if let Some(frame) = segment.frames.next() {
                    let frame = frame?;
                    let scores = frame
                        .unit_scores
                        .expect("timestamped frame has captured scores");
                    let samples = frame.audio.flatten_all()?.to_vec1::<f32>()?;
                    self.pending.extend(segment.alignment.process_frame(
                        &scores,
                        is_voiced(&samples),
                        start,
                    )?);
                    self.samples += frame.audio.dim(2)?;
                    self.pending
                        .push_back(TimestampEvent::AudioChunk(AudioChunk {
                            audio: frame.audio,
                            start_time: start,
                            end_time: self.samples as f64 / self.model.sample_rate as f64,
                        }));
                } else {
                    self.pending.extend(segment.alignment.finish(start));
                    self.segment = None;
                }
            } else if let Some(chunk) = self.chunks.next() {
                let text_start =
                    get_attention_cursor(&self.voice, "flow_lm.transformer.layers.0.self_attn").len;
                let capture = AttentionCapture::new(
                    self.model.timestamp_heads.clone(),
                    text_start,
                    chunk.token_to_unit,
                )?;
                self.segment = Some(Segment {
                    frames: self.model.generate_segment_frames(
                        chunk.text,
                        chunk.tail_guess,
                        &self.voice,
                        Some(capture),
                    ),
                    alignment: WordAlignment::new(chunk.units),
                });
            } else {
                return Ok(None);
            }
        }
    }
}

impl Iterator for TimestampStream {
    type Item = Result<TimestampEvent>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        match self.advance() {
            Ok(Some(event)) => Some(Ok(event)),
            result => {
                self.finished = true;
                self.segment = None;
                self.pending.clear();
                result.err().map(Err)
            }
        }
    }
}

impl std::iter::FusedIterator for TimestampStream {}
