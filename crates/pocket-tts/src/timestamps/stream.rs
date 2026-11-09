use std::collections::VecDeque;

use anyhow::Result;

use super::{
    AudioChunk, TimestampEvent,
    alignment::{WordAlignment, is_voiced},
    attention::AttentionCapture,
    text::TimestampTextChunk,
};
use crate::{
    ModelState, TTSModel,
    tts_model::{GeneratedFrame, SegmentContext},
    voice_state::get_attention_cursor,
};

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
    continuation: bool,
    capture: bool,
    previous: Option<SegmentContext>,
}

struct Segment {
    frames: Box<dyn Iterator<Item = Result<GeneratedFrame>>>,
    alignment: WordAlignment,
    raw_text: String,
    latents: Vec<candle_core::Tensor>,
}

impl TimestampStream {
    /// Batch mono audio without losing word events. Continue until `None`, even
    /// when a batch has no audio: it can carry the final word boundary.
    pub fn next_batch(&mut self, min_samples: usize) -> Result<Option<super::TimestampBatch>> {
        super::batch::next_batch(self, min_samples)
    }

    pub(crate) fn new(
        model: TTSModel,
        voice: ModelState,
        chunks: Vec<TimestampTextChunk>,
        continuation: bool,
        capture: bool,
    ) -> Self {
        Self {
            model,
            voice,
            chunks: chunks.into_iter(),
            segment: None,
            pending: VecDeque::new(),
            samples: 0,
            finished: false,
            continuation,
            capture,
            previous: None,
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
                    if let Some(latent) = frame.normalized_latent {
                        segment.latents.push(latent.unsqueeze(1)?);
                    }
                    if let Some(scores) = frame.unit_scores {
                        let samples = frame.audio.flatten_all()?.to_vec1::<f32>()?;
                        self.pending.extend(segment.alignment.process_frame(
                            &scores,
                            is_voiced(&samples),
                            start,
                            frame.context_dominates,
                        )?);
                    }
                    self.samples += frame.audio.dim(2)?;
                    self.pending
                        .push_back(TimestampEvent::AudioChunk(AudioChunk {
                            audio: frame.audio,
                            start_time: start,
                            end_time: self.samples as f64 / self.model.sample_rate as f64,
                        }));
                } else {
                    self.pending.extend(segment.alignment.finish(start));
                    self.previous = if self.continuation && !segment.latents.is_empty() {
                        Some(SegmentContext {
                            previous_text: segment.raw_text.clone(),
                            normalized_latents: candle_core::Tensor::cat(&segment.latents, 1)?,
                        })
                    } else {
                        None
                    };
                    self.segment = None;
                }
            } else if let Some(mut chunk) = self.chunks.next() {
                let history = if let Some(previous) = &self.previous {
                    chunk.continue_from(
                        &previous.previous_text,
                        &self.model.text_options(),
                        self.model.conditioner.tokenizer(),
                        self.capture,
                    )?
                } else {
                    vec![0.0; chunk.token_to_unit.len()]
                };
                let capture = if self.capture {
                    let text_start =
                        get_attention_cursor(&self.voice, "flow_lm.transformer.layers.0.self_attn")
                            .len;
                    Some(
                        AttentionCapture::new(
                            self.model.timestamp_heads.clone(),
                            text_start,
                            chunk.token_to_unit,
                        )?
                        .with_history(history)?,
                    )
                } else {
                    None
                };
                let frames = self.model.generate_segment_frames(
                    chunk.text,
                    chunk.tail_guess,
                    &self.voice,
                    capture,
                    self.previous.as_ref(),
                    self.continuation,
                );
                // Prefill owns its private caches. The old segment is no longer
                // needed after initialization; only the new emitted latents grow.
                self.previous = None;
                self.segment = Some(Segment {
                    frames,
                    alignment: WordAlignment::new(chunk.units),
                    raw_text: chunk.raw_text,
                    latents: Vec::new(),
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
                self.previous = None;
                self.pending.clear();
                result.err().map(Err)
            }
        }
    }
}

impl std::iter::FusedIterator for TimestampStream {}
