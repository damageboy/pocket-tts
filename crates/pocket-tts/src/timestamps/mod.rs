//! Opt-in word timestamps derived from dpm63/pocket-tts-timestamped (MIT).
//! See the crate's THIRD_PARTY_NOTICES.md for provenance and the license.

pub(crate) mod alignment;
pub(crate) mod attention;
mod batch;
mod stream;
pub(crate) mod text;

use candle_core::Tensor;
use serde::Serialize;

pub use batch::{TimestampBatch, WordEvent};
pub use stream::TimestampStream;

/// Audio on the same sample-derived timeline as the word events (seconds).
#[derive(Debug)]
pub struct AudioChunk {
    /// Shape `[1, channels, samples]`, as in the ordinary streaming API.
    pub audio: Tensor,
    pub start_time: f64,
    pub end_time: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct WordStart {
    pub word: String,
    pub word_index: usize,
    pub start_time: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct WordEnd {
    pub word: String,
    pub word_index: usize,
    pub start_time: f64,
    pub end_time: f64,
}

/// Events come from a single generation pass. Word boundaries precede their
/// corresponding audio frame; a final WordEnd may follow the last audio frame.
/// Boundaries are approximate frame-level estimates, not forced alignment.
#[derive(Debug)]
pub enum TimestampEvent {
    AudioChunk(AudioChunk),
    WordStart(WordStart),
    WordEnd(WordEnd),
}
