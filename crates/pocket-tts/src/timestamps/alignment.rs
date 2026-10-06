//! Incremental alignment adapted from pocket-tts-timestamped (MIT).
//! See THIRD_PARTY_NOTICES.md. No ASR or additional learned model is used.

use super::{TimestampEvent, WordEnd, WordStart, text::TextUnit};

pub(crate) fn is_voiced(samples: &[f32]) -> bool {
    let energy: f64 = samples.iter().map(|&s| f64::from(s).powi(2)).sum();
    energy > f64::from(1e-3f32).powi(2) * samples.len() as f64
}

pub(crate) struct WordAlignment {
    units: Vec<TextUnit>,
    word_units: Vec<usize>,
    next: usize,
    open: Option<(usize, f64)>,
}

impl WordAlignment {
    pub fn new(units: Vec<TextUnit>) -> Self {
        let word_units = units
            .iter()
            .enumerate()
            .filter_map(|(i, u)| u.word.as_ref().map(|_| i))
            .collect();
        Self {
            units,
            word_units,
            next: 0,
            open: None,
        }
    }

    fn close(&mut self, end_time: f64) -> TimestampEvent {
        let (position, start_time) = self.open.take().unwrap();
        let word = self.units[self.word_units[position]].word.as_ref().unwrap();
        TimestampEvent::WordEnd(WordEnd {
            word: word.text.clone(),
            word_index: word.word_index,
            start_time,
            end_time,
        })
    }

    fn open_next(&mut self, start_time: f64) -> TimestampEvent {
        let word = self.units[self.word_units[self.next]]
            .word
            .as_ref()
            .unwrap();
        self.open = Some((self.next, start_time));
        self.next += 1;
        TimestampEvent::WordStart(WordStart {
            word: word.text.clone(),
            word_index: word.word_index,
            start_time,
        })
    }

    pub fn process_frame(
        &mut self,
        scores: &[f32],
        voiced: bool,
        start: f64,
    ) -> anyhow::Result<Vec<TimestampEvent>> {
        anyhow::ensure!(
            scores.len() == self.units.len(),
            "Incorrect timestamp unit score count"
        );
        let mut events = Vec::new();
        if !voiced {
            if let Some((position, _)) = self.open {
                let current = self.word_units[position];
                let future_dominates = scores[current + 1..].iter().any(|&s| s > scores[current]);
                let final_word = position + 1 == self.word_units.len();
                let later_punctuation = self.units[current + 1..]
                    .iter()
                    .any(|u| u.word.is_none() && !u.synthetic);
                if future_dominates || (final_word && !later_punctuation) {
                    events.push(self.close(start));
                }
            }
        } else if self.open.is_none() {
            if self.next < self.word_units.len() {
                events.push(self.open_next(start));
            }
        } else if self.next < self.word_units.len() {
            let current = self.word_units[self.open.unwrap().0];
            let next_dominates = scores[self.word_units[self.next]] > scores[current];
            let later_dominates = scores[current] < 0.001
                && self.word_units[self.next + 1..]
                    .iter()
                    .any(|&i| scores[i] > scores[current]);
            if next_dominates || later_dominates {
                events.push(self.close(start));
                events.push(self.open_next(start));
            }
        }
        Ok(events)
    }

    pub fn finish(&mut self, end: f64) -> Vec<TimestampEvent> {
        let events = if self.open.is_some() {
            vec![self.close(end)]
        } else {
            Vec::new()
        };
        self.next = self.word_units.len();
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timestamps::text::SourceWord;

    fn units(synthetic: bool) -> Vec<TextUnit> {
        vec![
            TextUnit {
                word: Some(SourceWord {
                    text: "first".into(),
                    word_index: 3,
                }),
                synthetic: false,
            },
            TextUnit {
                word: None,
                synthetic: false,
            },
            TextUnit {
                word: Some(SourceWord {
                    text: "second".into(),
                    word_index: 5,
                }),
                synthetic: false,
            },
            TextUnit {
                word: None,
                synthetic,
            },
        ]
    }

    #[test]
    fn silence_never_opens_words_and_punctuation_closes_them() {
        let mut alignment = WordAlignment::new(units(false));
        assert!(
            alignment
                .process_frame(&[0.8, 0.1, 0.1, 0.0], false, 0.0)
                .unwrap()
                .is_empty()
        );
        assert!(matches!(
            alignment
                .process_frame(&[0.8, 0.1, 0.1, 0.0], true, 0.08)
                .unwrap()
                .as_slice(),
            [TimestampEvent::WordStart(WordStart {
                word_index: 3,
                start_time: 0.08,
                ..
            })]
        ));
        assert!(matches!(
            alignment
                .process_frame(&[0.1, 0.7, 0.2, 0.0], false, 0.16)
                .unwrap()
                .as_slice(),
            [TimestampEvent::WordEnd(WordEnd {
                word_index: 3,
                start_time: 0.08,
                end_time: 0.16,
                ..
            })]
        ));
        assert!(
            alignment
                .process_frame(&[0.0, 0.0, 1.0, 0.0], false, 0.24)
                .unwrap()
                .is_empty()
        );
        assert!(matches!(
            alignment
                .process_frame(&[0.0, 0.0, 1.0, 0.0], true, 0.32)
                .unwrap()
                .as_slice(),
            [TimestampEvent::WordStart(WordStart {
                word_index: 5,
                start_time: 0.32,
                ..
            })]
        ));
        // Real trailing punctuation keeps the word open until its score wins.
        assert!(
            alignment
                .process_frame(&[0.0, 0.0, 0.9, 0.1], false, 0.40)
                .unwrap()
                .is_empty()
        );
        assert!(matches!(
            alignment.finish(0.48).as_slice(),
            [TimestampEvent::WordEnd(WordEnd {
                word_index: 5,
                end_time: 0.48,
                ..
            })]
        ));
        assert!(alignment.finish(0.56).is_empty());
    }

    #[test]
    fn transitions_are_monotonic_and_finish_does_not_invent_unspoken_words() {
        let mut alignment = WordAlignment::new(units(true));
        alignment
            .process_frame(&[1.0, 0.0, 0.0, 0.0], true, 1.0)
            .unwrap();
        assert!(
            alignment
                .process_frame(&[0.5, 0.0, 0.5, 0.0], true, 1.08)
                .unwrap()
                .is_empty()
        );
        assert!(matches!(
            alignment
                .process_frame(&[0.1, 0.0, 0.9, 0.0], true, 1.16)
                .unwrap()
                .as_slice(),
            [
                TimestampEvent::WordEnd(WordEnd {
                    word_index: 3,
                    end_time: 1.16,
                    ..
                }),
                TimestampEvent::WordStart(WordStart {
                    word_index: 5,
                    start_time: 1.16,
                    ..
                })
            ]
        ));
        assert!(matches!(
            alignment
                .process_frame(&[0.0, 0.0, 1.0, 0.0], false, 1.24)
                .unwrap()
                .as_slice(),
            [TimestampEvent::WordEnd(WordEnd {
                word_index: 5,
                end_time: 1.24,
                ..
            })]
        ));
        assert!(alignment.finish(1.32).is_empty());
        let mut alignment = WordAlignment::new(units(true));
        alignment
            .process_frame(&[1.0, 0.0, 0.0, 0.0], true, 0.0)
            .unwrap();
        assert_eq!(alignment.finish(0.08).len(), 1);
        assert!(
            alignment
                .process_frame(&[0.0, 0.0, 1.0, 0.0], true, 0.16)
                .unwrap()
                .is_empty()
        );
        assert!(alignment.process_frame(&[], true, 0.24).is_err());
    }

    #[test]
    fn voiced_gate_uses_energy_not_mean_and_strict_threshold() {
        assert!(!is_voiced(&[]));
        assert!(!is_voiced(&[0.001, -0.001]));
        assert!(is_voiced(&[0.002, -0.002]));
    }
}
