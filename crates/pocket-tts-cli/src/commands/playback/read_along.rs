//! Terminal read-along driven by consumed source samples.

use crossterm::{
    cursor::{Hide, MoveTo, Show},
    execute, queue,
    style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor},
    terminal::{self, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen},
};
use pocket_tts::timestamps::WordEvent;
use regex::Regex;
use std::{
    collections::BTreeMap,
    io::{self, Write},
    ops::Range,
    sync::{
        LazyLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::Receiver,
    },
    thread,
    time::Duration,
};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

struct Timing {
    start: f64,
    end: Option<f64>,
}

struct ReadAlong {
    text: String,
    words: Vec<Range<usize>>,
    timings: BTreeMap<usize, Timing>,
}

impl ReadAlong {
    fn new(text: &str) -> Self {
        // Same lexical grammar as timestamps/text.rs and the web read-along.
        static WORDS: LazyLock<Regex> = LazyLock::new(|| {
            Regex::new(r"[\p{L}\p{N}][\p{L}\p{N}\p{M}]*(?:[-‐‑'’][\p{L}\p{N}][\p{L}\p{N}\p{M}]*)*")
                .unwrap()
        });
        let text = plain_text(&pocket_tts::pause::strip_pause_markers(text));
        let words = WORDS.find_iter(&text).map(|m| m.range()).collect();
        Self {
            text,
            words,
            timings: BTreeMap::new(),
        }
    }

    fn update(&mut self, event: WordEvent) {
        let (index, word, start, end) = match event {
            WordEvent::WordStart(w) => (w.word_index, w.word, w.start_time, None),
            WordEvent::WordEnd(w) => (w.word_index, w.word, w.start_time, Some(w.end_time)),
        };
        if self
            .words
            .get(index)
            .is_some_and(|span| self.text[span.clone()] == word)
        {
            self.timings.insert(index, Timing { start, end });
        }
    }

    fn active(&self, time: f64) -> Option<usize> {
        self.timings.iter().rev().find_map(|(&index, timing)| {
            (time >= timing.start && timing.end.is_none_or(|end| time < end)).then_some(index)
        })
    }

    fn draw(&self, out: &mut impl Write, time: f64, columns: u16, rows: u16) -> io::Result<()> {
        queue!(out, MoveTo(0, 0), Clear(ClearType::All))?;
        if columns < 4 || rows < 8 {
            return out.flush();
        }
        let width = (columns - 2) as usize;
        let active = self.active(time);
        let status = if let Some(index) = active {
            let timing = &self.timings[&index];
            format!(
                "{time:.2}s | word #{index} | {:.2}-{}",
                timing.start,
                timing
                    .end
                    .map_or_else(|| "pending".into(), |end| format!("{end:.2}s"))
            )
        } else {
            format!(
                "{time:.2}s  |  {} timed words  |  waiting / between words",
                self.timings.len()
            )
        };
        queue!(
            out,
            SetForegroundColor(Color::Cyan),
            SetAttribute(Attribute::Bold)
        )?;
        put(out, 1, "Pocket TTS  /  live read-along", width)?;
        queue!(out, ResetColor, SetAttribute(Attribute::Reset))?;
        put(out, 2, &status, width)?;
        put(
            out,
            3,
            "Ctrl-C stops  |  approximate word boundaries; device latency may apply",
            width,
        )?;

        let wrapped = lines(&self.text, width);
        let focus = self
            .timings
            .iter()
            .rev()
            .find(|(_, t)| time >= t.start)
            .map_or(0, |(&index, _)| self.words[index].start);
        let focus_line = wrapped
            .iter()
            .position(|line| line.contains(&focus))
            .unwrap_or(0);
        let height = (rows - 7) as usize;
        let top = focus_line
            .saturating_sub(height / 2)
            .min(wrapped.len().saturating_sub(height));
        let highlight = active.map(|index| &self.words[index]);
        for (row, line) in wrapped.iter().skip(top).take(height).enumerate() {
            queue!(out, MoveTo(1, row as u16 + 5))?;
            let overlap = highlight.map(|span| span.start.max(line.start)..span.end.min(line.end));
            if let Some(span) = overlap.filter(|span| span.start < span.end) {
                queue!(
                    out,
                    Print(&self.text[line.start..span.start]),
                    // Reverse video remains visible when NO_COLOR disables colors.
                    SetAttribute(Attribute::Reverse),
                    SetAttribute(Attribute::Bold),
                    Print(&self.text[span.clone()]),
                    ResetColor,
                    SetAttribute(Attribute::Reset),
                    Print(&self.text[span.end..line.end])
                )?;
            } else {
                queue!(out, Print(&self.text[line.clone()]))?;
            }
        }
        put(
            out,
            rows - 1,
            &format!(
                "Lines {}-{} / {}  |  highlighting follows audio, not generation",
                top + 1,
                (top + height).min(wrapped.len()),
                wrapped.len()
            ),
            width,
        )?;
        out.flush()
    }
}

/// Never interpret user text as terminal escape commands.
pub(super) fn plain_text(text: &str) -> String {
    text.chars()
        .map(|c| if c.is_control() && c != '\n' { ' ' } else { c })
        .collect()
}

fn lines(text: &str, width: usize) -> Vec<Range<usize>> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut used = 0;
    for (offset, grapheme) in text.grapheme_indices(true) {
        if grapheme == "\n" {
            result.push(start..offset);
            start = offset + grapheme.len();
            used = 0;
            continue;
        }
        let size = grapheme.width();
        if used + size > width && offset > start {
            result.push(start..offset);
            start = offset;
            used = 0;
        }
        used += size;
    }
    result.push(start..text.len());
    result
}

fn put(out: &mut impl Write, row: u16, text: &str, width: usize) -> io::Result<()> {
    let first = lines(text, width).into_iter().next().unwrap();
    queue!(out, MoveTo(1, row), Print(&text[first]))
}

struct Terminal(io::Stderr);

impl Drop for Terminal {
    fn drop(&mut self) {
        let _ = execute!(
            self.0,
            ResetColor,
            SetAttribute(Attribute::Reset),
            Show,
            LeaveAlternateScreen
        );
    }
}

pub(super) fn run(
    text: &str,
    events: Receiver<WordEvent>,
    played: &AtomicU64,
    rate: u32,
    done: &AtomicBool,
    cancelled: &AtomicBool,
) -> io::Result<()> {
    let mut terminal = Terminal(io::stderr());
    execute!(terminal.0, EnterAlternateScreen, Hide)?;
    let mut view = ReadAlong::new(text);
    while !done.load(Ordering::Relaxed) && !cancelled.load(Ordering::Relaxed) {
        for event in events.try_iter() {
            view.update(event);
        }
        let time = played.load(Ordering::Relaxed) as f64 / rate as f64;
        let (columns, rows) = terminal::size()?;
        view.draw(&mut terminal.0, time, columns, rows)?;
        thread::sleep(Duration::from_millis(33));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pocket_tts::timestamps::{WordEnd, WordStart};

    #[test]
    fn narrow_terminal_keeps_the_whole_timestamp_range_visible() {
        let mut view = ReadAlong::new("extraordinary");
        view.update(WordEvent::WordEnd(WordEnd {
            word: "extraordinary".into(),
            word_index: 0,
            start_time: 1.25,
            end_time: 1.75,
        }));
        let mut output = Vec::new();
        view.draw(&mut output, 1.5, 42, 10).unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(
            output.contains("1.75s"),
            "end of timestamp range was clipped: {output:?}"
        );
        assert!(
            output.contains("\x1b[7m"),
            "highlight must also work without color"
        );
    }

    #[test]
    fn repeated_and_skipped_words_keep_source_indices_and_exclusive_ends() {
        let mut view = ReadAlong::new("go, go… go!");
        view.update(WordEvent::WordStart(WordStart {
            word: "go".into(),
            word_index: 2,
            start_time: 1.25,
        }));
        assert_eq!(view.active(1.24), None);
        assert_eq!(view.active(1.25), Some(2));
        assert_eq!(&view.text[view.words[2].clone()], "go");
        view.update(WordEvent::WordEnd(WordEnd {
            word: "go".into(),
            word_index: 2,
            start_time: 1.25,
            end_time: 1.75,
        }));
        assert_eq!(view.active(1.74), Some(2));
        assert_eq!(view.active(1.75), None);
        view.update(WordEvent::WordStart(WordStart {
            word: "wrong".into(),
            word_index: 0,
            start_time: 2.0,
        }));
        assert_eq!(view.active(2.1), None);
    }

    #[test]
    fn unicode_and_pause_markers_match_timestamp_word_identity() {
        let view = ReadAlong::new("Été, l’été co‑op e\u{301}! [pause:1s] 世界 42.");
        let words = view
            .words
            .iter()
            .map(|span| &view.text[span.clone()])
            .collect::<Vec<_>>();
        assert_eq!(words, ["Été", "l’été", "co‑op", "e\u{301}", "世界", "42"]);
    }

    #[test]
    fn wraps_unicode_without_splitting_graphemes_or_losing_newlines() {
        let text = "A界e\u{301}B\nfox";
        let rows = lines(text, 3)
            .into_iter()
            .map(|range| &text[range])
            .collect::<Vec<_>>();
        assert_eq!(rows, ["A界", "e\u{301}B", "fox"]);
        let text = "abc\n\nx";
        let rows = lines(text, 3)
            .into_iter()
            .map(|range| &text[range])
            .collect::<Vec<_>>();
        assert_eq!(rows, ["abc", "", "x"]);
    }
}
