//! Terminal read-along driven by consumed source samples.

use crossterm::{
    cursor::{Hide, Show},
    execute, queue,
    style::{Attribute, ResetColor, SetAttribute},
    terminal::{
        self, BeginSynchronizedUpdate, EndSynchronizedUpdate, EnterAlternateScreen,
        LeaveAlternateScreen,
    },
};
use pocket_tts::timestamps::WordEvent;
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
};
use regex::Regex;
use std::{
    collections::BTreeMap,
    io::{self, BufWriter, Write},
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

    fn display(&self, time: f64, columns: u16, rows: u16) -> DisplayState {
        let active = self.active(time);
        // Only the counter is throttled; highlight boundaries use the precise audio clock.
        let counter = (time * 10.0).floor() / 10.0;
        let status = if let Some(index) = active {
            let timing = &self.timings[&index];
            format!(
                "{counter:.1}s | word #{index} | {:.2}-{}",
                timing.start,
                timing
                    .end
                    .map_or_else(|| "pending".into(), |end| format!("{end:.2}s"))
            )
        } else {
            format!(
                "{counter:.1}s  |  {} timed words  |  waiting / between words",
                self.timings.len()
            )
        };
        let focus = self
            .timings
            .iter()
            .rev()
            .find(|(_, t)| time >= t.start)
            .map_or(0, |(&index, _)| self.words[index].start);
        DisplayState {
            active,
            focus,
            status,
            columns,
            rows,
        }
    }

    fn render(&self, frame: &mut Frame, state: &DisplayState) {
        let area = frame.area();
        if area.width < 4 || area.height < 8 {
            return;
        }
        let width = (area.width - 2) as usize;
        let row_area = |row| Rect::new(area.x + 1, area.y + row, width as u16, 1);
        frame.render_widget(
            Line::from("Pocket TTS  /  live read-along").style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            row_area(1),
        );
        frame.render_widget(state.status.as_str(), row_area(2));
        frame.render_widget(
            "Ctrl-C stops  |  approximate word boundaries; device latency may apply",
            row_area(3),
        );

        let wrapped = lines(&self.text, width);
        let focus_line = wrapped
            .iter()
            .position(|line| line.contains(&state.focus))
            .unwrap_or(0);
        let height = (area.height - 7) as usize;
        let top = focus_line
            .saturating_sub(height / 2)
            .min(wrapped.len().saturating_sub(height));
        let highlight = state.active.map(|index| &self.words[index]);
        for (row, line) in wrapped.iter().skip(top).take(height).enumerate() {
            let overlap = highlight.map(|span| span.start.max(line.start)..span.end.min(line.end));
            let content = if let Some(span) = overlap.filter(|span| span.start < span.end) {
                Line::from(vec![
                    Span::raw(&self.text[line.start..span.start]),
                    // Reverse video remains visible when NO_COLOR disables colors.
                    Span::styled(
                        &self.text[span.clone()],
                        Style::default().add_modifier(Modifier::REVERSED | Modifier::BOLD),
                    ),
                    Span::raw(&self.text[span.end..line.end]),
                ])
            } else {
                Line::from(&self.text[line.clone()])
            };
            frame.render_widget(content, row_area(row as u16 + 5));
        }
        frame.render_widget(
            format!(
                "Lines {}-{} / {}  |  highlighting follows audio, not generation",
                top + 1,
                (top + height).min(wrapped.len()),
                wrapped.len()
            ),
            row_area(area.height - 1),
        );
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

#[derive(PartialEq, Eq)]
struct DisplayState {
    active: Option<usize>,
    focus: usize,
    status: String,
    columns: u16,
    rows: u16,
}

fn draw_if_changed<W: Write>(
    terminal: &mut Terminal<CrosstermBackend<W>>,
    view: &ReadAlong,
    state: DisplayState,
    previous: &mut Option<DisplayState>,
) -> io::Result<()> {
    if previous.as_ref() == Some(&state) {
        return Ok(());
    }
    // Unsupported terminals ignore these commands. On supported terminals the diff is
    // presented atomically, including the clear Ratatui performs on a resize.
    queue!(terminal.backend_mut(), BeginSynchronizedUpdate)?;
    let result = terminal
        .draw(|frame| view.render(frame, &state))
        .map(|_| ());
    let end = execute!(terminal.backend_mut(), EndSynchronizedUpdate);
    result?;
    end?;
    *previous = Some(state);
    Ok(())
}

struct Screen(BufWriter<io::Stderr>);

impl Drop for Screen {
    fn drop(&mut self) {
        let _ = execute!(
            self.0,
            EndSynchronizedUpdate,
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
    let mut screen = Screen(BufWriter::new(io::stderr()));
    execute!(screen.0, EnterAlternateScreen, Hide)?;
    // Keep cooked input so the existing Ctrl-C cancellation handler still receives SIGINT.
    let mut terminal = Terminal::new(CrosstermBackend::new(&mut screen.0))?;
    let mut previous = None;
    let mut view = ReadAlong::new(text);
    while !done.load(Ordering::Relaxed) && !cancelled.load(Ordering::Relaxed) {
        for event in events.try_iter() {
            view.update(event);
        }
        let time = played.load(Ordering::Relaxed) as f64 / rate as f64;
        let (columns, rows) = terminal::size()?;
        draw_if_changed(
            &mut terminal,
            &view,
            view.display(time, columns, rows),
            &mut previous,
        )?;
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
        let mut terminal = Terminal::new(ratatui::backend::TestBackend::new(42, 10)).unwrap();
        terminal
            .draw(|frame| view.render(frame, &view.display(1.5, 42, 10)))
            .unwrap();
        let buffer = terminal.backend().buffer();
        let status: String = (0..42).map(|x| buffer[(x, 2)].symbol()).collect();
        assert!(
            status.contains("1.75s"),
            "end of timestamp range was clipped: {status:?}"
        );
        assert!(buffer[(1, 5)].modifier.contains(Modifier::REVERSED));
    }

    #[derive(Clone, Default)]
    struct CapturedOutput(std::rc::Rc<std::cell::RefCell<Vec<u8>>>);

    impl Write for CapturedOutput {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.borrow_mut().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn unchanged_frames_write_nothing_and_clock_updates_do_not_repaint_text() {
        let output = CapturedOutput::default();
        let mut terminal = Terminal::with_options(
            CrosstermBackend::new(output.clone()),
            ratatui::TerminalOptions {
                viewport: ratatui::Viewport::Fixed(Rect::new(0, 0, 80, 12)),
            },
        )
        .unwrap();
        let view = ReadAlong::new("one two");
        let mut previous = None;
        draw_if_changed(
            &mut terminal,
            &view,
            view.display(0.01, 80, 12),
            &mut previous,
        )
        .unwrap();
        output.0.borrow_mut().clear();
        draw_if_changed(
            &mut terminal,
            &view,
            view.display(0.09, 80, 12),
            &mut previous,
        )
        .unwrap();
        assert!(output.0.borrow().is_empty());
        draw_if_changed(
            &mut terminal,
            &view,
            view.display(0.11, 80, 12),
            &mut previous,
        )
        .unwrap();
        let bytes = output.0.borrow();
        let commands = std::str::from_utf8(&bytes).unwrap();
        assert!(!commands.is_empty());
        assert!(
            !commands.contains("\x1b[2J"),
            "ordinary frames must not clear the screen"
        );
        assert!(!commands.contains("Pocket TTS"));
        assert!(!commands.contains("one two"));
        assert!(commands.starts_with("\x1b[?2026h"));
        assert!(commands.ends_with("\x1b[?2026l"));
    }

    #[test]
    fn highlight_changes_within_a_counter_tick_and_clears_at_exclusive_end() {
        let mut view = ReadAlong::new("one two");
        for (word_index, word, start_time, end_time) in
            [(0, "one", 0.01, 0.04), (1, "two", 0.04, 0.08)]
        {
            view.update(WordEvent::WordEnd(WordEnd {
                word: word.into(),
                word_index,
                start_time,
                end_time,
            }));
        }
        let mut terminal = Terminal::new(ratatui::backend::TestBackend::new(80, 12)).unwrap();
        let first = view.display(0.02, 80, 12);
        let second = view.display(0.04, 80, 12);
        assert!(first != second);
        terminal.draw(|frame| view.render(frame, &first)).unwrap();
        assert!(
            terminal.backend().buffer()[(1, 5)]
                .modifier
                .contains(Modifier::REVERSED)
        );
        terminal.draw(|frame| view.render(frame, &second)).unwrap();
        assert!(
            !terminal.backend().buffer()[(1, 5)]
                .modifier
                .contains(Modifier::REVERSED)
        );
        assert!(
            terminal.backend().buffer()[(5, 5)]
                .modifier
                .contains(Modifier::REVERSED)
        );
        terminal
            .draw(|frame| view.render(frame, &view.display(0.08, 80, 12)))
            .unwrap();
        assert!(
            !terminal.backend().buffer()[(5, 5)]
                .modifier
                .contains(Modifier::REVERSED)
        );
    }

    #[test]
    fn scrolling_and_resizing_keep_the_active_word_visible() {
        let mut view = ReadAlong::new("zero\none\ntwo\nthree\nfour\nfive\nsix");
        view.update(WordEvent::WordStart(WordStart {
            word: "five".into(),
            word_index: 5,
            start_time: 1.0,
        }));
        let mut terminal = Terminal::new(ratatui::backend::TestBackend::new(20, 10)).unwrap();
        terminal
            .draw(|frame| view.render(frame, &view.display(1.0, 20, 10)))
            .unwrap();
        assert_eq!(terminal.backend().buffer()[(1, 6)].symbol(), "f");
        assert!(
            terminal.backend().buffer()[(1, 6)]
                .modifier
                .contains(Modifier::REVERSED)
        );
        terminal.backend_mut().resize(4, 8);
        terminal
            .draw(|frame| view.render(frame, &view.display(1.0, 4, 8)))
            .unwrap();
        assert_eq!(terminal.backend().buffer()[(1, 5)].symbol(), "f");
        assert!(
            terminal.backend().buffer()[(1, 5)]
                .modifier
                .contains(Modifier::REVERSED)
        );
        terminal.backend_mut().resize(3, 7);
        terminal
            .draw(|frame| view.render(frame, &view.display(1.0, 3, 7)))
            .unwrap();
        assert!(
            terminal
                .backend()
                .buffer()
                .content
                .iter()
                .all(|cell| cell.symbol() == " ")
        );
        terminal.backend_mut().resize(20, 10);
        terminal
            .draw(|frame| view.render(frame, &view.display(1.0, 20, 10)))
            .unwrap();
        assert!(
            terminal.backend().buffer()[(1, 6)]
                .modifier
                .contains(Modifier::REVERSED)
        );
    }

    #[test]
    fn arriving_end_event_refreshes_pending_status_even_when_clock_is_paused() {
        let mut view = ReadAlong::new("one");
        view.update(WordEvent::WordStart(WordStart {
            word: "one".into(),
            word_index: 0,
            start_time: 1.0,
        }));
        let pending = view.display(1.5, 80, 12);
        view.update(WordEvent::WordEnd(WordEnd {
            word: "one".into(),
            word_index: 0,
            start_time: 1.0,
            end_time: 2.0,
        }));
        let complete = view.display(1.5, 80, 12);
        assert!(pending != complete);
        assert!(pending.status.contains("pending"));
        assert!(complete.status.contains("2.00s"));
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
