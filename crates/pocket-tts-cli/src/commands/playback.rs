//! Native streaming playback. Inference never runs in the audio callback.

mod read_along;

use anyhow::{Context, Result, bail, ensure};
use pocket_tts::{ModelState, TTSModel, timestamps::TimestampBatch};
use rodio::{DeviceSinkBuilder, Player, Source};
use std::{
    io::IsTerminal,
    num::{NonZeroU16, NonZeroU32},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError},
    },
    thread,
    time::{Duration, Instant},
};

use super::generate::{GenerateArgs, GenerationTimings};

struct PcmSource {
    rx: Receiver<Vec<f32>>,
    chunk: std::vec::IntoIter<f32>,
    rate: NonZeroU32,
    played: Arc<AtomicU64>,
    cancelled: Arc<AtomicBool>,
    tail: u32,
}

impl PcmSource {
    fn new(rx: Receiver<Vec<f32>>, rate: u32, played: Arc<AtomicU64>) -> Self {
        Self {
            rx,
            chunk: Vec::new().into_iter(),
            rate: NonZeroU32::new(rate).expect("model sample rate is positive"),
            played,
            cancelled: Arc::new(AtomicBool::new(false)),
            // Rodio signals EOF before hardware playback finishes. Keep pulling
            // silence beyond the final PCM so the normal device buffer drains.
            // This is a latency allowance, not a hardware drain acknowledgment.
            tail: rate / 5,
        }
    }
}

impl Iterator for PcmSource {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.cancelled.load(Ordering::Relaxed) {
            return None;
        }
        loop {
            if let Some(sample) = self.chunk.next() {
                self.played.fetch_add(1, Ordering::Relaxed);
                return Some(sample);
            }
            match self.rx.try_recv() {
                Ok(chunk) => self.chunk = chunk.into_iter(),
                Err(TryRecvError::Empty) => return Some(0.0),
                Err(TryRecvError::Disconnected) => {
                    if self.tail == 0 {
                        return None;
                    }
                    self.tail -= 1;
                    return Some(0.0);
                }
            }
        }
    }
}

impl Source for PcmSource {
    fn current_span_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> NonZeroU16 {
        NonZeroU16::new(1).unwrap()
    }

    fn sample_rate(&self) -> NonZeroU32 {
        self.rate
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }
}

/// Bounded backpressure belongs on the producer, never the audio callback.
fn enqueue(tx: &SyncSender<Vec<f32>>, mut chunk: Vec<f32>, cancelled: &AtomicBool) -> Result<()> {
    loop {
        ensure!(!cancelled.load(Ordering::Relaxed), "Playback interrupted");
        match tx.try_send(chunk) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Full(returned)) => {
                chunk = returned;
                thread::sleep(Duration::from_millis(5));
            }
            Err(TrySendError::Disconnected(_)) => bail!("Audio output stopped"),
        }
    }
}

pub(super) fn run(
    model: &TTSModel,
    args: &GenerateArgs,
    text: &str,
    voice: &ModelState,
) -> Result<GenerationTimings> {
    let rate = model.sample_rate as u32;
    let mut batches: Box<dyn Iterator<Item = Result<TimestampBatch>> + '_> = if args.no_highlight {
        Box::new(model.generate_stream_long(text, voice).map(|chunk| {
            Ok(TimestampBatch {
                audio: chunk?.flatten_all()?.to_vec1::<f32>()?,
                chunks_merged: 1,
                ..Default::default()
            })
        }))
    } else {
        let mut stream = model
            .generate_audio_with_timestamps_stream(text, voice)
            .context(
                "Word tracking unavailable; use --play --no-highlight for audio-only playback",
            )?;
        if !args.quiet && !pocket_tts::pause::parse_explicit_pauses(text).is_empty() {
            eprintln!(
                "Warning: word tracking strips [pause:…] markers without inserting silence. Use --no-highlight to preserve pauses."
            );
        }
        Box::new(std::iter::from_fn(move || stream.next_batch(1).transpose()))
    };

    let cancelled = Arc::new(AtomicBool::new(false));
    let signal = cancelled.clone();
    ctrlc::set_handler(move || signal.store(true, Ordering::Relaxed))
        .context("Could not install playback interrupt handler")?;
    let failed = Arc::new(AtomicBool::new(false));
    let audio_failed = failed.clone();
    let audio_cancelled = cancelled.clone();
    let mut device = DeviceSinkBuilder::from_default_device()
        .context("No default audio output device")?
        .with_error_callback(move |_| {
            audio_failed.store(true, Ordering::Relaxed);
            audio_cancelled.store(true, Ordering::Relaxed);
        })
        .open_stream()
        .context("Could not open the default audio output device")?;
    device.log_on_drop(false);
    let player = Player::connect_new(device.mixer());
    // Each queued piece is at most 80 ms: bounded even for long pause tensors.
    let (tx, rx) = mpsc::sync_channel(8);
    let played = Arc::new(AtomicU64::new(0));
    let mut source = PcmSource::new(rx, rate, played.clone());
    source.cancelled = cancelled.clone();
    player.append(source);

    let mut wav = args
        .output
        .as_ref()
        .map(|path| {
            hound::WavWriter::create(
                path,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: rate,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                },
            )
            .with_context(|| format!("Could not create {}", path.display()))
        })
        .transpose()?;
    let interactive = !args.quiet
        && !args.no_highlight
        && std::io::stderr().is_terminal()
        && std::env::var("TERM").is_ok_and(|term| term != "dumb");
    if !args.quiet && !interactive {
        eprintln!("Playing: {}", read_along::plain_text(text));
    }
    let done = AtomicBool::new(false);
    let (words_tx, words_rx) = mpsc::channel();
    let result = thread::scope(|scope| {
        let render = scope.spawn(|| {
            if !interactive {
                return Ok(());
            }
            let result = read_along::run(text, words_rx, &played, rate, &done, &cancelled);
            if result.is_err() {
                cancelled.store(true, Ordering::Relaxed);
            }
            result
        });
        let result = (|| {
            let mut timings = GenerationTimings {
                inference: Duration::ZERO,
                concat: Duration::ZERO,
                wav_write: Duration::ZERO,
                audio_duration_sec: 0.0,
                chunks: 0,
            };
            let mut samples = 0usize;
            loop {
                ensure!(!cancelled.load(Ordering::Relaxed), "Playback interrupted");
                let start = Instant::now();
                let batch = batches.next().transpose()?;
                timings.inference += start.elapsed();
                let Some(batch) = batch else { break };
                // Metadata-only batches (including final WordEnd) are not EOF.
                if interactive {
                    for event in batch.events {
                        words_tx.send(event).context("Read-along stopped")?;
                    }
                }
                samples += batch.audio.len();
                timings.chunks += batch.chunks_merged;
                if let Some(writer) = &mut wav {
                    let start = Instant::now();
                    for sample in &batch.audio {
                        writer.write_sample((sample.clamp(-1.0, 1.0) * 32767.0) as i16)?;
                    }
                    timings.wav_write += start.elapsed();
                }
                for chunk in batch.audio.chunks((rate as usize * 80 / 1000).max(1)) {
                    enqueue(&tx, chunk.to_vec(), &cancelled)?;
                }
            }
            model.device.synchronize()?;
            ensure!(
                samples > 0,
                "No audio generated - text may be too short or invalid"
            );
            drop(tx);
            while !player.empty() {
                ensure!(!cancelled.load(Ordering::Relaxed), "Playback interrupted");
                thread::sleep(Duration::from_millis(10));
            }
            if let Some(writer) = wav.take() {
                let start = Instant::now();
                writer.finalize()?;
                timings.wav_write += start.elapsed();
            }
            timings.audio_duration_sec = samples as f32 / rate as f32;
            Ok(timings)
        })();
        // Always release the display, including generation, device and file errors.
        player.stop();
        done.store(true, Ordering::Relaxed);
        render
            .join()
            .map_err(|_| anyhow::anyhow!("Read-along thread panicked"))??;
        result
    });
    ensure!(
        !failed.load(Ordering::Relaxed),
        "Audio output device failed during playback"
    );
    if let Ok(timings) = &result
        && !args.quiet
    {
        eprintln!("Playback finished ({:.2}s).", timings.audio_duration_sec);
        if let Some(path) = &args.output {
            eprintln!("Saved: {}", path.display());
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{atomic::Ordering, mpsc::sync_channel};

    #[test]
    fn underrun_freezes_content_clock_but_real_silence_counts() {
        let (tx, rx) = sync_channel(2);
        let played = Arc::new(AtomicU64::new(0));
        let mut source = PcmSource::new(rx, 100, played.clone());
        assert_eq!(source.next(), Some(0.0));
        assert_eq!(played.load(Ordering::Relaxed), 0);
        tx.send(vec![0.25, 0.0, -0.75]).unwrap();
        assert_eq!(
            source.by_ref().take(3).collect::<Vec<_>>(),
            [0.25, 0.0, -0.75]
        );
        assert_eq!(played.load(Ordering::Relaxed), 3);
        assert_eq!(source.next(), Some(0.0));
        assert_eq!(played.load(Ordering::Relaxed), 3);
        tx.send(vec![0.5]).unwrap();
        assert_eq!(source.next(), Some(0.5));
        assert_eq!(played.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn eof_drains_all_chunks_and_silent_tail_without_advancing_clock() {
        let (tx, rx) = sync_channel(3);
        let played = Arc::new(AtomicU64::new(0));
        let mut source = PcmSource::new(rx, 100, played.clone());
        tx.send(vec![0.1, 0.2]).unwrap();
        tx.send(vec![]).unwrap();
        tx.send(vec![0.3]).unwrap();
        drop(tx);
        assert_eq!(source.by_ref().take(3).collect::<Vec<_>>(), [0.1, 0.2, 0.3]);
        let tail = source.by_ref().collect::<Vec<_>>();
        assert_eq!(tail, vec![0.0; 20]); // 200 ms at 100 Hz
        assert_eq!(played.load(Ordering::Relaxed), 3);
        assert_eq!(source.next(), None);
    }
}
