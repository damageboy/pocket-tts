# Pocket TTS upstream audit — 9 November 2026

## Finding

The last **selective upstream port** used **`9a1b48922deb44cd4a805fde279eb33b71c1a473`**, upstream **v4.0.0**, dated **20 October 2026**. It was implemented by local commit **`c234d8a94d4f1ca185eb0f06d57c5c2661fc9c50`**, “Support small Pocket TTS models with streaming word timestamps,” on the same date.

This is not a complete v4.0.0 synchronization. The general catalog and several runtime behaviors still derive from **v3.3.0**, **`3dbee45d343d7dddd0d105468d17f8dcba14db3e`**, dated **24 September 2026**. The October 5 resync deliberately stopped there. The October 20 work selectively added the three small English models and their required operators.

The inspected checkout is `/Users/dmg/projects/pocket-tts`, at **`995f871a04ffd333232c0cf70b93c38de0578942`**. The chat's initial `/Users/dmg/Documents/pocket-tts` directory is an empty Git repository. Its similarly named directory is not the Rust source checkout. The Rust checkout's `upstream` remote is `babybirdprd/pocket-tts`; Kyutai is the Python/model upstream requested here.

## Evidence in this checkout

- `README.md:31–46` distinguishes the v3.3 catalog from the selectively imported v4 small models.
- `.amp/in/small-reference/{english_18m_2026-10,english_7m_2026-10,english_2m_2026-10}/manifest.json` records `upstream_commit: 9a1b48922deb44cd4a805fde279eb33b71c1a473`.
- `assets/upstream-v3.3/manifest.json` records `3dbee45d343d7dddd0d105468d17f8dcba14db3e`.
- `crates/pocket-tts/tests/upstream_model.rs` explicitly accepts those two reference revisions.
- `docs/superpowers/plans/2026-10-05-upstream-v3-resync.md` identifies `41cbc84af539ea78a804ffca5f9c6edc1a22ce44` as a separately evaluated snapshot, not the implemented baseline.
- `git show c234d8a` confirms small-model operators, three configs and calibrated timestamp heads. `995f871` subsequently archived historical development artifacts; it did not resync inference.

The local extensions include shared Rust/WASM attention-derived word timestamps (`1e0500d`, October 6), terminal playback and word tracking (`ce0ab8f`, October 6), and small-model timestamp calibration (`c234d8a`, October 20).

## Upstream changes after the last selective port

Verified upstream `main` is **`c91f50f38c6b914d49dbdff51b8d322edf843db2`**, dated **9 November 2026**. The exact [v4 baseline-to-main comparison](https://github.com/kyutai-labs/pocket-tts/compare/9a1b48922deb44cd4a805fde279eb33b71c1a473...c91f50f38c6b914d49dbdff51b8d322edf843db2) contains **21 commits / 39 changed files**. Seven commits affect inference/model selection, six affect training, one changes deployment configuration, and seven are documentation/version changes. File totals include documentation and tests.

### Changes relevant to this inference port

| Commit | Actual behavior change | Local impact |
|---|---|---|
| [`b300d2f`](https://github.com/kyutai-labs/pocket-tts/commit/b300d2f40334755c0974c1255d69d7818d184203), Oct 21, #398 | Adds `english_2026-10`, makes it the English default, and replaces all six small-model checkpoints, including REFL variants. Updates matching preset states. Targets dropped sentence-final numbers and poorly learned conversational words, digits and acronyms. | Required model/config update. Existing small-model names now refer to different weights upstream. Old timestamp heads are not validated for the replacement weights. |
| [`44cb63e`](https://github.com/kyutai-labs/pocket-tts/commit/44cb63eaa4144f32ff80863968515165a5554353), Oct 22, #399 | Checks decimal punctuation with a bounded token window instead of decoding the whole prefix and suffix repeatedly. Preserves the sentence boundary in `3. 14 people...`. | Our `text.rs::find_boundaries` still performs the repeated full decoding. Port the final combined chunker implementation. |
| [`2dae420`](https://github.com/kyutai-labs/pocket-tts/commit/2dae420ca04c361816c99fb4825605d033ce0d85), Oct 26, #401 | Optional `teacher_force_previous_chunk`: condition on previous + current text, replay the previous chunk's generated latents, and warm the audio decoder with its final 12 frames. Default chunk size becomes 25 tokens in this mode, otherwise 50. | Largest runtime change. Requires new segment orchestration and timestamp mapping. Default remains off. |
| [`4cc5753`](https://github.com/kyutai-labs/pocket-tts/commit/4cc5753319ceddc5cbf70598d9011f1f22858ba0), Nov 3, #409 | Keeps a segment ending in a decimal number with the next segment, even beyond the soft token limit; avoids turning `$822.95,` into a terminal chunk that drops the cents. | Shared text splitter and timestamp source mapping need regression coverage. It cannot attach a following segment when none exists. |
| [`725e12c`](https://github.com/kyutai-labs/pocket-tts/commit/725e12c70520a26dc417ef61044503e7dabaa699), Nov 4, #411 | Protects decimal commas, thousands separators, URLs, emails, inner abbreviation punctuation and specified titles. Updates Python and Kyutai's separate Rust implementation. | Port the algorithm into our Candle code; upstream's Rust backend is not a replacement for this repository. |
| [`f6104b5`](https://github.com/kyutai-labs/pocket-tts/commit/f6104b50e3b9308edfd0b7b3bb81e224ae07cc7c), Nov 5, #415 | Keeps closing quotes/brackets with the preceding sentence/clause. Uses bounded lookahead for multibyte quote tokens; opening quotes after spaces stay with the next sentence. | Important for source-word mapping and avoiding punctuation-only chunks. |
| [`cb77ab8`](https://github.com/kyutai-labs/pocket-tts/commit/cb77ab8561658221d6912cff6e83c09834a70819), Oct 20, #392 | Explicit UTF-8 when Python reads YAML and its HTML page. | Rust config reading already uses UTF-8 strings. Verify non-ASCII config/text paths; no corresponding Python I/O patch is needed. |

No post-v4.0.0 changes in this comparison alter the FlowLM transformer architecture, sampler implementation, or Mimi model architecture. The new inference work is text splitting, segment conditioning, and model/voice selection.

The replacement weight revisions are `db04f0e653473e5c7333c69686ac92b7460b0c1d` (cloning bundle) and `a7056f505a4a8ca72d8b5fb1440071d6f1a35739` (public bundle and upstream presets). Tokenizer pins must still be taken from each model's config. A repository revision alone does not mean every path's contents changed.

### Real work outside this inference port

| Commit(s) | Work | Treatment |
|---|---|---|
| [`edbd17c`](https://github.com/kyutai-labs/pocket-tts/commit/edbd17c8d9e55865849212a6fa604dbb27dc88f7), [`3396b66`](https://github.com/kyutai-labs/pocket-tts/commit/3396b66bf364f525df6a4641f66fe601ff5a265a), #394/#397 | Training loader preserves audio through an unaligned final spoken word, while ignoring punctuation-only trailing tokens when deciding the endpoint. | Benefit arrives through retrained weights; no inference-time alignment loader to port. |
| [`cef8e22`](https://github.com/kyutai-labs/pocket-tts/commit/cef8e2250662c91aa1ef64a014fbc8c5f55bfb1b), #396 | Removes weight decay from the text embedding table, preserving tokens absent from fine-tuning data. | Training-only; adopt the resulting checkpoints. |
| [`77c3d76`](https://github.com/kyutai-labs/pocket-tts/commit/77c3d76b63d6cfe08c674b4f728db6774cf01df6), #402 | Precomputed teacher-target cache, cache builder, sharded/resumable reading and cached-target distillation. | Training-only; do not confuse this with runtime previous-chunk teacher forcing. |
| [`2e117a1`](https://github.com/kyutai-labs/pocket-tts/commit/2e117a1e7df06323c9df0315c671f3cbc3b8d9fc), #414 | Graceful training stop through a `STOP` file and cleanup of loader processes on exit. | No runtime port. |
| [`0d20106`](https://github.com/kyutai-labs/pocket-tts/commit/0d2010682f3ec336890f55cb2acc0f22abbc1d8d), #416 | Muon becomes the default training optimizer, with a fine-tuning learning-rate adjustment. | No runtime port. |
| [`da629e6`](https://github.com/kyutai-labs/pocket-tts/commit/da629e60b68fc141a790daa05630735966b68720) | Removes `--quantize` from the two 2M deployment services. | Deployment choice, not removal of quantization support or a new quantization kernel. Keep our default unquantized; validate any explicit quantized use separately. |

The remaining seven commits are `a922b91` (version 4.0.1) and `1a9e9ef`, `b20753a`, `c2af77e`, `91a9e96`, `28770c0`, `c91f50f` (README/model-community/report links). They do not add runtime support for those community languages.

## Earlier differences still relevant

Because the October 20 import was selective, comparing only `9a1b489..main` misses these items. The [v3.3-to-v4 comparison](https://github.com/kyutai-labs/pocket-tts/compare/3dbee45d343d7dddd0d105468d17f8dcba14db3e...9a1b48922deb44cd4a805fde279eb33b71c1a473) and local source confirm:

1. **Prompt ends and output onsets:** `bdac156` adds voice-prompt trimming, a 20 ms fade-out and 80 ms silence; `523e8a6`/`a3325b1`/`80d24a3` ultimately settle on a **5 ms output fade-in**, not an 80 ms fade. Neither preprocessing nor output fade is in our current generation path. These matter before reproducing upstream continuation.
2. **Catalog/presets:** October multilingual refresh `41cbc84` and restored pause-ended preset selection `1df6b3d` were not adopted for the legacy catalog. Native and browser code retain a legacy preset revision and a separate small-model revision. Refresh weights and voice states together per model.
3. **Text replacements:** #387's small-model replacements exist locally; its changes to older English/German and other configs were not imported as a complete set. Existing `english.yaml` lacks the newer curly-apostrophe map; `german_24l.yaml` lacks dash-to-comma entries.
4. **Additional configurations:** `english_2026-09_2`, `english_drifting_26-09`, and the three `_refl_2026-10` configs are absent. The underlying drifting/pointwise machinery now exists, so these are chiefly bundle/catalog/validation work, not a second sampler implementation.
5. **Custom Mimi override:** upstream can replace a bundled Mimi with `mimi.weights_path`. Our loader still constructs everything from one bundle. This older custom-model capability is a separate completeness item; none of the shipped configs being updated requires it.

Kyutai also added an xn-based Rust CLI before v4.0.0. This plan keeps the existing Candle implementation, public APIs, WASM packaging and timestamp extension.

## Timestamp preservation requirements

`timestamps/stream.rs` currently creates a fresh segment, derives the text-key offset from the voice attention-cache length, and advances time only by emitted PCM samples. `timestamps/attention.rs` averages calibrated query-head distributions normalized over text keys only. `timestamps/text.rs` maps those tokens to original words, including normalization and synthetic punctuation. Ordinary and timestamped generation share `TTSModel::generate_segment_frames`.

Preserve that shared decoder path, original source indices, text-only attention normalization, event schema, cancellation/state isolation, and final event-only batches. Timestamps remain approximate, normally one 80 ms model frame apart.

For continuation, previous text becomes context rather than a second set of source words. Use an explicit token-to-current-word mapping for the combined prompt; ignore historical words without pretending their token positions do not exist. Forced frames and decoder warmup must produce **no audio events, no word events and no elapsed output time**. Warmup is at most 12 frames (0.96 s at 12.5 Hz) and is discarded. Reset alignment for the new segment while retaining the global emitted-sample clock.

Changing weights invalidates the old calibration evidence even when layer/head counts are identical. The documented old 18M/7M/2M selections `(5,3)`, `(5,3)+(4,3)`, `(5,2)+(5,0)` may be candidates, but cannot be carried forward as verified. Recalibrate and validate both independent and continuation modes before promoting refreshed timestamp-enabled bundles.

## Scope and verification status

This is a source/history audit and implementation plan. No inference tests, calibration runs or audio-quality measurements were performed during this audit. Historical logs in `.amp` are evidence of the previous implementation only. The existing modified `crates/pocket-tts/pkg/pocket_tts_bg.wasm` was left intact.

See [the implementation plan](../plans/2026-11-09-upstream-sync-preserve-timestamps.md) for sequencing, interfaces and acceptance checks. Training code, a backend migration, community-model ingestion, and custom split-bundle loading are excluded from the proposed inference update; the last item remains explicitly documented as an older parity gap.
