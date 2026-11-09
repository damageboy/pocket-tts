# Pocket TTS upstream synchronization with preserved timestamps — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the relevant inference changes through Kyutai commit `c91f50f38c6b914d49dbdff51b8d322edf843db2` while preserving single-pass Rust/WASM word timestamps and playback tracking.

**Architecture:** Keep Candle and the shared segment decoder. Update shared text preparation, make model/voice/calibration identities explicit, then add optional previous-chunk conditioning to both ordinary and timestamped streams. Existing entry points keep their signatures and default continuation off.

**Tech Stack:** Rust 2024, Candle 0.10.2, tokenizers 0.23.1, WASM/wasm-bindgen, TypeScript browser worker, Axum, PyO3; pinned Python/PyTorch only for reference generation and offline calibration.

**Spec:** `docs/superpowers/specs/2026-11-09-upstream-audit.md`, particularly “Timestamp preservation requirements.”

## Global Constraints

Run from `/Users/dmg/projects/pocket-tts`; the similarly named Documents workspace is empty. Starting HEAD is `995f871a04ffd333232c0cf70b93c38de0578942`. Preserve the existing modified `crates/pocket-tts/pkg/pocket_tts_bg.wasm`; its audit-time SHA-256 is `442f68ccf15623a83b8240556950c8911e481f5ecf0d163d47fd716251f7a4ca`. Recheck working state before implementation. Build new WASM artifacts into a separate output directory.

“Always use --release” and “Streaming first” are repository requirements. Keep current CPU optimizations, Rust/CLI/HTTP/Python/WASM interfaces and timestamp event fields. No runtime ASR, training subsystem, backend replacement, or silently floating model revisions. Preserve the timestamp fork's MIT attribution.

Use immutable bundle identity: config, model weights, tokenizer, preset states and timestamp calibration. A shape-compatible head selection is not calibration. Never update all preset pins globally while leaving some models on older weights. Do not promote new `english` or small-model weights if doing so removes an existing timestamp capability.

Output time is emitted PCM sample count divided by sample rate. Forced context and decoder warmup are excluded. At 24 kHz/12.5 Hz, one normal output frame is 1,920 samples/80 ms. Continue consuming batches until `None`; an empty audio buffer can carry the final word end. Preserve existing pause semantics: ordinary timestamp streams strip explicit pause markers, whereas `generate_stream_long` inserts silence.

## Review Focus

Decimal/quote splitting with SentencePiece and byte-based JSON tokenizers must preserve input text and source indices (Task 2).

Replacement weights with old cached voice states or attention-head choices must not masquerade as a validated bundle (Task 4).

Silent, shorter-than-20-ms, partially framed, and speech-ended clone prompts must match the pinned preprocessing without mutating the input (Task 3).

Repeated words across segment boundaries and historical-context attention must not duplicate or invent word events (Task 5).

Cancellation during continuation prefill/warmup and a final event-only batch must preserve reusable voice state and playback timing (Tasks 5–6).

## Progress

- [x] 2026-11-09: identify both source baselines and inspect all 21 subsequent commit/file changes.
- [x] 2026-11-09: trace current text, timestamp, decoder and model/preset selection paths.
- [ ] Task 1: establish reproducible current and target reference fixtures.
- [ ] Task 2: port chunker fixes with source-word mapping regressions.
- [ ] Task 3: port prompt pause normalization and output fade.
- [ ] Task 4: adopt matching bundles with validated timestamp heads.
- [ ] Task 5: implement continuation in the shared streaming core and timestamp path.
- [ ] Task 6: expose options consistently and complete native/WASM acceptance.
- [ ] Task 7: synchronize Cargo package/dependency/lockfile versions with local release tag v4.0.1 and add packaging guards.

Each task is one reviewable milestone. Write its behavioral tests first, run the named checks to establish the expected failures, implement, rerun, then commit only its intended paths. Fixture/bootstrap checks in Task 1 verify provenance rather than requiring an artificial failing test. Follow the repository's living-plan conventions in `.agent/PLANS.md`; maintain the decisions and outcome sections below.

## Context and File Structure

`crates/pocket-tts/src/text.rs` owns token-based splitting. `tts_model.rs` owns model construction, voice preparation and `generate_segment_frames`; both audio-only and timestamp streams consume that decoder. `models/flow_lm.rs` transforms normalized audio latents into the next latent. `models/mimi.rs` decodes them to PCM. A latent is one model-generated audio frame representation, not emitted sound until decoded.

`timestamps/text.rs` preserves source words through text preparation; `timestamps/attention.rs` reduces selected attention heads; `timestamps/stream.rs` associates scores with voiced PCM frames and the output sample clock. `timestamps/batch.rs` combines events without losing final boundaries. Keep those responsibilities.

Native preset selection is in `crates/pocket-tts-cli/src/voice.rs`; browser preset selection is in `crates/pocket-tts-cli/web/src/workers/model-config.ts`. Model lists also live in `commands/list.rs` and `web/src/lib/model-catalog.ts`; `scripts/generate-model-catalog.ts` exports the browser catalog. Changes must reach every selector, not just YAML files.

Use the existing tests under `crates/pocket-tts/tests`, plus new `upstream_continuation.rs` for the new state transitions. Add `scripts/generate_sync_reference.py` and `scripts/calibrate_timestamp_heads.py` as portable reference/calibration entry points. Save small deterministic fixtures under `assets/upstream-2026-11-09`; keep downloaded model weights and bulky calibration audio in ignored local storage. Preserve old v3.3 and v4 fixtures unchanged.

## Task 1: Establish pinned references and preservation checks

**Files:** Create `scripts/generate_sync_reference.py`, `scripts/test_sync_reference.py`, and `assets/upstream-2026-11-09/manifest.json`; modify `crates/pocket-tts/tests/upstream_model.rs`, `timestamps_model.rs`, `wasm_path_parity.rs`, and `scripts/test-wasm-timestamps.mjs` to accept explicit fixture config paths without abandoning old manifests.

**Interfaces:** The new generator accepts `--reference`, `--expected-commit`, `--output`, `--language`, `--voice`, `--text`, and `--teacher-force-previous-chunk`. Its manifest records source SHA, dependency versions, complete config, asset URI/revision/SHA-256, text, token IDs, options, and tensor names. Verify source HEAD before importing Python. Never use the checked-in v1.0.1 `python-reference` as the oracle.

First record current independent-stream PCM, event order and word indices using deterministic noise or temperature zero for `english` plus all three small models. Record old config/voice/calibration identities, not just language names. Make fixture config selection prefer an explicit manifest path; this allows old weights to remain testable after catalog promotion. Verify unknown revisions, missing assets, and config/asset mismatch are fatal.

Generate target text-splitting fixtures with both tokenizer families, prompt-preprocessing fixtures, and model fixtures containing voice/text-prefill caches, fixed noise, latents, EOS and raw decoder PCM. Keep output fade as a separate assertion from raw decoder parity. Add continuation fixtures showing combined text, forced prefix, warmup length and emitted frames. Regenerate deterministic fixtures twice and compare their hashes.

Run `python -m pytest scripts/test_sync_reference.py -q` in the pinned reference environment. Then run `POCKET_TTS_REFERENCE=<absolute-fixture-directory> cargo test --release -p pocket-tts --test upstream_model --locked -- --ignored --nocapture` for each retained baseline. Expected: source/asset validation passes and old parity remains within the existing per-tensor tolerances. Missing assets must not produce a passing skip. Commit the fixture harness and small reproducible fixtures.

## Task 2: Port final upstream text splitting

**Files:** Modify `crates/pocket-tts/src/text.rs`, `crates/pocket-tts/tests/upstream_text.rs`, `crates/pocket-tts/src/timestamps/text.rs`, and text fixtures from Task 1.

**Interfaces:** Preserve `TextOptions::split(&self, tokenizer: &Tokenizer, text: &str, max_tokens: usize) -> Result<Vec<String>>` and `TextOptions::prepare`. Internally replace full-prefix decimal checking with bounded neighboring-token checks and closer lookahead; both sentence and fallback clause boundaries receive the tokenizer.

First add tests `decimal_boundaries_preserve_numbers_and_spaces`, `titles_urls_and_emails_stay_intact`, `closers_stay_with_their_sentence`, `decimal_segments_keep_the_following_clause`, and `timestamp_indices_survive_upstream_chunking`. Compare exact chunks against the pinned Python fixtures. Include `3.14` versus `3. 14 people`, `$822.95, which...`, `822,95 €`, `1,994.42`, `Dr. Miller`, URLs/emails, `p.m.`, final `U.S.`, `He said “Stop.” Then...`, and `He left. “Hi.”`.

Implement the final #399/#409/#411/#415 behavior together. The title set in actual source is `Mr. Mrs. Ms. Dr. Prof. Rev. vs. e.g. i.e.`; do not add `St.` based solely on the commit description. Use a six-token title window and at most eight closer tokens, matching the pinned source. Handle partial multibyte closers without advancing to an incomplete character. Glue decimal-ended refined segments to the following segment before greedy grouping, retaining a soft token limit and final unpaired segments.

Verify repeated words, Unicode normalization expansions, replacement/deletion maps, synthetic punctuation and pause stripping retain original indices or explicitly omit ambiguous words as today. Assert concatenated source text is not corrupted. Instrument decoded token volume on a repeated long prompt to demonstrate linear rather than quadratic work; avoid a flaky wall-clock unit test.

Run `cargo test --release -p pocket-tts --test upstream_text --locked` and `cargo test --release -p pocket-tts --lib timestamps::text --locked`. Expected: new cases fail before the port and match target fixtures afterward; ordinary and timestamp preparation use the same chunks. Commit the splitter and mapping tests together.

## Task 3: Port prompt normalization and decoder onset fade

**Files:** Modify `crates/pocket-tts/src/audio.rs`, `tts_model.rs`, `tests/upstream_model.rs`, and `tests/timestamps_model.rs`; add `tests/upstream_audio.rs`.

**Interfaces:** Add `audio::end_on_pause(audio: &Tensor, sample_rate: usize) -> Result<Tensor>`. Call it once on the model-rate audio in `get_voice_state_from_tensor`, before codec frame padding/encoding; file/byte clone entry points already route there. Imported preset states bypass it. Add internal `fade_in_first_frame(audio: &Tensor, sample_rate: usize) -> Result<Tensor>` to the shared segment output path.

First test exact behavior: complete 20 ms analysis frames; RMS energy across batch/channel dimensions; strict `db > max_db - 35`; trim after the last active frame; fade out the last min(20 ms, retained length); append exactly 80 ms zeros. With no complete frame, upstream returns unchanged audio. All-zero audio containing complete frames follows the same numerical threshold rule, not an invented special case. Compare to Python and assert input tensors remain unchanged.

For fresh output decoders, fade exactly the first `sample_rate / 200` samples using an inclusive zero-to-one ramp (120 samples at 24 kHz), once per text segment. Do not remove samples or fade each transport packet. Keep fade above raw Mimi decoding so raw operator fixtures remain valid. Continued/warmed decoder segments in Task 5 skip this fresh-state fade.

Test clone PCM ending in speech, long silence, partial final analysis frame and more than 12 seconds of input. Test exact output ramp, unchanged remainder/next frame, identical sample count, and identical audio between timestamp-enabled and audio-only streams. Recheck voiced-frame gating against the emitted faded PCM.

Run `cargo test --release -p pocket-tts --test upstream_audio --locked`, relevant library tests, and model-backed `upstream_model`/`timestamps_model` with explicit fixtures. Expected: preprocessing tensors match Python and time is never shifted by the fade. Commit once the cloned-prompt and shared-output checks pass.

## Task 4: Refresh model bundles and recalibrate timestamps

**Files:** Modify affected `crates/pocket-tts/config/*.yaml`, `crates/pocket-tts/src/{config,tts_model}.rs`, `crates/pocket-tts-cli/src/voice.rs`, `commands/list.rs`, `web/src/lib/model-catalog.ts`, `web/src/workers/model-config.ts`, their tests, `scripts/generate-model-catalog.ts`, and README. Create missing configs `english_2026-10`, `english_2026-09_2`, `english_drifting_26-09`, and the three small `_refl_2026-10` variants. Create `scripts/calibrate_timestamp_heads.py` and calibration manifests.

**Interfaces:** Add optional `preset_voice_revision: Option<String>` to Rust `Config`, carry it through `TTSModel`, and add a matching YAML scalar consumed by native/browser catalog generation. Change the private native helper to `stock_voice_url(model: &TTSModel, name: &str) -> Result<String>` so it can use both model origin and preset revision. Preserve current fallback behavior for old/custom configs without it. Timestamp calibration records identify weights, tokenizer, preset revision, query-head pairs and supported generation modes. The existing `timestamp_heads` field remains the runtime input.

First add tests proving native and browser asset resolution agree for each bundle, old snapshots remain reproducible through explicit fixtures, cache URLs include revisions, and a new bundle cannot inherit verified timestamp status just from matching dimensions. Update config-equality tests to exclude only the documented Rust extension fields (`timestamp_heads`, `preset_voice_revision`), while comparing all upstream fields.

Copy target upstream config values at the pinned source SHA. Refresh existing English/small and October multilingual bundles atomically with matching presets; preserve tokenizer pins per config. Include earlier text-replacement maps. New English/small weights use cloning revision `db04f0e653473e5c7333c69686ac92b7460b0c1d` and public revision `a7056f505a4a8ca72d8b5fb1440071d6f1a35739`. Resolve other bundles from their target configs, never by assigning that weight revision indiscriminately.

Calibrate each changed timestamp-enabled checkpoint independently. Keep original heads only if held-out validation demonstrates their suitability. Use at least two voices and disjoint calibration/evaluation texts covering ordinary prose, numbers, punctuation, repeated words and multi-chunk input. Record matched-word coverage, mean and 95th-percentile boundary error, missing/repeated words and head candidates. ASR can propose alignment offline, but manually inspect a subset and never install ASR in the runtime path.

Promotion requires no regression in the existing event/PCM contract and no degradation of word coverage or boundary error against the old bundle on the same evaluation set. If model speech changes make an ASR reference ambiguous, adjudicate those cases before comparing. Revalidate continuation separately in Task 5; this milestone initially validates independent segments. New models may remain explicitly audio-only with missing heads, as existing unsupported configs do. Previously supported model names must retain timestamp capability before replacement/default promotion.

Run model-backed `upstream_model` and `timestamps_model` for every changed supported checkpoint, then native/catalog/browser asset tests. Existing tensor tolerances are the starting acceptance limits: embeddings `2e-6`, caches/latents `5e-4`, raw decoder PCM `2e-3`; investigate failures rather than relaxing them silently. Verify sentence-final years and conversational/numeric prompts audibly. Publish calibration evidence and update the README's stale statement that drifting is absent. Commit validated bundle changes in small model groups; keep `english` promotion until its calibration passes.

## Task 5: Add previous-chunk continuation without duplicate timestamps

**Files:** Modify `crates/pocket-tts/src/tts_model.rs`, `models/flow_lm.rs`, `timestamps/{text,attention,alignment,stream}.rs`, and `lib.rs`; create `crates/pocket-tts/tests/upstream_continuation.rs`; extend timestamp tests.

**Interfaces:** Add exported `GenerationOptions { max_tokens: Option<usize>, teacher_force_previous_chunk: bool }`, defaulting to `None` and `false`. Add `generate_stream_owned_with_options(&self, text: &str, voice_state: &ModelState, options: GenerationOptions) -> Box<dyn Iterator<Item = Result<Tensor>>>` and `generate_audio_with_timestamps_stream_with_options(&self, text: &str, voice_state: &ModelState, options: GenerationOptions) -> Result<TimestampStream>`. Existing methods delegate with defaults.

Add internal `SegmentContext { previous_text: String, normalized_latents: Tensor }`. Extend `GeneratedFrame` with `normalized_latent: Option<Tensor>` populated only when continuation needs retention. Extend the internal segment generator with optional context and retain-latents inputs. Use the same segment driver for audio-only and timestamped orchestration; retain only the immediately preceding emitted segment, not all earlier audio. Generalize timestamp text preparation to take the resolved token limit.

First add `continuation_off_preserves_independent_segments`, `continuation_replays_only_previous_emitted_latents`, `forced_context_does_not_trigger_eos_or_emit_audio`, `warmup_is_excluded_from_sample_clock`, `repeated_boundary_words_emit_once`, and `continuation_cancel_preserves_voice`. Assert the default limit is 50, continuation default is 25, and explicit limits override either. Limits remain soft, including decimal-glued segments.

For each continued segment, clone pristine voice state with private mutable key/value buffers. Prepare previous plus current text together. Prefill FlowLM with that text and BOS followed by all previous emitted latents except the last. Start new generation with the last previous latent. Ignore all prefill outputs/EOS, avoid sampling during prefill, and preserve RNG behavior. Add a backbone-only prefill helper in `models/flow_lm.rs` rather than running the sampler for replayed frames. Derive EOS tail length only from current text; apply the six-new-frame EOS guard to new frames, and retain only latents actually decoded/emitted before cutoff.

Warm a fresh Mimi state with the final min(12, previous-frame-count) latents, denormalized/quantized exactly as for emission. Drop this PCM, advance decoder state/positions correctly, then decode new frames without the fresh-state fade. Upstream's previous latents are normalized FlowLM outputs; do not feed already denormalized/quantized values back into FlowLM.

Build attention mapping from the actual combined tokenization, not by adding separately tokenized prefix lengths. The text-key offset remains after the original voice cache. Normalize over all combined text keys; give previous-text tokens zero mappings to new output words and separately retain their attention mass. Preserve synthetic punctuation and ambiguous-boundary rules. Capture only one-query newly generated frames; do not send multi-query prefill through the current capture method. Reset per-segment alignment, close the previous open word, and keep one global emitted-sample counter.

The current `WordAlignment::process_frame` opens a word on the first voiced frame regardless of attention. Zero-mapping historical words alone therefore does not prevent an early new-word event. Add an internal `context_dominates: bool` input to that method, default false for independent segments, and suppress opening/advancing words while it is true; ordinary silence/finalization still closes words. Initially set it when historical text attention mass is greater than or equal to current-text mass. Add synthetic-score tests for both dominance directions and repeated boundary words. This is a proposed local alignment rule, not upstream behavior: validate its start-time/coverage effects on held-out continuation speech before accepting it. If it fails, revise the rule and evidence in this plan before exposing continuation timestamps.

Compare these tensor/state transitions with Task 1's target Python fixtures. Then require bit-identical timestamp-on/off PCM within the same Rust backend under fixed noise/temperature zero, paired monotonic events with unique global source indices, no emitted forced audio, and no elapsed warmup time. Test empty/final event-only batches, forced-prefix EOS, streams exceeding three chunks, errors after a prefix, and cancellation before/after warmup. Rerun head evaluation for continuation: the changed context may require different calibrated head selection or a mode-specific validation gate. Do not ship continuation timestamps as validated solely because independent mode passed.

Run `cargo test --release -p pocket-tts --test upstream_continuation --locked`, then the explicit model-backed continuation and timestamp cases with `-- --ignored --nocapture`. Expected: matching target prefill/latent/decoder behavior, no duplicate words/audio, unchanged independent-mode behavior from the already-updated pipeline. Commit core and timestamp support together before exposing the option to users.

## Task 6: Integrate options and verify playback

**Files:** Modify `crates/pocket-tts/src/wasm.rs`, `crates/pocket-tts-cli/src/commands/{generate,playback}.rs`, `server/handlers.rs`, `crates/pocket-tts-bindings/src/lib.rs`, `web/src/workers/{wasm-tts-protocol,wasm-tts.worker}.ts`, `web/src/hooks/use-tts-stream.ts`, `scripts/test-wasm-timestamps.mjs`, relevant tests, README and generation/API docs.

**Interfaces:** CLI exposes `--teacher-force-previous-chunk` and optional `--max-tokens`. Add optional fields with the same snake_case names to applicable HTTP/Python options and browser worker start requests. Add WASM methods `start_stream_with_options(text, max_tokens, teacher_force_previous_chunk)` and `start_stream_with_timestamps_and_options(text, max_tokens, teacher_force_previous_chunk)` while retaining existing exports. Word event/batch formats do not change. HTTP remains its existing audio transport; this work does not invent a timestamp HTTP format.

First test old callers with omitted options, option propagation in ordinary/timestamp modes, explicit limits, invalid types and cancellation. Route continuation through both streaming and collected output paths. Preserve explicit-pause behavior: the long-stream path must emit its existing silence and clear continuation history at an explicit pause; timestamp streams continue stripping markers as documented. Add a regression for this boundary so a flag cannot silently bypass inserted silence.

Run release library, CLI and binding tests and `cargo check --release -p pocket-tts --target wasm32-unknown-unknown --features wasm --locked`. Build browser assets and a new WASM package outside the existing `pkg` directory. Run `node scripts/test-wasm-timestamps.mjs <new-pkg-directory> <absolute-fixture-manifest>` for the refreshed default and each supported small model, extended to cover continuation and changed text splitting. Run the repository's browser component/worklet tests using its existing test runner, plus `npm run build` in `crates/pocket-tts-cli/web`.

Use real Chrome for model download, cached reload, voice selection, generation, word tracking and cancellation. Test underflow: hardware/worklet silence must not advance the text clock. Use native `generate --play --teacher-force-previous-chunk` on a multi-chunk numeric/quoted prompt; compare with `--no-highlight` and saved PCM using the same generation settings. Record time to first audio and sustained throughput for continuation off/on; distinguish expected per-chunk prefill cost from regressions in the default path. Compilation alone is not runtime acceptance.

Run final `cargo test --release -p pocket-tts --locked`, `cargo test --release -p pocket-tts-cli --locked`, and `cargo test --release -p pocket-tts-bindings --locked`, resolving documented platform/environment needs rather than counting skipped model tests as passes. Repeat tests only after relevant changes or failures. Stage explicit changed paths, inspect the diff, commit the integration and validation record. Prepare local versioned test artifacts. The user will test WASM and CLI before public release; do not publish or push release tags in this implementation turn.

## Task 7: Synchronize release identity

**Files:** Modify root `Cargo.toml`, `Cargo.lock`, internal path dependency versions in both dependent crates, release packaging validation and its tests, and release documentation.

**Interfaces:** Release tag `v4.0.1` must match workspace/package version `4.0.1`, internal publishable dependency requirements, CLI `--version`, and WASM package metadata. Add a release guard that fails before packaging or publishing if the tag and Cargo versions differ. The upstream source SHA remains separately recorded; it is not inferred from the package version.

Write a packaging mismatch regression first, confirm rejection is missing, then implement the validation. Test matching and mismatching tags, malformed versions, and inherited workspace package versions. Build CLI/WASM artifacts and inspect their reported versions. Run `cargo package --list` / package metadata checks as applicable; do not claim crates.io publication or change unrelated packages.

After all implementation and final review checks pass, commit intended paths and create a new annotated local `v4.0.1` tag at that commit. Refuse to overwrite an existing tag. Leave public publishing for after the user's runtime tests. Record the exact commit, tag, artifact paths and validation in the handoff.

## Decision Log

2026-11-09: Keep Candle and shared audio/timestamp generation. The upstream xn CLI is a separate implementation and would discard established local integration if adopted wholesale.

2026-11-09: Treat `9a1b489` as the latest selective import, not a full parity claim. Include older prompt/fade/config gaps that affect the requested update; retain the custom split-bundle loader gap as an explicit separate follow-up.

2026-11-09: Keep continuation opt-in and protect existing public methods with default wrappers. Update defaults and same-name weight pins only with matching presets and validated timestamp behavior.

2026-11-09: Complete implementation in this dependency order. Subagents are an execution choice governed by the selected skill and user authorization, not part of this audit's work.

## Surprises & Discoveries

The small English models were replaced in place one day after the last port; the name alone cannot identify the calibration target. Existing preset resolution duplicates a two-revision switch in Rust and TypeScript.

The old fixture generator is hard-pinned to v3.3 while the v4 archive references scripts/assets in a sibling checkout. A portable new oracle is needed before further parity claims.

The README says drifting is absent, but the small-model import added the sampler implementation; what remains absent is the older named drifting configuration. The upstream commit message's title list also differs from the final source: implement actual source behavior.

## Idempotence and Recovery

Keep old fixture/config/asset manifests immutable; generate into new directories. Update bundles independently and retain recoverable old identities before promotion. Preserve unrelated working changes, especially the modified WASM binary. Use explicit-path commits after each milestone; revert only the relevant milestone if a later gate fails. No reset, clean, force push, deployment or release publication is required by this plan.

## Outcomes & Retrospective

Planning completed on 9 November 2026; implementation authorized later the same day, including release-version synchronization. Source history and implementation paths were inspected; no new runtime behavior, tests, calibration or audio-quality results have been produced. Implementation remains pending. Complete this section with per-model acceptance evidence, timing measurements and any retained exclusions when the work is executed.
