# Pocket TTS upstream v3 resync

## Purpose and approved scope

Bring the Rust/Candle inference port to released Python v3.3.0, then evaluate
the unreleased October changes separately. Preserve the port's Rust, CLI, HTTP,
Python-binding and WASM interfaces, streaming behavior and CPU optimizations
where they remain equivalent. Do not port training or PyTorch checkpoint loading.

The user approved the staged plan in
https://ampcode.com/threads/T-01a10d07-1759-73bb-9cb1-563ea1be8b8f.
This living execution record follows `.agent/PLANS.md`.

## Targets and context

The released reference is upstream kyutai-labs/pocket-tts v3.3.0 at
`3dbee45d343d7dddd0d105468d17f8dcba14db3e`. The separately considered main
snapshot is `41cbc84af539ea78a804ffca5f9c6edc1a22ce44`.

Rust starts at `65b03f6bf5d8fd1c900a570210b1b7c5bfa43b76`. Existing changes to
Cargo.toml and Cargo.lock point to a local Candle checkout and belong to the
user; preserve them. The checked-in `python-reference` is version 1.0.1, not
an oracle for the new models. Use an isolated, commit-pinned Python environment
instead of importing that directory or silently using the latest package.

Model identity includes the configuration, weight revision, tokenizer revision,
and preset-state revision. Presets contain transformer attention caches, not
portable speaker vectors. Never mix bundles or update floating Hub revisions.

## Progress

- [x] Investigated current code, upstream release history and numerical contracts.
- [x] User approved implementation.
- [x] Capture baseline and reproducible Python fixtures; regeneration is byte-identical.
- [x] Implement v3 config, native/byte tokenizers, open fallback and cloning guards.
- [x] Verify numerical operators and both samplers at one and three steps.
- [x] Port text preparation, token-based chunking, generation bounds and EOS rules.
- [x] Verify voice-cache isolation, terminal stream errors, zero-tail EOS and browser cancellation.
- [x] Rebuild WASM and execute real browser generation using pinned cached English assets.
- [x] Compare 32 teacher-forced frames and uneven decoder partitions for English, French, Dutch and French 24-layer.
- [x] Run Python bindings and HTTP generation, disconnect/reuse and asset-route checks.
- [x] Run final core/CLI release suites, explicit native/byte German parity, and normal release build.
- [x] Inspect short speech samples in all seven languages and a multi-chunk English comparison with Python.
- [x] Finish all 18 model comparisons, each with 32 teacher-forced frames and unequal decoder partitions; also pass full-model three-step flow matching.
- [ ] Complete current-model audio-cloning acceptance (blocked: current cloning bundles unavailable; cached legacy English bundles do not cover this gate).
- [ ] Complete release-quality/performance qualification: multiple prompts/voices, gated prompts and controlled regression benchmarks.
- [x] Evaluate main-only changes; retain them as separately gated follow-ups below.

## Milestones and acceptance

### 1. Reproducible reference

Install the exact released checkout in an isolated Python 3.12 environment with
pytest and PyYAML. Keep generation tools under `scripts/`, outside the upstream
tree. Store compact deterministic fixtures under `assets/` and heavyweight local
artifacts under ignored `.amp/in/`. Capture source revisions, asset hashes,
runtime versions and parameters in each fixture manifest. Fixture production
must fail when the reference revision or required assets are wrong.

Start with text/tokenizer and operator fixtures that need no model weights.
Then compare full tensors through imported voice state, text conditioning,
transformer, sampler, decoder and multi-step generation. Inject identical noise
instead of assuming Rust and Torch RNG seeds are equivalent. Required acceptance
tests must fail on missing assets, not return successful skips.

### 2. Configurations and loading

Update `crates/pocket-tts/src/config.rs`, `conditioners/text.rs`, `weights.rs`
and `config/*.yaml`. Test JSON loading through both paths, vocabulary mismatch,
sampler rejection, temperature overrides and the shipped config matrix before
implementation. Use official tokenizers with exact token-ID comparisons. Verify
fallback to open weights and distinguish cloning capability explicitly.

### 3. Numerical core

In `models/flow_lm.rs` and `modules/mlp.rs`, support LSD with two time conditions
and flow matching with one. Verify Candle activation and normalization semantics
against Python before changing them. Compare cached modulation with direct
evaluation. Test nonzero asymmetric inputs and multiple solver steps. Compare
Mimi on identical latent inputs, including the changed French codec. Fix the
first divergent stage rather than widening downstream tolerances.

### 4. Text and generation

Port upstream preparation order, per-model character translation, empty-text
errors, capitalization and terminal punctuation. Split token sequences at
sentence boundaries, preserving decimal points, and fall back to clause
boundaries. Fifty tokens remains a soft target; never truncate words.

Use `ceil((token_count / 3 + 2) * frame_rate)` for the generation bound. Ignore
EOS before step 6. Resolve tails as explicit override, model recommendation,
then the text heuristic (5 frames through four words, otherwise 3). Stop before
emitting the cutoff frame. Tests must distinguish steps 5/6, tails 0/1/3, four
and five words, and identical word counts with different token counts.

### 5. Stateful audio

Verify copied voice state and fresh Mimi state per text chunk. Serial, parallel,
owned and borrowed streams must agree given identical noise. Propagate errors;
disconnect/drop must stop work. Compare per-frame and partitioned latent decode
before adding batching. Never delay the first latent to fill a batch. Validate
cloning with mono 24 kHz first, then test resampling and channel handling
separately. Rubato and SciPy output need not be sample-identical.

### 6. Integrations

Update `crates/pocket-tts-cli/src/voice.rs`, command/server catalogs,
`crates/pocket-tts-bindings/src/lib.rs`, and the browser worker. Replace independent
browser config synthesis with canonical configurations or a generated view.
Preserve existing option names as aliases where upstream names changed. Test
CLI generation, HTTP disconnects, Python bindings and actual WASM execution.

### 7. Acceptance and follow-ups

Compare output with the same Python model, not only old Rust. Record first-audio
latency, generation time/audio duration, loading time and peak memory. Listen
for repeats, omissions, clipped endings, incorrect first words and chunk seams.
Run every advertised model variant; use deeper fixtures for each architecture
and tokenizer family. Keep model-quality changes separate from port regressions.

Only after the released checkpoint passes, evaluate main's end-on-pause voice
preprocessing with recomputed presets, 5 ms output fade, October multilingual
weights, and zero-time-condition drifting head as independent changes.

## Verification commands

Run from the repository root. All inference and performance checks use release:

    cargo test --release -p pocket-tts --lib --locked
    cargo test --release -p pocket-tts --locked
    cargo test --release -p pocket-tts-cli --locked
    cargo check --release -p pocket-tts-bindings --locked
    cargo fmt --all -- --check
    cargo clippy --release -p pocket-tts -p pocket-tts-cli --all-targets -- -D warnings
    bash scripts/build-wasm.sh

On this checkout, running combined test targets with `panic = "abort"` caused
Cargo output collisions and abort/unwind dependency errors. The final test runs
used `CARGO_PROFILE_RELEASE_PANIC=unwind` and separate package commands; the
ordinary `cargo build --release -p pocket-tts-cli --locked` also passed without
that override. The model-backed native/byte comparison is an explicit check:

    CARGO_PROFILE_RELEASE_PANIC=unwind cargo test --release -p pocket-tts --test wasm_path_parity --locked -- --ignored

Build the web UI with `bun install --frozen-lockfile` and `bun run build` from
`crates/pocket-tts-cli/web` before CLI checks that embed it. Model-backed checks
require the matching cached assets and gated access for cloning. Do not equate
network skips in the historical tests with numerical acceptance.

The isolated reference was installed in `.amp/in/reference-venv` from the clean
checkout `.amp/in/upstream-v3.3.0`. Its manifest records Python 3.12.8, Torch
2.14.1, tokenizers 0.23.2, NumPy 2.5.3 and safetensors 0.8.0. Native verification
used rustc 1.98.1 on macOS arm64 and the clean local Candle checkout at
[`6e0195c`](https://github.com/huggingface/candle/commit/6e0195c801ae2be90f856b807de4e00090bfd969).
Reproduce the small fixtures and compare them without overwriting expectations:

    .amp/in/reference-venv/bin/python -m pytest scripts/test_upstream_reference.py -q
    .amp/in/reference-venv/bin/python scripts/generate_upstream_reference.py --reference .amp/in/upstream-v3.3.0 --output .amp/in/artifacts/reproduced-fixtures
    diff -rq assets/upstream-v3.3 .amp/in/artifacts/reproduced-fixtures

Generate heavyweight model references separately (change language and voice
together for each configured bundle); the test fails if its manifest/assets are
missing. Setting `HF_HUB_DISABLE_XET=1` recovered a failing CDN download here.

    HF_HUB_DISABLE_XET=1 .amp/in/reference-venv/bin/python scripts/generate_model_reference.py --reference .amp/in/upstream-v3.3.0 --output .amp/in/artifacts/english-v3.3 --language english --voice alba
    POCKET_TTS_REFERENCE="$PWD/.amp/in/artifacts/english-v3.3" cargo test --release -p pocket-tts --test upstream_model --locked -- --ignored --nocapture

The WASM dependency and binding-generator versions must agree. This checkout's
lockfile uses 0.2.127, while the system tool was 0.2.121. Install the matching
generator under `.amp/in/tools` and prepend that directory to PATH; do not alter
the user's lockfile to accommodate an old global tool. On this macOS host, the
Python extension requires dynamic Python symbol lookup at link time:

    PYO3_PYTHON="$PWD/.amp/in/reference-venv/bin/python" cargo rustc --release -p pocket-tts-bindings --lib --locked -- -C link-arg=-undefined -C link-arg=dynamic_lookup

## Surprises and discoveries

The old parity suite accepts any English load failure as a skip. The streaming
loop emits a frame Python discards at EOS. Generation length uses words rather
than tokens. Native JSON tokenizers already work, but byte loading lacks the
native vocabulary-size validation. Avoid reimplementing existing support.

The first English free-running comparison exceeded 0.0005 latent absolute error
at frame 5 (0.00062537). Feeding each implementation the same preceding Python
latent reduced that frame's error to 0.00006843. The acceptance test now explicitly
names teacher forcing and independently feeds Python latents to Mimi. This
isolates stage correctness; it does not assert identical free-running waveforms.
The original failing log is retained at `.amp/in/artifacts/model-parity.log`.

Candle's `Tensor::clone` shares storage and `slice_set` mutates it. The old
streaming clone therefore changed audio-derived voice caches with spare capacity.
A sentinel-buffer regression failed before copying mutable KV buffers per
segment and passed afterwards. A separate failure showed `flat_map` resumed at
the next text segment after an error; all public streaming paths now stop after
their first error, including the pause-aware path. Evidence is in
`.amp/in/artifacts/state-tests-{red,green}.log`.

The full historical suite can pass while mixing new weights with old voice
states, and some tests silently skip unavailable assets. `v2_parity_test` now
uses frozen original YAML fixtures, while required v3 comparisons fail on missing
assets. No old successful skip counts as released-model acceptance.

Strict Clippy currently fails on pre-existing `chunks_exact(2)` calls in
`src/audio.rs` under Rust 1.98. Those unrelated lines remain untouched. Ordinary
Clippy completed with only those two warnings. Web build and seven Bun tests
passed, but full ESLint still reports nine existing errors (audio-worklet
`var`/`any` and mixed component/non-component exports). These are not a clean
lint result and were not suppressed.

Real HTTP startup exposed Axum's obsolete `/wasm/pkg/*path` route syntax. The
minimal `{*path}` correction restored startup; health, both catalogs, JS asset
serving and generation were then exercised. After closing a 100-sentence stream
after its first PCM frame, a follow-up generation finished in 1.191 seconds and
matched its pre-disconnect zero-temperature WAV byte for byte.

The old `wasm_path_parity` test also compared today's German native load with
hardcoded v2 weights/tokenizer and a synthesized config. It now uses canonical
YAML and matching released assets, compares every PCM sample at zero temperature,
and propagates errors instead of filtering them out. It is explicitly opt-in
because it requires model assets; its invoked run passed and missing assets
are fatal, unlike the previous successful network skip.

## Main-only follow-ups remain separate

Both the released and main snapshots call themselves version 3.3.0; the source
commit, not the package version, identifies the reference. The comparison is
https://github.com/kyutai-labs/pocket-tts/compare/3dbee45d343d7dddd0d105468d17f8dcba14db3e...41cbc84af539ea78a804ffca5f9c6edc1a22ce44.

First port end-on-pause from `pocket_tts/data/audio_utils.py`: complete 20 ms
frames, relative RMS threshold strictly above peak minus 35 dB, trimming after
the last active frame, 20 ms fade-out and 80 ms zero padding. Apply after
truncation and resampling, before Mimi. Imported states bypass preprocessing and
must be regenerated for the selected weights; changing the encoder path alone
does not fix presets. Test exact thresholds, partial frames, all-zero audio
longer and shorter than one frame, source immutability and resulting cache offsets.

Gate the output fade separately: the final main snapshot fades in only the first
120 samples at 24 kHz (5 ms) per freshly initialized decoder/text chunk. It is
not a fade on every HTTP/WASM packet and not the 20 ms prompt fade-out. Test the
exact ramp, unchanged remainder, unchanged second output, sample counts and
equivalence across decoder batching.

The October refresh changes only weight pins for Dutch, French, French 24-layer,
German, Italian, Italian 24-layer, Portuguese and Spanish. English and the other
24-layer models are not refreshed. Pair gated revision
`3e82814a68665eec246ff649b14c71331f955c06` or open revision
`1e08e6a23401048648a9fdcfde2f89348215c2a7` with the matching new presets; tokenizer
pins remain unchanged. Require strict loading, preset/latent/EOS parity and
long audio-prompt (>12 seconds) premature-ending regressions.

Add `english_drifting_26-09` last as an optional model, not the English default.
Its head has no time embeddings and returns `head(conditioning, noise)` directly;
it does not add that result to noise or integrate Euler steps. Verify no
`time_embed.*` weight requirements, identical output for different positive
decode-step counts, supplied-noise parity and unchanged LSD/flow-matching tests.

## Decision log

- 2026-10-05: Released v3.3 is the first acceptance target; main changes are
  separate because their model and preprocessing effects need independent checks.
- 2026-10-05: Preserve the user's dependency edits; work in the current checkout
  without moving or stashing their changes.
- 2026-10-05: Reference tools live outside the old vendored Python package to
  prevent accidentally generating v1 fixtures for a v3 acceptance test.
- 2026-10-05: Retain stage-local tensor tolerances rather than widening them to
  absorb autoregressive amplification. Track free-running speech quality separately.
- 2026-10-05: Copy only mutable KV buffers at segment creation; scalar state is
  replaced rather than mutated. Keep fresh Mimi state and lazy stream cancellation.
- 2026-10-05: Do not advance main-only weights/preprocessing before released
  acceptance, especially while gated cloning remains unverified.

## Recovery and delivery

Use additive fixtures and pinned assets. Do not overwrite old fixtures until
new fixtures have reproducible provenance. Keep previous model configurations
available by explicit name and keep each verified milestone reviewable. Never
stage the user's dependency edits. Pushing, publishing and deploying remain
outside the authorization for local implementation.

The reference-fixture, core-inference and CLI/HTTP/WASM milestones are committed
locally, followed by this documentation/validation record. Cargo.toml and
Cargo.lock remain the user's uncommitted local-Candle changes. All checks here
used that local Candle revision, not the published dependency version in HEAD.
The CodeTour and heavyweight verification artifacts remain local and ignored.

## Outcomes and retrospective

Local implementation is verified through CLI, HTTP, Python and actual WASM;
all 18 released model configurations passed numerical comparison. Full release
qualification remains incomplete. Nothing is pushed, published or deployed.
Small fixtures regenerate identically. Model comparisons cover prefill caches,
32 supplied-noise teacher-forced latent frames, EOS decisions and per-frame/
partitioned decoder output. Worst absolute errors across the matrix were
0.00001503 for prefill, 0.00012363 for latents and 0.00000266 for PCM, below the
unchanged tolerances of 0.0005 (prefill/latents) and 0.002 (PCM). A separate
three-step full-model flow-matching run also passed. Logs are
`.amp/in/artifacts/model-matrix-parity.log` and `fm3-parity.log`.

Sixteen model references used public preset-only bundles. January and April
six-layer English used available legacy cloning bundles, but the matrix exercised
their preset states, not audio-prompt encoding. Do not count those cases as
current-model cloning acceptance; that still requires access to the corresponding
current gated bundles and explicit encoder/prompt tests.

The English native smoke produced the exact requested sentence without detected artifacts:
6.52 seconds of audio in 1.91 seconds of inference, with approximately 658 MiB
peak resident memory. This is one run, not a performance regression benchmark.

Real Chrome/WASM English generation reported TTFC 367 ms, TTFA 446 ms and zero
rebuffers. A direct worker check produced 63,360 finite non-silent samples,
rejected WAV cloning for the public bundle, and acknowledged cancellation after
six chunks (the worker's scheduling yield interval). English assets were seeded
into browser Cache Storage from the exact reference files to avoid CDN variability;
neither inference nor the worker was mocked. The inspected image is
`.amp/in/artifacts/upstream-v3-browser-generated.png`.

All seven language-default native samples were transcribed as expected, without
detected repetitions, clipped endings or glitches. This is a short-prompt smoke,
not a comprehensive pronunciation evaluation. An English multi-chunk
comparison at temperature zero produced 20.48 seconds in both implementations,
with consistent speakers and complete endings. Initial independent inspections
disagreed on the decimal; a paired playback heard both say "three forty" for
`3.14`. Treat number pronunciation as a shared quality caveat, not a demonstrated
Rust-only regression. The two free-running waveforms are not sample-identical.

During concurrent verification, that passage took 2.64 seconds to generate in
Python and 6.14 seconds through the Rust binding; model-plus-voice loading took
1.51 and 0.48 seconds respectively. These observations do not establish a
controlled speed comparison or regression against the old port. Preserve the
performance qualification gate rather than claiming a speed improvement.

The final core suite reports 85 passing checks including its doctest, with
asset-dependent tests explicitly ignored or historically network-skipped; the
CLI suite reports 40 passing checks (including duplicated binary/library tests).
Explicit model comparisons are reported separately and never counted as passed
because they were ignored. A seven-concern CodeTour is available under
`.amp/in/artifacts/upstream-v3-resync.tour`; all source patterns were validated.

Plan updated during implementation to record numerical-drift isolation, cache
ownership failures, recovered download/toolchain issues and the concrete boundaries
between released compatibility and unreleased main follow-ups. The final-runtime
update adds HTTP recovery, native/byte parity, multilingual and long-form evidence,
and preserves the outstanding qualification limits. The final matrix update
records 18 passing variants and distinguishes available legacy cloning bundles
from still-unverified current-model audio cloning.
