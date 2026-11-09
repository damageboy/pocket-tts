# Timestamp bundle calibration — 2026-11-09

Only `english_7m_2026-10` is promoted. English/default, 18M and 2M retain historical weights, presets and heads because their target candidates failed the frozen gate. Their replacements remain reproducible. No heads were reselected after evaluation. Human listening and continuation acceptance remain pending.

## Protocol and evidence

`protocol.json` freezes six selection texts and six disjoint evaluation texts, Alba/Anna, exact normalization, ranking and promotion rules. There are 192 utterances: four models × old/new × two splits × six texts × two voices. Main generation uses default temperature 0.3 and one sampler step. Each case/voice has a SHA256-derived u64 seed shared old/new and reset before all-head capture, native timestamp generation and ordinary generation. All three emitted PCM vectors must match exactly. Scoped test-only RNG leaves production sampling and APIs unchanged. Cargo.lock pins rand 0.10 and rand_distr.

Rank all individual query heads, the historical pair, and all pairs among the six best calibration singles: coverage first, then pooled mean absolute error, then pooled p95. Freeze selection before heldout capture. Baselines use historical heads. Gate coverage and each start/end/pooled mean/p95 independently, without epsilon. JSON errors are seconds; tables below use milliseconds. P95 uses linear interpolation. Full-source coverage denominator includes unresolved lexical/numeric words (228 per heldout bundle). Common-source-word metrics are diagnostic only and cannot override the gate.

Each model directory contains frozen `selection.json` (all candidates), `evaluation.json` (full/per-case/common-source metrics and worst boundary traces), `adjudication.json` (review before comparison), and `utterances.json` (ASR words/transcript, selected/native events, settings and raw trace/WAV hashes/paths). The 192 original all-head traces and float32 WAVs remain in `/private/tmp/pocket-tts-calibration-defaults`; they are not duplicated in Git. Preserve that directory or regenerate it. No weights are copied into this evidence directory.

`capture-provenance.json` retains generation source hashes, Cargo.lock hash, authenticated model/tokenizer/preset hashes and ASR manifest. `analysis-provenance.json` records the later analysis script and final ruling; generation provenance is not rewritten. ASR is Systran/faster-whisper-base.en revision `3d3d5dee26484f91867d81cb899cfcf72b96be6c`, CPU int8, beam5, word timestamps, no previous-text conditioning. Package versions are in `asr-requirements.lock.txt`. `target-configs.json` records upstream fields at commit `c91f50f38c6b914d49dbdff51b8d322edf843db2`. `released-bundles.json` identifies all 27 final configs, both weight URIs, tokenizers, presets and heads.

Nine unchanged-weight timestamp bundles retain historical preset revision `4e1e0a3e611c51c0b4ed8174fc10f32a54644303`; refreshing their changed upstream states is deferred. English/default retains that preset; 18M/2M retain `a3055dcb533cf2ec90458008a01e7cef550bae2a`. Promoted 7M and audio-only bundles use target preset `a7056f505a4a8ca72d8b5fb1440071d6f1a35739`. Target text/config fields apply; only three failed replacements retain historical weight URIs. Six new English variants are audio-only.

The initial temperature-zero/three-step run at `/private/tmp/pocket-tts-calibration-run` was plumbing-only; its protocol is archived there as `protocol-plumbing-only.json`. No heldout was captured before the default-setting correction. The corpus was unchanged.

## Heldout results

| Model | Candidate heads | Coverage old → new | Start mean/p95 old → new (ms) | End mean/p95 old → new (ms) | Pooled mean/p95 old → new (ms) | Decision |
|---|---|---|---|---|---|---|
| english | [[3, 8], [3, 14]] | 228/228 → 228/228 | 109.2/373.0 → 94.8/459.0 | 111.6/300.0 → 62.5/280.0 | 110.4/325.0 → 78.7/340.0 | retain; failed start_p95, pooled_p95 |
| english_18m_2026-10 | [[5, 3]] | 221/228 → 228/228 | 104.9/580.0 → 93.2/439.0 | 60.7/220.0 → 62.1/240.0 | 82.8/320.0 → 77.6/260.0 | retain; failed end_mean, end_p95 |
| english_7m_2026-10 | [[5, 3]] | 220/228 → 227/228 | 109.8/561.0 → 95.2/498.0 | 71.7/280.0 → 67.5/240.0 | 90.8/343.0 → 81.4/260.0 | promote |
| english_2m_2026-10 | [[5, 2], [4, 2]] | 214/228 → 225/228 | 119.0/554.0 → 218.4/1176.0 | 92.1/327.0 → 196.8/1200.0 | 105.6/493.0 → 207.6/1200.0 | retain; failed start_mean, start_p95, end_mean, end_p95, pooled_mean, pooled_p95 |

| Model | ASR missing | ASR extra | ASR repeats | Timestamp missing | Common words | Common pooled mean/p95 (ms) |
|---|---|---|---|---|---|---|
| english | 0 → 0 | 2 → 2 | 0 → 0 | 0 → 0 | 228 | 110.4/325.0 → 78.7/340.0 |
| english_18m_2026-10 | 7 → 0 | 7 → 3 | 0 → 0 | 0 → 0 | 221 | 82.8/320.0 → 75.7/260.0 |
| english_7m_2026-10 | 8 → 1 | 20 → 3 | 0 → 0 | 1 → 0 | 220 | 90.8/343.0 → 78.6/260.0 |
| english_2m_2026-10 | 13 → 3 | 8 → 5 | 0 → 0 | 5 → 0 | 212 | 106.2/497.0 → 210.3/1302.0 |

## Limits and ambiguity

ASR boundaries are approximate, not human-labelled acoustic truth; generated frames are 80 ms. Two voices and one deterministic draw per case do not establish all-seed/all-voice performance. Cloning weights are pinned to upstream pairing, but measured audio uses public preset-only weights. These data validate independent segments; continuation is separate.

Exact numeric expansions and neighbours/neighbors, harbour/harbor were frozen from calibration evidence before heldout. Unresolved number speech remains unmatched. Cafe/calf/cave, the/to, Oh/All and no/know remain unmatched. Cafe/café also remains unmatched: no heldout-driven normalization was added. Adjacent extra ASR tokens equal to the preceding token count as repeats; suffix hallucinations remain extras. No boundaries are invented or holdouts dropped.

English baseline Alba multi-chunk repeats `ready`; frozen matching chooses the second occurrence (14.16–14.24 s), and the first (13.98–14.16 s) remains extra/repeated. The English tail-error failure remains. 18M's small end-error failure is retained without tolerance relaxation. 2M target Anna multi-chunk `and` (index 51) ends at 14.72 s versus ASR 16.64 s; common-word errors confirm the regression survives the coverage change. 2M Anna prose's extra `7` has ASR probability 0.001 and no voiced frames, supporting an ASR-artifact diagnosis without changing matching. Other ambiguous acoustic truth remains pending human listening.

## Reproduction

Run from repository root with authenticated fixtures in `assets/upstream-2026-11-09/{baseline,target}`. Every target retains a normative Python `config.yaml` and authenticated Rust `rust-config.yaml`, including deferred candidates. Fixture manifests and capture provenance identify cached assets. `prepare` currently expects the inventory at `/private/tmp/pocket-tts-calibration-assets` and the absolute asset paths; restore those paths when reproducing elsewhere.

Build the exporter, then set `NATIVE_TEST` to the printed release unit-test executable:

```sh
CARGO_TARGET_DIR=/private/tmp/pocket-tts-build cargo test --release --locked -p pocket-tts --lib --no-run
PY=/private/tmp/pocket-tts-calibration-venv/bin/python
OUT=/private/tmp/pocket-tts-calibration-rerun
$PY scripts/calibrate_timestamp_heads.py prepare --output "$OUT"
$PY scripts/calibrate_timestamp_heads.py capture --output "$OUT" --split calibration --native-test "$NATIVE_TEST"
$PY scripts/calibrate_timestamp_heads.py transcribe --output "$OUT" --split calibration
$PY scripts/calibrate_timestamp_heads.py rank --output "$OUT"
$PY scripts/calibrate_timestamp_heads.py capture --output "$OUT" --split evaluation --native-test "$NATIVE_TEST"
$PY scripts/calibrate_timestamp_heads.py transcribe --output "$OUT" --split evaluation
$PY scripts/calibrate_timestamp_heads.py evaluate --output "$OUT"
```

Use a fresh output directory. Add `--model english_7m_2026-10` to each capture/transcribe/rank/evaluate for one model. Existing captures cannot be overwritten and ranking refuses once heldout capture starts. `evaluate` computes metrics but leaves `promoted:false`; adjudication and event/PCM checks precede the explicit decision. Existing authoritative traces can be reevaluated without generation. Inspect lexical mismatches before deciding promotion.

Test each target candidate explicitly (substitute the other three model names as needed):

```sh
POCKET_TTS_REFERENCE="$PWD/assets/upstream-2026-11-09/target/english_7m_2026-10" CARGO_TARGET_DIR=/private/tmp/pocket-tts-build cargo test --release --locked -p pocket-tts --test upstream_model -- --ignored
POCKET_TTS_TIMESTAMP_MODEL="$PWD/assets/upstream-2026-11-09/target/english_7m_2026-10/manifest.json" CARGO_TARGET_DIR=/private/tmp/pocket-tts-build cargo test --release --locked -p pocket-tts --test timestamps_model -- --ignored
```

For listening, `utterances.json` records exact WAV paths/checksums. For new audio from a candidate YAML, pass an explicit matching preset state from capture provenance: native stock-voice inference intentionally rejects custom config paths.

Export schema_version 1 stores all layer/query-head pairs, native heads/events, sample rate/count, seed/settings, source units/global indices, segment text/end, and per-frame start/sample count/voiced flag/all-head unit probabilities. The ignored test takes a job with config/weight/tokenizer paths, voices, texts/seeds and output path. Offline replay uses float32 accumulation and production alignment branches and checks every capture against native events. PCM is the faded emitted stream; warmup/context is excluded. Hooks exist only in test builds. Whisper is offline-only.
