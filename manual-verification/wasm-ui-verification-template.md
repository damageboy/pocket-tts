# WASM UI Manual Verification Template

Date:
Tester:
Commit/Branch:
Hardware:
OS/Browser:

## Setup
- Build web assets: `cd crates/pocket-tts-cli/web && bun run build`
- Build WASM assets: `./scripts/build-wasm.sh` or `./scripts/build-wasm.ps1`
- Standard UI command: `cargo run --release -p pocket-tts-cli -- serve`
- WASM UI command: `cargo run --release -p pocket-tts-cli -- serve --ui wasm-experimental --port 8080`

## Standard UI Checks (`http://localhost:8000`)
- [ ] App loads and renders main controls
- [ ] Generate -> playback works
- [ ] Stop works during buffering and during playback
- [ ] Buffer status transitions are sensible (`buffering -> playing -> finished`)
- [ ] Preset voice switch works (e.g. `alba` -> `marius`)
- [ ] Voice clone WAV upload works
- [ ] Download WAV produces valid output file

## WASM UI Checks (`http://localhost:8080`)
- [ ] WASM initialization stages are visible and ordered
- [ ] HF repo/token inputs are visible
- [ ] Manual override section is collapsed by default
- [ ] Preset voice load works
- [ ] Switch presets and generate again: the next clip uses the new voice
- [ ] Select a preset after uploading a voice: both upload fields clear and Active voice names the preset
- [ ] An invalid uploaded embedding shows a visible error; selecting a valid preset allows generation again
- [ ] WAV cloning works
- [ ] Safetensors embedding upload works
- [ ] Generate -> playback works
- [ ] Stop works during buffering and during playback
- [ ] Download WAV produces valid output file

## Timestamped WASM Playback
- [ ] Initialize English (latest) with the updated WASM package and select Alba
- [ ] Enter `The quick brown fox jumps over the lazy dog.` and click **Generate + highlight words**
- [ ] The read-along highlights each word as it plays, not when generation finishes
- [ ] Hover a word to inspect its start/end seconds; compare the audible boundary (alignment is approximate)
- [ ] Stop clears the active highlight and freezes the playback clock; a new run resets both
- [ ] Stop remains available while queued audio plays after generation finishes
- [ ] Try repeated words, punctuation, longer text, and a slow device: skipped words must not shift later highlights and buffering must not advance the clock
- [ ] Generate Audio still works without the read-along; an uncalibrated checkpoint reports a timestamp error without disabling ordinary generation
- [ ] At a narrow viewport, the transcript wraps and both generation buttons remain usable

## HF Loading UX Cases
- [ ] Missing token for gated repo shows clear actionable error
- [ ] Invalid token shows clear actionable error
- [ ] Retry after failure works without page refresh
- [ ] Successful init reports source (`local`, `hf`, or `manual`)

## Notes / Bugs
- Issue:
  - Steps:
  - Expected:
  - Actual:
  - Severity:

## Result
- [ ] PASS
- [ ] FAIL
