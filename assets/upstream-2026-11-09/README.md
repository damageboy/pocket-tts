# Pinned upstream sync fixtures

Model weights remain external at the immutable URI/revision/SHA-256 recorded in each model manifest. Missing or mismatched assets are fatal.

The schema-2 `fixture_config_path` (or `config_path`) must identify a file covered by that manifest's `files` SHA-256 map. The selected YAML must equal `manifest.config` after removing only the documented Rust calibration field `timestamp_heads`. Its digest authenticates that calibration too. Validation occurs before the timestamp test overrides temperature. Legacy schema-1 calibration restoration keeps its existing behavior.

WASM fixture validation requires Python with PyYAML. Select the pinned reference interpreter explicitly:

```sh
export POCKET_TTS_REFERENCE_PYTHON=/private/tmp/pocket-tts-reference-venv/bin/python
node scripts/test-wasm-timestamps.mjs --validate-fixture assets/upstream-2026-11-09/baseline/english/manifest.json
node scripts/test-wasm-timestamps.mjs PATH_TO_NEW_WASM_PACKAGE assets/upstream-2026-11-09/baseline/english/manifest.json
```

`POCKET_TTS_REFERENCE_PYTHON` defaults to `python3`. The script reports a clear error when that interpreter or PyYAML is unavailable. The recorded `requirements.lock.txt` includes PyYAML; a fresh reference environment can install that file without model inference.

Focused identity regressions use tiny temporary files and require no models:

```sh
POCKET_TTS_REFERENCE_PYTHON=/private/tmp/pocket-tts-reference-venv/bin/python node --test scripts/test-wasm-timestamps.test.mjs
CARGO_TARGET_DIR=/private/tmp/pocket-tts-build cargo test --release -p pocket-tts --test fixture_integrity --locked
```

See `manifest.json` for source revisions, regeneration commands, model families and staged continuation limits.
