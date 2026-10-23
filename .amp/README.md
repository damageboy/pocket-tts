# Historical development artifacts

This archive preserves the local Pocket TTS research and verification artifacts
available on 2026-10-23. Original files are retained unchanged, including failed
runs and machine-local paths. These are historical evidence, not current test
results or a portable, self-contained reproduction environment.

The v4 implementation and verification were recorded in
[the development thread](https://ampcode.com/threads/T-01a15a23-5622-71af-9761-23b7f34a1f4d)
and released as [v4.0.0](https://github.com/damageboy/pocket-tts/releases/tag/v4.0.0).

## Preserved files

- `in/*.log`: calibration, intermediate and final parity checks, package tests,
  the failed combined workspace test run, formatting, WASM build and release CI.
- `in/small-calibration/`: 24 generated WAV clips, their NPZ calibration data and
  JSON transcriptions, plus head rankings for the 18M, 7M and 2M models.
- `in/small-reference/`: three sets of Python reference tensors, manifests and
  local configs. The `model.safetensors` files contain test tensors (embeddings,
  attention caches, noise, latents and PCM), not downloaded model weights.
- `in/small-pkg/`: the historical local WASM package and model catalog. Its catalog
  records the pre-commit source revision; this is not the published release archive.
- `v4-env-requirements.txt`: installed package versions exported from the original
  Python 3.12.8 environment with `uv pip freeze` (uv 0.6.14).

The original artifact set is 101 files totaling 24,618,701 bytes. Some manifests
and configs refer to assets and scripts in the sibling `pocket-speechify` checkout.
Those files are outside this archive; paths must be adapted to reproduce the work.
The small-model reference manifests identify upstream revision
[`9a1b489`](https://github.com/kyutai-labs/pocket-tts/commit/9a1b48922deb44cd4a805fde279eb33b71c1a473).

## Replaceable dependencies left local

No local files were deleted. These directories are intentionally not committed:

- `in/v4-env/`: installed Python environment (about 928 MB). Package versions are
  recorded above instead of vendoring executables and site-packages.
- `in/small-calibration/asr/`: Hugging Face cache for
  `Systran/faster-whisper-base.en`, revision
  `3d3d5dee26484f91867d81cb899cfcf72b96be6c` (about 141 MB).
- `in/upstream-v3.3.0/`: an unchanged nested checkout of
  [kyutai-labs/pocket-tts at `3dbee45`](https://github.com/kyutai-labs/pocket-tts/commit/3dbee45d343d7dddd0d105468d17f8dcba14db3e).
  There were no local changes to preserve.

New files under `in/` remain ignored. Add future historical artifacts explicitly
only after checking their contents for credentials, private data and caches.
