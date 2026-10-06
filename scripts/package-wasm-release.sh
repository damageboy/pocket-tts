#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

TAG="${1:-}"
OUT_DIR="${2:-dist/wasm-release}"
if [[ ! "$TAG" =~ ^v[0-9][a-zA-Z0-9._+-]*$ ]]; then
    echo "Error: expected a release tag such as v3.3.0." >&2
    exit 1
fi

PKG_SRC="crates/pocket-tts/pkg"
FILES=(pocket_tts.js pocket_tts_bg.wasm pocket_tts.d.ts pocket_tts_bg.wasm.d.ts models.json)
for file in "${FILES[@]}"; do
    if [[ ! -s "${PKG_SRC}/${file}" ]]; then
        echo "Error: required file missing or empty: ${PKG_SRC}/${file}" >&2
        exit 1
    fi
done

ARCHIVE="pocket-tts-${TAG}-wasm-web.tar.gz"
mkdir -p "$OUT_DIR"
# Avoid macOS AppleDouble metadata files in otherwise portable archives.
COPYFILE_DISABLE=1 tar -czf "${OUT_DIR}/${ARCHIVE}" -C "$PKG_SRC" "${FILES[@]}"
(
    cd "$OUT_DIR"
    shasum -a 256 "$ARCHIVE" > "${ARCHIVE}.sha256"
)
echo "WASM release packaged in ${OUT_DIR}/${ARCHIVE}"
