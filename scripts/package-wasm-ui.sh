#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

CHECK_MODE=0
if [[ "${1:-}" == "--check" ]]; then
	CHECK_MODE=1
	shift
fi

OUT_DIR="${1:-dist/wasm-ui}"
CNAME_VALUE="${POCKET_TTS_WASM_CNAME:-pocket-tts.houmus.org}"
INDEX_SRC="crates/pocket-tts/examples/wasm/index.html"
PKG_SRC="crates/pocket-tts/pkg"

require_file() {
	local path="$1"
	if [[ ! -f "$path" ]]; then
		echo "Error: required file not found: $path" >&2
		exit 1
	fi
}

package_site() {
	local out_dir="$1"

	require_file "$INDEX_SRC"
	require_file "${PKG_SRC}/pocket_tts.js"
	require_file "${PKG_SRC}/pocket_tts_bg.wasm"

	rm -rf "$out_dir"
	mkdir -p "${out_dir}/pkg"

	cp "$INDEX_SRC" "${out_dir}/index.html"
	python3 - "$out_dir/index.html" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = "../../pkg/pocket_tts.js"
new = "./pkg/pocket_tts.js"
if old not in text:
    raise SystemExit(f"expected wasm import path not found: {old}")
path.write_text(text.replace(old, new))
PY

	cp "${PKG_SRC}"/* "${out_dir}/pkg/"
	printf '%s\n' "$CNAME_VALUE" >"${out_dir}/CNAME"
	touch "${out_dir}/.nojekyll"
}

if [[ "$CHECK_MODE" -eq 1 ]]; then
	TMP_DIR="$(mktemp -d)"
	trap 'rm -rf "$TMP_DIR"' EXIT
	package_site "${TMP_DIR}/site"
	require_file "${TMP_DIR}/site/index.html"
	require_file "${TMP_DIR}/site/pkg/pocket_tts.js"
	require_file "${TMP_DIR}/site/pkg/pocket_tts_bg.wasm"
	require_file "${TMP_DIR}/site/CNAME"
	grep -q "./pkg/pocket_tts.js" "${TMP_DIR}/site/index.html"
	grep -qx "$CNAME_VALUE" "${TMP_DIR}/site/CNAME"
	echo "WASM UI package check passed."
	exit 0
fi

package_site "$OUT_DIR"

echo "WASM UI site packaged in ${OUT_DIR}:"
find "$OUT_DIR" -maxdepth 2 -type f | sort | sed 's#^#  - #'
