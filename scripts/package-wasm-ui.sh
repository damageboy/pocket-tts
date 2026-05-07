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
WEB_DIST="crates/pocket-tts-cli/web/dist"
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

    require_file "${WEB_DIST}/index.html"
    require_file "${PKG_SRC}/pocket_tts.js"
    require_file "${PKG_SRC}/pocket_tts_bg.wasm"

    rm -rf "$out_dir"
    mkdir -p "$out_dir"

    # Match `pocket-tts serve --ui wasm-experimental`: serve the built CLI React UI
    # and inject the same bootstrap mode the Axum server injects at runtime.
    cp -R "${WEB_DIST}/." "$out_dir/"
    python3 - "$out_dir/index.html" <<'PY'
from pathlib import Path
import json
import sys

path = Path(sys.argv[1])
html = path.read_text()
bootstrap = {
    "ui_mode": "wasm-experimental",
    "api_base": "",
    "wasm_base": "/wasm/pkg",
}
script = f"<script>window.__POCKET_TTS_BOOTSTRAP__ = {json.dumps(bootstrap)};</script>"
if "window.__POCKET_TTS_BOOTSTRAP__" not in html:
    if "</head>" in html:
        html = html.replace("</head>", f"{script}</head>", 1)
    else:
        html = f"{script}{html}"
path.write_text(html)
PY

    mkdir -p "${out_dir}/wasm/pkg"
    cp "${PKG_SRC}"/* "${out_dir}/wasm/pkg/"
    printf '%s\n' "$CNAME_VALUE" > "${out_dir}/CNAME"
    touch "${out_dir}/.nojekyll"
}

if [[ "$CHECK_MODE" -eq 1 ]]; then
    TMP_DIR="$(mktemp -d)"
    trap 'rm -rf "$TMP_DIR"' EXIT
    package_site "${TMP_DIR}/site"
    require_file "${TMP_DIR}/site/index.html"
    require_file "${TMP_DIR}/site/wasm/pkg/pocket_tts.js"
    require_file "${TMP_DIR}/site/wasm/pkg/pocket_tts_bg.wasm"
    require_file "${TMP_DIR}/site/CNAME"
    grep -q "window.__POCKET_TTS_BOOTSTRAP__" "${TMP_DIR}/site/index.html"
    grep -q '"ui_mode": "wasm-experimental"' "${TMP_DIR}/site/index.html"
    grep -q '"wasm_base": "/wasm/pkg"' "${TMP_DIR}/site/index.html"
    grep -qx "$CNAME_VALUE" "${TMP_DIR}/site/CNAME"
    echo "WASM UI package check passed."
    exit 0
fi

package_site "$OUT_DIR"

echo "WASM UI site packaged in ${OUT_DIR}:"
find "$OUT_DIR" -maxdepth 3 -type f | sort | sed 's#^#  - #'