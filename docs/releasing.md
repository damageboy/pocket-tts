# Release identity and local test builds

The Rust package version and release tag use the same version: `4.0.1` and
`v4.0.1`. The Kyutai source revision is recorded separately in
[`assets/upstream-2026-11-09/manifest.json`](../assets/upstream-2026-11-09/manifest.json).
A release number does not imply that every upstream subsystem was ported.

When preparing a release, update `[workspace.package].version`, both internal
`pocket-tts` dependency requirements, and the three local package entries in
`Cargo.lock` together. Member packages inherit the workspace version. The CLI
uses Cargo's version; both WASM build scripts generate `package.json` from it.

Run the release checks before creating the tag:

```sh
python3 scripts/test_package_wasm_release.py
python3 scripts/test_package_wasm_ui.py
python3 scripts/validate-release-version.py v4.0.1
cargo run --release -p pocket-tts-cli -- --version
```

Build into a separate directory to preserve an existing packaged WASM binary:

```sh
bash scripts/build-wasm.sh dist/wasm-v4.0.1/pkg
node scripts/test-wasm-init.mjs dist/wasm-v4.0.1/pkg
python3 scripts/validate-release-version.py v4.0.1 dist/wasm-v4.0.1/pkg
bash scripts/package-wasm-release.sh v4.0.1 dist/wasm-release dist/wasm-v4.0.1/pkg
```

PowerShell accepts the same build destination through
`scripts/build-wasm.ps1 -OutputDirectory dist/wasm-v4.0.1/pkg`. Build the web UI
first if packaging the demo site, then use
`bash scripts/package-wasm-ui.sh dist/wasm-ui dist/wasm-v4.0.1/pkg`.

The packaging guard rejects mismatches between the tag, Cargo manifests,
internal dependencies, lockfile and WASM metadata. Release CI also checks the
Cargo identity before compilation. It does not publish Rust crates.

After verification, create an annotated local tag at the final commit without
overwriting an existing tag. Pushing a `v*` tag triggers the GitHub release
workflow. Keep the tag local until CLI and browser testing is accepted;
publishing to GitHub or crates.io is a separate action.
