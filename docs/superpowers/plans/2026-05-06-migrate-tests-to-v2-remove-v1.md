# Migrate to v2-only: Remove v1 Code and Update Tests

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all v1-specific code paths and update tests to use the v2 production model (`english` / `english_2026-04`).

**Architecture:** The v2.1.0 port added v2 support while keeping v1 backward compatibility. Now that v2 is verified working, remove v1 cruft to simplify the codebase.

**Tech Stack:** Rust, Candle

---

### Task 1: Update Integration Tests to v2

**Files:**

- Modify: `crates/pocket-tts/tests/integration_tests.rs`

- [ ] Change `get_model()` and `get_model_with_params()` from `"b6369a24"` to `"english"`
- [ ] Update voice loading: use predefined voice name (e.g., `"alba"`) via `get_voice_state_from_prompt_file` instead of `ref.wav` audio encoding (or keep both paths tested)
- [ ] Update `test_generate_with_pauses_adds_silence` expected values for v2 generation characteristics
- [ ] Update quantized test from `"b6369a24"` to `"english"`
- [ ] Run all integration tests, fix any assertion failures
- [ ] Commit

### Task 2: Remove v1 Config and Assets

**Files:**

- Delete: `crates/pocket-tts/config/b6369a24.yaml`
- Delete: `crates/pocket-tts/assets/tokenizer.json` (v1 English tokenizer, 240KB embedded in WASM)
- Modify: `crates/pocket-tts/src/wasm.rs` — remove `include_bytes!("../assets/tokenizer.json")` fallback
- Modify: `crates/pocket-tts/src/config.rs` — remove `DEFAULT_VARIANT`

- [ ] Delete `b6369a24.yaml` (users should use `"english"` or `"english_2026-01"`)
- [ ] Delete `assets/tokenizer.json`
- [ ] In `wasm.rs`: make `tokenizer_bytes` required (error if empty instead of falling back to v1 tokenizer)
- [ ] Remove `DEFAULT_VARIANT` constant from `config::defaults` (keep `DEFAULT_LANGUAGE`)
- [ ] Update any remaining references to `b6369a24` in code/tests/docs
- [ ] Commit

### Task 3: Clean Up v1 Voice Loading Path

**Files:**

- Modify: `crates/pocket-tts-cli/src/voice.rs`

- [ ] Remove the legacy v1 fallback path in `resolve_predefined_voice` (the `else` branch that uses `embeddings/{name}.safetensors` without language prefix)
- [ ] All voices now require `model.language()` to be set
- [ ] Commit

### Task 4: Update Documentation

**Files:**

- Modify: `AGENTS.md`
- Modify: `crates/pocket-tts-cli/src/commands/generate.rs` — remove `--variant` hidden flag
- Modify: `crates/pocket-tts-cli/src/commands/serve.rs` — remove `--variant` hidden flag

- [ ] Remove all references to `b6369a24` and `DEFAULT_VARIANT` from docs
- [ ] Remove deprecated `--variant` CLI flag entirely
- [ ] Update AGENTS.md model weights section for v2 paths
- [ ] Commit

### Task 5: Verify CI

- [ ] Run full test suite locally
- [ ] Push and verify CI passes on all platforms
