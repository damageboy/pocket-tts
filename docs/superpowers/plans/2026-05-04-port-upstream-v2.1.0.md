# Port Upstream v2.1.0 Changes to Rust Codebase

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port all Python upstream changes from commit `ef69ab86` to `upstream/main` (v2.1.0) into the Rust/Candle codebase, enabling multi-language support and the new v2 model architecture.

**Architecture:** The v2.0/2.1 release introduces: (1) multi-language model support with per-language configs, weights, tokenizers, and voices, (2) a new Mimi codec with `inner_dim`/`outer_dim` for downsample/upsample layers, (3) BOS-before-voice injection in FlowLM, (4) weight normalization handling in weight loading, (5) updated voice resolution (per-language predefined voices), and (6) updated CLI/API with `--language` replacing `--variant`.

**Tech Stack:** Rust, Candle, serde_yaml, clap, axum

---

## Change Inventory & Effort Estimation

| Area                             | Upstream Changes                                                                                                                                                                                                                                                   | Rust Impact                                             | Effort |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------- | ------ |
| **Config schema**                | 6 new fields (`insert_bos_before_voice`, `inner_dim`, `outer_dim`, `pad_with_spaces_for_short_inputs`, `remove_semicolons`, `model_recommended_frames_after_eos`)                                                                                                  | `config.rs`                                             | S      |
| **Config files**                 | 11 new YAML configs (multi-language), rename `b6369a24.yaml` → `english_2026-01.yaml`                                                                                                                                                                              | Copy YAMLs, update `find_config_path`                   | S      |
| **ConvDownsample/Upsample**      | `out_dimension`/`in_dimension` params for Mimi v2                                                                                                                                                                                                                  | `modules/conv.rs`, `models/mimi.rs`                     | M      |
| **FlowLM BOS-before-voice**      | New `bos_before_voice` parameter + `insert_bos_before_voice` flag                                                                                                                                                                                                  | `models/flow_lm.rs`, `tts_model.rs`                     | M      |
| **Weight loading**               | Weight normalization (`weight_g`/`weight_v`), new key remapping (`fuser.padding_value` → `bos_before_voice`, `.self_attn.in_proj_weight` → `.self_attn.in_proj.weight`, `.conv.conv.` → `.conv.`), skip new keys (`_codebook`, `wavlm_*`, `num_ema_updates`, etc.) | `tts_model.rs::from_config` or new weight mapping       | L      |
| **Speaker proj weight**          | Dynamic dims from config instead of hardcoded `(1024, 512)`                                                                                                                                                                                                        | `tts_model.rs`                                          | S      |
| **Voice resolution**             | Per-language predefined voices via `get_predefined_voice(language, name)`, `origin` tracking, new voice names (giovanni, lola, juergen, rafael, estelle)                                                                                                           | `voice.rs`, `tts_model.rs`                              | M      |
| **prepare_text_prompt**          | Now takes `pad_with_spaces_for_short_inputs` and `remove_semicolons` params; conditional space-padding                                                                                                                                                             | `tts_model.rs`                                          | S      |
| **split_into_best_sentences**    | Passes through new params                                                                                                                                                                                                                                          | `tts_model.rs`                                          | S      |
| **CLI language support**         | `--language` replaces `--variant`, per-language default text/voice                                                                                                                                                                                                 | `commands/generate.rs`, `commands/serve.rs`, `voice.rs` | M      |
| **API changes**                  | Per-language default voice, no global model state, `HTMLResponse` with template substitution                                                                                                                                                                       | `commands/serve.rs`, server handler                     | M      |
| **Transformer quantization fix** | Handle `torch.ao` dynamic-quantized `in_proj.weight` callable                                                                                                                                                                                                      | N/A (Rust doesn't use torch.ao)                         | Skip   |
| **Debug/save weights env vars**  | `DEBUG_MIMI`, `POCKET_TTS_SAVE_WEIGHTS`                                                                                                                                                                                                                            | Nice-to-have, optional                                  | XS     |

### Total Effort Estimate

| Size      | Count        | Hours (est.)     |
| --------- | ------------ | ---------------- |
| XS        | 1            | 0.5              |
| S         | 4            | 4                |
| M         | 4            | 8                |
| L         | 1            | 4                |
| **Total** | **10 tasks** | **~16-20 hours** |

---

## Dependency Graph

```
Task 1 (Config Schema) ──┐
                          ├──→ Task 3 (Config Files) ──→ Task 9 (CLI language)
Task 2 (Conv dims)  ──────┤                              ↑
                          ├──→ Task 5 (Weight Loading) ──┤
Task 4 (FlowLM BOS) ─────┤                              │
                          ├──→ Task 6 (Speaker proj) ────┤
                          │                              │
                          └──→ Task 7 (Voice resolution) ┤
                                                         │
Task 8 (Text prep) ─────────────────────────────────────→│
                                                         └──→ Task 10 (API)
```

Tasks 1, 2, 4, 8 can be done in parallel. Tasks 5-7 depend on 1-4. Tasks 9-10 depend on everything.

---

## Task 1: Update Config Schema (S)

**Files:**

- Modify: `crates/pocket-tts/src/config.rs`

- [ ] **Step 1: Add new fields to `FlowLMConfig`**

```rust
// In FlowLMConfig:
#[serde(default)]
pub insert_bos_before_voice: bool,
```

- [ ] **Step 2: Add new fields to `MimiConfig`**

```rust
// In MimiConfig:
#[serde(default)]
pub inner_dim: Option<usize>,
#[serde(default)]
pub outer_dim: Option<usize>,
```

- [ ] **Step 3: Add new fields to `Config`**

```rust
// In Config:
#[serde(default)]
pub pad_with_spaces_for_short_inputs: bool,
#[serde(default)]
pub remove_semicolons: bool,
#[serde(default)]
pub model_recommended_frames_after_eos: Option<usize>,
```

- [ ] **Step 4: Update `defaults` module**

Replace `DEFAULT_VARIANT` with `DEFAULT_LANGUAGE`:

```rust
pub const DEFAULT_LANGUAGE: &str = "english";
```

Add per-language default text and voice mappings.

- [ ] **Step 5: Run tests**

Run: `cargo test -p pocket-tts --release -- config`
Expected: existing tests pass (with updated config path references)

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: update config schema for v2.1.0 multi-language support"
```

---

## Task 2: Update ConvDownsample/ConvTrUpsample Dimensions (M)

**Files:**

- Modify: `crates/pocket-tts/src/modules/conv.rs`
- Modify: `crates/pocket-tts/src/models/mimi.rs`

- [ ] **Step 1: Add `out_dimension` param to `ConvDownsample1d::new`**

```rust
pub fn new(stride: usize, dimension: usize, out_dimension: Option<usize>, name: &str, vb: VarBuilder) -> Result<Self> {
    let out_dim = out_dimension.unwrap_or(dimension);
    let conv = StreamingConv1d::new(
        dimension,
        out_dim,  // was: dimension
        ...
    )?;
```

- [ ] **Step 2: Add `in_dimension` param to `ConvTrUpsample1d::new`**

```rust
pub fn new(stride: usize, dimension: usize, in_dimension: Option<usize>, name: &str, vb: VarBuilder) -> Result<Self> {
    let in_dim = in_dimension.unwrap_or(dimension);
    let convtr = StreamingConvTranspose1d::new(
        in_dim,   // was: dimension
        dimension,
        ...
    )?;
```

- [ ] **Step 3: Update `MimiModel::new` to pass `inner_dim`/`outer_dim`**

In `models/mimi.rs`, pass config values through:

```rust
ConvDownsample1d::new(stride, dimension, config_inner_dim, ...)
ConvTrUpsample1d::new(stride, dimension, config_outer_dim, ...)
```

- [ ] **Step 4: Update `tts_model.rs::from_config_and_vb`** to pass `inner_dim`/`outer_dim` from config to MimiModel constructor.

- [ ] **Step 5: Run tests**

Run: `cargo test -p pocket-tts --release`

- [ ] **Step 6: Commit**

---

## Task 3: Add Multi-Language Config Files (S)

**Files:**

- Create: `crates/pocket-tts/config/english.yaml`
- Create: `crates/pocket-tts/config/english_2026-01.yaml` (rename from `b6369a24.yaml`)
- Create: `crates/pocket-tts/config/english_2026-04.yaml`
- Create: `crates/pocket-tts/config/french_24l.yaml`
- Create: `crates/pocket-tts/config/german.yaml`
- Create: `crates/pocket-tts/config/german_24l.yaml`
- Create: `crates/pocket-tts/config/italian.yaml`
- Create: `crates/pocket-tts/config/italian_24l.yaml`
- Create: `crates/pocket-tts/config/portuguese.yaml`
- Create: `crates/pocket-tts/config/portuguese_24l.yaml`
- Create: `crates/pocket-tts/config/spanish.yaml`
- Create: `crates/pocket-tts/config/spanish_24l.yaml`
- Modify: `crates/pocket-tts/src/tts_model.rs` (`find_config_path`)
- Modify: `crates/pocket-tts/src/config.rs` (also look in `pocket_tts/config/` alongside Rust config dir)

- [ ] **Step 1: Copy config YAMLs from upstream**

Copy all YAML files from `pocket_tts/config/` (Python upstream) into `crates/pocket-tts/config/`.
Keep `b6369a24.yaml` as a symlink or copy of `english_2026-01.yaml` for backward compat.

- [ ] **Step 2: Update `find_config_path`**

Change to accept either a language name or a full path:

- If input ends with `.yaml`/`.yml`, treat as path
- Otherwise, look for `{language}.yaml` in config directories
- Keep backward compat: `b6369a24` still resolves

- [ ] **Step 3: Run tests**

- [ ] **Step 4: Commit**

---

## Task 4: Add BOS-Before-Voice to FlowLM (M)

**Files:**

- Modify: `crates/pocket-tts/src/models/flow_lm.rs`
- Modify: `crates/pocket-tts/src/tts_model.rs`

- [ ] **Step 1: Add `bos_before_voice` field to `FlowLMModel`**

```rust
pub struct FlowLMModel {
    // ... existing fields ...
    pub insert_bos_before_voice: bool,
    pub bos_before_voice: Option<Tensor>,  // [1, 1, dim] - loaded from weights
}
```

- [ ] **Step 2: Conditionally load `bos_before_voice` weight**

In `FlowLMModel::new` or the builder, if `insert_bos_before_voice` is true, load the parameter:

```rust
let bos_before_voice = if insert_bos_before_voice {
    Some(vb.get((1, 1, dim), "bos_before_voice")?)
} else {
    None
};
```

- [ ] **Step 3: Inject BOS before voice conditioning in `get_voice_state_from_tensor`**

After computing `conditioning`, before `run_flow_lm_prompt`:

```rust
let conditioning = if let Some(bos) = &self.flow_lm.bos_before_voice {
    Tensor::cat(&[bos, &conditioning], 1)?
} else {
    conditioning
};
```

- [ ] **Step 4: Pass config flag through model construction**

Update `from_config_and_vb` to pass `config.flow_lm.insert_bos_before_voice` to `FlowLMModel::new`.

- [ ] **Step 5: Run tests**

- [ ] **Step 6: Commit**

---

## Task 5: Update Weight Loading / Key Remapping (L)

**Files:**

- Modify: `crates/pocket-tts/src/tts_model.rs` (weight loading section)

This is the most complex task. The upstream now uses a unified `model.safetensors` per language with different key naming conventions. The Rust code currently uses VarBuilder which auto-maps keys. We need to handle:

- [ ] **Step 1: Understand current weight loading**

The Rust code uses `VarBuilder::from_mmaped_safetensors` which auto-strips prefixes when you call `vb.pp("mimi.encoder")`. This means most key remapping should work automatically IF the weight file keys match the Rust model's expected names.

- [ ] **Step 2: Handle `.conv.conv.` → `.conv.` and `.convtr.convtr.` → `.convtr.` remapping**

The v2 weights have doubled prefixes (e.g., `mimi.encoder.model.0.conv.conv.weight`). The Rust VarBuilder doesn't auto-remap these. Options:

1. Pre-process the safetensors keys before building VarBuilder
2. Use a custom weight loader that applies remapping

Implement a `remap_weight_keys()` function that loads safetensors, remaps keys, and returns a new tensor map.

- [ ] **Step 3: Handle weight normalization (`weight_g` + `weight_v` → `weight`)**

v2 weights use PyTorch weight normalization. The file contains `*.weight_g` and `*.weight_v` tensors that must be combined:

```rust
// weight = weight_g * normalize(weight_v, dim=0)
fn apply_weight_norm(weight_v: &Tensor, weight_g: &Tensor) -> Tensor {
    let norm = weight_v.normalize(0);  // L2 norm along dim 0
    weight_g * norm
}
```

- [ ] **Step 4: Handle `fuser.padding_value` → `bos_before_voice` key remap**

- [ ] **Step 5: Handle `.self_attn.in_proj_weight` → `.self_attn.in_proj.weight`**

- [ ] **Step 6: Skip irrelevant keys**

Skip keys containing: `_codebook`, `wavlm`, `quantizer.vq.`, `quantizer.logvar`, `num_ema_updates`

- [ ] **Step 7: Test with actual v2 weights**

Download the `english` model weights and verify loading succeeds:

```bash
HF_TOKEN=... cargo test -p pocket-tts --release -- test_load
```

- [ ] **Step 8: Commit**

---

## Task 6: Dynamic Speaker Projection Dimensions (S)

**Files:**

- Modify: `crates/pocket-tts/src/tts_model.rs`

- [ ] **Step 1: Use config dimensions for speaker_proj_weight**

Currently hardcoded to `(1024, 512)`. Change to:

```rust
let speaker_proj_dim_out = config.flow_lm.transformer.d_model;
let speaker_proj_dim_in = config.mimi.inner_dim.unwrap_or(config.mimi.seanet.dimension);
let speaker_proj_weight = vb.get(
    (speaker_proj_dim_out, speaker_proj_dim_in),
    "flow_lm.speaker_proj_weight"
)?;
```

Wait — actually, the Python code uses `config.mimi.inner_dim or config.mimi.seanet.dimension` for the second dim. But the speaker projection maps from Mimi's quantizer output space. Need to verify this matches the actual weight tensor shape in v2 models.

- [ ] **Step 2: Verify against v2 weight shapes**

- [ ] **Step 3: Commit**

---

## Task 7: Update Voice Resolution for Multi-Language (M)

**Files:**

- Modify: `crates/pocket-tts-cli/src/voice.rs`
- Modify: `crates/pocket-tts/src/tts_model.rs` (add `origin` field)

- [ ] **Step 1: Add `origin` field to `TTSModel`**

```rust
pub struct TTSModel {
    // ...
    pub origin: Option<std::path::PathBuf>,  // config path used to load model
}
```

- [ ] **Step 2: Update `PREDEFINED_VOICES` to include new multi-language voices**

```rust
pub const PREDEFINED_VOICES: &[&str] = &[
    "alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma",
    "giovanni", "lola", "juergen", "rafael", "estelle",
];
```

- [ ] **Step 3: Update `resolve_predefined_voice` to use per-language paths**

The new URL pattern is:

```
hf://kyutai/pocket-tts-without-voice-cloning/languages/{language}/embeddings/{name}.safetensors@{revision}
```

Need to extract language from `model.origin` (the config path stem).

- [ ] **Step 4: Add per-language default voice selection**

```rust
pub fn default_voice_for_language(language: &str) -> &str {
    match language {
        l if l.contains("italian") => "giovanni",
        l if l.contains("spanish") => "lola",
        l if l.contains("german") => "juergen",
        l if l.contains("portuguese") => "rafael",
        l if l.contains("french") => "estelle",
        _ => "alba",
    }
}
```

- [ ] **Step 5: Run tests**

- [ ] **Step 6: Commit**

---

## Task 8: Update `prepare_text_prompt` and `split_into_best_sentences` (S)

**Files:**

- Modify: `crates/pocket-tts/src/tts_model.rs`

- [ ] **Step 1: Add `remove_semicolons` and `pad_with_spaces_for_short_inputs` to TTSModel struct**

- [ ] **Step 2: Update `prepare_text_prompt`**

Add semicolons → commas replacement when `remove_semicolons` is true.
Make space-padding conditional on `pad_with_spaces_for_short_inputs`.

Currently the Rust code unconditionally adds 8 spaces. The new behavior:

- Only pad with spaces if `pad_with_spaces_for_short_inputs` is true AND word count < 5

- [ ] **Step 3: Update `split_into_best_sentences` to pass through new params**

- [ ] **Step 4: Update `generate_audio_stream` to use `model_recommended_frames_after_eos`**

If `frames_after_eos` is None, fall back to `self.model_recommended_frames_after_eos`.

- [ ] **Step 5: Run tests (including prepare_text_prompt tests)**

Run: `cargo test -p pocket-tts --release -- test_prepare_text_prompt`

- [ ] **Step 6: Commit**

---

## Task 9: Update CLI for Language-Based Model Selection (M)

**Files:**

- Modify: `crates/pocket-tts-cli/src/commands/generate.rs`
- Modify: `crates/pocket-tts-cli/src/commands/serve.rs`
- Modify: `crates/pocket-tts-cli/src/voice.rs`

- [ ] **Step 1: Add `--language` arg to `GenerateArgs`**

```rust
/// Language for the TTS model (e.g., "english", "french_24l", "german")
#[arg(long)]
pub language: Option<String>,
```

Change `--variant` default to be optional / deprecated.

- [ ] **Step 2: Add `--language` arg to `ServeArgs`**

- [ ] **Step 3: Update model loading to use language OR config path**

Implement the Python logic: if both `language` and `config` are set, error. If neither, default to `"english"`.

- [ ] **Step 4: Add per-language default text**

Port `DEFAULT_TEXT_FOR_LANGUAGE` dict from Python.

- [ ] **Step 5: Add per-language default voice selection in generate/serve**

- [ ] **Step 6: Run CLI tests**

- [ ] **Step 7: Commit**

---

## Task 10: Update API/Server for Multi-Language (M)

**Files:**

- Modify: `crates/pocket-tts-cli/src/commands/serve.rs`
- Modify: server handler files

- [ ] **Step 1: Remove global model state pre-loading**

The API should no longer pre-compute a default voice state at startup. Instead, resolve per-request.

- [ ] **Step 2: Add per-language default voice in TTS endpoint**

If no voice specified, use `default_voice_for_language(model.origin)`.

- [ ] **Step 3: Update HTML serving to template default text**

Replace placeholder in `index.html` with language-appropriate default text.

- [ ] **Step 4: Run server tests**

- [ ] **Step 5: Commit**

---

## Key Risks & Open Questions

1. **Weight compatibility**: The biggest risk is that v2 weight files have different key naming/structure. Task 5 (weight loading) is the critical path. Recommend downloading actual v2 weights early to test.

2. **Backward compatibility**: The `b6369a24` variant is used in existing tests and by users. We should keep it working (as an alias for `english_2026-01`).

3. **Voice embedding URLs changed**: The old URL pattern (`embeddings_v3/{name}.safetensors`) is replaced by per-language paths (`languages/{language}/embeddings/{name}.safetensors`). Old voices won't resolve with the new model.

4. **Mimi quantizer dimension change**: v2 models have `quantizer.dimension: 32` (was 512). The `inner_dim`/`outer_dim` on the downsample/upsample layers handle this, but the interaction with `speaker_proj_weight` dimensions needs careful verification.

5. **Weight normalization in Candle**: Candle doesn't have a built-in `_weight_norm` equivalent. We need to implement `weight = weight_g * F.normalize(weight_v, dim=0)` manually, which is straightforward but needs testing.

6. **WASM impact**: The `load_from_bytes` path also needs updating for the new config fields and weight format.

## Recommended Execution Order

**Phase 1 (Foundation, parallel):** Tasks 1, 2, 4, 8 — pure model/config changes, no weight deps
**Phase 2 (Weight loading):** Task 5 — test with real v2 weights
**Phase 3 (Integration):** Tasks 3, 6, 7 — config files + voice resolution
**Phase 4 (CLI/API):** Tasks 9, 10 — user-facing changes
