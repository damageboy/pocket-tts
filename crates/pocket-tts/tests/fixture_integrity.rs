//! Small local identity checks. No model files or inference are required.
#[path = "common/fixture.rs"]
mod fixture;
use pocket_tts::anyhow::Result;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

struct LocalFixture {
    directory: PathBuf,
    manifest: Value,
}
impl LocalFixture {
    fn new() -> Result<Self> {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "pocket-fixture-{}-{unique}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir(&directory)?;
        let config = json!({"weights_path": "hf://repo/weights@fixed", "weights_path_without_voice_cloning": "hf://repo/open@fixed", "flow_lm": {"lookup_table": {"tokenizer_path": "hf://repo/tokenizer@fixed"}}});
        let source = serde_yaml::to_string(&config)?;
        std::fs::write(directory.join("config.yaml"), &source)?;
        std::fs::write(
            directory.join("selected.yaml"),
            format!("{source}timestamp_heads: [[0, 0]]\n"),
        )?;
        let mut manifest = json!({"schema_version": 2, "language": "local-test", "config": config, "config_path": "config.yaml", "fixture_config_path": "selected.yaml", "files": {"config.yaml": digest(source.as_bytes())}});
        for (key, uri) in [
            ("weights", "hf://repo/weights@fixed"),
            ("tokenizer", "hf://repo/tokenizer@fixed"),
            ("voice_state", "hf://repo/voice@fixed"),
        ] {
            std::fs::write(directory.join(key), key)?;
            manifest[key] = json!({"path": key, "uri": uri, "sha256": digest(key.as_bytes())});
        }
        Ok(Self {
            directory,
            manifest,
        })
    }
    fn authenticate_selected(&mut self) -> Result<()> {
        self.manifest["files"]["selected.yaml"] = json!(digest(&std::fs::read(
            self.directory.join("selected.yaml")
        )?));
        Ok(())
    }
}
impl Drop for LocalFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

#[test]
fn selected_calibration_requires_its_own_digest() -> Result<()> {
    let local = LocalFixture::new()?;
    let result = fixture::config(&local.directory, &local.manifest);
    assert!(
        result.is_err(),
        "unlisted external calibration was accepted"
    );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("selected config SHA-256")
    );
    Ok(())
}

#[test]
fn authenticated_selected_calibration_is_accepted() -> Result<()> {
    let mut local = LocalFixture::new()?;
    local.authenticate_selected()?;
    fixture::config(&local.directory, &local.manifest)?;
    Ok(())
}

#[test]
fn changing_selected_calibration_is_fatal() -> Result<()> {
    let mut local = LocalFixture::new()?;
    local.authenticate_selected()?;
    let selected = local.directory.join("selected.yaml");
    std::fs::write(
        &selected,
        std::fs::read_to_string(&selected)?.replace("[[0, 0]]", "[[1, 1]]"),
    )?;
    assert!(fixture::config(&local.directory, &local.manifest).is_err());
    Ok(())
}

#[test]
fn legacy_schema_still_accepts_unlisted_calibration() -> Result<()> {
    let mut local = LocalFixture::new()?;
    local.manifest["schema_version"] = json!(1);
    fixture::config(&local.directory, &local.manifest)?;
    Ok(())
}
