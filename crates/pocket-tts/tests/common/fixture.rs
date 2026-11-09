//! Shared local fixture resolution. Explicit paths are relative to the manifest.
use pocket_tts::anyhow::{Result, ensure};
use sha2::{Digest, Sha256};
use std::{
    io::Read,
    path::{Path, PathBuf},
};

pub fn path(directory: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        directory.join(path)
    }
}

pub fn config(directory: &Path, manifest: &serde_json::Value) -> Result<Vec<u8>> {
    let explicit = manifest["fixture_config_path"]
        .as_str()
        .or_else(|| manifest["config_path"].as_str());
    let selected = if let Some(value) = explicit {
        path(directory, value)
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("config")
            .join(format!("{}.yaml", manifest["language"].as_str().unwrap()))
    };
    let mut bytes = std::fs::read(selected)?;
    // Historical config_path names Python YAML, which predates timestamp heads.
    // Restore the archived Rust calibration only when the complete source
    // config matches; future catalog changes cannot change this selection.
    if manifest["schema_version"].as_u64().unwrap_or(1) < 2 {
        let actual: serde_yaml::Value = serde_yaml::from_slice(&bytes)?;
        if actual.get("timestamp_heads").is_none() {
            let archived = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../assets/upstream-2026-11-09/baseline")
                .join(manifest["language"].as_str().unwrap())
                .join("rust-config.yaml");
            if archived.is_file() {
                let archived_bytes = std::fs::read(archived)?;
                let mut expected: serde_yaml::Value = serde_yaml::from_slice(&archived_bytes)?;
                expected.as_mapping_mut().unwrap().remove("timestamp_heads");
                ensure!(actual == expected, "legacy config/calibration mismatch");
                bytes = archived_bytes;
            }
        }
    }
    validate(directory, manifest)?;
    if let Some(expected) = manifest.get("config") {
        let mut actual: serde_json::Value = serde_yaml::from_slice(&bytes)?;
        actual.as_object_mut().unwrap().remove("timestamp_heads");
        ensure!(
            &actual == expected,
            "fixture config/asset identity mismatch"
        );
    }
    Ok(bytes)
}

fn hash(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(hash
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

pub fn validate(directory: &Path, manifest: &serde_json::Value) -> Result<()> {
    if let Some(commit) = manifest["upstream_commit"].as_str() {
        ensure!(
            [
                "3dbee45d343d7dddd0d105468d17f8dcba14db3e",
                "9a1b48922deb44cd4a805fde279eb33b71c1a473",
                "c91f50f38c6b914d49dbdff51b8d322edf843db2"
            ]
            .contains(&commit),
            "unknown fixture source revision: {commit}"
        );
    }
    for key in ["weights", "tokenizer", "voice_state"] {
        let asset = &manifest[key];
        let asset_path = path(
            directory,
            asset["path"]
                .as_str()
                .ok_or_else(|| pocket_tts::anyhow::anyhow!("missing fixture asset {key}"))?,
        );
        ensure!(
            asset_path.is_file(),
            "missing fixture asset: {}",
            asset_path.display()
        );
        if let Some(expected) = asset["sha256"].as_str() {
            ensure!(
                hash(&asset_path)? == expected,
                "fixture asset hash mismatch: {key}"
            );
        } else {
            ensure!(
                manifest["schema_version"].as_u64().unwrap_or(1) < 2,
                "fixture asset hash missing: {key}"
            );
        }
    }
    if manifest["schema_version"].as_u64().unwrap_or(1) >= 2 {
        let config = &manifest["config"];
        let weights = manifest["weights"]["uri"].as_str();
        ensure!(
            weights.is_some()
                && (weights == config["weights_path"].as_str()
                    || weights == config["weights_path_without_voice_cloning"].as_str()),
            "fixture config/asset mismatch: weights"
        );
        ensure!(
            manifest["tokenizer"]["uri"] == config["flow_lm"]["lookup_table"]["tokenizer_path"],
            "fixture config/asset mismatch: tokenizer"
        );
        for (name, digest) in manifest["files"]
            .as_object()
            .ok_or_else(|| pocket_tts::anyhow::anyhow!("missing fixture file hashes"))?
        {
            ensure!(
                hash(&directory.join(name))? == digest.as_str().unwrap(),
                "fixture file hash mismatch: {name}"
            );
        }
    }
    Ok(())
}
