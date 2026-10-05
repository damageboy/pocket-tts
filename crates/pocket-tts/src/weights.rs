use anyhow::Result;
use std::path::PathBuf;

#[cfg(not(target_arch = "wasm32"))]
use candle_core::Device;

#[cfg(not(target_arch = "wasm32"))]
use hf_hub::{HFClientSync, split_id};

/// Resolve the configured bundle and its voice-cloning capability. Like the
/// reference, only download failures trigger fallback, not invalid weights.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn download_model_weights(
    config: &crate::config::Config,
    mut download: impl FnMut(&str) -> Result<PathBuf>,
) -> Result<(PathBuf, bool)> {
    if let Some(path) = &config.weights_path {
        match download(path) {
            Ok(file) => return Ok((file, true)),
            Err(error) if config.weights_path_without_voice_cloning.is_some() => {
                tracing::warn!("Cloning weights unavailable ({error}); using preset-only weights");
            }
            Err(error) => return Err(error),
        }
    }
    let path = config
        .weights_path_without_voice_cloning
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("No model weights specified in config"))?;
    Ok((download(path)?, false))
}

/// Download a file from HuggingFace Hub if necessary.
///
/// Supports the format: `hf://owner/repo/filename@revision`
/// where `@revision` is optional.
///
/// Note: Not available on wasm32 targets (use local file loading instead).
#[cfg(not(target_arch = "wasm32"))]
pub fn download_if_necessary(file_path: &str) -> Result<PathBuf> {
    if file_path.starts_with("hf://") {
        let path = file_path.trim_start_matches("hf://");
        let parts: Vec<&str> = path.split('/').collect();
        if parts.len() < 3 {
            anyhow::bail!(
                "Invalid hf:// path: {}. Expected hf://repo_owner/repo_name/filename[@revision]",
                file_path
            );
        }
        let repo_id = format!("{}/{}", parts[0], parts[1]);
        let filename_with_revision = parts[2..].join("/");

        // Parse optional revision from filename (e.g., "file.safetensors@abc123")
        let (filename, revision) = if let Some(at_pos) = filename_with_revision.rfind('@') {
            let (f, r) = filename_with_revision.split_at(at_pos);
            (f.to_string(), Some(r[1..].to_string())) // Skip the '@'
        } else {
            (filename_with_revision, None)
        };

        let client = HFClientSync::new()?;
        let (owner, name) = split_id(&repo_id);
        let repo = client.model(owner, name);
        let path = repo
            .download_file()
            .filename(filename)
            .maybe_revision(revision)
            .send()?;
        Ok(path)
    } else {
        Ok(PathBuf::from(file_path))
    }
}

/// WASM version: Only supports local file paths
#[cfg(target_arch = "wasm32")]
pub fn download_if_necessary(file_path: &str) -> Result<PathBuf> {
    if file_path.starts_with("hf://") {
        anyhow::bail!("HuggingFace Hub downloads not supported on WASM. Use local file paths.");
    }
    Ok(PathBuf::from(file_path))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_weights(
    file_path: &str,
    _device: &Device,
) -> Result<candle_core::safetensors::MmapedSafetensors> {
    let path = download_if_necessary(file_path)?;
    let safetensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };
    Ok(safetensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_if_necessary_local() {
        let path = "test.safetensors";
        let res = download_if_necessary(path).unwrap();
        assert_eq!(res, PathBuf::from(path));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn model_download_fallback_tracks_capability_and_preserves_errors() -> Result<()> {
        let mut config = crate::config::load_config(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/config/english.yaml"
        ))?;
        config.weights_path = Some("gated".into());
        config.weights_path_without_voice_cloning = Some("open".into());
        let mut requested = Vec::new();
        let (path, cloning) = download_model_weights(&config, |path| {
            requested.push(path.to_owned());
            if path == "gated" {
                anyhow::bail!("access denied")
            }
            Ok(path.into())
        })?;
        assert_eq!(requested, ["gated", "open"]);
        assert_eq!(path, PathBuf::from("open"));
        assert!(!cloning);
        let (_, cloning) = download_model_weights(&config, |path| Ok(path.into()))?;
        assert!(cloning);
        config.weights_path_without_voice_cloning = None;
        let error =
            download_model_weights(&config, |_| anyhow::bail!("access denied")).unwrap_err();
        assert!(error.to_string().contains("access denied"));
        config.weights_path = None;
        config.weights_path_without_voice_cloning = Some("open".into());
        assert!(!download_model_weights(&config, |path| Ok(path.into()))?.1);
        Ok(())
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_invalid_hf_path() {
        let path = "hf://invalid";
        let res = download_if_necessary(path);
        assert!(res.is_err());
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_parse_revision() {
        // Test parsing logic (doesn't actually download)
        let path = "hf://kyutai/pocket-tts/file.safetensors@abc123def";
        // This will fail to download but we're testing the parsing
        let res = download_if_necessary(path);
        // We expect a network error, not a parsing error
        assert!(res.is_err());
    }
}
