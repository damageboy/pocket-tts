"""Reference integrity failures must happen before importing model code."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).with_name('generate_sync_reference.py')


def generator():
    assert SCRIPT.exists(), 'the sync fixture generator is missing'
    spec = importlib.util.spec_from_file_location('sync_reference', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wrong_source_rejected_before_output(tmp_path):
    output = tmp_path / 'fixtures'
    result = subprocess.run([sys.executable, str(SCRIPT), '--reference', str(Path(__file__).parents[1]),
                             '--expected-commit', 'c91f50f38c6b914d49dbdff51b8d322edf843db2', '--output', str(output)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'reference revision' in result.stderr
    assert not output.exists()


def test_unknown_revision_is_fatal(tmp_path):
    with pytest.raises(ValueError, match='Unknown reference revision'):
        generator().verify_reference(tmp_path, '0' * 40)


def test_missing_asset_is_fatal(tmp_path):
    with pytest.raises(FileNotFoundError):
        generator().validate_asset({'path': str(tmp_path / 'absent'), 'sha256': '0' * 64})


def test_asset_mismatch_is_fatal(tmp_path):
    asset = tmp_path / 'asset'
    asset.write_bytes(b'wrong revision')
    with pytest.raises(ValueError, match='asset hash mismatch'):
        generator().validate_asset({'path': str(asset), 'sha256': '0' * 64})


def test_config_asset_mismatch_is_fatal(tmp_path):
    asset = tmp_path / 'asset'
    asset.write_bytes(b'weights')
    module = generator()
    with pytest.raises(ValueError, match='config/asset mismatch'):
        module.validate_asset({'path': str(asset), 'sha256': module.sha256(asset),
                               'uri': 'hf://repo/wrong@' + 'a' * 40},
                              'hf://repo/expected@' + 'b' * 40)


def test_checked_in_fixture_integrity():
    module = generator()
    root = Path(__file__).parents[1] / 'assets/upstream-2026-11-09'
    assert (root / 'manifest.json').exists(), 'generated target fixtures are missing'
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert module.sha256(root / name) == digest
    assert manifest['upstream_commit'] in module.KNOWN_COMMITS
    assert {entry['tokenizer_family'] for entry in manifest['models']} == {'tokenizers', 'sentencepiece'}


def test_raw_decoder_fade_and_prompt_preprocessing_are_separate():
    import torch
    from safetensors.torch import load_file
    root = Path(__file__).parents[1] / 'assets/upstream-2026-11-09/target'
    for directory in sorted(root.iterdir()):
        tensors = load_file(directory / 'model.safetensors')
        raw = tensors['step_0.pcm']
        expected = raw.clone()
        expected[..., :120] *= torch.linspace(0, 1, 120)
        assert torch.equal(expected, tensors['output_fade.first_frame'])
        assert raw.shape == expected.shape
        assert not torch.equal(raw, expected)
        manifest = json.loads((directory / 'manifest.json').read_text())
        assert tensors['continuation.forced_prefix'].shape[1] == manifest['steps']
        assert manifest['continuation']['warmup_frames'] == min(12, manifest['steps'])
        assert torch.isnan(tensors['continuation.forced_prefix'][:, 0]).all()
        assert torch.equal(tensors['continuation.forced_prefix'][:, 1:],
                           tensors['continuation.forced_latents'][:, :-1])
        assert torch.equal(tensors['continuation.first_input'],
                           tensors['continuation.forced_latents'][:, -1:])
        prompt = load_file(directory / 'prompt.safetensors')
        assert torch.equal(prompt['short.input'], prompt['short.output'])
        assert prompt['all_zero.output'].shape[-1] == 3360
        assert torch.count_nonzero(prompt['all_zero.output']) == 0


def test_all_model_manifests_validate_pinned_config_and_assets():
    module = generator()
    root = Path(__file__).parents[1] / 'assets/upstream-2026-11-09'
    manifests = sorted(root.glob('*/*/manifest.json'))
    assert len(manifests) == 8
    for manifest in manifests:
        module.validate_fixture(manifest)
