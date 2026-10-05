"""Offline contract tests for the reference generator's provenance gate."""

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


class ReferenceProvenanceTests(unittest.TestCase):
    def test_rejects_an_unrelated_checkout_before_importing_or_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "-c", "user.name=Test", "-c",
                 "user.email=test@example.invalid", "commit", "--allow-empty", "-qm", "test"],
                check=True,
            )
            output = root / "fixtures"
            result = subprocess.run(
                ["python3", str(Path(__file__).with_name("generate_upstream_reference.py")),
                 "--reference", str(root), "--output", str(output)],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("reference revision", result.stderr)
            self.assertFalse(output.exists())

    def test_fixture_hash_detects_content_changes(self):
        path = Path(__file__).with_name("generate_upstream_reference.py")
        spec = importlib.util.spec_from_file_location("reference", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "asset"
            file.write_bytes(b"abc")
            self.assertEqual(
                module.sha256(file),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )
            file.write_bytes(b"abd")
            self.assertNotEqual(
                module.sha256(file),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )


if __name__ == "__main__":
    unittest.main()
