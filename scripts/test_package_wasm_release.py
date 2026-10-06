"""Check release archives using distinct build fixtures, without Rust or model weights."""

import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest


class PackageReleaseTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        script = Path(__file__).with_name("package-wasm-release.sh")
        self.assertTrue(script.is_file(), "release packaging script is missing")
        (self.repo / "scripts").mkdir()
        shutil.copy(script, self.repo / "scripts")
        self.pkg = self.repo / "crates/pocket-tts/pkg"
        self.pkg.mkdir(parents=True)
        self.files = {
            "pocket_tts.js": b"export default function init() {}",
            "pocket_tts_bg.wasm": b"\x00asm\x01\x00\x00\x00",
            "pocket_tts.d.ts": b"export default function init(): Promise<void>;",
            "pocket_tts_bg.wasm.d.ts": b"export const memory: WebAssembly.Memory;",
        }
        for name, data in self.files.items():
            (self.pkg / name).write_bytes(data)
        (self.pkg / "stale-file.txt").write_text("must not ship")

    def package(self, tag="v3.3.0"):
        return subprocess.run(
            ["bash", "scripts/package-wasm-release.sh", tag],
            cwd=self.repo, capture_output=True, text=True,
        )

    def test_archive_preserves_matching_files_and_has_verifiable_checksum(self):
        result = self.package()
        self.assertEqual(result.returncode, 0, result.stderr)
        archive = self.repo / "dist/wasm-release/pocket-tts-v3.3.0-wasm-web.tar.gz"
        with tarfile.open(archive) as tar:
            self.assertEqual(set(tar.getnames()), set(self.files))
            for name, data in self.files.items():
                self.assertEqual(tar.extractfile(name).read(), data)
        checksum = Path(str(archive) + ".sha256").read_text().split()
        self.assertEqual(checksum, [hashlib.sha256(archive.read_bytes()).hexdigest(), archive.name])

    def test_missing_wasm_or_js_fails_without_an_archive(self):
        for name in ["pocket_tts.js", "pocket_tts_bg.wasm"]:
            with self.subTest(name=name):
                (self.pkg / name).unlink()
                result = self.package()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(name, result.stderr)
                self.assertFalse(list(self.repo.glob("dist/**/*.tar.gz")))
                (self.pkg / name).write_bytes(self.files[name])

    def test_rejects_non_release_or_path_like_names(self):
        for tag in ["main", "", "v3/../../escape"]:
            with self.subTest(tag=tag):
                result = self.package(tag)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(list(self.repo.glob("dist/**/*.tar.gz")))


if __name__ == "__main__":
    unittest.main()
