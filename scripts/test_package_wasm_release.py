"""Check release archives using distinct build fixtures, without Rust or model weights."""

import hashlib
import json
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
        (self.repo / "Cargo.toml").write_text(
            '[workspace]\nmembers = ["crates/pocket-tts", "crates/pocket-tts-cli"]\n'
            '[workspace.package]\nversion = "3.3.0"\n'
        )
        for name in ["pocket-tts", "pocket-tts-cli"]:
            member = self.repo / "crates" / name
            member.mkdir(parents=True)
            manifest = f'[package]\nname = "{name}"\nversion.workspace = true\n'
            if name == "pocket-tts-cli":
                manifest += '[dependencies]\npocket-tts = { path = "../pocket-tts", version = "3.3.0" }\n'
            (member / "Cargo.toml").write_text(manifest)
        (self.repo / "Cargo.lock").write_text(
            'version = 4\n[[package]]\nname = "pocket-tts"\nversion = "3.3.0"\n'
            '[[package]]\nname = "pocket-tts-cli"\nversion = "3.3.0"\n'
        )
        validator = Path(__file__).with_name("validate-release-version.py")
        if validator.exists():
            shutil.copy(validator, self.repo / "scripts")
        self.pkg = self.repo / "crates/pocket-tts/pkg"
        self.pkg.mkdir(parents=True)
        self.files = {
            "pocket_tts.js": b"export default function init() {}",
            "pocket_tts_bg.wasm": b"\x00asm\x01\x00\x00\x00",
            "pocket_tts.d.ts": b"export default function init(): Promise<void>;",
            "pocket_tts_bg.wasm.d.ts": b"export const memory: WebAssembly.Memory;",
            "models.json": b'{"schemaVersion":1,"sourceRevision":"fixture"}',
            "package.json": b'{"name":"pocket-tts","version":"3.3.0","type":"module"}',
            "THIRD_PARTY_NOTICES.md": b"Timestamp fork attribution and MIT permission notice.",
        }
        for name, data in self.files.items():
            destination = self.pkg.parent if name == "THIRD_PARTY_NOTICES.md" else self.pkg
            (destination / name).write_bytes(data)
        (self.pkg / "stale-file.txt").write_text("must not ship")

    def package(self, tag="v3.3.0"):
        return subprocess.run(
            ["bash", "scripts/package-wasm-release.sh", tag],
            cwd=self.repo, capture_output=True, text=True,
        )

    def set_release_version(self, version, previous="3.3.0"):
        paths = [self.repo / "Cargo.toml", self.repo / "Cargo.lock", self.pkg / "package.json"]
        paths.extend(self.repo.glob("crates/*/Cargo.toml"))
        for path in paths:
            path.write_text(path.read_text().replace(previous, version))

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

    def test_rejects_tag_that_disagrees_with_cargo(self):
        result = self.package("v4.0.1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("version", result.stderr.lower())
        self.assertFalse(list(self.repo.glob("dist/**/*.tar.gz")))

    def test_rejects_stale_publishable_dependency(self):
        manifest = self.repo / "crates/pocket-tts-cli/Cargo.toml"
        manifest.write_text(manifest.read_text().replace('version = "3.3.0"', 'version = "0.8.1"'))
        result = self.package()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pocket-tts", result.stderr)
        self.assertFalse(list(self.repo.glob("dist/**/*.tar.gz")))

    def test_accepts_external_workspace_dependency_with_string_version(self):
        root = self.repo / "Cargo.toml"
        root.write_text(root.read_text() + '[workspace.dependencies]\nanyhow = "1.0"\n')
        manifest = self.repo / "crates/pocket-tts-cli/Cargo.toml"
        manifest.write_text(manifest.read_text() + 'anyhow = { workspace = true }\n')
        result = self.package()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_stale_lockfile_package(self):
        lock = self.repo / "Cargo.lock"
        lock.write_text(lock.read_text().replace('version = "3.3.0"', 'version = "0.8.1"'))
        result = self.package()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Cargo.lock", result.stderr)
        self.assertFalse(list(self.repo.glob("dist/**/*.tar.gz")))

    def test_rejects_stale_wasm_package_version(self):
        (self.pkg / "package.json").write_text(json.dumps({"version": "0.8.1"}))
        result = self.package()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("package.json", result.stderr)
        self.assertFalse(list(self.repo.glob("dist/**/*.tar.gz")))

    def test_packages_explicit_build_directory(self):
        build = self.repo / "fresh-wasm"
        self.pkg.rename(build)
        result = subprocess.run(
            ["bash", "scripts/package-wasm-release.sh", "v3.3.0", "release-output", str(build)],
            cwd=self.repo, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.repo / "release-output/pocket-tts-v3.3.0-wasm-web.tar.gz").is_file())

    def test_missing_wasm_or_js_fails_without_an_archive(self):
        for name in ["pocket_tts.js", "pocket_tts_bg.wasm", "models.json"]:
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

    def test_rejects_consistently_malformed_semver_without_an_archive(self):
        previous = "3.3.0"
        for version in [
            "3.3.0-01", "3.3.0-alpha.01", "3.3.0-a..b", "3.3.0-.alpha",
            "3.3.0-alpha.", "3.3.0+.", "3.3.0+build..meta", "3.3.0+build.",
            "3.3.0+.build", "3.3.0-", "3.3.0+", "03.3.0", "3.03.0",
            "3.3.00", "3.3.0-α", "3.3.0+build_meta",
        ]:
            self.set_release_version(version, previous)
            previous = version
            with self.subTest(version=version):
                shutil.rmtree(self.repo / "dist", ignore_errors=True)
                result = self.package(f"v{version}")
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("release", result.stderr.lower())
                self.assertFalse(list(self.repo.glob("dist/**/*.tar.gz")))

    def test_accepts_valid_prerelease_and_build_identifiers(self):
        previous = "3.3.0"
        for version in [
            "3.3.0-0", "3.3.0-alpha.1", "3.3.0-01alpha.0.-",
            "3.3.0+001", "3.3.0+build.01.-", "3.3.0-rc.1+build.001",
        ]:
            self.set_release_version(version, previous)
            previous = version
            with self.subTest(version=version):
                result = self.package(f"v{version}")
                self.assertEqual(result.returncode, 0, result.stderr)
                archive = self.repo / f"dist/wasm-release/pocket-tts-v{version}-wasm-web.tar.gz"
                self.assertTrue(archive.is_file())


if __name__ == "__main__":
    unittest.main()
