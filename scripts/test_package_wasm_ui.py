"""Exercise version stamping with isolated Git repositories and tiny build fixtures."""

import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


class PackageVersionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        (self.repo / "scripts").mkdir()
        shutil.copy(Path(__file__).with_name("package-wasm-ui.sh"), self.repo / "scripts")
        web = self.repo / "crates/pocket-tts-cli/web/dist"
        web.mkdir(parents=True)
        (web / "index.html").write_text("<html><head></head><body></body></html>")
        pkg = self.repo / "crates/pocket-tts/pkg"
        pkg.mkdir(parents=True)
        for name in ["pocket_tts.js", "pocket_tts_bg.wasm"]:
            (pkg / name).write_bytes(b"fixture")
        self.git("init", "-q")
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.invalid")
        self.git("-c", "commit.gpgsign=false", "commit", "--allow-empty", "-qm", "initial")

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.repo, text=True).strip()

    def packaged_version(self):
        subprocess.run(["bash", "scripts/package-wasm-ui.sh", "site"],
                       cwd=self.repo, check=True, capture_output=True)
        html = (self.repo / "site/index.html").read_text()
        match = re.search(r"window\.__POCKET_TTS_BOOTSTRAP__ = (.*?);</script>", html)
        self.assertIsNotNone(match)
        return json.loads(match[1]).get("build_version")

    def test_untagged_commit_uses_short_sha(self):
        self.assertEqual(self.packaged_version(), "git-" + self.git("rev-parse", "--short=12", "HEAD"))

    def test_exact_lightweight_tag_then_descendant_commit(self):
        self.git("tag", "v3.3.0")
        self.assertEqual(self.packaged_version(), "v3.3.0")
        self.git("-c", "commit.gpgsign=false", "commit", "--allow-empty", "-qm", "after release")
        self.assertEqual(self.packaged_version(), "git-" + self.git("rev-parse", "--short=12", "HEAD"))

    def test_annotated_tag_on_detached_head(self):
        self.git("-c", "tag.gpgsign=false", "tag", "-a", "upstream-v3.3.0", "-m", "release")
        self.git("checkout", "--detach", "-q")
        self.assertEqual(self.packaged_version(), "upstream-v3.3.0")


if __name__ == "__main__":
    unittest.main()
