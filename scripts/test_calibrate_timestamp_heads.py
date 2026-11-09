"""Offline calibration contracts: provenance gates, disjoint data, numeric references."""

import importlib.util
import unittest
import tempfile
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "calibration", Path(__file__).with_name("calibrate_timestamp_heads.py")
)
module = importlib.util.module_from_spec(spec)


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec.loader.exec_module(module)

    def test_shape_compatible_identity_cannot_inherit_verified_calibration(self):
        old = {
            "config_sha256": "config",
            "protocol_sha256": "protocol",
            "weights": "old",
            "tokenizer": "same",
            "preset_revision": "old",
            "heads": [[3, 8]],
            "modes": ["independent_segments"],
        }
        record = {"identity": old, "promoted": True}
        self.assertTrue(module.calibration_matches(record, old))
        for field in (
            "config_sha256",
            "protocol_sha256",
            "weights",
            "tokenizer",
            "preset_revision",
            "heads",
            "modes",
        ):
            changed = dict(old, **{field: "different"})
            self.assertFalse(module.calibration_matches(record, changed))

    def test_numeric_source_word_owns_all_expanded_asr_boundaries(self):
        words = [
            {"word": "in", "start": 0.0, "end": 0.1},
            {"word": "twenty", "start": 0.1, "end": 0.3},
            {"word": "twenty", "start": 0.3, "end": 0.5},
            {"word": "five", "start": 0.5, "end": 0.7},
        ]
        matched, missing, extra = module.match_reference(["in", "2025"], words)
        self.assertEqual(missing, [])
        self.assertEqual(extra, [])
        self.assertEqual(matched[1], (0.1, 0.7))

    def test_protocol_is_disjoint_and_contains_both_voices(self):
        protocol = module.load_protocol(
            Path(__file__).parent.parent
            / "assets/timestamp-calibration-2026-11-09/protocol.json"
        )
        self.assertEqual(len(protocol["calibration"]), 6)
        self.assertEqual(len(protocol["evaluation"]), 6)
        self.assertEqual(protocol["generation"]["voices"], ["alba", "anna"])

    def test_selection_cannot_change_after_heldout_capture_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "target/english/evaluation").mkdir(parents=True)
            with self.assertRaisesRegex(AssertionError, "heldout"):
                module.rank(root, ["english"])

    def test_each_boundary_metric_can_independently_block_promotion(self):
        old = {
            "coverage": 1.0,
            "start_mean": 0.1,
            "start_p95": 0.2,
            "end_mean": 0.1,
            "end_p95": 0.2,
            "pooled_mean": 0.1,
            "pooled_p95": 0.2,
        }
        for key in old:
            new = dict(old)
            new[key] += -0.01 if key == "coverage" else 0.01
            self.assertFalse(module.promotion_gate(old, new)["passed"], key)


if __name__ == "__main__":
    unittest.main()
