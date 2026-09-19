"""Preregistration consistency: manifest integrity vs protocol constants."""

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v4_confirm_analysis as confirm
import v4_protocol_common as common

MANIFEST = ROOT / "results" / "v4" / "prereg" / "seed_manifest.json"


class PreregConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MANIFEST.is_file():
            raise unittest.SkipTest("seed manifest not frozen yet")
        cls.manifest = json.loads(MANIFEST.read_text())

    def test_manifest_sha_matches_detached_signature(self):
        digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
        recorded = (MANIFEST.with_name("seed_manifest.sha256").read_text())
        self.assertTrue(recorded.startswith(digest))

    def test_blocks_match_protocol_constants(self):
        self.assertEqual(self.manifest["training_seeds"], common.TRAINING_SEEDS)
        self.assertEqual(self.manifest["faultfree_sim_seeds"]["seeds"],
                         common.FAULTFREE_SEEDS)
        self.assertEqual(self.manifest["severity_sim_seeds"]["seeds"],
                         common.SEVERITY_SEEDS)
        prefixes = self.manifest["severity_prefixes"]
        self.assertEqual(prefixes["sweep"], common.SWEEP_SEEDS)
        self.assertEqual(prefixes["robustness"], common.ROBUSTNESS_SEEDS)
        self.assertEqual(prefixes["det_subset"], common.DET_SEEDS)
        self.assertEqual(self.manifest["dataset_seed"], 62026)
        self.assertEqual(self.manifest["smoke_seeds"]["sim"], common.SMOKE_SEEDS)

    def test_new_blocks_avoid_used_and_forbidden(self):
        new = (set(self.manifest["faultfree_sim_seeds"]["seeds"])
               | set(self.manifest["severity_sim_seeds"]["seeds"]))
        for name, block in (("12026", range(12026, 12031)),
                            ("19026", range(19026, 19031)),
                            ("29026", range(29026, 29031)),
                            ("39026", range(39026, 39031)),
                            ("49026", range(49026, 49031)),
                            ("59026", range(59026, 59031)),
                            ("91000", range(91000, 91060))):
            with self.subTest(block=name):
                self.assertTrue(new.isdisjoint(block))

    def test_hypothesis_registry_has_nine_entries(self):
        self.assertEqual([h for h, _, _ in confirm.HYPOTHESES],
                         [f"H{i}" for i in range(1, 10)])
        self.assertEqual(confirm.ALPHA, 0.05)
        self.assertEqual(confirm.BOOTSTRAP_REPLICATES, 20000)


if __name__ == "__main__":
    unittest.main()
