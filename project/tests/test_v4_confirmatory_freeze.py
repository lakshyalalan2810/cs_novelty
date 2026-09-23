"""Read-only checks for the frozen V4 H1-H9 result manifest."""

import hashlib
import json
import sqlite3
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "v4" / "confirmatory" / "result_manifest.json"
MANIFEST_SHA256 = "0591d54f1e5c45fbb7c5a1910b6cd01a045424848a81ae33d9b477f021ed522e"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ConfirmatoryFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST.read_text())

    def test_manifest_and_all_bound_artifacts_are_unchanged(self):
        self.assertEqual(sha256(MANIFEST), MANIFEST_SHA256)
        for relative, expected in self.manifest["artifacts"].items():
            with self.subTest(path=relative):
                self.assertEqual(sha256(ROOT / relative), expected)
        correction = self.manifest["statistical_correction"]
        self.assertEqual(sha256(ROOT / correction["source_path"]),
                         correction["source_sha256"])
        self.assertEqual(
            sha256(ROOT / correction["correction_record_path"]),
            correction["correction_record_sha256"])
        for relative, expected in correction["corrected_artifacts"].items():
            with self.subTest(path=relative):
                self.assertEqual(sha256(ROOT / relative), expected)

    def test_provenance_and_analysis_settings(self):
        provenance = self.manifest["administrative_provenance"]
        self.assertEqual(
            provenance["administratively_refrozen_plan_sha256"],
            sha256(ROOT / "results/v4/prereg/h1_h9_execution_plan.json"))
        self.assertEqual(provenance["core_runner_sha256"],
                         sha256(ROOT / "scripts/v4_protocol_core.py"))
        self.assertEqual(provenance["preexecution_manifest_sha256"],
                         sha256(ROOT / "results/v4/prereg/preexecution_hashes.json"))
        self.assertEqual(self.manifest["analysis"]["bootstrap_replicates"],
                         20000)
        self.assertEqual(self.manifest["analysis"]["family_alpha"], 0.05)
        self.assertEqual(self.manifest["analysis"]["run_count"], 1)
        self.assertEqual(
            self.manifest["statistical_correction"]["scientific_simulations_rerun"],
            0)

    def test_final_summary_scope_and_holm_survivors(self):
        summary = json.loads((ROOT / self.manifest["result_report"]["path"])
                             .read_text())
        self.assertEqual(summary["scope"]["executed_h1_h9_core_cells"], 12200)
        self.assertEqual(summary["scope"]["deferred_umbrella_cells"], 257260)
        self.assertEqual(summary["holm_survivors"], ["H8", "H9"])
        self.assertEqual(summary["integrity"]["independent_verifier"], "PASS")

    def test_ekf_repair_provenance_counts(self):
        repaired = "39f1f08802baea34024ad3a323ae0b4a0c4095f13eb8a8e4752906073d2e6a50"
        old = "7979c98f08431d51c9828bc282f706893abee75de16dc9483d948c16e6a8eed1"
        expected = {
            "faultfree": {(repaired, 2400), (old, 7200)},
            "sweep": {(repaired, 300), (old, 1700)},
            "recovery": {(old, 600)},
        }
        for protocol, counts in expected.items():
            connection = sqlite3.connect(
                ROOT / "results" / "v4" / protocol / "checkpoint.sqlite3")
            try:
                actual = set(connection.execute(
                    "SELECT provenance, count(*) FROM results GROUP BY provenance"))
                unique = connection.execute(
                    "SELECT count(*), count(DISTINCT key) FROM results").fetchone()
            finally:
                connection.close()
            self.assertEqual(actual, counts)
            self.assertEqual(unique[0], unique[1])


if __name__ == "__main__":
    unittest.main()
