"""Fast, read-only checks for public-facing context and frozen V4 authority."""

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "project"
CONTEXT = ROOT / "context"
TITLE = "Disturbance-Aware Witness Gating for Reliability-Aware LSTM-MPC of a Nonlinear DC Motor"


class ContextConsistencyTests(unittest.TestCase):
    def test_required_context_and_title(self):
        required = (
            "START_HERE.md", "PROJECT_SNAPSHOT.md", "ARCHITECTURE.md",
            "CODE_CONNECTIONS.md", "EXPERIMENTS.md", "V4_CONFIRMATORY.md",
            "STATISTICS.md", "RESULTS_AUTHORITY.md", "PROVENANCE.md",
            "OPEN_QUESTIONS.md", "AI_HANDOFF.md", "DO_NOT_TOUCH.md",
            "context_manifest.json", "file_relationships.json",
        )
        self.assertTrue(all((CONTEXT / name).is_file() for name in required))
        for path in (ROOT / "README.md", CONTEXT / "START_HERE.md",
                     PROJECT / "paper" / "main.tex"):
            self.assertIn(TITLE, path.read_text(encoding="utf-8"))

    def test_frozen_scope_and_decisions(self):
        summary = json.loads((PROJECT / "results" / "v4" / "confirmatory" /
                              "final_summary_corrected.json").read_text())
        self.assertEqual(summary["scope"]["executed_h1_h9_core_cells"], 12200)
        self.assertEqual(summary["scope"]["deferred_umbrella_cells"], 257260)
        self.assertEqual(summary["holm_survivors"], ["H8", "H9"])
        decisions = {row["id"]: row["decision"] for row in summary["hypotheses"]}
        self.assertTrue(all(decisions[f"H{i}"] == "do-not-reject"
                            for i in range(1, 8)))
        self.assertEqual([decisions["H8"], decisions["H9"]], ["reject", "reject"])
        h8 = next(row for row in summary["hypotheses"] if row["id"] == "H8")
        self.assertEqual((h8["actual_pairs"], h8["excluded_pairs"]), (124, 26))

    def test_historical_statuses_and_manifest_bytes(self):
        summary = json.loads((PROJECT / "results" / "v4" / "confirmatory" /
                              "final_summary_corrected.json").read_text())
        statuses = summary["historical_verifiers"]
        self.assertIn("NO_GO", statuses["C4_development_negative_ablation"])
        self.assertIn("BASELINE_ONLY", statuses["EKF_baseline"])
        self.assertIn("TRAINING-SEED-SENSITIVE", statuses["training_seed_robustness"])
        manifest = PROJECT / "results" / "v4" / "confirmatory" / "result_manifest.json"
        manifest_bytes = manifest.read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(hashlib.sha256(manifest_bytes).hexdigest(),
                         "0591d54f1e5c45fbb7c5a1910b6cd01a045424848a81ae33d9b477f021ed522e")


if __name__ == "__main__":
    unittest.main()
