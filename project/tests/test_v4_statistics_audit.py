"""Focused checks for the frozen-data V4 statistics correction audit."""

import json
import math
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v4_statistics_audit as audit  # noqa: E402


class StatisticsAuditTests(unittest.TestCase):
    def test_bootstrap_keeps_reference_rows_in_simulation_clusters(self):
        frame = pd.DataFrame({
            "training_seed": [2026] * 4,
            "simulation_seed": [1, 1, 2, 2],
            "delta": [0.0, 0.0, 1.0, 1.0],
        })
        result = audit.hierarchical_bootstrap(frame, seed=7, replicates=200)
        self.assertEqual(result["observed_mean_delta"], 0.5)
        self.assertEqual(result["bootstrap_mean_delta"], 0.5325)
        self.assertEqual(result["p_value_two_sided"], 0.46)

    def test_corrected_outputs_preserve_decisions_and_exclusions(self):
        out = ROOT / "results" / "v4" / "confirmatory"
        table = pd.read_csv(out / "hypothesis_table_corrected.csv")
        self.assertEqual(table.loc[table.holm_reject, "hypothesis"].tolist(),
                         ["H8", "H9"])
        self.assertEqual(table.set_index("hypothesis").loc[
            ["H3", "H4", "H5", "H8", "H9"],
            "n_excluded_pairs"].tolist(), [21, 21, 36, 26, 26])
        exclusions = json.loads((out / "exclusion_audit.json").read_text())
        self.assertTrue(exclusions["matched_variant_sets"]["H3_equals_H4"])
        self.assertTrue(exclusions["matched_variant_sets"]["H8_equals_H9"])
        expected = {
            "H1": (0.2097696875, 0.049166666666666664,
                   0.42833333333333334, 0.0627, 0.2508),
            "H2": (0.20812314583333333, 0.04958333333333333,
                   0.42833333333333334, 0.0659, 0.2508),
            "H7": (-0.06352613722947986, -0.14888930391392288,
                   -0.013248945668225584, 0.0744, 0.2508),
        }
        indexed = table.set_index("hypothesis")
        for hypothesis, values in expected.items():
            actual = indexed.loc[hypothesis]
            for field, value in zip(("bootstrap_mean_delta", "ci_lo", "ci_hi",
                                     "p_value_two_sided", "holm_adjusted_p"),
                                    values):
                self.assertTrue(math.isclose(actual[field], value,
                                             rel_tol=0, abs_tol=1e-14))


if __name__ == "__main__":
    unittest.main()
