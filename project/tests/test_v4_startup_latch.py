"""Tests for the V4 post-hoc startup-latch decomposition."""

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import v4_analyze_startup_latch as posthoc


class WilsonCITests(unittest.TestCase):
    def test_matches_frozen_documented_intervals(self):
        # Independent oracles from FINAL_ROBUSTNESS_AND_SEVERITY_STUDY.md:
        # pooled 11/15 -> [0.4805, 0.8910]; 0/15 -> [0.0000, 0.2039].
        lo, hi = posthoc.wilson_ci(11, 15)
        self.assertAlmostEqual(lo, 0.4805, places=4)
        self.assertAlmostEqual(hi, 0.8910, places=4)
        lo, hi = posthoc.wilson_ci(0, 15)
        self.assertAlmostEqual(lo, 0.0, places=4)
        self.assertAlmostEqual(hi, 0.2039, places=4)

    def test_degenerate_cases(self):
        lo, hi = posthoc.wilson_ci(0, 0)
        self.assertTrue(pd.isna(lo) and pd.isna(hi))
        lo, hi = posthoc.wilson_ci(5, 5)
        self.assertLess(lo, 1.0)
        self.assertEqual(hi, 1.0)
        with self.assertRaises(ValueError):
            posthoc.wilson_ci(3, 2)


class ClassifyRunsTests(unittest.TestCase):
    def _frame(self):
        runs = pd.DataFrame({
            "key": ["pre", "post", "none", "both"],
            "event_start_s": [2.0, 2.0, 2.0, 2.0],
            "reliability_entries": [1, 1, 0, 2],
            "post_event_reliability_entries": [0, 1, 0, 1],
        })
        entries = pd.DataFrame({
            "key": ["pre", "post", "both", "both"],
            "time_s": [0.5, 2.02, 0.6, 2.5],
            "event": ["entry", "entry", "entry", "entry"],
        })
        return runs, entries

    def test_labels(self):
        runs, entries = self._frame()
        out = posthoc.classify_runs(runs, entries, key_cols=["key"],
                                    event_start_col="event_start_s",
                                    label_cols=[])
        labels = dict(zip(out["key"], out["label"]))
        self.assertEqual(labels, {"pre": "PRE_EVENT_LATCH",
                                  "post": "POST_EVENT_ENTRY",
                                  "none": "NO_ENTRY", "both": "PRE_EVENT_LATCH"})
        both = out[out["key"] == "both"].iloc[0]
        self.assertTrue(both["pre_and_post_entry"])
        phases = dict(zip(out["key"], out["latch_phase"]))
        self.assertEqual(phases["pre"], "STARTUP")
        self.assertEqual(phases["post"], "NONE")
        self.assertEqual(phases["none"], "NONE")

    def test_mid_run_phase(self):
        runs = pd.DataFrame({"key": ["m"], "event_start_s": [3.0],
                             "reliability_entries": [1],
                             "post_event_reliability_entries": [0]})
        entries = pd.DataFrame({"key": ["m"], "time_s": [2.03],
                                "event": ["entry"]})
        out = posthoc.classify_runs(runs, entries, key_cols=["key"],
                                    event_start_col="event_start_s",
                                    label_cols=[])
        self.assertEqual(out.iloc[0]["label"], "PRE_EVENT_LATCH")
        self.assertEqual(out.iloc[0]["latch_phase"], "MID_RUN_PRE_EVENT")

    def test_count_mismatch_fails_loud(self):
        runs, entries = self._frame()
        runs.loc[runs["key"] == "pre", "reliability_entries"] = 2
        with self.assertRaises(SystemExit):
            posthoc.classify_runs(runs, entries, key_cols=["key"],
                                  event_start_col="event_start_s",
                                  label_cols=[])


class FrozenConventionTests(unittest.TestCase):
    def test_part_a_event_starts_match_frozen_v3(self):
        import evaluate_v3_closed_loop as frozen_v3
        for scenario, onset in posthoc.PART_A_EVENT_START.items():
            self.assertEqual(frozen_v3.SCENARIOS[scenario]["event_start"], onset)

    def test_envelope_rules_reproduce_frozen_row(self):
        # bias_10pct frozen inputs -> LIMITED under the frozen rules.
        label, sup, lim = posthoc.apply_envelope_rules(
            0.733333, [0.8, 0.4, 1.0], -4.547573,
            [-4.473579, -4.557370, -4.611770], 15, 3, 0)
        self.assertEqual(label, "LIMITED")
        self.assertFalse(sup["all_training_seed_detection_probability_ge_threshold"])
        self.assertTrue(all(lim.values()))
        # Conditional-on-clean inputs for the same row -> SUPPORTED (POST-HOC).
        label, sup, _ = posthoc.apply_envelope_rules(
            1.0, [1.0, 1.0, 1.0], -4.547573,
            [-4.473579, -4.557370, -4.611770], 15, 3, 0)
        self.assertEqual(label, "SUPPORTED")
        self.assertTrue(all(sup.values()))


class PosthocOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outdir = ROOT / "results" / "v4" / "posthoc"
        for name in ("partA_run_classification.csv",
                     "partB_run_classification.csv", "per_seed_summary.csv",
                     "envelope_relabelled_posthoc.csv", "posthoc_summary.json"):
            if not (cls.outdir / name).is_file():
                raise unittest.SkipTest(f"missing {name}; run "
                                        "scripts/v4_analyze_startup_latch.py")

    def test_known_bias_latch_pairs(self):
        part_b = pd.read_csv(self.outdir / "partB_run_classification.csv")
        bias = part_b[part_b["family"] == "bias"]
        latched = set(zip(bias[bias["label"] == "PRE_EVENT_LATCH"]["training_seed"],
                          bias[bias["label"] == "PRE_EVENT_LATCH"]["simulation_seed"]))
        self.assertEqual(latched, {(2026, 59030), (2027, 59026),
                                   (2027, 59027), (2027, 59030)})
        clean = bias[bias["label"] != "PRE_EVENT_LATCH"]
        self.assertEqual(len(clean), 44)  # 11 pairs x 4 levels
        self.assertTrue((clean["label"] == "POST_EVENT_ENTRY").all())

    def test_preregistered_labels_unchanged(self):
        relabeled = pd.read_csv(self.outdir / "envelope_relabelled_posthoc.csv")
        envelope = pd.read_csv(ROOT / "results/final_robustness"
                               / "operating_envelope_classification.csv")
        merged = relabeled.merge(envelope[["condition_label", "classification"]],
                                 on="condition_label")
        self.assertTrue((merged["preregistered_classification"]
                         == merged["classification"]).all())
        self.assertTrue((relabeled["analysis"]
                         == "POST-HOC (conditional-on-clean-start)").all())


if __name__ == "__main__":
    unittest.main()
