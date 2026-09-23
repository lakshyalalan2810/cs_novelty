"""Tests for V4 protocols: grids, fault math, stats, PI engine smoke.

No native MPC solves here (this container aborts in scipy SLSQP via OMP
Error #15); MPC-controller cells are covered by dry-run plan tests, and
the engine rollout is smoke-tested with PI (<=2 seeds, short horizon).
"""

import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v4_analysis as analysis
import v4_confirm_analysis as confirm
import v4_protocol_faultfree as faultfree
import v4_protocol_core as core
import v4_protocol_common as common
import v4_protocol_recovery as recovery
import v4_protocol_robustness as robustness
import v4_protocol_sweep as sweep
import v4_protocol_timing as timing
from v4_closed_loop import (
    PARAM_PRESETS,
    Fault,
    RunConfig,
    apply_param_preset,
    reference_profile,
    run_v4_cell,
    sigma_to_rad_s,
)
from v4_protocol_common import (
    C3_LEGACY_SEEDS,
    FAULTFREE_SEEDS,
    ROBUSTNESS_SEEDS,
    SEVERITY_SEEDS,
    SMOKE_SEEDS,
    SWEEP_SEEDS,
    TRAINING_SEEDS,
    training_seeds_for,
)


class PlanCountTests(unittest.TestCase):
    def test_faultfree_grid(self):
        cells = faultfree.plan()
        n_full = (len(faultfree.CONTROLLERS) - 1) * 11 * 200 * 4
        self.assertEqual(len(cells), n_full + 3 * 200 * 4)
        self.assertTrue({cell.simulation_seed for cell in cells}
                        <= set(FAULTFREE_SEEDS))
        c3_seeds = {cell.training_seed for cell in cells
                    if cell.controller == "C3"}
        self.assertEqual(c3_seeds, set(C3_LEGACY_SEEDS))
        self.assertTrue(all(cell.fault.kind == "none" for cell in cells))
        self.assertTrue(all(cell.fault.onset_s == float("inf")
                            for cell in cells))

    def test_sweep_grid(self):
        cells = sweep.plan()
        n_faults = 6 + 5 + 5 + 4 + 2
        main = sum(len(training_seeds_for(controller)) * len(SWEEP_SEEDS)
                   * n_faults for controller in sweep.CONTROLLERS)
        det = sum(len(training_seeds_for(controller)) * 20 * 4 * 3
                  for controller in sweep.CONTROLLERS)
        self.assertEqual(len(cells), main + det)
        det_cells = [cell for cell in cells if cell.threshold_scale != 1.0]
        self.assertEqual(len(det_cells), det)
        self.assertTrue({cell.threshold_scale for cell in det_cells}
                        == {0.5, 0.75, 1.5, 2.0})

    def test_recovery_grid(self):
        cells = recovery.plan()
        expected = sum(len(training_seeds_for(controller)) * 100 * 3
                       for controller in recovery.CONTROLLERS)
        self.assertEqual(len(cells), expected)
        self.assertTrue(all(cell.duration == 10.0 for cell in cells))

    def test_robustness_grid(self):
        n_corners = len(robustness.corners())
        self.assertEqual(n_corners, 1 + 3 + 2 + 3 + 7)
        cells = robustness.plan()
        expected = sum(len(training_seeds_for(controller))
                       * len(ROBUSTNESS_SEEDS) * n_corners * 2
                       for controller in robustness.CONTROLLERS)
        self.assertEqual(len(cells), expected)

    def test_timing_grid(self):
        cells = timing.plan()
        expected = sum(len(training_seeds_for(controller)) * 5 * 2
                       for controller in timing.CONTROLLERS)
        self.assertEqual(len(cells), expected)
        self.assertEqual({cell.control_stride for cell in cells}, {5, 10})


class FaultMathTests(unittest.TestCase):
    def test_sigma_conversion(self):
        self.assertEqual(sigma_to_rad_s(8.0), 2.0)
        self.assertEqual(sigma_to_rad_s(0.5), 0.125)

    def test_bias_window(self):
        fault = Fault(kind="bias", onset_s=2.0, end_s=4.0,
                      magnitude_rad_s=2.0)
        self.assertEqual(fault.apply_speed(35.0, 1.0), 35.0)
        self.assertEqual(fault.apply_speed(35.0, 2.0), 37.0)
        self.assertEqual(fault.apply_speed(35.0, 3.99), 37.0)
        self.assertEqual(fault.apply_speed(35.0, 4.0), 35.0)

    def test_dropout_and_drift(self):
        dropout = Fault(kind="dropout", onset_s=2.0, dropout_duration_s=0.5)
        self.assertEqual(dropout.apply_speed(35.0, 2.2), 0.0)
        self.assertEqual(dropout.apply_speed(35.0, 2.6), 35.0)
        drift = Fault(kind="drift", onset_s=2.0, end_s=6.0,
                      drift_rate_rad_s2=1.0)
        self.assertEqual(drift.apply_speed(35.0, 1.0), 35.0)
        self.assertEqual(drift.apply_speed(35.0, 4.0), 37.0)
        self.assertEqual(drift.apply_speed(35.0, 6.0), 39.0)
        self.assertEqual(drift.apply_speed(35.0, 8.0), 39.0)

    def test_load_and_window(self):
        fault = Fault(kind="load", onset_s=3.0, end_s=None,
                      load_step_Nm=0.15)
        self.assertEqual(fault.load_at(2.0), 0.03)
        self.assertEqual(fault.load_at(3.0), 0.15)
        time = np.arange(0.0, 6.0, 0.01)
        self.assertEqual(fault.event_window(time).sum(), 300)

    def test_none_fault_is_inert(self):
        fault = Fault(kind="none", onset_s=float("inf"), end_s=None)
        self.assertEqual(fault.apply_speed(35.0, 3.0), 35.0)
        self.assertEqual(fault.load_at(3.0), 0.03)


class ReferenceAndParamTests(unittest.TestCase):
    def test_reference_profiles(self):
        time = np.arange(0.0, 6.0, 0.01)
        self.assertTrue((reference_profile("nominal", time) == 35.0).all())
        step = reference_profile("step", time)
        self.assertEqual(step[0], 20.0)
        self.assertEqual(step[-1], 40.0)
        wide = reference_profile("wide", time)
        self.assertEqual(set(np.unique(wide)), {10.0, 50.0, 90.0})
        with self.assertRaises(SystemExit):
            reference_profile("nope", time)

    def test_param_presets(self):
        from motor_model import DCMotorParams
        nominal = DCMotorParams()
        same = apply_param_preset(nominal, "nominal")
        self.assertEqual(same, nominal)
        up30 = apply_param_preset(nominal, "uniform_p30")
        self.assertAlmostEqual(up30.resistance, 2.0 * 1.3)
        self.assertAlmostEqual(up30.inertia, 0.01 * 1.3)
        self.assertAlmostEqual(
            up30.friction_smoothing_speed,
            nominal.friction_smoothing_speed)
        shifted = apply_param_preset(nominal, "frozen_shifted")
        self.assertAlmostEqual(shifted.resistance, 2.4)
        self.assertAlmostEqual(shifted.torque_constant, 0.085)
        with self.assertRaises(SystemExit):
            apply_param_preset(nominal, "nope")


class StatsOracleTests(unittest.TestCase):
    def test_wilson_matches_frozen_values(self):
        lo, hi = analysis.wilson_ci(11, 15)
        self.assertAlmostEqual(lo, 0.4805, places=4)
        self.assertAlmostEqual(hi, 0.8910, places=4)

    def test_holm(self):
        # Ordered 0.01/0.03/0.04: 0.01 <= .05/3 rejects, but 0.03 > .05/2
        # stops the step-down; only the first hypothesis rejects.
        decisions = analysis.holm([0.01, 0.04, 0.03], alpha=0.05)
        self.assertEqual([decision["reject"] for decision in decisions],
                         [True, False, False])
        adjusted = [decision["holm_adjusted_p"] for decision in decisions]
        self.assertAlmostEqual(adjusted[0], 0.03)
        self.assertAlmostEqual(adjusted[1], 0.06)
        self.assertAlmostEqual(adjusted[2], 0.06)
        decisions = analysis.holm([0.01, 0.02, 0.03], alpha=0.05)
        self.assertTrue(all(decision["reject"] for decision in decisions))

    def test_cohens_dz(self):
        deltas = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(
            analysis.cohens_dz(deltas),
            float(np.mean(deltas) / np.std(deltas, ddof=1)))
        self.assertTrue(np.isnan(analysis.cohens_dz(np.ones(5))))

    def test_hierarchical_bootstrap_deterministic_and_degenerate(self):
        frame = pd.DataFrame({
            "training_seed": [2026, 2026, 2027, 2027],
            "simulation_seed": [1, 2, 1, 2],
            "delta": [1.0, 2.0, 3.0, 4.0]})
        first = analysis.hierarchical_bootstrap_mean(frame, "delta", 500, 7)
        second = analysis.hierarchical_bootstrap_mean(frame, "delta", 500, 7)
        self.assertEqual(first["ci_lo"], second["ci_lo"])
        const = frame.copy()
        const["delta"] = 2.5
        degenerate = analysis.hierarchical_bootstrap_mean(
            const, "delta", 200, 7)
        self.assertEqual(
            (degenerate["ci_lo"], degenerate["ci_hi"]), (2.5, 2.5))

    def test_mdm_interpolation(self):
        result = analysis.minimum_detectable_magnitude(
            [1.0, 2.0, 4.0, 8.0], [0.2, 0.7, 1.0, 1.0], target=0.9)
        self.assertEqual(result["flag"], "interpolated")
        # log2-linear between (2, 0.7) and (4, 1.0) at 0.9.
        expected = 2.0 ** (1.0 + (0.9 - 0.7) / 0.3)
        self.assertAlmostEqual(result["mdm"], expected)
        self.assertEqual(
            analysis.minimum_detectable_magnitude(
                [1.0, 2.0], [0.1, 0.2])["flag"], "above_grid")
        self.assertEqual(
            analysis.minimum_detectable_magnitude(
                [1.0, 2.0], [0.95, 1.0])["flag"], "below_grid")

    def test_det_table_sorted(self):
        table = analysis.det_table([0.1, 0.01], [0.2, 0.5], ["a", "b"])
        self.assertEqual(table["false_alarm_rate"].tolist(), [0.01, 0.1])
        self.assertEqual(table["detection_rate"].tolist(), [0.5, 0.8])


class ConfirmPairingTests(unittest.TestCase):
    def _synthetic_inputs(self):
        trains = [2026, 2027, 2028]
        sims = [71000, 71001]
        faultfree_rows = []
        for train in trains:
            for sim in sims:
                for controller, entries, rmse in (
                        ("C3", 1, 2.0), ("V4_full_aux", 0, 1.0),
                        ("V4_full_ekf", 0, 1.1), ("B", 0, 1.0)):
                    faultfree_rows.append({
                        "controller": controller, "training_seed": train,
                        "simulation_seed": sim, "reference": "nominal",
                        "reliability_entries": entries, "overall_rmse": rmse})
        faultfree = pd.DataFrame(faultfree_rows)
        sweep_rows = []
        for train in trains:
            for sim in sims:
                # One pre-latched pair (excluded from clean-pair hypotheses).
                pre = 1 if (train, sim) == (2026, 71000) else 0
                for controller in ("C3", "V4_full_aux", "V4_full_ekf", "B"):
                    detected = controller != "C3"
                    sweep_rows.append({
                        "controller": controller, "training_seed": train,
                        "simulation_seed": sim, "fault_kind": "bias",
                        "fault_magnitude_sigma": 2.0,
                        "sensor_fault_detected": detected,
                        "fault_window_rmse": 0.5 if detected else 2.0,
                        "pre_event_reliability_entries": pre,
                        "post_event_reliability_entries": int(detected),
                        "load_step_Nm": 0.0})
                    sweep_rows.append({
                        "controller": controller, "training_seed": train,
                        "simulation_seed": sim, "fault_kind": "bias",
                        "fault_magnitude_sigma": 8.0,
                        "sensor_fault_detected": True,
                        "fault_window_rmse": (0.4 if controller == "V4_full_aux"
                                              else 2.0),
                        "pre_event_reliability_entries": pre,
                        "post_event_reliability_entries": 1,
                        "load_step_Nm": 0.0})
                    sweep_rows.append({
                        "controller": controller, "training_seed": train,
                        "simulation_seed": sim, "fault_kind": "load",
                        "fault_magnitude_sigma": np.nan,
                        "sensor_fault_detected": False,
                        "fault_window_rmse": 1.0,
                        "pre_event_reliability_entries": pre,
                        "post_event_reliability_entries": (
                            1 if controller == "C3" else 0),
                        "load_step_Nm": 0.15})
        sweep = pd.DataFrame(sweep_rows)
        recovery_rows = []
        for train in trains:
            for sim in sims:
                for controller in ("C3", "V4_full_aux"):
                    recovery_rows.append({
                        "controller": controller, "training_seed": train,
                        "simulation_seed": sim, "fault_kind": "bias",
                        "pre_event_reliability_entries": 0,
                        "recovery_latency_s": (
                            1.0 if controller == "V4_full_aux" else np.nan)})
        recovery = pd.DataFrame(recovery_rows)
        return faultfree, sweep, recovery

    def test_c3_comparisons_explicitly_use_common_seed_population(self):
        faultfree, sweep, recovery = self._synthetic_inputs()
        extra = faultfree[faultfree["controller"] != "C3"].copy()
        extra["training_seed"] = 2029
        frames = confirm.hypothesis_frames(
            pd.concat([faultfree, extra], ignore_index=True), sweep, recovery)
        self.assertEqual(set(frames["H1"]["training_seed"]),
                         set(C3_LEGACY_SEEDS))
        self.assertEqual(set(frames["H7"]["training_seed"]),
                         set(C3_LEGACY_SEEDS))

    def test_hypothesis_signs_and_holm_integration(self):
        faultfree, sweep, recovery = self._synthetic_inputs()
        frames = confirm.hypothesis_frames(faultfree, sweep, recovery)
        self.assertEqual(set(frames), {f"H{i}" for i in range(1, 10)})
        # Constructed so every paired delta is positive.
        for hypothesis, frame in frames.items():
            with self.subTest(hypothesis=hypothesis):
                self.assertTrue((frame["delta"] > 0).all())
        # The pre-latched pair is excluded from clean-pair hypotheses.
        self.assertEqual(frames["H3"].attrs["n_excluded_pairs"], 1)
        self.assertEqual(len(frames["H3"]), 5)
        table = confirm.evaluate(frames, 500)
        self.assertEqual(len(table), 9)
        self.assertTrue((table["holm_adjusted_p"] >= 0).all())
        self.assertEqual(
            table[table["hypothesis"] == "H8"]["n_excluded_pairs"].iloc[0], 1)

    def test_unpaired_cells_fail_loud(self):
        faultfree, _, _ = self._synthetic_inputs()
        faultfree = faultfree[faultfree["controller"] != "C3"]
        with self.assertRaises(SystemExit):
            confirm.paired_deltas(faultfree, "C3", "V4_full_aux",
                                  "reliability_entries",
                                  ["training_seed", "simulation_seed"])


class PIEngineSmokeTests(unittest.TestCase):
    def test_pi_clean_and_biased_runs(self):
        clean = RunConfig(controller="PI", simulation_seed=SMOKE_SEEDS[0],
                          reference="nominal", duration=2.0,
                          fault=Fault(kind="none", onset_s=float("inf"),
                                      end_s=None),
                          record_traces=True)
        row, events, traces = run_v4_cell(clean)
        self.assertTrue(row["run_complete"])
        self.assertTrue(np.isfinite(row["overall_rmse"]))
        self.assertEqual(row["reliability_entries"], 0)
        self.assertEqual(events, [])
        # Clean measurement offset is pure noise (mean ~0).
        offset = traces["measured"] - traces["true_speed"]
        self.assertAlmostEqual(float(np.mean(offset)), 0.0, delta=0.1)
        biased = RunConfig(
            controller="PI", simulation_seed=SMOKE_SEEDS[1],
            reference="nominal",
            duration=2.0,
            fault=Fault(kind="bias", onset_s=1.0, end_s=2.0,
                        magnitude_rad_s=2.0, magnitude_sigma=8.0),
            record_traces=True)
        biased_row, _, biased_traces = run_v4_cell(biased)
        window = (biased_traces["time_s"] >= 1.0) & (
            biased_traces["time_s"] < 2.0)
        injected = (biased_traces["measured"][window]
                    - biased_traces["true_speed"][window])
        self.assertAlmostEqual(float(np.mean(injected)), 2.0, delta=0.15)
        pre = biased_traces["time_s"] < 1.0
        pre_offset = (biased_traces["measured"][pre]
                      - biased_traces["true_speed"][pre])
        self.assertAlmostEqual(float(np.mean(pre_offset)), 0.0, delta=0.1)

    def test_pi_with_delay_and_quantization(self):
        config = RunConfig(controller="PI", simulation_seed=SMOKE_SEEDS[1],
                           reference="step", duration=2.0,
                           fault=Fault(kind="none", onset_s=float("inf"),
                                       end_s=None),
                           current_noise_std=0.05,
                           current_quantization_a=0.01, sample_delay=2)
        row, _, _ = run_v4_cell(config)
        self.assertTrue(row["run_complete"])
        self.assertTrue(np.isfinite(row["overall_rmse"]))
        self.assertEqual(row["sample_delay"], 2)


class EKFWitnessRegressionTests(unittest.TestCase):
    def test_witness_initializes_on_first_post_warmup_use(self):
        config = RunConfig(
            controller="V4_full_ekf", training_seed=2026,
            simulation_seed=SMOKE_SEEDS[0], duration=0.25,
            fault=Fault(kind="none", onset_s=float("inf"), end_s=None))
        row, _, _ = run_v4_cell(config)
        self.assertEqual(row["observer_failures"], 0)


class CorePlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = core.hypothesis_cells()
        cls.cells = core.core_cells(cls.frames)

    def test_exact_hypothesis_and_union_counts(self):
        self.assertEqual({key: len(value) for key, value in self.frames.items()},
                         {"H1": 4800, "H2": 4800, "H3": 300,
                          "H4": 300, "H5": 600, "H6": 1100,
                          "H7": 7200, "H8": 300, "H9": 300})
        self.assertEqual(len(self.cells), 12200)

    def test_seed_contract_and_no_duplicate_keys(self):
        keys = [common.scientific_key(cell) for cell in self.cells]
        self.assertEqual(len(keys), len(set(keys)))
        for hypothesis, cells in self.frames.items():
            expected = (set(TRAINING_SEEDS) if hypothesis == "H6"
                        else set(C3_LEGACY_SEEDS))
            self.assertEqual({cell.training_seed for cell in cells}, expected)
        self.assertTrue(all(
            cell.simulation_seed in (set(FAULTFREE_SEEDS)
                                     | set(SEVERITY_SEEDS))
            for cell in self.cells))

    def test_dry_run_never_executes(self):
        plan = {
                                   "hypothesis_cell_counts": {},
                                   "union_count": 0,
                                   "original_umbrella_count": 268910,
                                   "original_umbrella_required_intersection": 0,
                                   "original_umbrella_deferred": 268910,
                                   "required_missing_from_original_umbrella": 0,
                               }
        with mock.patch.object(core, "validated_frozen_plan",
                               return_value=(plan, [], "plan-hash")), \
             mock.patch.object(core, "execute") as execute_mock:
            core.main(["--dry-run"])
        execute_mock.assert_not_called()


class CheckpointResumeTests(unittest.TestCase):
    @staticmethod
    def _result(cell):
        return ({"controller": cell.controller,
                 "training_seed": cell.training_seed,
                 "simulation_seed": cell.simulation_seed,
                 "overall_rmse": float(cell.simulation_seed)},
                [{"time_s": 1.0, "event": "test"}], None)

    def test_resume_matches_uninterrupted_and_skips_completed(self):
        cells = [RunConfig("PI", simulation_seed=seed,
                           reference="nominal", duration=0.01)
                 for seed in SMOKE_SEEDS]
        calls = []

        def interrupted(batch, workers=1):
            calls.extend(cell.simulation_seed for cell in batch)
            if len(calls) > 1:
                raise RuntimeError("simulated interruption")
            return [self._result(cell) for cell in batch]

        with tempfile.TemporaryDirectory() as partial_dir, \
             tempfile.TemporaryDirectory() as clean_dir, \
             mock.patch.object(common, "_provenance_fingerprint",
                               return_value="test-provenance"):
            with mock.patch.object(common, "run_batch", side_effect=interrupted):
                with self.assertRaises(RuntimeError):
                    common.execute("test", Path(partial_dir), cells,
                                   checkpoint_every=1, progress_interval_s=0)
            resumed_calls = []

            def resumed(batch, workers=1):
                resumed_calls.extend(cell.simulation_seed for cell in batch)
                return [self._result(cell) for cell in batch]

            with mock.patch.object(common, "run_batch", side_effect=resumed):
                common.execute("test", Path(partial_dir), cells,
                               checkpoint_every=1, progress_interval_s=0)
            self.assertEqual(resumed_calls, [SMOKE_SEEDS[1]])
            with mock.patch.object(common, "run_batch",
                                   side_effect=lambda batch, workers=1: [
                                       self._result(cell) for cell in batch]):
                common.execute("test", Path(clean_dir), cells,
                               checkpoint_every=1, progress_interval_s=0)
            for filename in ("runs.csv", "events.csv"):
                self.assertEqual((Path(partial_dir) / filename).read_bytes(),
                                 (Path(clean_dir) / filename).read_bytes())

    def test_core_runner_never_mutates_frozen_plan(self):
        cells = [RunConfig("PI", simulation_seed=seed,
                           reference="nominal", duration=0.01)
                 for seed in SMOKE_SEEDS]
        plan = {
            "generated_utc": "frozen",
            "hypothesis_cell_counts": {},
            "union_count": len(cells),
            "original_umbrella_count": 268910,
            "original_umbrella_required_intersection": len(cells),
            "original_umbrella_deferred": 268908,
            "required_missing_from_original_umbrella": 0,
            "cells": [common.scientific_key(cell) for cell in cells],
        }
        calls = []

        def interrupted(batch, workers=1):
            calls.extend(cell.simulation_seed for cell in batch)
            if len(calls) > 1:
                raise RuntimeError("simulated interruption")
            return [self._result(cell) for cell in batch]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            hashes_path = root / "hashes.json"
            status_path = root / "status.json"
            plan_path.write_text(json.dumps(plan, indent=2) + "\n")
            frozen_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
            hashes_path.write_text(json.dumps({"files": {
                core.PLAN_RELATIVE_PATH: frozen_hash}}))

            def generated_plan():
                return ({**plan, "generated_utc": "runtime"}, cells)

            def execute_smoke(name, out_dir, selected, workers=1):
                if selected:
                    common.execute(name, out_dir, selected, workers,
                                   checkpoint_every=1,
                                   progress_interval_s=0)

            patches = (
                mock.patch.object(core, "PROJECT", root),
                mock.patch.object(core, "PLAN_PATH", plan_path),
                mock.patch.object(core, "HASHES_PATH", hashes_path),
                mock.patch.object(core, "STATUS_PATH", status_path),
                mock.patch.object(core, "build_plan",
                                  side_effect=generated_plan),
                mock.patch.object(core, "execute",
                                  side_effect=execute_smoke),
                mock.patch.object(common, "_provenance_fingerprint",
                                  return_value="test-provenance"),
            )
            with patches[0], patches[1], patches[2], patches[3], \
                 patches[4], patches[5], patches[6]:
                with mock.patch.object(common, "run_batch",
                                       side_effect=interrupted):
                    with self.assertRaises(RuntimeError):
                        core.main(["--workers", "1"])
                self.assertEqual(hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                                 frozen_hash)
                self.assertEqual(json.loads(status_path.read_text())["state"],
                                 "interrupted")

                with mock.patch.object(common, "run_batch",
                                       side_effect=lambda batch, workers=1: [
                                           self._result(cell) for cell in batch]):
                    core.main(["--workers", "1"])

            self.assertEqual(hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                             frozen_hash)
            self.assertEqual(json.loads(status_path.read_text())["state"],
                             "complete")
            checkpoint = root / "results" / "v4" / "faultfree" / "checkpoint.sqlite3"
            connection = sqlite3.connect(checkpoint)
            try:
                self.assertEqual(connection.execute(
                    "SELECT COUNT(*) FROM results").fetchone()[0], 2)
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
