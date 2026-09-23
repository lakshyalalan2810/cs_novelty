"""Read-only integrity checks for prepared V4 artifacts."""

import hashlib
import json
import math
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "v4"
MODELS = ROOT / "results" / "v4" / "models"
CALIBRATION = ROOT / "results" / "v4" / "calibration"
SEEDS = range(2026, 2037)


def assert_finite_numbers(test, value, path="root"):
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_numbers(test, item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite_numbers(test, item, f"{path}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        test.assertTrue(math.isfinite(value), path)


class PreparedArtifactTests(unittest.TestCase):
    def test_dataset_manifest_and_serialization(self):
        manifest_path = DATA / "dc_motor_v4_dataset_manifest.json"
        dataset_path = DATA / "dc_motor_v4_dataset.npz"
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["n_trajectories"], 120)
        self.assertEqual({key: len(value) for key, value in
                          manifest["split_run_ids"].items()},
                         {"train": 72, "validation": 24, "test": 24})
        self.assertEqual(hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                         manifest["sha256"])
        with np.load(dataset_path) as dataset:
            self.assertEqual(dataset["run_ids"].shape, (120,))
            self.assertEqual(dataset["split_run_ids_train"].shape, (72,))
            self.assertEqual(dataset["split_run_ids_validation"].shape, (24,))
            self.assertEqual(dataset["split_run_ids_test"].shape, (24,))
            for key in ("X_train", "y_train", "X_validation",
                        "y_validation", "X_test", "y_test"):
                self.assertTrue(np.isfinite(dataset[key]).all(), key)

    def test_all_training_artifacts_are_readable_and_seed_bound(self):
        summary = json.loads((MODELS / "training_summary.json").read_text())
        self.assertEqual(set(summary), {str(seed) for seed in SEEDS})
        assert_finite_numbers(self, summary)
        for seed in SEEDS:
            for kind in ("main", "aux"):
                prefix = MODELS / f"{kind}_seed_{seed}"
                paths = [prefix.with_suffix(".pt"),
                         Path(f"{prefix}_config.json"),
                         Path(f"{prefix}_metrics.json"),
                         Path(f"{prefix}_history.npz")]
                self.assertTrue(all(path.is_file() and path.stat().st_size > 0
                                    for path in paths))
                config = json.loads(paths[1].read_text())
                metrics = json.loads(paths[2].read_text())
                self.assertEqual(config["training"]["seed"], seed)
                self.assertEqual(metrics["seed"], seed)
                assert_finite_numbers(self, config)
                assert_finite_numbers(self, metrics)
                state = torch.load(paths[0], map_location="cpu",
                                   weights_only=True)
                self.assertTrue(state)
                self.assertTrue(all(torch.isfinite(tensor).all().item()
                                    for tensor in state.values()
                                    if torch.is_tensor(tensor)))
                with np.load(paths[3]) as history:
                    self.assertTrue(history.files)
                    self.assertTrue(all(np.isfinite(history[key]).all()
                                        for key in history.files))

    def test_all_v4_calibrations_are_readable_finite_and_seed_bound(self):
        summary = json.loads(
            (CALIBRATION / "calibration_summary.json").read_text())
        self.assertEqual(set(summary), {str(seed) for seed in SEEDS})
        assert_finite_numbers(self, summary)
        for seed in SEEDS:
            path = CALIBRATION / f"sensor_seed{seed}_v4_calibration.json"
            self.assertTrue(path.is_file() and path.stat().st_size > 0)
            document = json.loads(path.read_text())
            self.assertEqual(document["training_seed"], seed)
            self.assertEqual(document["pair_id"], f"P{seed}")
            self.assertEqual(document["computed_values"], summary[str(seed)])
            assert_finite_numbers(self, document)

    def test_baseline_calibration_is_readable_and_finite(self):
        path = CALIBRATION / "baselines_v4_calibration.json"
        self.assertTrue(path.is_file() and path.stat().st_size > 0)
        assert_finite_numbers(self, json.loads(path.read_text()))


if __name__ == "__main__":
    unittest.main()
