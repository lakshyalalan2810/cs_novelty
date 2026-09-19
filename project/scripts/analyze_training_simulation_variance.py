"""Retrospective training-seed x simulation-seed variance analysis.

This supplementary Part A analysis reads the completed frozen
training-seed robustness matrix and produces only new final-robustness
artifacts.  It does not train, calibrate, simulate, or modify any controller.

For each scenario, paired fault-window RMSE differences are defined as::

    Delta[t, s] = RMSE(C3_arbitration_MPC, t, s)
                - RMSE(B_plain_MPC, t, s)

Negative Delta favors C3.  A balanced additive two-way decomposition is used
descriptively:

    Delta[t, s] = mu + alpha_training[t] + beta_simulation[s] + residual[t, s]

There is one observation per training x simulation cell, so the residual term
contains training-by-simulation interaction/nonadditivity and any remaining
unexplained variation.  It is not interpreted as an independently identified
pure error term.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = PROJECT / "results" / "training_seed_robustness"
RUNS_PATH = SOURCE_ROOT / "training_seed_runs.csv"
PRESTUDY_MANIFEST = PROJECT / "results" / "final_robustness" / "prestudy_frozen_hash_manifest.json"
OUTPUT_ROOT = PROJECT / "results" / "final_robustness"
DECOMPOSITION_PATH = OUTPUT_ROOT / "variance_decomposition.csv"
SUMMARY_PATH = OUTPUT_ROOT / "variance_decomposition_summary.json"
BOOTSTRAP_PATH = OUTPUT_ROOT / "variance_bootstrap.csv"
REPORT_PATH = PROJECT / "TRAINING_SIMULATION_VARIANCE_ANALYSIS.md"

TRAINING_SEEDS = (2026, 2027, 2028)
SIMULATION_SEEDS = (49026, 49027, 49028, 49029, 49030)
CONTROLLERS = ("B_plain_MPC", "C3_arbitration_MPC")
SCENARIOS = (
    "combined_fault_load",
    "sensor_bias_5",
    "sensor_dropout",
    "load_disturbance",
    "parameter_variation",
)
EXPECTED_RUN_COUNT = 150
EXPECTED_PAIR_COUNT = len(TRAINING_SEEDS) * len(SIMULATION_SEEDS) * len(SCENARIOS)

DEFAULT_ANALYSIS_SEED = 20260917
DEFAULT_BOOTSTRAP_REPLICATES = 20_000

PAIR_BINDING_COLUMNS = (
    "main_model_sha256",
    "main_config_sha256",
    "aux_model_sha256",
    "aux_config_sha256",
    "sensor_calibration_sha256",
    "v3_calibration_sha256",
    "pair_config_sha256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT.resolve()).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(relative(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{relative(path)} must contain a JSON object")
    return payload


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def validate_frozen_source_hash() -> str:
    manifest = load_json(PRESTUDY_MANIFEST)
    if manifest.get("study") != "final_robustness_hardening":
        raise RuntimeError("prestudy final-robustness manifest belongs to another study")
    frozen = manifest.get("frozen_artifacts")
    if not isinstance(frozen, dict):
        raise RuntimeError("prestudy manifest lacks frozen_artifacts")
    key = relative(RUNS_PATH)
    entry = frozen.get(key)
    if not isinstance(entry, dict):
        raise RuntimeError(f"prestudy manifest does not bind {key}")
    expected = str(entry.get("sha256", "")).lower()
    actual = sha256(RUNS_PATH)
    if len(expected) != 64 or actual != expected:
        raise RuntimeError(f"frozen robustness matrix hash mismatch: {actual} != {expected}")
    if "bytes" in entry and RUNS_PATH.stat().st_size != int(entry["bytes"]):
        raise RuntimeError("frozen robustness matrix byte count changed")
    return actual


def validate_and_pair_matrix(runs: pd.DataFrame) -> pd.DataFrame:
    required = {
        "training_seed",
        "simulation_seed",
        "controller",
        "scenario",
        "fault_window_rmse",
    }
    missing = sorted(required - set(runs.columns))
    if missing:
        raise RuntimeError(f"training_seed_runs.csv missing required columns: {missing}")
    if len(runs) != EXPECTED_RUN_COUNT:
        raise RuntimeError(f"expected exactly {EXPECTED_RUN_COUNT} rows, found {len(runs)}")

    frame = runs.copy()
    for column in ("training_seed", "simulation_seed"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or not np.array_equal(numeric.to_numpy(), numeric.astype(int).to_numpy()):
            raise RuntimeError(f"{column} contains missing or non-integer values")
        frame[column] = numeric.astype(int)

    rmse = pd.to_numeric(frame["fault_window_rmse"], errors="coerce").to_numpy(float)
    if not np.isfinite(rmse).all():
        raise RuntimeError("fault_window_rmse contains missing or nonfinite values")
    frame["fault_window_rmse"] = rmse

    if set(frame["training_seed"]) != set(TRAINING_SEEDS):
        raise RuntimeError("training seed set is not exactly {2026,2027,2028}")
    if set(frame["simulation_seed"]) != set(SIMULATION_SEEDS):
        raise RuntimeError("simulation seed set is not exactly {49026..49030}")
    if set(frame["controller"].astype(str)) != set(CONTROLLERS):
        raise RuntimeError("controller set is not exactly B_plain_MPC/C3_arbitration_MPC")
    if set(frame["scenario"].astype(str)) != set(SCENARIOS):
        raise RuntimeError("scenario set is not the exact five-scenario robustness design")

    key_columns = ["training_seed", "scenario", "simulation_seed", "controller"]
    if frame.duplicated(key_columns).any():
        duplicate = frame.loc[frame.duplicated(key_columns, keep=False), key_columns].head(5)
        raise RuntimeError(f"duplicate primary keys detected: {duplicate.to_dict(orient='records')}")

    expected = {
        (training_seed, scenario, simulation_seed, controller)
        for training_seed in TRAINING_SEEDS
        for scenario in SCENARIOS
        for simulation_seed in SIMULATION_SEEDS
        for controller in CONTROLLERS
    }
    actual = set(map(tuple, frame[key_columns].itertuples(index=False, name=None)))
    if actual != expected:
        raise RuntimeError(
            "primary matrix does not match the exact 3x5x5x2 design: "
            f"missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}"
        )

    binding_columns = [column for column in PAIR_BINDING_COLUMNS if column in frame.columns]
    pair_rows: list[dict[str, Any]] = []
    for (training_seed, scenario, simulation_seed), group in frame.groupby(
        ["training_seed", "scenario", "simulation_seed"], sort=False
    ):
        if len(group) != 2 or set(group["controller"].astype(str)) != set(CONTROLLERS):
            raise RuntimeError(
                f"incomplete B/C3 pair for {training_seed}/{scenario}/{simulation_seed}"
            )
        for column in binding_columns:
            if group[column].nunique(dropna=False) != 1:
                raise RuntimeError(
                    f"B/C3 pair mixes {column} for {training_seed}/{scenario}/{simulation_seed}"
                )
        b = group[group["controller"] == "B_plain_MPC"].iloc[0]
        c3 = group[group["controller"] == "C3_arbitration_MPC"].iloc[0]
        pair_rows.append(
            {
                "training_seed": int(training_seed),
                "scenario": str(scenario),
                "simulation_seed": int(simulation_seed),
                "b_fault_window_rmse": float(b["fault_window_rmse"]),
                "c3_fault_window_rmse": float(c3["fault_window_rmse"]),
                "delta": float(c3["fault_window_rmse"] - b["fault_window_rmse"]),
            }
        )

    paired = pd.DataFrame(pair_rows)
    if len(paired) != EXPECTED_PAIR_COUNT:
        raise RuntimeError(f"expected exactly {EXPECTED_PAIR_COUNT} paired Delta cells, found {len(paired)}")
    for scenario in SCENARIOS:
        scenario_rows = paired[paired["scenario"] == scenario]
        if len(scenario_rows) != len(TRAINING_SEEDS) * len(SIMULATION_SEEDS):
            raise RuntimeError(f"{scenario} does not contain exactly 15 paired Delta values")
    return paired


def scenario_matrix(paired: pd.DataFrame, scenario: str) -> pd.DataFrame:
    matrix = paired[paired["scenario"] == scenario].pivot(
        index="training_seed", columns="simulation_seed", values="delta"
    )
    matrix = matrix.reindex(index=TRAINING_SEEDS, columns=SIMULATION_SEEDS)
    if matrix.isna().any().any():
        raise RuntimeError(f"{scenario} 3x5 paired Delta matrix contains a missing cell")
    return matrix.astype(float)


def decompose_scenario(matrix_frame: pd.DataFrame, scenario: str) -> tuple[dict[str, Any], dict[str, Any]]:
    values = matrix_frame.to_numpy(float)
    n_training, n_simulation = values.shape
    if (n_training, n_simulation) != (len(TRAINING_SEEDS), len(SIMULATION_SEEDS)):
        raise RuntimeError(f"{scenario} is not a 3x5 matrix")

    grand_mean = float(values.mean())
    training_means = values.mean(axis=1)
    simulation_means = values.mean(axis=0)
    training_effects = training_means - grand_mean
    simulation_effects = simulation_means - grand_mean
    residual = values - grand_mean - training_effects[:, None] - simulation_effects[None, :]

    centered = values - grand_mean
    ss_total = float(np.sum(centered**2))
    ss_training = float(n_simulation * np.sum(training_effects**2))
    ss_simulation = float(n_training * np.sum(simulation_effects**2))
    ss_residual = float(np.sum(residual**2))
    if not np.isclose(
        ss_total,
        ss_training + ss_simulation + ss_residual,
        rtol=1e-11,
        atol=1e-11,
    ):
        raise RuntimeError(f"{scenario} two-way SS decomposition does not close")

    if ss_total > 0.0:
        training_prop = ss_training / ss_total
        simulation_prop = ss_simulation / ss_total
        residual_prop = ss_residual / ss_total
    else:
        training_prop = simulation_prop = residual_prop = 0.0

    favorable = values < 0.0
    unfavorable = values > 0.0
    zero = values == 0.0
    training_favorable_counts = favorable.sum(axis=1).astype(int)
    simulation_favorable_counts = favorable.sum(axis=0).astype(int)

    row: dict[str, Any] = {
        "scenario": scenario,
        "paired_delta_count": int(values.size),
        "grand_mean_delta": grand_mean,
        "delta_min": float(values.min()),
        "delta_max": float(values.max()),
        "delta_range": float(np.ptp(values)),
        "ss_total_centered": ss_total,
        "ss_training_seed": ss_training,
        "ss_simulation_seed": ss_simulation,
        "ss_residual_interaction_unexplained": ss_residual,
        "prop_training_seed": float(training_prop),
        "prop_simulation_seed": float(simulation_prop),
        "prop_residual_interaction_unexplained": float(residual_prop),
        "training_seed_effect_range": float(np.ptp(training_effects)),
        "simulation_seed_effect_range": float(np.ptp(simulation_effects)),
        "residual_range": float(np.ptp(residual)),
        "residual_max_abs": float(np.max(np.abs(residual))),
        "favorable_delta_count": int(favorable.sum()),
        "unfavorable_delta_count": int(unfavorable.sum()),
        "zero_delta_count": int(zero.sum()),
        "favorable_delta_fraction": float(favorable.mean()),
        "favorable_training_seed_mean_count": int(np.sum(training_means < 0.0)),
        "favorable_simulation_seed_mean_count": int(np.sum(simulation_means < 0.0)),
    }
    for index, training_seed in enumerate(TRAINING_SEEDS):
        row[f"training_seed_{training_seed}_mean_delta"] = float(training_means[index])
        row[f"training_seed_{training_seed}_effect"] = float(training_effects[index])
        row[f"training_seed_{training_seed}_favorable_simulation_count"] = int(
            training_favorable_counts[index]
        )
    for index, simulation_seed in enumerate(SIMULATION_SEEDS):
        row[f"simulation_seed_{simulation_seed}_mean_delta"] = float(simulation_means[index])
        row[f"simulation_seed_{simulation_seed}_effect"] = float(simulation_effects[index])
        row[f"simulation_seed_{simulation_seed}_favorable_training_count"] = int(
            simulation_favorable_counts[index]
        )

    details = {
        "scenario": scenario,
        "grand_mean_delta": grand_mean,
        "training_seed_means": {
            str(seed): float(training_means[index]) for index, seed in enumerate(TRAINING_SEEDS)
        },
        "simulation_seed_means": {
            str(seed): float(simulation_means[index]) for index, seed in enumerate(SIMULATION_SEEDS)
        },
        "training_seed_effects": {
            str(seed): float(training_effects[index]) for index, seed in enumerate(TRAINING_SEEDS)
        },
        "simulation_seed_effects": {
            str(seed): float(simulation_effects[index]) for index, seed in enumerate(SIMULATION_SEEDS)
        },
        "sum_of_squares": {
            "total_centered": ss_total,
            "training_seed": ss_training,
            "simulation_seed": ss_simulation,
            "residual_interaction_unexplained": ss_residual,
        },
        "proportion_of_total_centered_variation": {
            "training_seed": float(training_prop),
            "simulation_seed": float(simulation_prop),
            "residual_interaction_unexplained": float(residual_prop),
        },
        "effect_ranges": {
            "training_seed": float(np.ptp(training_effects)),
            "simulation_seed": float(np.ptp(simulation_effects)),
            "residual": float(np.ptp(residual)),
        },
        "delta_sign_counts": {
            "negative_favorable": int(favorable.sum()),
            "positive_unfavorable": int(unfavorable.sum()),
            "zero_tie": int(zero.sum()),
        },
        "per_training_seed_favorable_simulation_count": {
            str(seed): int(training_favorable_counts[index])
            for index, seed in enumerate(TRAINING_SEEDS)
        },
        "per_simulation_seed_favorable_training_count": {
            str(seed): int(simulation_favorable_counts[index])
            for index, seed in enumerate(SIMULATION_SEEDS)
        },
        "delta_matrix": {
            str(training_seed): {
                str(simulation_seed): float(values[row_index, column_index])
                for column_index, simulation_seed in enumerate(SIMULATION_SEEDS)
            }
            for row_index, training_seed in enumerate(TRAINING_SEEDS)
        },
    }
    return row, details


def hierarchical_bootstrap(
    paired: pd.DataFrame,
    *,
    analysis_seed: int,
    replicates: int,
) -> pd.DataFrame:
    if replicates < 1000:
        raise ValueError("bootstrap replicate count must be at least 1000")
    rng = np.random.default_rng(analysis_seed)
    rows: list[dict[str, Any]] = []

    for scenario in SCENARIOS:
        matrix = scenario_matrix(paired, scenario).to_numpy(float)
        point_estimate = float(matrix.mean())
        bootstrap_means = np.empty(replicates, dtype=float)
        for replicate in range(replicates):
            sampled_training_indices = rng.integers(0, len(TRAINING_SEEDS), size=len(TRAINING_SEEDS))
            sampled_cluster_means = np.empty(len(TRAINING_SEEDS), dtype=float)
            for cluster_index, training_index in enumerate(sampled_training_indices):
                simulation_indices = rng.integers(
                    0, len(SIMULATION_SEEDS), size=len(SIMULATION_SEEDS)
                )
                sampled_cluster_means[cluster_index] = float(
                    matrix[int(training_index), simulation_indices].mean()
                )
            bootstrap_means[replicate] = float(sampled_cluster_means.mean())

        lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
        rows.append(
            {
                "scenario": scenario,
                "point_estimate_mean_delta": point_estimate,
                "bootstrap_mean_delta": float(bootstrap_means.mean()),
                "bootstrap_sd_delta": float(bootstrap_means.std(ddof=1)),
                "ci95_low": float(lower),
                "ci95_high": float(upper),
                "prob_mean_delta_lt_zero": float(np.mean(bootstrap_means < 0.0)),
                "analysis_rng_seed": int(analysis_seed),
                "bootstrap_replicates": int(replicates),
                "training_seed_clusters": len(TRAINING_SEEDS),
                "simulation_seeds_per_cluster": len(SIMULATION_SEEDS),
            }
        )
    return pd.DataFrame(rows)


def format_num(value: float, digits: int = 6, signed: bool = False) -> str:
    if signed:
        return f"{value:+.{digits}f}"
    return f"{value:.{digits}f}"


def write_report(
    decomposition: pd.DataFrame,
    details: list[dict[str, Any]],
    bootstrap: pd.DataFrame,
    *,
    analysis_seed: int,
    replicates: int,
    source_hash: str,
) -> None:
    by_scenario = {row["scenario"]: row for row in details}
    combined = by_scenario["combined_fault_load"]

    lines: list[str] = [
        "# Training × Simulation Variance Analysis",
        "",
        "This retrospective Part A analysis uses the completed frozen 3-training-seed × 5-simulation-seed robustness matrix. For every cell, `Delta = fault_window_RMSE(C3_arbitration_MPC) - fault_window_RMSE(B_plain_MPC)`, so negative Delta favors C3.",
        "",
        "## Design verification",
        "",
        f"The source matrix contains exactly `{EXPECTED_RUN_COUNT}` rows and yields `{EXPECTED_PAIR_COUNT}` exact B/C3 pairs: 3 training seeds × 5 simulation seeds × 5 scenarios. Every scenario therefore contains exactly 15 paired Delta values, with no missing or duplicate training×simulation combinations. The frozen source SHA-256 is `{source_hash}`.",
        "",
        "Training seeds are `2026, 2027, 2028`; simulation seeds are `49026, 49027, 49028, 49029, 49030`.",
        "",
        "## Descriptive decomposition",
        "",
        "For each scenario, the balanced additive decomposition is `Delta[t,s] = mu + alpha_training[t] + beta_simulation[s] + residual[t,s]`, where `alpha_training` is the training-seed mean minus the grand mean and `beta_simulation` is the simulation-seed mean minus the grand mean. The centered total sum of squares is decomposed exactly into training-seed SS, simulation-seed SS, and residual SS.",
        "",
        "Because there is one observation per training×simulation cell, the residual contains training-by-simulation interaction/nonadditivity together with any remaining unexplained variation. These proportions are descriptive variance attribution and are not inferential ANOVA significance claims.",
        "",
        "| Scenario | Grand mean Δ | Training SS proportion | Simulation SS proportion | Residual proportion | Training effect range | Simulation effect range | Favorable / 15 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in decomposition.itertuples(index=False):
        lines.append(
            "| "
            f"{row.scenario} | {format_num(row.grand_mean_delta, signed=True)} | "
            f"{row.prop_training_seed:.3f} | {row.prop_simulation_seed:.3f} | "
            f"{row.prop_residual_interaction_unexplained:.3f} | "
            f"{row.training_seed_effect_range:.6f} | {row.simulation_seed_effect_range:.6f} | "
            f"{int(row.favorable_delta_count)} / 15 |"
        )

    lines.extend(
        [
            "",
            "## Combined fault + load detail",
            "",
            f"The combined-fault grand mean is `{format_num(combined['grand_mean_delta'], signed=True)}`. Its centered variation is attributed descriptively as `{combined['proportion_of_total_centered_variation']['training_seed']:.1%}` training seed, `{combined['proportion_of_total_centered_variation']['simulation_seed']:.1%}` simulation seed, and `{combined['proportion_of_total_centered_variation']['residual_interaction_unexplained']:.1%}` residual interaction/unexplained variation.",
            "",
            "| Training seed | Mean Δ | Favorable simulation seeds |",
            "|---:|---:|---:|",
        ]
    )
    for seed in TRAINING_SEEDS:
        lines.append(
            f"| {seed} | {format_num(combined['training_seed_means'][str(seed)], signed=True)} | "
            f"{combined['per_training_seed_favorable_simulation_count'][str(seed)]} / 5 |"
        )

    lines.extend(
        [
            "",
            "| Simulation seed | Mean Δ across training seeds | Favorable training seeds |",
            "|---:|---:|---:|",
        ]
    )
    for seed in SIMULATION_SEEDS:
        lines.append(
            f"| {seed} | {format_num(combined['simulation_seed_means'][str(seed)], signed=True)} | "
            f"{combined['per_simulation_seed_favorable_training_count'][str(seed)]} / 3 |"
        )

    lines.extend(
        [
            "",
            f"Across the 15 combined-fault cells, `{combined['delta_sign_counts']['negative_favorable']}` are favorable, `{combined['delta_sign_counts']['positive_unfavorable']}` are unfavorable, and `{combined['delta_sign_counts']['zero_tie']}` are exact ties. The training-seed effect range is `{combined['effect_ranges']['training_seed']:.6f}` RMSE and the simulation-seed effect range is `{combined['effect_ranges']['simulation_seed']:.6f}` RMSE.",
            "",
            "## Scenario-level sign structure",
            "",
            "| Scenario | Training-seed means favorable | Favorable cells | Unfavorable cells | Exact ties |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIOS:
        item = by_scenario[scenario]
        favorable_training_means = sum(value < 0 for value in item["training_seed_means"].values())
        signs = item["delta_sign_counts"]
        lines.append(
            f"| {scenario} | {favorable_training_means} / 3 | {signs['negative_favorable']} | "
            f"{signs['positive_unfavorable']} | {signs['zero_tie']} |"
        )

    lines.extend(
        [
            "",
            "The sensor-dropout result is favorable in every paired cell. Sensor bias remains favorable in most cells and has a negative mean for all three training seeds. Load disturbance has a positive mean Delta for all three training seeds, preserving the previously documented C3 load-mismatch weakness. Parameter variation remains heterogeneous, with several exact ties and large simulation-seed effects.",
            "",
            "## Hierarchical bootstrap",
            "",
            f"The bootstrap uses the fixed new Part A analysis RNG seed **{analysis_seed}** and **{replicates:,} replicates**. Each replicate samples the three training-seed clusters with replacement, then samples five simulation-seed Delta values with replacement inside each selected training cluster, and averages the sampled cluster means. This preserves the training→simulation hierarchy rather than treating all 15 cells as independent.",
            "",
            "| Scenario | Point mean Δ | Descriptive 95% CI | Pr(bootstrap mean Δ < 0) |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in bootstrap.itertuples(index=False):
        lines.append(
            f"| {row.scenario} | {format_num(row.point_estimate_mean_delta, signed=True)} | "
            f"[{format_num(row.ci95_low, signed=True)}, {format_num(row.ci95_high, signed=True)}] | "
            f"{row.prob_mean_delta_lt_zero:.4f} |"
        )

    lines.extend(
        [
            "",
            "Only three training seeds are observed. The hierarchical bootstrap is therefore descriptive; the training-level uncertainty is poorly estimated and the 20,000 replicates improve Monte Carlo stability conditional on these three observed model realizations rather than creating additional independent training evidence.",
            "",
            "## Outputs and scope",
            "",
            "Machine-readable results are in `results/final_robustness/variance_decomposition.csv`, `variance_decomposition_summary.json`, and `variance_bootstrap.csv`. This analysis is retrospective and introduces no controller, model, calibration, MPC, training, or simulation changes.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-seed", type=int, default=DEFAULT_ANALYSIS_SEED)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only this script's four Part A outputs if they already exist",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    targets = (DECOMPOSITION_PATH, SUMMARY_PATH, BOOTSTRAP_PATH, REPORT_PATH)
    existing = [relative(path) for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise RuntimeError(f"Part A outputs already exist: {existing}; use --overwrite to replace them")

    source_hash = validate_frozen_source_hash()
    runs = pd.read_csv(RUNS_PATH, float_precision="round_trip")
    paired = validate_and_pair_matrix(runs)

    decomposition_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        row, detail = decompose_scenario(scenario_matrix(paired, scenario), scenario)
        decomposition_rows.append(row)
        details.append(detail)
    decomposition = pd.DataFrame(decomposition_rows)

    bootstrap = hierarchical_bootstrap(
        paired,
        analysis_seed=args.analysis_seed,
        replicates=args.bootstrap_replicates,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    decomposition.to_csv(DECOMPOSITION_PATH, index=False)
    bootstrap.to_csv(BOOTSTRAP_PATH, index=False)

    summary = {
        "study": "final_robustness_hardening_part_a",
        "analysis_role": "retrospective_descriptive_training_simulation_variance_attribution",
        "source": {
            "path": relative(RUNS_PATH),
            "sha256": source_hash,
            "prestudy_manifest": relative(PRESTUDY_MANIFEST),
            "prestudy_manifest_sha256": sha256(PRESTUDY_MANIFEST),
        },
        "design": {
            "training_seeds": list(TRAINING_SEEDS),
            "simulation_seeds": list(SIMULATION_SEEDS),
            "controllers": list(CONTROLLERS),
            "scenarios": list(SCENARIOS),
            "source_run_count": int(len(runs)),
            "paired_delta_count_total": int(len(paired)),
            "paired_delta_count_per_scenario": len(TRAINING_SEEDS) * len(SIMULATION_SEEDS),
            "no_missing_combinations": True,
            "no_duplicate_combinations": True,
            "exact_b_c3_pairing_verified": True,
            "pair_binding_columns_verified": [
                column for column in PAIR_BINDING_COLUMNS if column in runs.columns
            ],
        },
        "delta_definition": "fault_window_RMSE(C3_arbitration_MPC) - fault_window_RMSE(B_plain_MPC); negative favors C3",
        "decomposition": {
            "model": "Delta[t,s] = mu + alpha_training[t] + beta_simulation[s] + residual[t,s]",
            "training_effect": "training-seed mean minus grand mean",
            "simulation_effect": "simulation-seed mean minus grand mean",
            "ss_training": "n_simulation * sum(alpha_training^2)",
            "ss_simulation": "n_training * sum(beta_simulation^2)",
            "ss_residual": "sum((Delta - mu - alpha_training - beta_simulation)^2)",
            "interpretation": (
                "descriptive balanced two-way variance attribution; with one observation per cell, residual "
                "contains interaction/nonadditivity and unexplained variation and is not a separately identified pure error term"
            ),
            "scenario_results": details,
        },
        "hierarchical_bootstrap": {
            "analysis_rng_seed": int(args.analysis_seed),
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "scenario_order": list(SCENARIOS),
            "method": (
                "For each scenario and replicate, sample 3 training-seed clusters with replacement; "
                "within each selected cluster sample 5 paired simulation-seed Delta values with replacement; "
                "average within sampled clusters and then across the 3 sampled clusters."
            ),
            "interval": "descriptive percentile 95% CI",
            "probability": "empirical proportion of bootstrap replicate mean Delta values < 0",
            "training_level_limitation": (
                "Only three training seeds are observed, so training-level uncertainty is poorly estimated; "
                "the bootstrap is descriptive conditional on these observed model realizations."
            ),
            "results": json_ready(bootstrap.to_dict(orient="records")),
        },
        "outputs": {
            "variance_decomposition_csv": {
                "path": relative(DECOMPOSITION_PATH),
                "sha256": sha256(DECOMPOSITION_PATH),
            },
            "variance_bootstrap_csv": {
                "path": relative(BOOTSTRAP_PATH),
                "sha256": sha256(BOOTSTRAP_PATH),
            },
            "markdown_report": {
                "path": relative(REPORT_PATH),
            },
        },
    }

    write_report(
        decomposition,
        details,
        bootstrap,
        analysis_seed=args.analysis_seed,
        replicates=args.bootstrap_replicates,
        source_hash=source_hash,
    )
    summary["outputs"]["markdown_report"]["sha256"] = sha256(REPORT_PATH)
    SUMMARY_PATH.write_text(
        json.dumps(json_ready(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"PASS: exact crossed design verified ({len(paired)} paired Delta cells)")
    print(f"ANALYSIS_RNG_SEED: {args.analysis_seed}")
    print(f"BOOTSTRAP_REPLICATES: {args.bootstrap_replicates}")
    print(f"SOURCE_SHA256: {source_hash}")
    print(f"WROTE: {relative(DECOMPOSITION_PATH)}")
    print(f"WROTE: {relative(SUMMARY_PATH)}")
    print(f"WROTE: {relative(BOOTSTRAP_PATH)}")
    print(f"WROTE: {relative(REPORT_PATH)}")


if __name__ == "__main__":
    main()
