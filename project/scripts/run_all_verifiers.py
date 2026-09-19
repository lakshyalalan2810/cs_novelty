"""Run every repository verifier and fail loudly on any failure.

Usage:
    python scripts/run_all_verifiers.py

Runs each verifier as a subprocess with the current interpreter, streams a
per-verifier PASS/FAIL summary, and exits nonzero if any verifier fails.
V4 verifiers matching scripts/v4_verify_*.py are picked up automatically so
new phases plug in without editing this runner.
"""

import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

# Explicit frozen list: every verifier that ships with the repo.
VERIFIERS = [
    "scripts/verify_results.py",
    "scripts/verify_v3_results.py",
    "scripts/verify_c4_results.py",
    "scripts/verify_ekf_results.py",
    "scripts/verify_training_seed_robustness.py",
    "scripts/verify_final_robustness_study.py",
    "scripts/validate_recovery.py",
]

V4_GLOB = "scripts/v4_verify_*.py"


def fail(message: str) -> None:
    raise SystemExit(f"RUN_ALL_VERIFIERS FAILED: {message}")


def collect_verifiers() -> list[str]:
    names = list(VERIFIERS)
    names.extend(sorted(p.as_posix() for p in PROJECT.glob(V4_GLOB)))
    seen: set[str] = set()
    unique: list[str] = []
    for name in names:
        if name in seen:
            fail(f"duplicate verifier entry: {name}")
        seen.add(name)
        unique.append(name)
    missing = [name for name in unique if not (PROJECT / name).is_file()]
    if missing:
        fail(f"missing verifier scripts: {missing}")
    return unique


def run_one(name: str) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, name],
        cwd=PROJECT,
        capture_output=True,
        text=True,
    )
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    tail_text = "\n".join(tail[-15:])
    return proc.returncode == 0, tail_text


def main() -> int:
    verifiers = collect_verifiers()
    failures: list[str] = []
    for name in verifiers:
        ok, tail_text = run_one(name)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}")
        if tail_text:
            for line in tail_text.splitlines():
                print(f"    {line}")
        if not ok:
            failures.append(name)
    print(f"verifiers run: {len(verifiers)}, failures: {len(failures)}")
    if failures:
        print(f"FAILED verifiers: {failures}")
        return 1
    print("RUN_ALL_VERIFIERS RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
