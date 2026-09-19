import ast
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_SCRIPTS = (
    ROOT / "scripts" / "verify_results.py",
    ROOT / "scripts" / "validate_recovery.py",
)


class FailLoudVerifierTests(unittest.TestCase):
    def test_verifiers_contain_no_bare_assert_statements(self):
        for script in VERIFIER_SCRIPTS:
            with self.subTest(script=script.name):
                tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
                bare_asserts = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)]
                self.assertEqual(bare_asserts, [])

    def test_verification_failures_survive_optimized_python(self):
        code = """
from scripts import validate_recovery, verify_results

for module in (validate_recovery, verify_results):
    try:
        module.require(False)
    except AssertionError as exc:
        if exc.args:
            raise SystemExit(f"unexpected no-message args for {module.__name__}: {exc.args!r}")
    else:
        raise SystemExit(f"optimized Python skipped failure in {module.__name__}")

message = ("sentinel", 7)
try:
    verify_results.require(False, message)
except AssertionError as exc:
    if exc.args != (message,):
        raise SystemExit(f"message changed: {exc.args!r}")
else:
    raise SystemExit("optimized Python skipped messaged failure")

try:
    verify_results.assert_close(1.0, 2.0)
except AssertionError as exc:
    if exc.args != ((1.0, 2.0),):
        raise SystemExit(f"assert_close message changed: {exc.args!r}")
else:
    raise SystemExit("optimized Python skipped assert_close failure")
"""
        completed = subprocess.run(
            [sys.executable, "-O", "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)


if __name__ == "__main__":
    unittest.main()
