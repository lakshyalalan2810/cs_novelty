"""Regression tests for the pandas-3-safe bool_series normalization.

Covers scripts/verify_final_robustness_study.py::bool_series across bool,
object, and nullable-string columns containing NaN/None/NA spellings.
"""

import unittest

import numpy as np
import pandas as pd

from scripts.verify_final_robustness_study import bool_series


class BoolSeriesRegressionTests(unittest.TestCase):
    def test_numpy_bool_column(self):
        parsed = bool_series(pd.Series([True, False, True]), "t")
        self.assertEqual(parsed.tolist(), [True, False, True])
        self.assertTrue(pd.api.types.is_bool_dtype(parsed.dtype))

    def test_nullable_boolean_with_na_requires_allow_nan(self):
        values = pd.Series([True, pd.NA, False], dtype="boolean")
        parsed = bool_series(values, "t", allow_nan=True)
        self.assertEqual(parsed.tolist(), [True, pd.NA, False])
        with self.assertRaises((RuntimeError, ValueError)):
            bool_series(values, "t")

    def test_object_column_with_nan_none_and_spellings(self):
        values = pd.Series(
            ["True", " false ", "YES", "No", "1", "0", np.nan, None, "", "nan", "NONE", "<NA>"],
            dtype=object,
        )
        parsed = bool_series(values, "t", allow_nan=True)
        self.assertEqual(
            list(parsed),
            [True, False, True, False, True, False, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
        )

    def test_object_column_missing_forbidden_without_allow_nan(self):
        values = pd.Series(["true", np.nan], dtype=object)
        with self.assertRaises(RuntimeError):
            bool_series(values, "t")

    def test_nullable_string_column_with_na(self):
        values = pd.Series(["True", "False", None], dtype="string")
        parsed = bool_series(values, "t", allow_nan=True)
        self.assertEqual(list(parsed), [True, False, pd.NA])

    def test_invalid_tokens_fail_loud(self):
        values = pd.Series(["true", "maybe"], dtype=object)
        with self.assertRaises(RuntimeError):
            bool_series(values, "t", allow_nan=True)

    def test_mixed_bool_nan_object_column(self):
        # The CSV-read shape that broke under pandas 3: object column mixing
        # booleans/strings with float NaN must still parse with allow_nan.
        values = pd.Series([True, False, float("nan"), "True"], dtype=object)
        parsed = bool_series(values, "t", allow_nan=True)
        self.assertEqual(list(parsed), [True, False, pd.NA, True])


if __name__ == "__main__":
    unittest.main()
