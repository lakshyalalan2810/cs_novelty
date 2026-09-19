# bool_series pandas-3 compatibility repair

The frozen Part B scientific evidence (735 labeled rows) and the 43 prestudy
frozen artifacts remain scientifically unchanged.

## Defect

`bool_series` in `scripts/verify_final_robustness_study.py` normalized
boolean-like CSV columns with

```python
values.astype(str).str.strip().str.lower()
```

and then mapped the tokens `"nan"`, `"none"`, `""`, and `"<na>"` to missing.
On pandas 3.x, `astype(str)` no longer renders missing values as the string
`"nan"`: NA stays NA through the `.str` accessors, so
`normalized.isin(missing_tokens)` is `False` on missing entries and the
verifier fails loudly on `NaN`-containing boolean columns (e.g.
`sensor_fault_detected` with `allow_nan=True`) instead of parsing them.

## Repair

One normalization line, behavior-preserving on pandas 2.x and correct on
pandas 3.x:

```python
normalized = values.astype("string").str.strip().str.lower().fillna("nan")
```

Missing values of every input kind (float `NaN`, `None`, `pd.NA`, nullable
`string` NA) now deterministically become the `"nan"` token, which the
unchanged mapping already treats as missing. Literal `"None"`/`"<NA>"`
spellings still map to missing as before. No metric, threshold,
classification rule, seed, or scientific artifact changed; only the
verifier's NA normalization changed.

## Validation

- `python scripts/verify_final_robustness_study.py` passes:
  `PASS: final robustness study independently verified`,
  `HISTORICAL_FROZEN_ARTIFACTS: 43/43 unchanged`,
  `PART_B_TOTAL_RUNS: 735`, zero safety counts.
- New regression coverage in `tests/test_bool_series_regression.py` (7 tests)
  pins parsing of numpy-bool, nullable-boolean, object, and nullable-string
  columns with `NaN`/`None`/`pd.NA`/`""`/`"<NA>"` spellings, plus fail-loud
  behavior on invalid tokens and on missing values without `allow_nan`.
- This environment runs pandas 2.3.3, where the old line still passed; the
  pandas 3.0.5 failure mode was verified by code inspection against the
  reported `astype(str)` NA-rendering change, and the new line is
  version-robust on both.
