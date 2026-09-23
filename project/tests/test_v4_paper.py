"""Paper artifact tests: tables/figures regenerate from result CSVs."""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PaperArtifactTests(unittest.TestCase):
    def test_tables_regenerate(self):
        proc = subprocess.run(
            [sys.executable, "scripts/v4_paper_tables.py"], cwd=ROOT,
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for name in ("tab_posthoc_per_seed.tex", "tab_envelope_relabel.tex",
                     "tab_frozen_detection.tex", "tab_h1_h9.tex"):
            path = ROOT / "paper" / "tables" / name
            self.assertTrue(path.is_file(), name)
            text = path.read_text()
            self.assertIn("\\begin{tabular}", text)
            self.assertIn("AUTO-GENERATED", text)

    def test_figures_regenerate(self):
        proc = subprocess.run(
            [sys.executable, "scripts/v4_paper_figures.py"], cwd=ROOT,
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for name in ("fig_latch_timeline.pdf",
                     "fig_conditional_detection.pdf",
                     "fig_architecture.pdf",
                     "fig_research_evolution.pdf",
                     "fig_hypothesis_effects.pdf",
                     "fig_load_false_entries.pdf"):
            path = ROOT / "paper" / "figures" / name
            self.assertTrue(path.is_file(), name)
            # Vector PDFs with real content (not empty stubs).
            self.assertGreater(path.stat().st_size, 5000, name)
            self.assertEqual(path.read_bytes()[:5], b"%PDF-")

    def test_manuscript_skeleton_sections(self):
        text = (ROOT / "paper" / "main.tex").read_text()
        for section in ("Introduction", "Related Work", "Method",
                        "Experimental Protocol", "Results", "Discussion",
                        "Limitations", "Conclusion"):
            self.assertIn(f"\\section{{{section}}}", text)
        self.assertIn("\\documentclass[conference]{IEEEtran}", text)

    def test_bibliography_has_only_verified_entries(self):
        text = (ROOT / "paper" / "references.bib").read_text()
        for key in ("Hochreiter1997", "Page1954", "Holm1979", "Mayne2000"):
            self.assertIn(key, text)
        self.assertNotIn("TODO", text)


if __name__ == "__main__":
    unittest.main()
