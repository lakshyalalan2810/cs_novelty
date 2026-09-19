"""Build the V3 notebook as a view of canonical generated evidence."""

import json
from pathlib import Path

from generate_v3_report import build_report

PROJECT = Path(__file__).resolve().parent.parent


def main() -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\n" for line in build_report().splitlines()],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import json\n",
                    "import pandas as pd\n",
                    "project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                    "runs = pd.read_csv(project_root / 'results/metrics/v3_final_11scenario_runs.csv')\n",
                    "summary = json.loads((project_root / 'results/metrics/v3_final_11scenario_summary.json').read_text())\n",
                    "print(len(runs), summary['final_holdout_seeds'])\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = PROJECT / "notebooks/04e_dual_virtual_sensor_arbitration.ipynb"
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Generated {path}")


if __name__ == "__main__":
    main()
