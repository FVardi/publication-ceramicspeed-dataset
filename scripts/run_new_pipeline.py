"""
run_new_pipeline.py
===================
Run the NEW (leak-free, simplified) analysis pipeline end to end.

This does NOT touch the legacy pipeline (scripts 01-10): those remain runnable
unchanged. The new pipeline reuses the features/metadata produced by
01_feature_generation.py and the candidate feature schema (``all_columns``)
from 02_feature_analysis.py -- it deliberately ignores the leaky ``retained``
selection -- then runs:

    11_featureset_comparison.py --lightgbm   full vs selected feature sets,
                                             ElasticNet + LightGBM, grouped CV
                                             on the 80% train + grouped holdout
    12_fullset_decomposition.py              operating-point decomposition and
                                             marginal/conditional correlations
                                             on the full feature sets

Prerequisites
-------------
Run 01_feature_generation.py and 02_feature_analysis.py first (they are shared
with the legacy pipeline and produce features.parquet, metadata.parquet, and
feature_selection.json).

Usage
-----
    python scripts/run_new_pipeline.py
    python scripts/run_new_pipeline.py --config alt.yaml   # forwarded to steps
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

STEPS = [
    ("Feature-set comparison (full vs selected; ElasticNet + LightGBM)",
     ["11_featureset_comparison.py"]),
    ("Operating-point decomposition + marginal/conditional correlations",
     ["12_fullset_decomposition.py"]),
    ("Group-paired Diebold-Mariano tests (complementarity, full vs selected)",
     ["13_group_paired_tests.py"]),
    ("Presentation figures + tables",
     ["14_new_method_figures.py"]),
    ("Channel mechanism (sub-band/feature-type coupling + SHAP)",
     ["16_channel_mechanism.py"]),
    ("Per-feature operating-condition correlation table",
     ["17_feature_oc_table.py"]),
]


def main() -> None:
    extra = sys.argv[1:]  # e.g. --config alt.yaml, forwarded to each step
    for desc, cmd in STEPS:
        print("\n" + "=" * 72)
        print(f">>> {desc}")
        print("=" * 72)
        result = subprocess.run(
            [sys.executable, str(HERE / cmd[0]), *cmd[1:], *extra]
        )
        if result.returncode != 0:
            print(f"\nFAILED: {cmd[0]} (exit code {result.returncode})")
            sys.exit(result.returncode)
    print("\n" + "=" * 72)
    print("New pipeline complete. Outputs in outputs/11_featureset_comparison/ "
          "and outputs/12_fullset_decomposition/.")
    print("=" * 72)


if __name__ == "__main__":
    main()
