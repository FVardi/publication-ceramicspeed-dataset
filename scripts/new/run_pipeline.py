"""
run_pipeline.py
===============
Run the NEW (leak-free, soft-sensing) pipeline end to end. Lives in scripts/new/
and drives the stage subfolders:

    signal_processing/   feature generation + characterisation of the feature
                         space (PCA, operating-condition correlations,
                         sub-band/feature-type coupling, per-feature OC table)
    modelling/           kappa regression, operating-point decomposition,
                         complementarity tests, clustered SHAP, figures

The legacy pipeline (scripts/legacy/, 01-10) is untouched and unaffected. The
new pipeline owns its own feature generation and writes everything under
outputs/new/.

Feature generation (signal_processing/01_feature_generation.py) reads raw HDF5
and is slow, so it is NOT run by default -- pass --with-feature-generation to
(re)build outputs/new/features.parquet. All other steps read that file.

Usage
-----
    python scripts/new/run_pipeline.py
    python scripts/new/run_pipeline.py --with-feature-generation
    python scripts/new/run_pipeline.py --config alt.yaml   # forwarded to steps
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

FEATURE_GENERATION = ("Feature generation (raw HDF5 -> outputs/new/features.parquet)",
                      "signal_processing/01_feature_generation.py")

STEPS = [
    # --- signal processing / characterisation ---
    ("Operating-condition correlations (marginal vs conditional)",
     "signal_processing/03_correlations.py"),
    ("Channel mechanism (sub-band / feature-type coupling + SHAP)",
     "signal_processing/04_channel_mechanism.py"),
    ("Per-feature operating-condition correlation table",
     "signal_processing/05_feature_oc_table.py"),
    # --- modelling ---
    ("kappa regression, full feature set (ElasticNet + LightGBM)",
     "modelling/01_regression.py"),
    ("Operating-point decomposition (two-stage vs direct)",
     "modelling/02_decomposition.py"),
    ("Group-paired Diebold-Mariano tests (complementarity)",
     "modelling/03_group_paired_tests.py"),
    ("Clustered SHAP feature importance",
     "modelling/04_clustered_shap.py"),
    ("Predicted-vs-true figures + comparison table",
     "modelling/05_figures.py"),
    ("Inspection figures (p-values, grouping/folds)",
     "modelling/06_inspect_results.py"),
]


def main() -> None:
    argv = sys.argv[1:]
    with_featgen = "--with-feature-generation" in argv
    extra = [a for a in argv if a != "--with-feature-generation"]  # forwarded to steps

    steps = ([FEATURE_GENERATION] if with_featgen else []) + STEPS
    for desc, rel in steps:
        print("\n" + "=" * 72)
        print(f">>> {desc}")
        print("=" * 72)
        result = subprocess.run([sys.executable, str(HERE / rel), *extra])
        if result.returncode != 0:
            print(f"\nFAILED: {rel} (exit code {result.returncode})")
            sys.exit(result.returncode)
    print("\n" + "=" * 72)
    print("New pipeline complete. Outputs under outputs/new/.")
    print("=" * 72)


if __name__ == "__main__":
    main()
