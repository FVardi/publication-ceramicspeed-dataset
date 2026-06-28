"""
13_group_paired_tests.py
========================
Group-paired significance tests on the new-pipeline holdout predictions.

Reads the per-sweep holdout predictions saved by 11_featureset_comparison.py
(file, sweep, group, y_true, y_pred) and runs modified Diebold-Mariano tests
for two questions:

  A. Complementarity -- does combining channels beat the best single channel?
     Combined vs AE  and  Combined vs US   (per model, per feature set).
  B. Full vs selected -- does dropping the selection gate change accuracy?
     full vs selected (per model, per target).

The test unit must be the acquisition-hold GROUP, not the window: windows within
a hold are near-duplicates, so a window-level test treats ~1326 correlated
points as independent and grossly overstates significance. We therefore average
the squared-error differential within each of the 138 holdout groups and apply
the Harvey et al. (1997) corrected Diebold-Mariano t-test to the 138 group means.
The naive window-level p-value is reported alongside to make the inflation
explicit.

Sign convention: d = e_A^2 - e_B^2, so a NEGATIVE mean differential means model A
has lower squared error (A is better).

Usage
-----
    python scripts/13_group_paired_tests.py
"""

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy import stats

from ceramicspeed.config import load_config, get_output_dir
from ceramicspeed.evaluation import diebold_mariano_test  # window-level reference

_p = argparse.ArgumentParser()
_p.add_argument("--config", type=str, default=None)
_args, _ = _p.parse_known_args()
cfg = load_config(_args.config)
OUTPUT_DIR = get_output_dir(cfg)
PRED_DIR = OUTPUT_DIR / "11_featureset_comparison" / "predictions"
SCRIPT_DIR = OUTPUT_DIR / "13_group_paired_tests"
SCRIPT_DIR.mkdir(parents=True, exist_ok=True)


def _load(model, target, mode):
    return pd.read_csv(PRED_DIR / f"holdout_{model}_{target}_{mode}.csv")


def _harvey_dm(d_group: np.ndarray, alternative: str = "two-sided"):
    """Modified DM (Harvey et al. 1997, h=1) on a group-level loss differential."""
    d = np.asarray(d_group, dtype=float)
    T = len(d)
    var_d = d.var(ddof=1)
    if var_d <= 0.0:
        return 0.0, 1.0
    dm = d.mean() / np.sqrt(var_d / T)
    dm *= np.sqrt((T - 1) / T)  # h=1 small-sample correction
    if alternative == "two-sided":
        p = float(2 * stats.t.sf(abs(dm), df=T - 1))
    elif alternative == "less":      # A better than B
        p = float(stats.t.cdf(dm, df=T - 1))
    else:                            # 'greater': A worse than B
        p = float(stats.t.sf(dm, df=T - 1))
    return float(dm), p


def group_paired_test(df_a, df_b, label_a, label_b):
    """Align two prediction sets on (file, sweep) and test A vs B, grouped."""
    m = df_a.merge(df_b, on=["file", "sweep"], suffixes=("_a", "_b"))
    e_a = m["y_true_a"].values - m["y_pred_a"].values
    e_b = m["y_true_b"].values - m["y_pred_b"].values
    groups = m["group_a"].values

    # Window-level (naive — overstates because windows are correlated)
    _, p_window = diebold_mariano_test(e_a, e_b)

    # Group-level: mean squared-error differential per acquisition hold
    d_win = e_a**2 - e_b**2
    gdf = pd.DataFrame({"g": groups, "d": d_win}).groupby("g")["d"].mean()
    dm, p_group = _harvey_dm(gdf.values)

    return {
        "A": label_a, "B": label_b,
        "n_windows": len(m), "n_groups": int(gdf.size),
        "mean_dMSE(A-B)": round(float(gdf.mean()), 5),
        "dm_stat": round(dm, 3),
        "p_group": round(p_group, 4),
        "p_window_naive": round(p_window, 6),
        "better": label_a if gdf.mean() < 0 else label_b,
    }


MODELS = ["ElasticNet", "LightGBM"]
MODES = ["full", "selected"]

# --- A. Complementarity: Combined vs each single channel --------------------
comp_rows = []
for model in MODELS:
    for mode in MODES:
        comb = _load(model, "Combined", mode)
        for single in ("AE", "US"):
            sdf = _load(model, single, mode)
            r = group_paired_test(comb, sdf, f"{model}_Combined_{mode}",
                                  f"{model}_{single}_{mode}")
            r = {"model": model, "feature_set": mode, "contrast": f"Combined vs {single}", **r}
            comp_rows.append(r)

comp = pd.DataFrame(comp_rows)
comp.to_csv(SCRIPT_DIR / "complementarity_tests.csv", index=False)

# --- B. Full vs selected: same model + target -------------------------------
fs_rows = []
for model in MODELS:
    for target in ("AE", "US", "Combined"):
        full = _load(model, target, "full")
        sel = _load(model, target, "selected")
        r = group_paired_test(full, sel, f"{model}_{target}_full",
                              f"{model}_{target}_selected")
        r = {"model": model, "target": target, "contrast": "full vs selected", **r}
        fs_rows.append(r)

fs = pd.DataFrame(fs_rows)
fs.to_csv(SCRIPT_DIR / "full_vs_selected_tests.csv", index=False)

# --- Report -----------------------------------------------------------------
_cols_comp = ["model", "feature_set", "contrast", "n_groups",
              "mean_dMSE(A-B)", "dm_stat", "p_group", "p_window_naive", "better"]
_cols_fs = ["model", "target", "n_groups",
            "mean_dMSE(A-B)", "dm_stat", "p_group", "p_window_naive", "better"]

print("\n" + "=" * 90)
print("A. COMPLEMENTARITY  (Combined vs single channel; negative dMSE = Combined better)")
print("=" * 90)
print(comp[_cols_comp].to_string(index=False))

print("\n" + "=" * 90)
print("B. FULL vs SELECTED  (negative dMSE = full better)")
print("=" * 90)
print(fs[_cols_fs].to_string(index=False))

print(f"\nSaved: {SCRIPT_DIR / 'complementarity_tests.csv'}")
print(f"Saved: {SCRIPT_DIR / 'full_vs_selected_tests.csv'}")

if __name__ == "__main__":
    print("\n13_group_paired_tests complete.")
