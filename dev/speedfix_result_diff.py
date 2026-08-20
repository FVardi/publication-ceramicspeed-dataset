"""
speedfix_result_diff.py
=======================
Compare pipeline results before and after the VFD-based speed reconstruction.

Baseline: outputs/baseline_pre_speedfix/   (OGT-based, pre-fix snapshot)
Current:  outputs/new/                      (reconstructed speeds)

Prints side-by-side: regression performance per configuration, DM contrasts,
and top SHAP clusters -- the numbers that decide whether the paper's
conclusions survive the fix.
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "outputs" / "baseline_pre_speedfix"
NEW = ROOT / "outputs" / "new"

pd.set_option("display.width", 140)


def _load(path):
    return pd.read_csv(path) if path.exists() else None


print("=" * 70)
print("1. Regression performance (holdout R2 / RMSE per configuration)")
print("=" * 70)
old = _load(OLD / "regression" / "featureset_comparison.csv")
new = _load(NEW / "regression" / "featureset_comparison.csv")
if old is not None and new is not None:
    key = ["model", "target"]
    cmp = old.merge(new, on=key, suffixes=("_old", "_new"))
    cmp["dR2"] = (cmp["holdout_r2_new"] - cmp["holdout_r2_old"]).round(3)
    cols = ["model", "target", "holdout_r2_old", "holdout_r2_new", "dR2",
            "holdout_rmse_old", "holdout_rmse_new", "n_pooled_old", "n_pooled_new"]
    print(cmp[cols].round(3).to_string(index=False))
else:
    print("  (missing files)")

print()
print("=" * 70)
print("2. DM contrasts (channel comparison + complementarity)")
print("=" * 70)
for fname in ("channel_comparison.csv", "complementarity_tests.csv"):
    old = _load(OLD / "group_paired_tests" / fname)
    new = _load(NEW / "group_paired_tests" / fname)
    if old is None or new is None:
        print(f"  {fname}: (missing)")
        continue
    key = ["model", "contrast"]
    cmp = old.merge(new, on=key, suffixes=("_old", "_new"))
    cols = ["model", "contrast", "dm_stat_old", "dm_stat_new",
            "p_group_old", "p_group_new", "better_old", "better_new",
            "n_groups_old", "n_groups_new"]
    cols = [c for c in cols if c in cmp.columns]
    print(f"\n-- {fname} --")
    print(cmp[cols].round(3).to_string(index=False))

print()
print("=" * 70)
print("3. Top SHAP clusters per channel (rep feature, group_shap)")
print("=" * 70)
for ch in ("ae", "us", "combined"):
    old = _load(OLD / "clustered_shap" / f"clustered_shap_{ch}.csv")
    new = _load(NEW / "clustered_shap" / f"clustered_shap_{ch}.csv")
    if old is None or new is None:
        print(f"  {ch}: (missing)")
        continue
    val_old = "group_shap" if "group_shap" in old.columns else "total_shap"
    o = old.nlargest(3, val_old)[["rep", val_old]].values
    n = new.nlargest(3, "group_shap")[["rep", "group_shap"]].values
    print(f"\n-- {ch.upper()} --")
    for i in range(3):
        print(f"  old #{i+1}: {o[i][0]:45s} {o[i][1]:.3f}   |   "
              f"new #{i+1}: {n[i][0]:45s} {n[i][1]:.3f}")
