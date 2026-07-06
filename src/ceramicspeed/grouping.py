"""Acquisition-hold grouping for leak-free, group-aware splitting/CV.

Shared between the new-pipeline scripts (11_featureset_comparison.py,
12_fullset_decomposition.py, ...) so the grouping logic and its
operating-point-twin fix live in one place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def derive_hold_groups(meta_df: pd.DataFrame) -> np.ndarray:
    """One group per contiguous acquisition hold (same file, same rounded
    RPM step, chronologically adjacent sweeps). Prevents near-duplicate
    windows within a single hold from crossing a train/test split."""
    sweep_no = meta_df["sweep"].str.split("_").str[1].astype(int).values
    files = meta_df["file"].values
    step = np.round(meta_df["rpm"].values / 100.0)
    order = np.lexsort((sweep_no, files))
    gid = np.empty(len(meta_df), dtype=int)
    g, prev = 0, None
    for pos in order:
        key = (files[pos], step[pos])
        if prev is None or key[0] != prev[0] or key[1] != prev[1]:
            g += 1
        gid[pos] = g
        prev = key
    return gid


def merge_twin_groups(
    meta_df: pd.DataFrame,
    base_groups: np.ndarray,
    rpm_bin_width: float = 100.0,
    temp_bin_width: float = 1.0,
    verbose: bool = True,
) -> np.ndarray:
    """Merge hold groups that revisit ~the same (rpm, temperature) operating
    point -- e.g. the up-sweep and down-sweep pass through the same RPM
    within one temperature block -- into a single group.

    Without this, kappa (a deterministic function of rpm and temperature
    alone, given fixed bearing/lubricant/load) is nearly identical between
    two "twin" groups even though they are temporally and acquisition-wise
    distinct holds. A model that has only learned to proxy the operating
    point, with no genuine sensitivity to lubrication-film state, could
    still predict a held-out twin's kappa accurately if its non-twin sibling
    was in the training set -- inflating apparent generalisation. Forcing
    twins onto the same side of every split removes that channel.
    """
    tmp = pd.DataFrame({
        "g": base_groups,
        "rpm": meta_df["rpm"].values,
        "temp": meta_df["temperature_c"].values,
    })
    gmean = tmp.groupby("g").agg(rpm_mean=("rpm", "mean"), temp_mean=("temp", "mean"))
    gmean["rpm_bin"] = (gmean["rpm_mean"] / rpm_bin_width).round() * rpm_bin_width
    gmean["temp_bin"] = (gmean["temp_mean"] / temp_bin_width).round() * temp_bin_width
    op_point_id = gmean.groupby(["rpm_bin", "temp_bin"]).ngroup()
    g_to_merged = op_point_id.to_dict()
    merged = np.array([g_to_merged[g] for g in base_groups])
    if verbose:
        n_before, n_after = len(np.unique(base_groups)), len(np.unique(merged))
        print(f"merge_twin_groups: {n_before} hold groups -> {n_after} operating-point "
              f"groups (rpm bin={rpm_bin_width}, temp bin={temp_bin_width}C); "
              f"{n_before - n_after} groups merged into existing operating-point twins")
    return merged
