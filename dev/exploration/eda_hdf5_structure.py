"""
eda_hdf5_structure.py
=====================
Discover the structure and content of HDF5 data files without assuming
any schema. Walks the full group/dataset tree, prints attributes, dataset
shapes/dtypes, and summarizes repeated groups (e.g. sweeps) so new data
drops can be compared against the expected layout.

Usage
-----
    python dev/exploration/eda_hdf5_structure.py
    python dev/exploration/eda_hdf5_structure.py --config alt.yaml
    python dev/exploration/eda_hdf5_structure.py --file path/to/data.h5
    python dev/exploration/eda_hdf5_structure.py --full          # no tree truncation
"""

# %%
import argparse
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from ceramicspeed.config import get_input_dir, load_config
from ceramicspeed.loading import discover_hdf5_files

# %%
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--file", type=str, default=None,
                        help="Inspect a single HDF5 file instead of discovering via config")
    parser.add_argument("--full", action="store_true",
                        help="Print every child of every group (no truncation)")
    parser.add_argument("--max-children", type=int, default=3,
                        help="Max children to expand per group before truncating (default 3)")
    args, _ = parser.parse_known_args()
    return args

args = parse_args()

if args.file:
    files = [Path(args.file)]
else:
    cfg = load_config(args.config)
    FILE_PATTERNS = cfg.get("filters", {}).get("file_patterns") or None
    files = discover_hdf5_files(get_input_dir(cfg), file_patterns=FILE_PATTERNS)

print(f"Found {len(files)} HDF5 file(s)")
for f in files:
    print(f"  {f}")


# %%
# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _fmt_attr(value) -> str:
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    if isinstance(value, np.ndarray):
        if value.size > 8:
            return f"array{value.shape} {value.dtype}"
        return np.array2string(value, precision=4)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _print_attrs(obj, indent: str) -> None:
    for key in sorted(obj.attrs):
        print(f"{indent}@ {key} = {_fmt_attr(obj.attrs[key])}")


def _dataset_line(ds: h5py.Dataset) -> str:
    parts = [f"shape={ds.shape}", f"dtype={ds.dtype}"]
    if ds.compression:
        parts.append(f"compression={ds.compression}")
    nbytes = ds.size * ds.dtype.itemsize
    parts.append(f"~{nbytes / 1e6:.1f} MB" if nbytes > 1e6 else f"{nbytes} B")
    return ", ".join(parts)


def print_tree(obj, indent: str = "", max_children: int | None = 3) -> None:
    """Recursively print groups, datasets, and their attributes.

    Groups with many similarly-named children (sweep_0, sweep_1, ...) are
    truncated to the first `max_children`, since siblings share structure.
    """
    _print_attrs(obj, indent)
    if isinstance(obj, h5py.Dataset):
        return

    names = list(obj.keys())
    shown = names if max_children is None else names[:max_children]
    for name in shown:
        child = obj[name]
        if isinstance(child, h5py.Group):
            print(f"{indent}+ {name}/  ({len(child)} children)")
        else:
            print(f"{indent}- {name}  [{_dataset_line(child)}]")
        print_tree(child, indent + "    ", max_children)
    if len(names) > len(shown):
        print(f"{indent}... ({len(names) - len(shown)} more: "
              f"{names[len(shown)]} .. {names[-1]})")


def summarize_siblings(grp: h5py.Group, label: str) -> None:
    """Check whether all children of a group share the same structure."""
    dataset_sets: Counter = Counter()
    attr_sets: Counter = Counter()
    shapes: dict[str, Counter] = {}
    for name in grp:
        child = grp[name]
        if not isinstance(child, h5py.Group):
            continue
        dataset_sets[tuple(sorted(child.keys()))] += 1
        attr_sets[tuple(sorted(child.attrs.keys()))] += 1
        for ds_name in child:
            if isinstance(child[ds_name], h5py.Dataset):
                shapes.setdefault(ds_name, Counter())[child[ds_name].shape] += 1

    print(f"\n  Sibling summary for '{label}' ({len(grp)} children):")
    print(f"    Distinct dataset sets: {len(dataset_sets)}")
    for keys, count in dataset_sets.most_common():
        print(f"      {count:4d} x {list(keys)}")
    print(f"    Distinct attr sets:    {len(attr_sets)}")
    for keys, count in attr_sets.most_common(3):
        print(f"      {count:4d} x {list(keys)}")
    if len(attr_sets) > 3:
        print(f"      ... ({len(attr_sets) - 3} more variants)")
    for ds_name, shape_counts in sorted(shapes.items()):
        desc = ", ".join(f"{c} x {s}" for s, c in shape_counts.most_common(4))
        print(f"    '{ds_name}' shapes: {desc}")


# %%
# -----------------------------------------------------------------------------
# Walk each file
# -----------------------------------------------------------------------------

max_children = None if args.full else args.max_children

for fpath in files:
    size_mb = fpath.stat().st_size / 1e6
    print(f"\n{'=' * 79}")
    print(f"FILE: {fpath.name}  ({size_mb:,.1f} MB)")
    print("=" * 79)
    with h5py.File(fpath, "r") as f:
        print_tree(f, max_children=max_children)

        # Summarize any group whose children are many similar subgroups
        for name in f:
            child = f[name]
            if isinstance(child, h5py.Group) and len(child) > args.max_children:
                n_subgroups = sum(isinstance(child[k], h5py.Group) for k in child)
                if n_subgroups > args.max_children:
                    summarize_siblings(child, name)

print("\nDone.")
