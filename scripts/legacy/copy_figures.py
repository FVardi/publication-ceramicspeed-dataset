"""
copy_figures.py
===============
Copy all figures referenced in paper/main.tex from the outputs/ directories
into paper/figures/ so the paper folder is self-contained for journal submission.

Searches outputs/ recursively for each filename found in \\includegraphics{}
calls in main.tex. Warns if a figure cannot be found.

Usage
-----
    python scripts/copy_figures.py
    python scripts/copy_figures.py --dry-run
"""

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
TEX_FILE = ROOT / "paper" / "main.tex"
OUTPUT_DIR = ROOT / "outputs"
DEST_DIR = ROOT / "paper" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be copied without copying.")
    return parser.parse_args()


def extract_figure_names(tex_path: Path) -> list[str]:
    """Scan main.tex and all sections/*.tex for includegraphics names."""
    text = tex_path.read_text(encoding="utf-8")
    sections_dir = tex_path.parent / "sections"
    if sections_dir.is_dir():
        for sec in sorted(sections_dir.glob("*.tex")):
            text += sec.read_text(encoding="utf-8")
    # Match \includegraphics[...]{filename} — strip any leading path component
    raw = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
    names = []
    for r in raw:
        name = Path(r).name  # drop any figures/ prefix already in the tex
        if name:
            names.append(name)
    return sorted(set(names))


def find_in_outputs(name: str) -> Path | None:
    # Prefer the styled 06_plots render when one exists.
    preferred = OUTPUT_DIR / "06_plots" / name
    if preferred.exists():
        return preferred
    for match in OUTPUT_DIR.rglob(name):
        return match
    return None


def main() -> None:
    args = parse_args()
    DEST_DIR.mkdir(exist_ok=True)

    figure_names = extract_figure_names(TEX_FILE)
    print(f"Found {len(figure_names)} unique figures in {TEX_FILE.name}")

    copied, skipped, missing = 0, 0, 0
    for name in figure_names:
        dest = DEST_DIR / name
        src = find_in_outputs(name)

        if src is None:
            # Already committed in paper/figures and not in outputs — keep it
            if dest.exists():
                print(f"  [keep]    {name}  (not in outputs, already in paper/figures)")
                skipped += 1
            else:
                print(f"  [MISSING] {name}  — not found in outputs/ or paper/figures/")
                missing += 1
            continue

        if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
            print(f"  [up-to-date] {name}")
            skipped += 1
            continue

        rel_src = src.relative_to(ROOT)
        if args.dry_run:
            print(f"  [would copy] {rel_src}  →  paper/figures/{name}")
        else:
            shutil.copy2(src, dest)
            print(f"  [copied]  {rel_src}  →  paper/figures/{name}")
        copied += 1

    print(f"\nDone: {copied} copied, {skipped} skipped, {missing} missing")
    if missing:
        print("Run the relevant pipeline scripts to generate missing figures.")


if __name__ == "__main__":
    main()
