"""
config.py
=========
Central configuration loader for the CeramicSpeed analysis pipeline.

Usage
-----
    from own_utils.config import load_config
    cfg = load_config()            # loads config.yaml next to this package
    cfg = load_config("alt.yaml")  # loads a specific config file
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Default location: config.yaml in the project root (parent of own_utils/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the YAML configuration file.

    Parameters
    ----------
    path:
        Path to the YAML file.  When ``None`` the default
        ``<project_root>/config.yaml`` is used.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If required keys are missing.
    """
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Copy config.yaml.example to config.yaml and edit the paths section."
        )

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    _resolve_machine_paths(cfg)
    _validate(cfg)
    return cfg


def _resolve_machine_paths(cfg: dict[str, Any]) -> None:
    """Overwrite cfg['paths'] with the entry matching the current machine.

    Compares Path.home() against the 'home' key of each entry in
    cfg['machines'].  If a match is found the machine's 'paths' dict is
    merged into cfg['paths'], so the rest of the pipeline is unaffected.
    """
    machines = cfg.get("machines")
    if not machines:
        return

    current_home = Path.home().as_posix().lower()
    for _label, machine in machines.items():
        machine_home = Path(machine.get("home", "")).as_posix().lower()
        if machine_home and current_home == machine_home:
            cfg["paths"] = dict(machine["paths"])
            return


def _validate(cfg: dict[str, Any]) -> None:
    """Check that required top-level keys are present."""
    required = ["paths", "sensors", "band_width_hz"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(
            f"Configuration is missing required keys: {missing}"
        )


# ---------------------------------------------------------------------------
# Convenience helpers so scripts don't reimplement the same logic
# ---------------------------------------------------------------------------

def get_input_dir(cfg: dict[str, Any]) -> Path:
    """Return the resolved input directory from config."""
    return Path(cfg["paths"]["input_dir"])


def get_output_dir(cfg: dict[str, Any]) -> Path:
    """Return the resolved output directory from config, creating it if needed."""
    p = Path(cfg["paths"]["output_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_sensor_names(cfg: dict[str, Any]) -> list[str]:
    """Return sensor names defined in config."""
    return list(cfg["sensors"].keys())


def get_sensor_freq_limits(cfg: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Return {sensor_name: (f_min, f_max)} from config."""
    return {
        name: (float(s["f_min"]), float(s["f_max"]))
        for name, s in cfg["sensors"].items()
    }


def make_frequency_bands(
    cfg: dict[str, Any],
) -> dict[str, list[tuple[float, float, str]]]:
    """Build fixed-width frequency bands for every sensor from config.

    Returns
    -------
    dict
        ``{sensor_name: [(f_lo, f_hi, label), ...]}``
    """
    width = float(cfg["band_width_hz"])
    bands: dict[str, list[tuple[float, float, str]]] = {}
    for sensor_name, limits in get_sensor_freq_limits(cfg).items():
        f_min, f_max = limits
        edges = np.arange(f_min, f_max + width, width)
        bands[sensor_name] = [
            (float(lo), float(hi), f"{lo / 1e3:.0f}-{hi / 1e3:.0f}kHz")
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
    return bands
