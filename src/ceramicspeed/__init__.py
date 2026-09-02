# ceramicspeed — shared utilities for CeramicSpeed analysis pipeline

from pathlib import Path

import matplotlib.pyplot as plt

# Applied once here so every figure in the project -- EDA scripts and the
# modelling pipeline alike -- shares the same text sizes, layout, and chart
# chrome, regardless of which script imports ceramicspeed first. See
# plot_style.mplstyle for the actual values.
plt.style.use(Path(__file__).parent / "plot_style.mplstyle")

from . import (  # noqa: F401
    loading,
    cleaning,
    features,
    analysis,
    modelling,
    visualization,
    calculate_kappa,
    config,
    eda,
)
