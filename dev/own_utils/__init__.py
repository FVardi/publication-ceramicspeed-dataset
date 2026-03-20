# own_utils — shared utilities for CeramicSpeed analysis pipeline
#
# Submodules
# ----------
# loading        — HDF5 and Parquet data loading
# cleaning       — NaN/Inf handling, outlier removal, operational filters
# features       — time-domain and frequency-domain feature extraction
# analysis       — correlation, redundancy, feature selection
# modelling      — model training (ElasticNet, BayesianRidge, LightGBM)
# visualization  — all plotting functions
# calculate_kappa — ISO 281 viscosity ratio calculation
# config         — YAML configuration loader
#
# Backward-compatible aliases
# ---------------------------
# The old module names still work via lazy imports so existing notebooks
# and scripts do not break immediately.  New code should import from the
# specific submodules above.

from . import (  # noqa: F401
    loading,
    cleaning,
    features,
    analysis,
    modelling,
    visualization,
    calculate_kappa,
    config,
)

# Backward-compatible aliases for old import paths
from . import features as own_utils  # noqa: F401  — for `from own_utils import own_utils`
from . import analysis as feature_correlation_helpers  # noqa: F401
from . import modelling as modelling_helpers  # noqa: F401
