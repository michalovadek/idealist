"""
Idealist: Fast Bayesian Ideal Point Estimation

Idealist implements 2-parameter Item Response Theory (IRT) models for estimating
ideal points from binary, ordinal, and continuous response data. Applications include:
- Political science: legislative voting, judicial decisions
- Psychometrics: test responses, ability estimation
- Marketing: product preferences, ratings
- Social science: survey responses, attitude measurement

The package uses JAX and NumPyro for fast Bayesian inference with full uncertainty
quantification.

Main Classes
------------
IdealPointEstimator : Main estimation class
IdealPointConfig : Model configuration
IdealPointResults : Results with uncertainty
IdealPointData : Data container

Data Loader
-----------
load_data : Load person-item response data (one row per observation)

Logging
-------
setup_logger : Configure logging output
get_logger : Get logger instance

Example
-------
>>> from idealist import IdealPointEstimator, IdealPointConfig, setup_logger
>>> from idealist.data import load_data
>>>
>>> # Optional: Enable logging to see progress
>>> setup_logger(enable=True)
>>>
>>> # Load data
>>> data = load_data(df, person_col='person',
...                  item_col='item', response_col='response')
>>>
>>> # Estimate ideal points
>>> config = IdealPointConfig(n_dims=1)
>>> estimator = IdealPointEstimator(config)
>>> results = estimator.fit(data, inference='vi')
>>>
>>> # Get results with uncertainty
>>> results_df = results.to_dataframe()
"""

__version__ = "0.1.0"

# IMPORTANT: Configure XLA CPU cores BEFORE any JAX imports
# This must happen first to ensure JAX recognizes all available CPU cores
# Import data utilities
from idealist import data
from idealist.core import device  # noqa: F401 - triggers XLA configuration

# Import core classes
from idealist.core.base import (
    IdealPointConfig,
    IdealPointResults,
    IdentificationConstraint,
    PriorFamily,
    ResponseType,
)

# Import device configuration utilities (optional advanced usage)
from idealist.core.device import configure_cpu_cores
from idealist.core.diagnostics import check_installation, print_device_info
from idealist.models.ideal_point import IdealPointEstimator

# Import logging utilities
from idealist.utils.logging import get_logger, setup_logger

__all__ = [
    # Main API
    "IdealPointEstimator",
    "IdealPointConfig",
    "IdealPointResults",
    # Enums
    "ResponseType",
    "IdentificationConstraint",
    "PriorFamily",
    # Submodules
    "data",
    # Logging
    "setup_logger",
    "get_logger",
    # Device configuration and diagnostics
    "configure_cpu_cores",
    "check_installation",
    "print_device_info",
    # Version
    "__version__",
]
