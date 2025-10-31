"""
Advanced features demonstration for idealist package.

This script shows advanced usage including:
- Custom informative priors
- Comparison with default priors

NOTE: Ordinal and continuous response types are supported by the package
but require proper data formatting. See the test suite for working examples.

To run this example from the project root directory:
    python -m examples.advanced_features

Or install the package and run from anywhere:
    pip install -e .
    python examples/advanced_features.py
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax  # Import JAX first on Windows
import numpy as np
import pandas as pd

from idealist import IdealPointEstimator, IdealPointConfig, ResponseType
from idealist.data import load_data

# ============================================================================
# Example 1: Custom Informative Priors vs Default Priors
# ============================================================================

print("=" * 80)
print("Example 1: Custom Informative Priors vs Default Priors")
print("=" * 80)

# Generate synthetic data with known parameters
np.random.seed(42)
n_persons = 30
n_items = 20

persons = []
items = []
responses = []

# True parameters (we'll use informative priors based on these)
true_ideal_points = np.random.normal(-0.5, 0.3, n_persons)  # Centered at -0.5
true_difficulty = np.random.normal(0, 0.8, n_items)
true_discrimination = np.random.normal(2.0, 0.5, n_items)  # High discrimination

for i in range(n_persons):
    for j in range(n_items):
        if np.random.rand() < 0.7:  # 70% observed
            linear_pred = true_difficulty[j] + true_discrimination[j] * true_ideal_points[i]
            prob = 1 / (1 + np.exp(-linear_pred))
            response = int(np.random.rand() < prob)

            persons.append(f'Person_{i:02d}')
            items.append(f'Item_{j:02d}')
            responses.append(response)

df = pd.DataFrame({'person': persons, 'item': items, 'response': responses})
data = load_data(df, person_col='person', item_col='item', response_col='response')

print(f"\n{data.summary()}")

# ============================================================================
# Model 1: Default weakly informative priors
# ============================================================================

print("\n" + "-" * 80)
print("Model 1: Default Weakly Informative Priors")
print("-" * 80)

config_default = IdealPointConfig(
    n_dims=1,
    response_type=ResponseType.BINARY
)

print("\nDefault priors:")
print(f"  Ideal points: N({config_default.prior_ideal_point_mean}, {config_default.prior_ideal_point_scale}²)")
print(f"  Difficulty: N({config_default.prior_difficulty_mean}, {config_default.prior_difficulty_scale}²)")
print(f"  Discrimination: N({config_default.prior_discrimination_mean}, {config_default.prior_discrimination_scale}²)")

estimator_default = IdealPointEstimator(config_default)
results_default = estimator_default.fit(
    data,
    inference='vi',
    vi_steps=2000,
    num_samples=500,
    device='cpu',
    progress_bar=True
)

print(f"\nModel fitted in {results_default.computation_time:.2f} seconds")

corr_default = np.corrcoef(results_default.ideal_points[:, 0], true_ideal_points)[0, 1]
print(f"Correlation with true ideal points: {np.abs(corr_default):.3f}")

# ============================================================================
# Model 2: Custom informative priors (based on domain knowledge)
# ============================================================================

print("\n" + "-" * 80)
print("Model 2: Custom Informative Priors")
print("-" * 80)

config_custom = IdealPointConfig(
    n_dims=1,
    response_type=ResponseType.BINARY,
    # Prior means (location) - informed by domain knowledge
    prior_ideal_point_mean=-0.5,      # We know ideal points are centered around -0.5
    prior_difficulty_mean=0.0,
    prior_discrimination_mean=2.0,     # We expect high discrimination
    # Prior scales (dispersion) - tighter than default
    prior_ideal_point_scale=0.5,       # Tighter prior (default: 1.0)
    prior_difficulty_scale=1.0,        # Tighter prior (default: 2.5)
    prior_discrimination_scale=0.8     # Tighter prior (default: 2.5)
)

print("\nCustom informative priors:")
print(f"  Ideal points: N({config_custom.prior_ideal_point_mean}, {config_custom.prior_ideal_point_scale}²)")
print(f"  Difficulty: N({config_custom.prior_difficulty_mean}, {config_custom.prior_difficulty_scale}²)")
print(f"  Discrimination: N({config_custom.prior_discrimination_mean}, {config_custom.prior_discrimination_scale}²)")

estimator_custom = IdealPointEstimator(config_custom)
results_custom = estimator_custom.fit(
    data,
    inference='vi',
    vi_steps=2000,
    num_samples=500,
    device='cpu',
    progress_bar=True
)

print(f"\nModel fitted in {results_custom.computation_time:.2f} seconds")

corr_custom = np.corrcoef(results_custom.ideal_points[:, 0], true_ideal_points)[0, 1]
print(f"Correlation with true ideal points: {np.abs(corr_custom):.3f}")

# ============================================================================
# Comparison
# ============================================================================

print("\n" + "=" * 80)
print("Comparison: Default vs Custom Priors")
print("=" * 80)

print(f"\nParameter Recovery (absolute correlation with truth):")
print(f"  Default priors: {np.abs(corr_default):.3f}")
print(f"  Custom priors:  {np.abs(corr_custom):.3f}")
print(f"  Improvement:    {np.abs(corr_custom) - np.abs(corr_default):+.3f}")

print(f"\nUncertainty (mean standard error of ideal points):")
print(f"  Default priors: {results_default.ideal_points_std.mean():.3f}")
print(f"  Custom priors:  {results_custom.ideal_points_std.mean():.3f}")
print(f"  Reduction:      {(1 - results_custom.ideal_points_std.mean() / results_default.ideal_points_std.mean())*100:.1f}%")

print("\n" + "=" * 80)
print("Key Takeaways:")
print("=" * 80)
print("1. Custom informative priors can improve parameter recovery")
print("2. Tighter priors reduce posterior uncertainty")
print("3. Priors should be based on domain knowledge, not data snooping")
print("4. Always compare prior vs posterior to check prior influence")
