"""
Quick start example for idealist package.

This script demonstrates basic usage of the idealist package for
ideal point estimation from binary choice data.

To run this example from the project root directory:
    python -m examples.quick_start

Or install the package and run from anywhere:
    pip install -e .
    python examples/quick_start.py
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax  # Import JAX first on Windows
import numpy as np
import pandas as pd

from idealist import IdealPointEstimator, IdealPointConfig, ResponseType
from idealist.data import load_data

# ============================================================================
# Example 1: Quick start with synthetic data
# ============================================================================

print("=" * 80)
print("Example 1: Quick Start with Synthetic Data")
print("=" * 80)

# Create sample binary choice data
np.random.seed(42)
data = {
    'person': ['P_A', 'P_A', 'P_A', 'P_B', 'P_B', 'P_B',
               'P_C', 'P_C', 'P_C', 'P_D', 'P_D', 'P_D'],
    'item': ['Item_1', 'Item_2', 'Item_3', 'Item_1', 'Item_2', 'Item_3',
             'Item_1', 'Item_2', 'Item_3', 'Item_1', 'Item_2', 'Item_3'],
    'response': [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1]
}
df = pd.DataFrame(data)

print("\nSample response data:")
print(df.head(6))

# Load data using idealist's data loader
response_data = load_data(
    df,
    person_col='person',
    item_col='item',
    response_col='response'
)

print(f"\n{response_data.summary()}")

# Create model configuration (what the model IS)
config = IdealPointConfig(
    n_dims=1,  # 1-dimensional ideal points
    response_type=ResponseType.BINARY,
)

# Initialize estimator
estimator = IdealPointEstimator(config)

# Fit model (how to estimate it)
print("\nFitting model with Variational Inference...")
results = estimator.fit(
    response_data,
    inference='vi',
    vi_steps=1000,
    num_samples=500,
    device='cpu',  # Runtime choice
    progress_bar=True,
)

print(f"\nModel fitted in {results.computation_time:.2f} seconds")

# Convert results to DataFrames
results_dict = results.to_dataframe()

print("\nEstimated Ideal Points (Persons):")
print(results_dict['persons'])

print("\nEstimated Item Parameters:")
print(results_dict['items'])

# ============================================================================
# Example 2: With more realistic data
# ============================================================================

print("\n" + "=" * 80)
print("Example 2: Larger Scale Simulation")
print("=" * 80)

# Generate larger synthetic dataset
np.random.seed(123)

n_persons = 50
n_items = 30

# True parameters
true_ideal_points = np.random.normal(0, 1, n_persons)
true_difficulty = np.random.normal(0, 1.5, n_items)
true_discrimination = np.abs(np.random.normal(1, 0.5, n_items))

# Generate responses
persons = []
items = []
responses = []

for i in range(n_persons):
    for j in range(n_items):
        # Sparse data (70% observed)
        if np.random.rand() < 0.7:
            # 2PL model
            linear_pred = true_difficulty[j] + true_discrimination[j] * true_ideal_points[i]
            prob = 1 / (1 + np.exp(-linear_pred))
            response = int(np.random.rand() < prob)

            persons.append(f'Person_{i:02d}')
            items.append(f'Item_{j:02d}')
            responses.append(response)

df_large = pd.DataFrame({
    'person': persons,
    'item': items,
    'response': responses,
})

print(f"\nGenerated {len(df_large)} responses from {n_persons} persons on {n_items} items")

# Load and fit
response_data_large = load_data(
    df_large,
    person_col='person',
    item_col='item',
    response_col='response'
)

print(f"{response_data_large.summary()}")

# Fit with VI (fast for this size)
print("\nFitting larger model...")
config_large = IdealPointConfig(
    n_dims=1,
    response_type=ResponseType.BINARY
)
estimator_large = IdealPointEstimator(config_large)

results_large = estimator_large.fit(
    response_data_large,
    inference='vi',
    vi_steps=2000,
    num_samples=1000,
    device='cpu',
    progress_bar=True,
)

print(f"\nModel fitted in {results_large.computation_time:.2f} seconds")

# Check parameter recovery
estimated_ideal_points = results_large.ideal_points[:, 0]
correlation = np.corrcoef(estimated_ideal_points, true_ideal_points)[0, 1]

print(f"\nParameter recovery:")
print(f"  Correlation with true ideal points: {np.abs(correlation):.3f}")
print(f"  (Note: Sign may flip due to reflection invariance)")

# Show top 10 persons
results_dict_large = results_large.to_dataframe()
top_10 = results_dict_large['persons'].sort_values('ideal_point', ascending=False).head(10)

print("\nTop 10 persons with highest ideal points:")
print(top_10[['person', 'ideal_point', 'ideal_point_se']])

print("\n" + "=" * 80)
print("Quick start examples completed!")
print("=" * 80)
