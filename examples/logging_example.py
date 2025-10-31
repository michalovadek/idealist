"""
Example: Using optional logging in idealist

This example demonstrates how to enable logging to see progress messages
during model fitting.
"""

import numpy as np
import pandas as pd
from idealist import IdealPointEstimator, IdealPointConfig, setup_logger
from idealist.data import load_data

# Enable logging to see progress messages
# By default, logging is disabled to avoid spamming users
setup_logger(enable=True)

# You can also set different log levels:
# setup_logger(enable=True, level=logging.DEBUG)  # More detailed output
# setup_logger(enable=True, level=logging.WARNING)  # Only warnings and errors

# Generate some example voting data
np.random.seed(42)
n_persons = 50
n_items = 20
n_obs = 500

# Create random person-item pairs
person_ids = np.random.randint(0, n_persons, size=n_obs)
item_ids = np.random.randint(0, n_items, size=n_obs)

# Generate binary responses (votes)
# Simulate some ideal points
true_ideal_points = np.random.randn(n_persons)
true_difficulties = np.random.randn(n_items)
true_discriminations = np.abs(np.random.randn(n_items)) + 0.5

# Calculate probabilities
logits = true_difficulties[item_ids] + true_discriminations[item_ids] * true_ideal_points[person_ids]
probs = 1 / (1 + np.exp(-logits))
responses = (np.random.rand(n_obs) < probs).astype(int)

# Load data
df = pd.DataFrame({
    'person': person_ids,
    'item': item_ids,
    'vote': responses
})

data = load_data(df, person_col='person', item_col='item', response_col='vote')

print("\n" + "="*70)
print("Example: Using logging to monitor model fitting")
print("="*70 + "\n")

# Fit model with Variational Inference
# With logging enabled, you'll see messages about:
# - Auto-detected response type
# - Device selection (CPU/GPU)
# - Inference method being used
print("Fitting model with Variational Inference...")
config = IdealPointConfig(n_dims=1)
estimator = IdealPointEstimator(config)
results = estimator.fit(data, inference='vi', vi_steps=1000, progress_bar=True)

print("\nModel fitting complete!")
print(f"Estimated {len(results.ideal_points)} ideal points")
print(f"Mean ideal point: {results.ideal_points.mean():.3f}")
print(f"Std ideal point: {results.ideal_points.std():.3f}")

print("\n" + "="*70)
print("To disable logging in your own code, use:")
print("  setup_logger(enable=False)")
print("or simply don't call setup_logger() at all (disabled by default)")
print("="*70 + "\n")
