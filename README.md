# Idealist: Bayesian Ideal Point Estimation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/michalovadek/idealist/workflows/Tests/badge.svg)](https://github.com/michalovadek/idealist/actions)

`idealist` is a Python package for fast Bayesian ideal point estimation using JAX and NumPyro. It implements 2-parameter Item Response Theory (IRT) models for estimating latent positions from binary, ordinal, and continuous response data.


**Note**: This package is an experiment in vibe-coding. All the code and documentation was written by Claude Code responding to prompts and feature requests.

## Installation

```bash
pip install idealist
```

### For GPU (CUDA) support:
```bash
pip install idealist jax[cuda12]  
```

## Quick Start

```python
import pandas as pd
from idealist import IdealPointEstimator, IdealPointConfig
from idealist.data import load_data

# Load your data (one row per observation)
df = pd.read_csv('responses.csv')  # Columns: person, item, response
data = load_data(df, person_col='person', item_col='item', response_col='response')

# Configure the MODEL (what it is)
config = IdealPointConfig(
    n_dims=1,                    # 1-dimensional ideal points
    response_type='binary'       # Binary responses (0/1)
)

# Create estimator
estimator = IdealPointEstimator(config)

# Fit the model (how to estimate it)
results = estimator.fit(
    data,
    inference='vi',              # Variational inference (fast)
    device='auto',               # Auto-select CPU/GPU
    num_samples=1000             # Number of posterior samples
)

# Get results with uncertainty
results_df = results.to_dataframe()
print(results_df['persons'].head())
```

## Data Format

Your data should be in standard format with **one row per observation**:

```
person | item   | response | person_group | item_category
-------|--------|----------|--------------|---------------
P001   | Item1  | 1        | GroupA       | TypeX
P001   | Item2  | 0        | GroupA       | TypeY
P002   | Item1  | 1        | GroupB       | TypeX
```

**Required columns:**
- `person`: Person identifier
- `item`: Item identifier
- `response`: Response value (binary, ordinal, continuous, or count)

**Optional columns:**
- Person-level covariates (e.g., demographics, groups)
- Item-level covariates (e.g., categories, properties)

```python
from idealist.data import load_data

# Basic usage
data = load_data(df, person_col='person', item_col='item', response_col='response')

# With covariates
data = load_data(
    df,
    person_col='person',
    item_col='item',
    response_col='response',
    person_covariate_cols=['person_group', 'age'],
    item_covariate_cols=['item_category']
)
```

## Model Specification

By default `idealist` implements a 2-parameter logistic ideal point model:

```
P(y_ij = 1) = logit^{-1}(α_j + β_j^T θ_i)
```

Where:
- `θ_i`: Ideal point for person i 
- `α_j`: Difficulty (intercept) for item j
- `β_j`: Discrimination (slope) for item j
- `y_ij`: Response of person i to item j

Supports 1D and 2D ideal point spaces.

## Advanced Features

For more advanced usage including temporal models, hierarchical models, custom priors, and ordinal/continuous responses, see the [examples](examples/) directory.

### Running Tests
```bash
pip install -e ".[dev]"
pytest
```