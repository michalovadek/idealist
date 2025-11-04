# idealist: Bayesian Ideal Point Estimation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/michalovadek/idealist/workflows/Tests/badge.svg)](https://github.com/michalovadek/idealist/actions)

Fast Bayesian ideal point estimation using JAX and NumPyro. Implements 2-parameter IRT models for binary, ordinal, and continuous response data.

**Note**: This package is an experiment in vibe-coding. All code was written by Claude Code.

## Installation

```bash
pip install idealist
```

For GPU support:
```bash
pip install idealist[cuda12]  # CUDA 12.x
pip install idealist[cuda11]  # CUDA 11.x
```

See [Installation Guide](docs/installation.md) for details.

## Quick Start

```python
from idealist import IdealPointEstimator, IdealPointConfig
from idealist.data import load_data

# Load data
data = load_data(df, person_col='person', item_col='item', response_col='response')

# Fit model
config = IdealPointConfig(n_dims=1, response_type='binary')
estimator = IdealPointEstimator(config)
results = estimator.fit(data, inference='vi')

# Get results
results_df = results.to_dataframe()
```

## Features

- **Response types**: Binary, ordinal, continuous, count, bounded continuous
- **Dimensions**: 1D or 2D ideal point spaces
- **Inference**: Variational inference (VI), MCMC, or MAP
- **Performance**: Automatic CPU/GPU detection and optimization
- **Uncertainty**: Full posterior distributions for all parameters

## Documentation

- [Installation Guide](docs/installation.md) - CPU, GPU, and TPU installation
- [Configuration Guide](docs/configuration.md) - Device settings and performance tuning
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions
- [Examples](examples/) - Working code examples

## License

MIT License - see [LICENSE](LICENSE) file for details.