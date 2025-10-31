# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive test suite with 94 tests covering all major functionality
- GitHub Actions CI/CD workflows for automated testing and deployment
- Support for multiple response types: binary, ordinal, continuous, bounded continuous, and count data
- Model persistence with NPZ and JSON formats
- Prior distribution helpers (default, weakly_informative, conservative, vague, centered, rasch, hierarchical)
- Multi-dimensional ideal point estimation (1D and 2D)
- Extensive documentation and examples
- Data validation and error handling
- Automatic response type detection
- Integration tests for complete workflows
- Edge case testing for robustness

### Changed
- Improved test organization with conftest.py fixtures
- Enhanced error messages for better user experience
- Optimized fixture generation to eliminate code duplication

### Fixed
- Ordinal data generation probability normalization
- Persistence metadata handling for different formats

## [0.1.0] - 2025-XX-XX

### Added
- Initial release of idealist package
- Variational Inference (VI) using JAX and NumPyro
- MCMC sampling with NUTS algorithm
- MAP estimation via optimization
- Binary response support for ideal point models
- 1D and 2D ideal point models
- Flexible data loading from pandas DataFrames or CSV files
- Full Bayesian uncertainty quantification with credible intervals
- GPU/TPU acceleration support via JAX
- DataFrame conversion for easy results analysis
- Person and item covariate support
- Customizable prior distributions
- Progress bars for long-running estimations
- Automatic device selection (CPU/GPU/TPU)

### Documentation
- Comprehensive README with quick start guide
- Example scripts demonstrating basic and advanced usage
- API documentation in docstrings
- Installation instructions for CPU and GPU

### Infrastructure
- MIT License
- GitHub repository setup
- Automated testing with pytest
- Code coverage reporting
- Linting with Black and Ruff
- Type checking with mypy
