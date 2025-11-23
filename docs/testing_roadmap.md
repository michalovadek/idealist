# Testing Roadmap for Idealist

**Current Coverage**: ~40%
**Target Coverage**: ~70%
**Status**: Last updated 2025-11-06

## Overview

This document tracks test coverage gaps and provides a roadmap for improving test quality. The idealist package has strong coverage of core inference functionality (~85%) but significant gaps in advanced features and edge cases.

## Critical Gaps (0% Coverage)

### 1. **Predictions** 🔴 CRITICAL
- **File**: `tests/test_prediction.py` (created, skeleton only)
- **Lines Needed**: ~250 lines, 12 tests
- **Status**: Skeleton created with detailed TODOs
- **Priority**: HIGHEST - Core inference capability
- **Features**:
  - Basic prediction on new data
  - Posterior predictive distributions
  - Uncertainty quantification
  - Predictions with temporal models and covariates

### 2. **Temporal Dynamics** 🔴 CRITICAL
- **File**: `tests/test_temporal.py` (created, skeleton only)
- **Lines Needed**: ~250 lines, 12 tests
- **Status**: Skeleton created with detailed TODOs
- **Priority**: HIGHEST - Major advertised feature
- **Features**:
  - Random walk ideal point evolution
  - Multi-timepoint data handling
  - Temporal predictions and trajectories
  - Integration with VI/MCMC/MAP

### 3. **Covariates (Hierarchical Models)** 🔴 CRITICAL
- **File**: `tests/test_covariates.py` (created, skeleton only)
- **Lines Needed**: ~300 lines, 15 tests
- **Status**: Skeleton created with detailed TODOs
- **Priority**: HIGHEST - Core hierarchical modeling
- **Features**:
  - Person-level covariates
  - Item-level covariates
  - Combined covariate models
  - Covariate effects estimation

### 4. **Device Management** 🔴 CRITICAL
- **File**: `tests/test_device.py` (created, skeleton only)
- **Lines Needed**: ~300 lines, 15 tests
- **Status**: Skeleton created with detailed TODOs
- **Priority**: HIGH - GPU/TPU validation
- **Features**:
  - CPU/GPU/TPU detection
  - Multi-core configuration
  - Auto device selection
  - JAX installation diagnostics

## Important Gaps (Partial Coverage)

### 5. **VI Guide Types** 🟡
- **Coverage**: 33% (1 of 3 guides tested)
- **File**: `tests/test_vi_guides.py` (not created)
- **Lines Needed**: ~200 lines, 10 tests
- **Untested**:
  - `guide_type='mvn'` (AutoMultivariateNormal)
  - `guide_type='lowrank_mvn'` (AutoLowRankMultivariateNormal)

### 6. **Optimizers** 🟡
- **Coverage**: 33% (1 of 3 optimizers tested)
- **File**: `tests/test_optimizers.py` (not created)
- **Lines Needed**: ~250 lines, 15 tests
- **Untested**:
  - VI: `vi_optimizer='sgd'` and `'adagrad'`
  - MAP: `map_optimizer='sgd'` and `'adagrad'`

### 7. **MCMC Diagnostics** 🟡
- **Coverage**: ~10% (convergence computed but not validated)
- **File**: Extend `tests/test_integration.py` or create `tests/test_mcmc_diagnostics.py`
- **Lines Needed**: ~150 lines, 8 tests
- **Untested**:
  - `print_mcmc_diagnostics()` output
  - Divergence detection
  - ESS/Rhat thresholds
  - Diagnostic warnings

## Implementation Priority

### Phase 1: Critical Features (Start Immediately)
**Goal**: Test all advertised major features
**Estimated Work**: 2-3 weeks, ~1000 lines, 49 tests

1. ✅ `test_prediction.py` - Skeleton created
2. ✅ `test_temporal.py` - Skeleton created
3. ✅ `test_covariates.py` - Skeleton created
4. ✅ `test_device.py` - Skeleton created

**Next Steps**:
- Implement fixtures for each test file
- Fill in test bodies following TODOs
- Run and debug each test suite
- Ensure all tests pass on CI (Linux, macOS, Windows)

### Phase 2: Important Features (Next Priority)
**Goal**: Validate all inference options
**Estimated Work**: 1-2 weeks, ~650 lines, 33 tests

5. Create `test_vi_guides.py`
6. Create `test_optimizers.py`
7. Expand MCMC diagnostics tests

### Phase 3: Polish (Future Work)
**Goal**: Achieve 70%+ coverage
**Estimated Work**: 1 week, ~300 lines, 18 tests

8. Expand multi-dimensional tests (n_dims > 2)
9. Add response type combinations (MCMC with ordinal, MAP with count, etc.)
10. Test prior effects on posterior distributions

## Test File Status

| File | Status | Lines | Tests | Coverage | Priority |
|------|--------|-------|-------|----------|----------|
| `test_basic_estimation.py` | ✅ Complete | 290 | 14 | 85% | - |
| `test_validation.py` | ✅ Complete | 550 | 30 | 85% | - |
| `test_response_types.py` | ✅ Complete | 230 | 9 | 70% | - |
| `test_integration.py` | ✅ Complete | 320 | 9 | 75% | - |
| `test_persistence.py` | ✅ Complete | 300 | 13 | 80% | - |
| `test_edge_cases.py` | ✅ Complete | 430 | 13 | 60% | - |
| `test_multidimensional.py` | ✅ Complete | 270 | 10 | 65% | - |
| `test_priors.py` | ✅ Complete | 430 | 19 | 50% | - |
| `test_prediction.py` | 🚧 Skeleton | 0 | 0 | 0% | 🔴 CRITICAL |
| `test_temporal.py` | 🚧 Skeleton | 0 | 0 | 0% | 🔴 CRITICAL |
| `test_covariates.py` | 🚧 Skeleton | 0 | 0 | 0% | 🔴 CRITICAL |
| `test_device.py` | 🚧 Skeleton | 0 | 0 | 0% | 🔴 CRITICAL |
| `test_vi_guides.py` | ❌ Missing | 0 | 0 | 0% | 🟡 Important |
| `test_optimizers.py` | ❌ Missing | 0 | 0 | 0% | 🟡 Important |

**Total Existing**: 2,820 lines, 117 tests
**Total Needed**: ~2,000 additional lines, ~100 additional tests

## Coverage Goals

| Component | Current | Target | Gap |
|-----------|---------|--------|-----|
| Core Inference (MAP/VI/MCMC) | 85% | 90% | +5% |
| Data Loading & Validation | 85% | 90% | +5% |
| Response Types | 70% | 85% | +15% |
| Predictions | 0% | 80% | +80% ⚠️ |
| Temporal Models | 0% | 80% | +80% ⚠️ |
| Hierarchical/Covariates | 0% | 80% | +80% ⚠️ |
| Device Management | 0% | 70% | +70% ⚠️ |
| Diagnostics | 0% | 70% | +70% ⚠️ |
| VI Guides | 33% | 90% | +57% |
| Optimizers | 33% | 90% | +57% |
| **Overall** | **~40%** | **~70%** | **+30%** |

## Notes for Contributors

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_prediction.py

# Run with coverage report
pytest --cov=idealist --cov-report=term-missing

# Run only non-skipped tests
pytest -v -k "not skip"
```

### Adding New Tests
1. Choose the appropriate test file or create a new one
2. Follow the existing test structure and naming conventions
3. Use fixtures from `conftest.py` when possible
4. Add `@pytest.mark.skip` with TODO for incomplete tests
5. Document what the test should verify in docstrings
6. Run `black` and `ruff` before committing

### Test Categories
- **Unit tests**: Test individual functions/methods
- **Integration tests**: Test complete workflows (marked with `@pytest.mark.integration`)
- **Slow tests**: Long-running tests (marked with `@pytest.mark.slow`)

### CI/CD
Tests run on GitHub Actions for:
- Python 3.9, 3.10, 3.11, 3.12
- Ubuntu, macOS, Windows
- All tests must pass before merge

## References

- **Main Implementation**: `idealist/models/ideal_point.py`
- **Data Loading**: `idealist/data/loaders.py`
- **Device Management**: `idealist/core/device.py`
- **Diagnostics**: `idealist/core/diagnostics.py`
- **Test Fixtures**: `tests/conftest.py`
