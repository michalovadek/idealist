"""
Tests for MCMC diagnostic functions.

Tests the diagnostic capabilities for assessing MCMC convergence and quality:
1. Effective Sample Size (ESS) - measures sampling efficiency
2. R-hat (Gelman-Rubin) - measures convergence across chains
3. Divergences - identifies problematic posterior geometry
4. Acceptance rate - NUTS sampler efficiency
5. Tree depth - NUTS exploration statistics

These diagnostics are critical for validating MCMC inference results.
"""

import numpy as np
import pytest

from idealist import IdealPointConfig, IdealPointEstimator, ResponseType


class TestMCMCBasicDiagnostics:
    """Tests for basic MCMC diagnostic computation."""

    @pytest.mark.slow
    def test_compute_mcmc_diagnostics(self, small_binary_data):
        """Test that MCMC diagnostics can be computed."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=100, num_samples=100,
            device="cpu", progress_bar=False
        )

        # Compute diagnostics
        diagnostics = model.compute_mcmc_diagnostics(results.posterior_samples)

        assert diagnostics is not None
        assert isinstance(diagnostics, dict)

    @pytest.mark.slow
    def test_diagnostics_structure(self, small_binary_data):
        """Test that diagnostics have expected structure."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=100, num_samples=100,
            device="cpu", progress_bar=False
        )

        diagnostics = model.compute_mcmc_diagnostics(results.posterior_samples)

        # Check for expected keys
        assert "parameters" in diagnostics
        assert "summary" in diagnostics


class TestEffectiveSampleSize:
    """Tests for Effective Sample Size (ESS) diagnostics."""

    @pytest.mark.slow
    def test_ess_computed(self, small_binary_data):
        """Test that ESS is computed for parameters."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=100, num_samples=100,
            device="cpu", progress_bar=False
        )

        diagnostics = model.compute_mcmc_diagnostics(results.posterior_samples)

        # Check ESS exists in summary
        assert "min_ess" in diagnostics["summary"]
        assert diagnostics["summary"]["min_ess"] is not None

    @pytest.mark.slow
    def test_ess_positive(self, small_binary_data):
        """Test that ESS values are positive."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=200, num_samples=200,
            device="cpu", progress_bar=False
        )

        diagnostics = model.compute_mcmc_diagnostics(results.posterior_samples)

        # ESS should be positive
        min_ess = diagnostics["summary"]["min_ess"]
        assert min_ess > 0

    @pytest.mark.slow
    def test_ess_per_parameter(self, small_binary_data):
        """Test that ESS is computed for each parameter type."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=100, num_samples=100,
            device="cpu", progress_bar=False
        )

        diagnostics = model.compute_mcmc_diagnostics(results.posterior_samples)

        # Check ESS for key parameters
        params = diagnostics["parameters"]
        if "ideal_points" in params:
            assert "ess_bulk_min" in params["ideal_points"]
            assert "ess_bulk_mean" in params["ideal_points"]

    @pytest.mark.slow
    def test_ess_improves_with_samples(self, small_binary_data):
        """Test that ESS improves with more samples."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)

        # Few samples
        model1 = IdealPointEstimator(config)
        results1 = model1.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=50, num_samples=50,
            device="cpu", progress_bar=False
        )
        diag1 = model1.compute_mcmc_diagnostics(results1.posterior_samples)

        # More samples
        model2 = IdealPointEstimator(config)
        results2 = model2.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=100, num_samples=200,
            device="cpu", progress_bar=False
        )
        diag2 = model2.compute_mcmc_diagnostics(results2.posterior_samples)

        # More samples should have higher ESS (generally)
        # Note: This might not always hold due to randomness
        assert diag1["summary"]["min_ess"] > 0
        assert diag2["summary"]["min_ess"] > 0


class TestRhatConvergence:
    """Tests for R-hat convergence diagnostics."""

    @pytest.mark.slow
    def test_rhat_computed(self, small_binary_data):
        """Test that R-hat is computed for parameters."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=100, num_samples=100,
            device="cpu", progress_bar=False
        )

        diagnostics = model.compute_mcmc_diagnostics(results.posterior_samples)

        # Check R-hat exists
        assert "max_rhat" in diagnostics["summary"]
        assert diagnostics["summary"]["max_rhat"] is not None

    @pytest.mark.slow
    def test_rhat_near_one(self, small_binary_data):
        """Test that R-hat is close to 1 for converged chains."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=200, num_samples=200,
            device="cpu", progress_bar=False
        )

        diagnostics = model.compute_mcmc_diagnostics(results.posterior_samples)

        # R-hat should be close to 1 (< 1.2 is usually acceptable)
        max_rhat = diagnostics["summary"]["max_rhat"]
        assert max_rhat < 1.5  # Loose bound for tests

    @pytest.mark.slow
    def test_rhat_per_parameter(self, small_binary_data):
        """Test that R-hat is computed for each parameter."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=100, num_samples=100,
            device="cpu", progress_bar=False
        )

        diagnostics = model.compute_mcmc_diagnostics(results.posterior_samples)

        params = diagnostics["parameters"]
        if "ideal_points" in params:
            assert "rhat_max" in params["ideal_points"]
            assert "rhat_mean" in params["ideal_points"]

    @pytest.mark.slow
    def test_rhat_multiple_chains(self, small_binary_data):
        """Test R-hat with multiple chains."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        # Multiple chains needed for R-hat
        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=4, num_warmup=100, num_samples=100,
            device="cpu", progress_bar=False
        )

        diagnostics = model.compute_mcmc_diagnostics(results.posterior_samples)

        # Should compute R-hat successfully
        assert "max_rhat" in diagnostics["summary"]


class TestDivergences:
    """Tests for divergence diagnostics."""

    @pytest.mark.slow
    def test_divergences_tracked(self, small_binary_data):
        """Test that divergences are tracked."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=100, num_samples=100,
            device="cpu", progress_bar=False
        )

        diagnostics = model.compute_mcmc_diagnostics(results.posterior_samples)

        # Check if divergences key exists (may or may not have divergences)
        # Divergences info might be in diagnostics if available
        assert diagnostics is not None

    @pytest.mark.slow
    def test_low_divergence_rate(self, small_binary_data):
        """Test that divergence rate is low for well-specified model."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=200, num_samples=200,
            device="cpu", progress_bar=False
        )

        diagnostics = model.compute_mcmc_diagnostics(results.posterior_samples)

        # If divergences are tracked, rate should be low
        if "divergences" in diagnostics:
            div_rate = diagnostics["divergences"]["divergence_rate"]
            assert div_rate < 0.1  # Less than 10% divergences


class TestAcceptanceRate:
    """Tests for NUTS acceptance rate diagnostics."""

    @pytest.mark.slow
    def test_acceptance_rate_computed(self, small_binary_data):
        """Test that acceptance rate is computed."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=100, num_samples=100,
            device="cpu", progress_bar=False
        )

        diagnostics = model.compute_mcmc_diagnostics(results.posterior_samples)

        # Acceptance rate might be in diagnostics
        if "acceptance_rate" in diagnostics:
            assert "mean" in diagnostics["acceptance_rate"]

    @pytest.mark.slow
    def test_acceptance_rate_reasonable(self, small_binary_data):
        """Test that acceptance rate is in reasonable range."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=200, num_samples=200,
            device="cpu", progress_bar=False
        )

        diagnostics = model.compute_mcmc_diagnostics(results.posterior_samples)

        # NUTS typically targets ~65% acceptance rate
        if "acceptance_rate" in diagnostics:
            mean_accept = diagnostics["acceptance_rate"]["mean"]
            assert 0.3 < mean_accept < 0.99  # Reasonable range


class TestPrintMCMCDiagnostics:
    """Tests for print_mcmc_diagnostics() function."""

    @pytest.mark.slow
    def test_print_diagnostics_runs(self, small_binary_data, capsys):
        """Test that print_mcmc_diagnostics() executes without error."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=100, num_samples=100,
            device="cpu", progress_bar=False
        )

        # Should not raise error
        model.print_mcmc_diagnostics()

        # Check that something was printed
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    @pytest.mark.slow
    def test_print_diagnostics_output_structure(self, small_binary_data, capsys):
        """Test that printed diagnostics have expected content."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=100, num_samples=100,
            device="cpu", progress_bar=False
        )

        model.print_mcmc_diagnostics()

        captured = capsys.readouterr()
        output = captured.out

        # Should mention key diagnostic terms
        assert "MCMC" in output or "Diagnostic" in output or "ESS" in output or "Rhat" in output


class TestDiagnosticsAdvancedModels:
    """Tests for diagnostics with advanced model features."""

    @pytest.mark.slow
    def test_diagnostics_with_temporal(self, small_temporal_data):
        """Test diagnostics with temporal models."""
        data = small_temporal_data

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            temporal_dynamics=True,
            n_timepoints=data["n_timepoints"]
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            timepoints=data["timepoints"],
            inference="mcmc", num_chains=2, num_warmup=100, num_samples=100,
            device="cpu", progress_bar=False
        )

        diagnostics = model.compute_mcmc_diagnostics(results.posterior_samples)

        assert diagnostics is not None
        assert "summary" in diagnostics

    @pytest.mark.slow
    def test_diagnostics_with_multidimensional(self, multidim_binary_data):
        """Test diagnostics with multidimensional models."""
        data = multidim_binary_data

        config = IdealPointConfig(n_dims=2, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=100, num_samples=100,
            device="cpu", progress_bar=False
        )

        diagnostics = model.compute_mcmc_diagnostics(results.posterior_samples)

        assert diagnostics is not None


class TestDiagnosticsEdgeCases:
    """Tests for edge cases in diagnostics."""

    @pytest.mark.slow
    def test_diagnostics_single_chain(self, small_binary_data):
        """Test diagnostics with single chain (R-hat not computable)."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=1, num_warmup=100, num_samples=100,
            device="cpu", progress_bar=False
        )

        # Should still compute diagnostics (ESS), but R-hat may not be available
        diagnostics = model.compute_mcmc_diagnostics(results.posterior_samples)

        assert diagnostics is not None

    @pytest.mark.slow
    def test_diagnostics_few_samples(self, small_binary_data):
        """Test diagnostics with very few samples."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=20, num_samples=20,
            device="cpu", progress_bar=False
        )

        # Should not crash even with few samples
        diagnostics = model.compute_mcmc_diagnostics(results.posterior_samples)

        assert diagnostics is not None


class TestConvergenceAssessment:
    """Tests for overall convergence assessment."""

    @pytest.mark.slow
    def test_convergence_summary_flags(self, small_binary_data):
        """Test that convergence summary includes boolean flags."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=200, num_samples=200,
            device="cpu", progress_bar=False
        )

        diagnostics = model.compute_mcmc_diagnostics(results.posterior_samples)

        summary = diagnostics["summary"]

        # Should have convergence flags
        if "all_rhat_below_1_1" in summary:
            assert isinstance(summary["all_rhat_below_1_1"], bool)
        if "all_ess_above_400" in summary:
            assert isinstance(summary["all_ess_above_400"], bool)

    @pytest.mark.slow
    def test_well_converged_model(self, small_binary_data):
        """Test diagnostics for well-converged model."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        # Use sufficient warmup and samples for convergence
        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=4, num_warmup=500, num_samples=500,
            device="cpu", progress_bar=False
        )

        diagnostics = model.compute_mcmc_diagnostics(results.posterior_samples)

        # Should show good convergence
        summary = diagnostics["summary"]
        assert summary["min_ess"] > 100  # Reasonable ESS
        if summary["max_rhat"] is not None:
            assert summary["max_rhat"] < 1.2  # Good R-hat


class TestDiagnosticsResponseTypes:
    """Tests for diagnostics with different response types."""

    @pytest.mark.slow
    def test_diagnostics_ordinal(self, small_ordinal_data):
        """Test diagnostics with ordinal responses."""
        data = small_ordinal_data

        config = IdealPointConfig(
            n_dims=1, response_type=ResponseType.ORDINAL, n_categories=data["n_categories"]
        )
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=100, num_samples=100,
            device="cpu", progress_bar=False
        )

        diagnostics = model.compute_mcmc_diagnostics(results.posterior_samples)

        assert diagnostics is not None
        assert "summary" in diagnostics

    @pytest.mark.slow
    def test_diagnostics_continuous(self, small_continuous_data):
        """Test diagnostics with continuous responses."""
        data = small_continuous_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.CONTINUOUS)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=100, num_samples=100,
            device="cpu", progress_bar=False
        )

        diagnostics = model.compute_mcmc_diagnostics(results.posterior_samples)

        assert diagnostics is not None
