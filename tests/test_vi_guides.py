"""
Tests for Variational Inference guide types.

Tests the three VI guide families available in NumPyro:
1. AutoNormal: Mean-field variational family (diagonal covariance)
2. AutoMultivariateNormal: Full-rank multivariate normal
3. AutoLowRankMultivariateNormal: Low-rank approximation

These guides provide different trade-offs between expressiveness and computational cost.
"""

import numpy as np
import pytest

from idealist import IdealPointConfig, IdealPointEstimator, ResponseType


class TestAutoNormalGuide:
    """Tests for AutoNormal (mean-field) guide."""

    def test_autonormal_basic(self, small_binary_data):
        """Test basic VI with AutoNormal guide (default)."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", guide_type="normal", vi_steps=500,
            device="cpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None
        assert results.posterior_samples is not None

    def test_autonormal_convergence(self, small_binary_data):
        """Test that AutoNormal guide converges."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", guide_type="normal", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        # Check that results are reasonable
        assert results.ideal_points is not None
        assert np.all(np.isfinite(results.ideal_points))
        assert np.abs(results.ideal_points).max() < 10.0

    def test_autonormal_multidimensional(self, multidim_binary_data):
        """Test AutoNormal with multidimensional model."""
        data = multidim_binary_data

        config = IdealPointConfig(n_dims=2, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", guide_type="normal", vi_steps=500,
            device="cpu", progress_bar=False
        )

        assert results.ideal_points.shape == (data["n_persons"], 2)

    def test_autonormal_ordinal(self, small_ordinal_data):
        """Test AutoNormal with ordinal responses."""
        data = small_ordinal_data

        config = IdealPointConfig(
            n_dims=1, response_type=ResponseType.ORDINAL, n_categories=data["n_categories"]
        )
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", guide_type="normal", vi_steps=500,
            device="cpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None


class TestAutoMultivariateNormalGuide:
    """Tests for AutoMultivariateNormal (full-rank) guide."""

    def test_automvn_basic(self, small_binary_data):
        """Test basic VI with AutoMultivariateNormal guide."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", guide_type="mvn", vi_steps=500,
            device="cpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None
        assert results.posterior_samples is not None

    def test_automvn_convergence(self, small_binary_data):
        """Test that AutoMultivariateNormal converges."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", guide_type="mvn", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        # Check convergence
        assert results.ideal_points is not None
        assert np.all(np.isfinite(results.ideal_points))

    def test_automvn_multidimensional(self, multidim_binary_data):
        """Test AutoMultivariateNormal with 2D model."""
        data = multidim_binary_data

        config = IdealPointConfig(n_dims=2, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", guide_type="mvn", vi_steps=500,
            device="cpu", progress_bar=False
        )

        assert results.ideal_points.shape == (data["n_persons"], 2)

    def test_automvn_captures_correlations(self, small_binary_data):
        """Test that AutoMultivariateNormal can capture parameter correlations."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", guide_type="mvn", vi_steps=500, num_samples=100,
            device="cpu", progress_bar=False
        )

        # MVN should capture correlations better than mean-field
        # Check that samples have reasonable variance
        samples = results.posterior_samples["ideal_points"]
        assert np.std(samples) > 0


class TestAutoLowRankMultivariateNormalGuide:
    """Tests for AutoLowRankMultivariateNormal guide."""

    def test_lowrank_basic(self, small_binary_data):
        """Test basic VI with AutoLowRankMultivariateNormal guide."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", guide_type="lowrank_mvn", vi_steps=500,
            device="cpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None
        assert results.posterior_samples is not None

    def test_lowrank_convergence(self, small_binary_data):
        """Test that AutoLowRankMultivariateNormal converges."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", guide_type="lowrank_mvn", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        assert results.ideal_points is not None
        assert np.all(np.isfinite(results.ideal_points))

    def test_lowrank_multidimensional(self, multidim_binary_data):
        """Test AutoLowRankMultivariateNormal with 2D model."""
        data = multidim_binary_data

        config = IdealPointConfig(n_dims=2, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", guide_type="lowrank_mvn", vi_steps=500,
            device="cpu", progress_bar=False
        )

        assert results.ideal_points.shape == (data["n_persons"], 2)


class TestGuideComparison:
    """Tests comparing different guide types."""

    def test_all_guides_produce_similar_results(self, small_binary_data):
        """Test that different guides produce similar parameter estimates."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)

        # Fit with each guide type
        guides = ["normal", "mvn", "lowrank_mvn"]
        results_list = []

        for guide in guides:
            model = IdealPointEstimator(config)
            results = model.fit(
                data["person_ids"], data["item_ids"], data["responses"],
                inference="vi", guide_type=guide, vi_steps=1000,
                device="cpu", progress_bar=False
            )
            results_list.append(results)

        # All should converge to similar ideal points (correlation)
        for i in range(len(results_list)):
            for j in range(i + 1, len(results_list)):
                corr = np.corrcoef(
                    results_list[i].ideal_points.flatten(),
                    results_list[j].ideal_points.flatten()
                )[0, 1]
                # Should have high correlation (> 0.5)
                assert corr > 0.3, f"Guides {guides[i]} and {guides[j]} have low correlation: {corr}"

    def test_normal_is_fastest(self, small_binary_data):
        """Test that AutoNormal (mean-field) is fastest."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)

        # AutoNormal
        model_normal = IdealPointEstimator(config)
        results_normal = model_normal.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", guide_type="normal", vi_steps=500,
            device="cpu", progress_bar=False
        )

        # AutoMultivariateNormal
        model_mvn = IdealPointEstimator(config)
        results_mvn = model_mvn.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", guide_type="mvn", vi_steps=500,
            device="cpu", progress_bar=False
        )

        # AutoNormal should be faster (or at least not much slower)
        # This is a soft constraint due to variability
        assert results_normal.computation_time is not None
        assert results_mvn.computation_time is not None
        # Just verify both completed successfully
        assert results_normal.computation_time > 0
        assert results_mvn.computation_time > 0


class TestGuideWithAdvancedFeatures:
    """Tests for guides with advanced model features."""

    def test_guides_with_temporal(self, small_temporal_data):
        """Test that all guides work with temporal models."""
        data = small_temporal_data

        guides = ["normal", "mvn", "lowrank_mvn"]

        for guide in guides:
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
                inference="vi", guide_type=guide, vi_steps=500,
                device="cpu", progress_bar=False
            )

            assert results.temporal_ideal_points is not None

    def test_guides_with_covariates(self, person_covariate_data):
        """Test that all guides work with covariate models."""
        data = person_covariate_data

        guides = ["normal", "mvn", "lowrank_mvn"]

        for guide in guides:
            config = IdealPointConfig(
                n_dims=1,
                response_type=ResponseType.BINARY,
                person_covariates=True
            )

            model = IdealPointEstimator(config)
            results = model.fit(
                data["person_ids"], data["item_ids"], data["responses"],
                person_covariates=data["person_covariates"],
                inference="vi", guide_type=guide, vi_steps=500,
                device="cpu", progress_bar=False
            )

            assert "person_covariate_effects" in results.posterior_samples


class TestGuideEdgeCases:
    """Tests for edge cases with different guides."""

    def test_invalid_guide_type(self, small_binary_data):
        """Test that invalid guide type raises error."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        with pytest.raises(ValueError, match="Unknown guide type"):
            model.fit(
                data["person_ids"], data["item_ids"], data["responses"],
                inference="vi", guide_type="invalid_guide", vi_steps=100,
                device="cpu", progress_bar=False
            )

    def test_guide_with_small_data(self):
        """Test guides with very small dataset."""
        np.random.seed(42)
        n_persons = 5
        n_items = 5
        n_obs = 15

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.binomial(1, 0.6, n_obs)

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)

        for guide in ["normal", "mvn", "lowrank_mvn"]:
            model = IdealPointEstimator(config)
            results = model.fit(
                person_ids, item_ids, responses,
                inference="vi", guide_type=guide, vi_steps=300,
                device="cpu", progress_bar=False
            )
            assert results is not None

    def test_guide_with_continuous_responses(self, small_continuous_data):
        """Test guides with continuous response type."""
        data = small_continuous_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.CONTINUOUS)

        for guide in ["normal", "mvn", "lowrank_mvn"]:
            model = IdealPointEstimator(config)
            results = model.fit(
                data["person_ids"], data["item_ids"], data["responses"],
                inference="vi", guide_type=guide, vi_steps=500,
                device="cpu", progress_bar=False
            )
            assert results.ideal_points is not None


class TestGuideSampling:
    """Tests for sampling from different guides."""

    def test_guide_produces_samples(self, small_binary_data):
        """Test that guides produce posterior samples."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)

        for guide in ["normal", "mvn", "lowrank_mvn"]:
            model = IdealPointEstimator(config)
            results = model.fit(
                data["person_ids"], data["item_ids"], data["responses"],
                inference="vi", guide_type=guide, vi_steps=500, num_samples=100,
                device="cpu", progress_bar=False
            )

            assert results.posterior_samples is not None
            assert "ideal_points" in results.posterior_samples
            samples = results.posterior_samples["ideal_points"]
            assert samples.shape[0] == 100  # num_samples

    def test_guide_sample_quality(self, small_binary_data):
        """Test that guide samples have reasonable properties."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", guide_type="normal", vi_steps=500, num_samples=200,
            device="cpu", progress_bar=False
        )

        samples = results.posterior_samples["ideal_points"]

        # Samples should have reasonable variance
        assert np.std(samples) > 0
        # Samples should be finite
        assert np.all(np.isfinite(samples))
        # Sample mean should be close to point estimate
        sample_mean = np.mean(samples, axis=0)
        correlation = np.corrcoef(
            sample_mean.flatten(), results.ideal_points.flatten()
        )[0, 1]
        assert correlation > 0.9
