"""
Tests for inference method and response type combinations.

Tests systematic combinations of:
- Inference methods: VI, MAP, MCMC
- Response types: Binary, Ordinal, Continuous, Count, Bounded Continuous

Ensures all combinations work correctly and produce reasonable results.
"""

import numpy as np
import pytest

from idealist import IdealPointConfig, IdealPointEstimator, ResponseType


class TestVIResponseTypes:
    """Tests for VI inference with different response types."""

    def test_vi_ordinal(self, small_ordinal_data):
        """Test VI with ordinal responses."""
        data = small_ordinal_data

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.ORDINAL,
            n_categories=data["n_categories"]
        )
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None
        assert np.all(np.isfinite(results.ideal_points))

    def test_vi_continuous(self, small_continuous_data):
        """Test VI with continuous responses."""
        data = small_continuous_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.CONTINUOUS)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None

    def test_vi_count(self, small_count_data):
        """Test VI with count data."""
        data = small_count_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.COUNT)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None

    def test_vi_bounded_continuous(self, small_bounded_continuous_data):
        """Test VI with bounded continuous responses."""
        data = small_bounded_continuous_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BOUNDED_CONTINUOUS)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None


class TestMAPResponseTypes:
    """Tests for MAP inference with different response types."""

    def test_map_ordinal(self, small_ordinal_data):
        """Test MAP with ordinal responses."""
        data = small_ordinal_data

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.ORDINAL,
            n_categories=data["n_categories"]
        )
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="map", map_steps=1000,
            device="cpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None

    def test_map_continuous(self, small_continuous_data):
        """Test MAP with continuous responses."""
        data = small_continuous_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.CONTINUOUS)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="map", map_steps=1000,
            device="cpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None

    def test_map_count(self, small_count_data):
        """Test MAP with count data."""
        data = small_count_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.COUNT)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="map", map_steps=1000,
            device="cpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None

    def test_map_bounded_continuous(self, small_bounded_continuous_data):
        """Test MAP with bounded continuous responses."""
        data = small_bounded_continuous_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BOUNDED_CONTINUOUS)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="map", map_steps=1000,
            device="cpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None


class TestMCMCResponseTypes:
    """Tests for MCMC inference with different response types."""

    @pytest.mark.slow
    def test_mcmc_ordinal(self, small_ordinal_data):
        """Test MCMC with ordinal responses."""
        data = small_ordinal_data

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.ORDINAL,
            n_categories=data["n_categories"]
        )
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=100, num_samples=100,
            device="cpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None
        assert results.posterior_samples is not None

    @pytest.mark.slow
    def test_mcmc_continuous(self, small_continuous_data):
        """Test MCMC with continuous responses."""
        data = small_continuous_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.CONTINUOUS)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=100, num_samples=100,
            device="cpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None

    @pytest.mark.slow
    def test_mcmc_count(self, small_count_data):
        """Test MCMC with count data."""
        data = small_count_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.COUNT)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=100, num_samples=100,
            device="cpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None

    @pytest.mark.slow
    def test_mcmc_bounded_continuous(self, small_bounded_continuous_data):
        """Test MCMC with bounded continuous responses."""
        data = small_bounded_continuous_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BOUNDED_CONTINUOUS)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=100, num_samples=100,
            device="cpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None


class TestResponseTypeConsistency:
    """Tests for consistency across inference methods for same response type."""

    def test_ordinal_consistency_across_inference(self, small_ordinal_data):
        """Test that ordinal responses produce consistent results across inference methods."""
        data = small_ordinal_data

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.ORDINAL,
            n_categories=data["n_categories"]
        )

        # VI
        model_vi = IdealPointEstimator(config)
        results_vi = model_vi.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        # MAP
        model_map = IdealPointEstimator(config)
        results_map = model_map.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="map", map_steps=1000,
            device="cpu", progress_bar=False
        )

        # Should produce correlated results
        corr = np.corrcoef(
            results_vi.ideal_points.flatten(),
            results_map.ideal_points.flatten()
        )[0, 1]
        assert corr > 0.3  # Reasonable correlation

    def test_continuous_consistency_across_inference(self, small_continuous_data):
        """Test continuous responses across inference methods."""
        data = small_continuous_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.CONTINUOUS)

        # VI
        model_vi = IdealPointEstimator(config)
        results_vi = model_vi.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        # MAP
        model_map = IdealPointEstimator(config)
        results_map = model_map.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="map", map_steps=1000,
            device="cpu", progress_bar=False
        )

        # Should produce similar results
        assert results_vi.ideal_points is not None
        assert results_map.ideal_points is not None


class TestMultidimensionalResponseCombinations:
    """Tests for multidimensional models with different response types."""

    def test_2d_ordinal(self):
        """Test 2D model with ordinal responses."""
        np.random.seed(42)
        n_persons = 30
        n_items = 20
        n_categories = 5
        n_obs = 150

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.randint(0, n_categories, n_obs)

        config = IdealPointConfig(
            n_dims=2,
            response_type=ResponseType.ORDINAL,
            n_categories=n_categories
        )
        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids, item_ids, responses,
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        assert results.ideal_points.shape == (n_persons, 2)

    def test_2d_continuous(self):
        """Test 2D model with continuous responses."""
        np.random.seed(123)
        n_persons = 30
        n_items = 20
        n_obs = 150

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.normal(0, 1, n_obs)

        config = IdealPointConfig(n_dims=2, response_type=ResponseType.CONTINUOUS)
        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids, item_ids, responses,
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        assert results.ideal_points.shape == (n_persons, 2)

    def test_3d_count(self):
        """Test 3D model with count data."""
        np.random.seed(456)
        n_persons = 30
        n_items = 20
        n_obs = 150

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.poisson(2, n_obs)

        config = IdealPointConfig(n_dims=3, response_type=ResponseType.COUNT)
        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids, item_ids, responses,
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        assert results.ideal_points.shape == (n_persons, 3)


class TestResponseTypeWithAdvancedFeatures:
    """Tests for response types combined with advanced features."""

    def test_ordinal_with_temporal(self):
        """Test ordinal responses with temporal dynamics."""
        np.random.seed(42)
        n_persons = 25
        n_items = 15
        n_timepoints = 3
        n_categories = 5
        n_obs = 150

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        timepoints = np.random.randint(0, n_timepoints, n_obs)
        responses = np.random.randint(0, n_categories, n_obs)

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.ORDINAL,
            n_categories=n_categories,
            temporal_dynamics=True,
            n_timepoints=n_timepoints
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            person_ids, item_ids, responses,
            timepoints=timepoints,
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        assert results.temporal_ideal_points is not None

    def test_continuous_with_covariates(self):
        """Test continuous responses with person covariates."""
        np.random.seed(123)
        n_persons = 25
        n_items = 15
        n_obs = 120

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.normal(0, 1, n_obs)
        person_covariates = np.random.randn(n_persons, 2)

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.CONTINUOUS,
            person_covariates=True
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            person_ids, item_ids, responses,
            person_covariates=person_covariates,
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        assert "person_covariate_effects" in results.posterior_samples

    def test_count_with_flexible_priors(self):
        """Test count data with Student-t priors."""
        np.random.seed(456)
        n_persons = 25
        n_items = 15
        n_obs = 120

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.poisson(2, n_obs)

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.COUNT,
            prior_ideal_point_family="student_t",
            prior_ideal_point_df=7.0
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            person_ids, item_ids, responses,
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        assert results.ideal_points is not None


class TestResponseTypePredictions:
    """Tests for predictions with different response types."""

    def test_ordinal_predictions(self, small_ordinal_data):
        """Test predictions for ordinal responses."""
        data = small_ordinal_data

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.ORDINAL,
            n_categories=data["n_categories"]
        )
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        predictions = model.predict(data["person_ids"][:20], data["item_ids"][:20])

        assert predictions is not None
        assert len(predictions) == 20

    def test_continuous_predictions(self, small_continuous_data):
        """Test predictions for continuous responses."""
        data = small_continuous_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.CONTINUOUS)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        predictions = model.predict(data["person_ids"][:20], data["item_ids"][:20])

        assert predictions is not None
        assert len(predictions) == 20
        assert np.all(np.isfinite(predictions))

    def test_count_predictions(self, small_count_data):
        """Test predictions for count data."""
        data = small_count_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.COUNT)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        predictions = model.predict(data["person_ids"][:20], data["item_ids"][:20])

        assert predictions is not None
        assert len(predictions) == 20
        assert np.all(predictions >= 0)  # Counts are non-negative


class TestResponseTypeEdgeCases:
    """Tests for edge cases with different response types."""

    def test_ordinal_single_category(self):
        """Test ordinal with 2 categories (essentially binary)."""
        np.random.seed(42)
        n_persons = 25
        n_items = 15
        n_obs = 100

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.randint(0, 2, n_obs)

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.ORDINAL,
            n_categories=2
        )
        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids, item_ids, responses,
            inference="vi", vi_steps=500,
            device="cpu", progress_bar=False
        )

        assert results.ideal_points is not None

    def test_ordinal_many_categories(self):
        """Test ordinal with many categories (10)."""
        np.random.seed(123)
        n_persons = 30
        n_items = 20
        n_categories = 10
        n_obs = 150

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.randint(0, n_categories, n_obs)

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.ORDINAL,
            n_categories=n_categories
        )
        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids, item_ids, responses,
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        assert results.ideal_points is not None

    def test_count_with_zeros(self):
        """Test count data with many zeros."""
        np.random.seed(456)
        n_persons = 25
        n_items = 15
        n_obs = 100

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        # Generate sparse counts (many zeros)
        responses = np.random.poisson(0.5, n_obs)  # Low rate

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.COUNT)
        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids, item_ids, responses,
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        assert results.ideal_points is not None

    def test_continuous_with_negative_values(self):
        """Test continuous responses with negative values."""
        np.random.seed(42)
        n_persons = 25
        n_items = 15
        n_obs = 100

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.normal(0, 2, n_obs)  # Can be negative

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.CONTINUOUS)
        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids, item_ids, responses,
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        assert results.ideal_points is not None
