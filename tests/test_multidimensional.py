"""
Tests for multi-dimensional ideal point estimation.

Tests that the IdealPointEstimator can handle 2D and higher dimensional
latent spaces.
"""

import numpy as np
import pytest

from idealist import IdealPointConfig, IdealPointEstimator, ResponseType


class TestMultiDimensional:
    """Tests for multi-dimensional ideal point models."""

    def test_2d_estimation(self, multidim_binary_data):
        """Test 2-dimensional ideal point estimation."""
        data = multidim_binary_data

        assert data["n_dims"] == 2, "Fixture should have 2 dimensions"

        config = IdealPointConfig(
            n_dims=data["n_dims"],
            response_type=ResponseType.BINARY,
        )

        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids=data["person_ids"],
            item_ids=data["item_ids"],
            responses=data["responses"],
            inference="vi",
            vi_steps=1500,
            num_samples=300,
            device="cpu",
            progress_bar=False,
        )

        # Check results structure
        assert results is not None
        assert results.ideal_points.shape == (data["n_persons"], 2)
        assert results.discrimination.shape == (data["n_items"], 2)
        assert results.difficulty.shape == (data["n_items"],)

        # Check uncertainty quantification for both dimensions
        assert results.ideal_points_std.shape == (data["n_persons"], 2)

        print(f"\n  2D estimation completed in {results.computation_time:.2f}s")

    def test_2d_parameter_recovery(self, multidim_binary_data):
        """Test that we can recover parameters in 2D space."""
        data = multidim_binary_data

        config = IdealPointConfig(
            n_dims=2,
            response_type=ResponseType.BINARY,
        )

        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids=data["person_ids"],
            item_ids=data["item_ids"],
            responses=data["responses"],
            inference="vi",
            vi_steps=2000,
            num_samples=500,
            device="cpu",
            progress_bar=False,
        )

        # Check that we got reasonable estimates for both dimensions
        # Note: dimensions may be rotated/reflected, so we check general properties
        assert np.abs(results.ideal_points[:, 0]).max() < 10.0
        assert np.abs(results.ideal_points[:, 1]).max() < 10.0

        # Check that both dimensions have non-zero variance
        var_dim1 = np.var(results.ideal_points[:, 0])
        var_dim2 = np.var(results.ideal_points[:, 1])
        assert var_dim1 > 0.1, "First dimension should have meaningful variation"
        assert var_dim2 > 0.1, "Second dimension should have meaningful variation"

        print(f"\n  2D parameter recovery: var_dim1={var_dim1:.2f}, var_dim2={var_dim2:.2f}")

    @pytest.mark.slow
    def test_2d_map_estimation(self, multidim_binary_data):
        """Test MAP estimation with 2D data."""
        data = multidim_binary_data

        config = IdealPointConfig(
            n_dims=2,
            response_type=ResponseType.BINARY,
        )

        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids=data["person_ids"],
            item_ids=data["item_ids"],
            responses=data["responses"],
            inference="map",
            map_steps=1500,
            device="cpu",
            progress_bar=False,
        )

        assert results is not None
        assert results.ideal_points.shape == (data["n_persons"], 2)

        print(f"\n  2D MAP completed in {results.computation_time:.2f}s")

    def test_dataframe_with_multidim(self, multidim_binary_data):
        """Test DataFrame conversion with multi-dimensional results."""
        data = multidim_binary_data

        config = IdealPointConfig(
            n_dims=2,
            response_type=ResponseType.BINARY,
        )

        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids=data["person_ids"],
            item_ids=data["item_ids"],
            responses=data["responses"],
            inference="vi",
            vi_steps=1000,
            device="cpu",
            progress_bar=False,
        )

        df_dict = results.to_dataframe()

        # Check that DataFrame has columns for both dimensions
        persons_df = df_dict["persons"]
        items_df = df_dict["items"]

        # Should have ideal_point columns (might be named differently for multidim)
        assert len(persons_df) == data["n_persons"]
        assert len(items_df) == data["n_items"]

        print(f"\n  2D DataFrame: {persons_df.shape}, {items_df.shape}")


class TestHigherDimensional:
    """Tests for 3D and higher dimensional models."""

    def test_3d_estimation(self):
        """Test 3-dimensional ideal point estimation."""
        np.random.seed(42)
        n_persons = 30
        n_items = 20
        n_dims = 3
        n_obs = 200

        # Generate 3D data
        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.binomial(1, 0.6, n_obs)

        config = IdealPointConfig(n_dims=3, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids, item_ids, responses,
            inference="vi", vi_steps=1000, device="cpu", progress_bar=False
        )

        # Check 3D structure
        assert results.ideal_points.shape == (n_persons, 3)
        assert results.discrimination.shape == (n_items, 3)
        assert results.ideal_points_std.shape == (n_persons, 3)

        # All dimensions should have variation
        for dim in range(3):
            assert np.var(results.ideal_points[:, dim]) > 0.05

    def test_4d_estimation(self):
        """Test 4-dimensional ideal point estimation."""
        np.random.seed(123)
        n_persons = 35
        n_items = 25
        n_dims = 4
        n_obs = 250

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.binomial(1, 0.6, n_obs)

        config = IdealPointConfig(n_dims=4, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids, item_ids, responses,
            inference="vi", vi_steps=1000, device="cpu", progress_bar=False
        )

        # Check 4D structure
        assert results.ideal_points.shape == (n_persons, 4)
        assert results.discrimination.shape == (n_items, 4)

    def test_5d_estimation(self):
        """Test 5-dimensional ideal point estimation."""
        np.random.seed(456)
        n_persons = 40
        n_items = 30
        n_dims = 5
        n_obs = 300

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.binomial(1, 0.6, n_obs)

        config = IdealPointConfig(n_dims=5, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids, item_ids, responses,
            inference="vi", vi_steps=1000, device="cpu", progress_bar=False
        )

        assert results.ideal_points.shape == (n_persons, 5)

    @pytest.mark.slow
    def test_3d_mcmc(self):
        """Test MCMC with 3D model."""
        np.random.seed(42)
        n_persons = 25
        n_items = 15
        n_obs = 150

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.binomial(1, 0.6, n_obs)

        config = IdealPointConfig(n_dims=3, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids, item_ids, responses,
            inference="mcmc", num_chains=2, num_warmup=100, num_samples=100,
            device="cpu", progress_bar=False
        )

        assert results.ideal_points.shape == (n_persons, 3)
        assert results.posterior_samples is not None


class TestMultidimensionalWithFeatures:
    """Tests for multidimensional models with advanced features."""

    def test_3d_with_temporal(self):
        """Test 3D model with temporal dynamics."""
        np.random.seed(42)
        n_persons = 25
        n_items = 15
        n_dims = 3
        n_timepoints = 3
        n_obs = 150

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        timepoints = np.random.randint(0, n_timepoints, n_obs)
        responses = np.random.binomial(1, 0.6, n_obs)

        config = IdealPointConfig(
            n_dims=3,
            response_type=ResponseType.BINARY,
            temporal_dynamics=True,
            n_timepoints=n_timepoints
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            person_ids, item_ids, responses,
            timepoints=timepoints,
            inference="vi", vi_steps=1000, device="cpu", progress_bar=False
        )

        # Check temporal 3D structure
        assert results.temporal_ideal_points.shape == (n_timepoints, n_persons, 3)

    def test_3d_with_covariates(self):
        """Test 3D model with person covariates."""
        np.random.seed(123)
        n_persons = 30
        n_items = 20
        n_dims = 3
        n_obs = 180

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.binomial(1, 0.6, n_obs)
        person_covariates = np.random.randn(n_persons, 2)

        config = IdealPointConfig(
            n_dims=3,
            response_type=ResponseType.BINARY,
            person_covariates=True
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            person_ids, item_ids, responses,
            person_covariates=person_covariates,
            inference="vi", vi_steps=1000, device="cpu", progress_bar=False
        )

        # Covariate effects should have shape (n_samples, n_covariates, n_dims)
        gamma = results.posterior_samples["person_covariate_effects"]
        assert gamma.shape[2] == 3  # n_dims

    def test_3d_ordinal(self):
        """Test 3D model with ordinal responses."""
        np.random.seed(456)
        n_persons = 30
        n_items = 20
        n_dims = 3
        n_categories = 5
        n_obs = 180

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.randint(0, n_categories, n_obs)

        config = IdealPointConfig(
            n_dims=3,
            response_type=ResponseType.ORDINAL,
            n_categories=n_categories
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            person_ids, item_ids, responses,
            inference="vi", vi_steps=1000, device="cpu", progress_bar=False
        )

        assert results.ideal_points.shape == (n_persons, 3)


class TestMultidimensionalPredictions:
    """Tests for predictions with multidimensional models."""

    def test_3d_predictions(self):
        """Test predictions with 3D model."""
        np.random.seed(42)
        n_persons = 30
        n_items = 20
        n_obs = 180

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.binomial(1, 0.6, n_obs)

        config = IdealPointConfig(n_dims=3, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids, item_ids, responses,
            inference="vi", vi_steps=1000, device="cpu", progress_bar=False
        )

        # Make predictions
        predictions = model.predict(person_ids[:20], item_ids[:20])

        assert predictions is not None
        assert len(predictions) == 20
        assert np.all(predictions >= 0) and np.all(predictions <= 1)

    def test_4d_predictions_with_samples(self):
        """Test predictions with samples from 4D model."""
        np.random.seed(123)
        n_persons = 30
        n_items = 20
        n_obs = 180

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.binomial(1, 0.6, n_obs)

        config = IdealPointConfig(n_dims=4, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids, item_ids, responses,
            inference="vi", vi_steps=1000, num_samples=100,
            device="cpu", progress_bar=False
        )

        # Predictions with samples
        predictions = model.predict(person_ids[:10], item_ids[:10], return_samples=True)

        assert predictions.shape == (100, 10)  # (n_samples, n_predictions)


class TestMultidimensionalEdgeCases:
    """Tests for edge cases with multidimensional models."""

    def test_single_dimension_still_works(self):
        """Test that n_dims=1 still works (regression test)."""
        np.random.seed(42)
        n_persons = 25
        n_items = 15
        n_obs = 100

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.binomial(1, 0.6, n_obs)

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids, item_ids, responses,
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        assert results.ideal_points.shape == (n_persons, 1)

    def test_very_high_dimensional(self):
        """Test with very high dimensionality (n_dims=10)."""
        np.random.seed(42)
        n_persons = 50
        n_items = 40
        n_dims = 10
        n_obs = 400

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.binomial(1, 0.6, n_obs)

        config = IdealPointConfig(n_dims=10, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids, item_ids, responses,
            inference="vi", vi_steps=1000, device="cpu", progress_bar=False
        )

        assert results.ideal_points.shape == (n_persons, 10)
        assert results.discrimination.shape == (n_items, 10)

    def test_multidim_with_sparse_data(self):
        """Test multidimensional model with sparse observations."""
        np.random.seed(42)
        n_persons = 40
        n_items = 30
        n_dims = 3
        n_obs = 80  # Very sparse

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.binomial(1, 0.6, n_obs)

        config = IdealPointConfig(n_dims=3, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids, item_ids, responses,
            inference="vi", vi_steps=1000, device="cpu", progress_bar=False
        )

        # Should still complete despite sparsity
        assert results.ideal_points.shape == (n_persons, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
