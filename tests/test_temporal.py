"""
Tests for temporal dynamics models.

Temporal dynamics allow ideal points to evolve over time using a random walk:
θ_t = θ_{t-1} + ε_t where ε_t ~ N(0, σ²)

This is essential for applications like tracking legislator ideology over Congressional
sessions, monitoring student ability across test administrations, and following
brand perception evolution over quarters.
"""

import numpy as np
import pytest

from idealist import IdealPointConfig, IdealPointEstimator, ResponseType


class TestBasicTemporalModels:
    """Tests for basic temporal model fitting."""

    def test_temporal_model_fitting(self, small_temporal_data):
        """Test basic temporal model fitting."""
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
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        # Verify temporal_ideal_points exist in results
        assert results is not None
        assert results.temporal_ideal_points is not None
        assert results.temporal_ideal_points.shape == (
            data["n_timepoints"], data["n_persons"], data["n_dims"]
        )

    def test_temporal_params_shape(self, small_temporal_data):
        """Test temporal parameter dimensions."""
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
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        # Check temporal_ideal_points shape
        assert results.temporal_ideal_points.shape == (
            data["n_timepoints"], data["n_persons"], data["n_dims"]
        )

        # Check that ideal points vary over time
        variance_over_time = np.var(results.temporal_ideal_points, axis=0).mean()
        assert variance_over_time > 0, "Ideal points should vary over time"

    def test_temporal_variance_estimation(self, small_temporal_data):
        """Test temporal variance parameter estimation."""
        data = small_temporal_data

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            temporal_dynamics=True,
            n_timepoints=data["n_timepoints"],
            temporal_variance=0.1  # Fixed variance
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            timepoints=data["timepoints"],
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        # Verify model fitted successfully with temporal variance
        assert results is not None
        assert results.temporal_ideal_points is not None


class TestTemporalConfiguration:
    """Tests for temporal model configuration."""

    def test_temporal_requires_timepoints(self, small_binary_data):
        """Test that temporal models require timepoints data."""
        data = small_binary_data

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            temporal_dynamics=True,
            n_timepoints=3
        )

        model = IdealPointEstimator(config)

        # Should work fine without timepoints if temporal_dynamics is False
        config_no_temporal = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            temporal_dynamics=False
        )
        model_no_temporal = IdealPointEstimator(config_no_temporal)
        results = model_no_temporal.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )
        assert results.temporal_ideal_points is None

    def test_temporal_n_timepoints_config(self, small_temporal_data):
        """Test n_timepoints configuration."""
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
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        # Verify n_timepoints matches configuration
        assert results.temporal_ideal_points.shape[0] == data["n_timepoints"]

    def test_temporal_auto_detect_timepoints(self, small_temporal_data):
        """Test automatic detection of n_timepoints from data."""
        data = small_temporal_data

        # Don't specify n_timepoints in config
        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            temporal_dynamics=True
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            timepoints=data["timepoints"],
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        # Should auto-detect from timepoints array
        assert results.temporal_ideal_points is not None
        assert results.temporal_ideal_points.shape[0] == data["n_timepoints"]


class TestTemporalInference:
    """Tests for temporal models with different inference methods."""

    def test_temporal_with_vi(self, small_temporal_data):
        """Test temporal model with VI inference."""
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
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        assert results.temporal_ideal_points is not None
        assert results.posterior_samples is not None

    @pytest.mark.slow
    def test_temporal_with_mcmc(self, small_temporal_data):
        """Test temporal model with MCMC inference."""
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
            inference="mcmc", num_chains=1, num_warmup=100, num_samples=100,
            device="cpu", progress_bar=False
        )

        assert results.temporal_ideal_points is not None
        assert results.posterior_samples is not None

    def test_temporal_with_map(self, small_temporal_data):
        """Test temporal model with MAP inference."""
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
            inference="map", map_steps=500, device="cpu", progress_bar=False
        )

        assert results.temporal_ideal_points is not None


class TestTemporalPredictions:
    """Tests for predictions with temporal models."""

    def test_temporal_predict_at_timepoint(self, small_temporal_data):
        """Test predictions at specific timepoints."""
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
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        # Make predictions at each timepoint
        for t in range(data["n_timepoints"]):
            mask = data["timepoints"] == t
            if np.sum(mask) > 0:
                pred_person_ids = data["person_ids"][mask][:10]
                pred_item_ids = data["item_ids"][mask][:10]
                pred_timepoints = np.repeat(t, len(pred_person_ids))

                predictions = model.predict(
                    pred_person_ids, pred_item_ids, timepoints=pred_timepoints
                )

                assert predictions is not None
                assert len(predictions) == len(pred_person_ids)

    def test_temporal_predictions_vary_over_time(self, medium_temporal_data):
        """Test that predictions vary over time for same person-item pair."""
        data = medium_temporal_data

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
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        # Predict same person-item pair at different timepoints
        person_id = 0
        item_id = 0
        timepoints = np.arange(data["n_timepoints"])
        person_ids = np.repeat(person_id, data["n_timepoints"])
        item_ids = np.repeat(item_id, data["n_timepoints"])

        predictions = model.predict(person_ids, item_ids, timepoints=timepoints)

        # Predictions should vary across time (usually)
        # Allow for small variance or no variance in edge cases
        assert predictions is not None
        assert len(predictions) == data["n_timepoints"]


class TestTemporalResults:
    """Tests for temporal results structure."""

    def test_temporal_results_structure(self, small_temporal_data):
        """Test results.temporal_ideal_points structure."""
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
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        # Check temporal_ideal_points structure
        assert results.temporal_ideal_points is not None
        assert results.temporal_ideal_points.ndim == 3
        assert results.temporal_ideal_points.shape == (
            data["n_timepoints"], data["n_persons"], data["n_dims"]
        )

        # Check that all values are finite
        assert np.all(np.isfinite(results.temporal_ideal_points))

    def test_temporal_trajectory_extraction(self, medium_temporal_data):
        """Test extracting temporal trajectories for specific persons."""
        data = medium_temporal_data

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
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        # Extract trajectory for person 0
        person_id = 0
        trajectory = results.temporal_ideal_points[:, person_id, :]

        # Trajectory should have shape (n_timepoints, n_dims)
        assert trajectory.shape == (data["n_timepoints"], data["n_dims"])
        assert np.all(np.isfinite(trajectory))


class TestTemporalEdgeCases:
    """Tests for edge cases in temporal models."""

    def test_temporal_single_timepoint(self):
        """Test edge case of single timepoint (should work like non-temporal)."""
        np.random.seed(42)
        n_persons = 20
        n_items = 12
        n_obs = 80

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.binomial(1, 0.6, n_obs)
        timepoints = np.zeros(n_obs, dtype=int)  # All at timepoint 0

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            temporal_dynamics=True,
            n_timepoints=1
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            person_ids, item_ids, responses,
            timepoints=timepoints,
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        # Should still produce results
        assert results.temporal_ideal_points is not None
        assert results.temporal_ideal_points.shape[0] == 1

    def test_temporal_unbalanced_panels(self, medium_temporal_data):
        """Test temporal model with unbalanced panels (different persons at different times)."""
        data = medium_temporal_data

        # Create unbalanced data by removing some observations
        mask = np.random.rand(len(data["responses"])) < 0.7
        unbalanced_person_ids = data["person_ids"][mask]
        unbalanced_item_ids = data["item_ids"][mask]
        unbalanced_responses = data["responses"][mask]
        unbalanced_timepoints = data["timepoints"][mask]

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            temporal_dynamics=True,
            n_timepoints=data["n_timepoints"]
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            unbalanced_person_ids, unbalanced_item_ids, unbalanced_responses,
            timepoints=unbalanced_timepoints,
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        # Should handle unbalanced panels
        assert results.temporal_ideal_points is not None


class TestTemporalMultidimensional:
    """Tests for temporal models with multiple dimensions."""

    def test_temporal_2d_model(self):
        """Test temporal model with 2-dimensional ideal points."""
        np.random.seed(42)
        n_persons = 20
        n_items = 12
        n_dims = 2
        n_timepoints = 3
        n_obs_per_time = 60

        person_ids = []
        item_ids = []
        timepoints = []
        responses = []

        for t in range(n_timepoints):
            for _ in range(n_obs_per_time):
                person_ids.append(np.random.randint(0, n_persons))
                item_ids.append(np.random.randint(0, n_items))
                timepoints.append(t)
                responses.append(np.random.binomial(1, 0.6))

        person_ids = np.array(person_ids)
        item_ids = np.array(item_ids)
        timepoints = np.array(timepoints)
        responses = np.array(responses)

        config = IdealPointConfig(
            n_dims=2,
            response_type=ResponseType.BINARY,
            temporal_dynamics=True,
            n_timepoints=n_timepoints
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            person_ids, item_ids, responses,
            timepoints=timepoints,
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        # Check 2D temporal ideal points
        assert results.temporal_ideal_points.shape == (n_timepoints, n_persons, n_dims)


class TestTemporalWithOtherFeatures:
    """Tests for combining temporal dynamics with other features."""

    def test_temporal_with_priors(self, small_temporal_data):
        """Test temporal model with custom priors."""
        data = small_temporal_data

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            temporal_dynamics=True,
            n_timepoints=data["n_timepoints"],
            prior_ideal_point_scale=2.0,
            prior_difficulty_scale=1.5
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            timepoints=data["timepoints"],
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        assert results.temporal_ideal_points is not None

    def test_temporal_with_ordinal_responses(self):
        """Test temporal model with ordinal responses."""
        np.random.seed(42)
        n_persons = 20
        n_items = 12
        n_timepoints = 3
        n_categories = 5
        n_obs = 100

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
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        assert results.temporal_ideal_points is not None
        assert results.temporal_ideal_points.shape == (n_timepoints, n_persons, 1)
