"""
Tests for prediction functionality.

Tests the predict() method which is essential for making predictions
on new data after model fitting, generating posterior predictive distributions,
and computing uncertainty estimates.
"""

import numpy as np
import pytest

from idealist import IdealPointConfig, IdealPointEstimator, ResponseType


class TestBasicPrediction:
    """Tests for basic prediction functionality."""

    def test_predict_basic_vi(self, small_binary_data):
        """Test basic prediction after VI inference."""
        data = small_binary_data

        # Fit model on subset of data
        train_mask = np.random.rand(len(data["responses"])) < 0.8
        train_person_ids = data["person_ids"][train_mask]
        train_item_ids = data["item_ids"][train_mask]
        train_responses = data["responses"][train_mask]

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            train_person_ids, train_item_ids, train_responses,
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        # Make predictions on held-out data
        test_mask = ~train_mask
        test_person_ids = data["person_ids"][test_mask]
        test_item_ids = data["item_ids"][test_mask]

        predictions = model.predict(test_person_ids, test_item_ids, return_samples=False)

        # Check predictions are valid
        assert predictions is not None
        assert len(predictions) == np.sum(test_mask)
        assert np.all(predictions >= 0) and np.all(predictions <= 1)

    def test_predict_returns_correct_shape(self, small_binary_data):
        """Test that predict() returns arrays with correct dimensions."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        # Fit model
        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        # Make predictions on subset
        n_pred = 50
        pred_person_ids = data["person_ids"][:n_pred]
        pred_item_ids = data["item_ids"][:n_pred]

        predictions = model.predict(pred_person_ids, pred_item_ids)

        # Check shape matches number of predictions
        assert len(predictions) == n_pred

    def test_predict_with_samples(self, small_binary_data):
        """Test predict() with return_samples=True."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        # Fit model
        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=500, num_samples=100, device="cpu", progress_bar=False
        )

        # Make predictions with samples
        n_pred = 20
        pred_person_ids = data["person_ids"][:n_pred]
        pred_item_ids = data["item_ids"][:n_pred]

        predictions = model.predict(pred_person_ids, pred_item_ids, return_samples=True)

        # Check that samples are returned
        assert isinstance(predictions, np.ndarray)
        assert predictions.ndim == 2  # (n_samples, n_predictions)
        assert predictions.shape[1] == n_pred

    def test_predict_mean_only(self, small_binary_data):
        """Test predict() with return_samples=False (default)."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        predictions = model.predict(
            data["person_ids"][:10], data["item_ids"][:10], return_samples=False
        )

        # Should return mean predictions only
        assert isinstance(predictions, np.ndarray)
        assert predictions.ndim == 1  # 1D array of mean predictions
        assert len(predictions) == 10


class TestPredictionResponseTypes:
    """Tests for predictions across different response types."""

    def test_predict_binary(self, small_binary_data):
        """Test predictions for binary responses."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        predictions = model.predict(data["person_ids"][:20], data["item_ids"][:20])

        # Binary predictions should be probabilities in [0, 1]
        assert np.all(predictions >= 0) and np.all(predictions <= 1)

    def test_predict_ordinal(self, small_ordinal_data):
        """Test predictions for ordinal responses."""
        data = small_ordinal_data

        config = IdealPointConfig(
            n_dims=1, response_type=ResponseType.ORDINAL, n_categories=data["n_categories"]
        )
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        predictions = model.predict(data["person_ids"][:20], data["item_ids"][:20])

        # Ordinal predictions should be in valid range
        assert predictions is not None
        assert len(predictions) == 20

    def test_predict_continuous(self, small_continuous_data):
        """Test predictions for continuous responses."""
        data = small_continuous_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.CONTINUOUS)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        predictions = model.predict(data["person_ids"][:20], data["item_ids"][:20])

        # Continuous predictions should be real-valued
        assert predictions is not None
        assert len(predictions) == 20
        assert np.all(np.isfinite(predictions))

    def test_predict_count(self, small_count_data):
        """Test predictions for count data."""
        data = small_count_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.COUNT)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        predictions = model.predict(data["person_ids"][:20], data["item_ids"][:20])

        # Count predictions should be non-negative
        assert predictions is not None
        assert len(predictions) == 20
        assert np.all(predictions >= 0)


class TestPredictionInferenceMethods:
    """Tests for predictions after different inference methods."""

    def test_predict_after_map(self, small_binary_data):
        """Test predictions after MAP inference."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="map", map_steps=500, device="cpu", progress_bar=False
        )

        predictions = model.predict(data["person_ids"][:20], data["item_ids"][:20])

        assert predictions is not None
        assert len(predictions) == 20
        assert np.all(predictions >= 0) and np.all(predictions <= 1)

    def test_predict_after_vi(self, small_binary_data):
        """Test predictions after VI inference."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        predictions = model.predict(data["person_ids"][:20], data["item_ids"][:20])

        assert predictions is not None
        assert len(predictions) == 20

    @pytest.mark.slow
    def test_predict_after_mcmc(self, small_binary_data):
        """Test predictions after MCMC inference."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=1, num_warmup=100, num_samples=100,
            device="cpu", progress_bar=False
        )

        predictions = model.predict(data["person_ids"][:20], data["item_ids"][:20])

        assert predictions is not None
        assert len(predictions) == 20


class TestPredictionAdvancedFeatures:
    """Tests for predictions with advanced model features."""

    def test_predict_temporal(self, small_temporal_data):
        """Test predictions with temporal dynamics models."""
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

        # Make predictions at specific timepoint
        n_pred = 20
        pred_person_ids = data["person_ids"][:n_pred]
        pred_item_ids = data["item_ids"][:n_pred]
        pred_timepoints = data["timepoints"][:n_pred]

        predictions = model.predict(
            pred_person_ids, pred_item_ids, timepoints=pred_timepoints
        )

        assert predictions is not None
        assert len(predictions) == n_pred

    def test_predict_with_person_covariates(self, person_covariate_data):
        """Test predictions when model includes person covariates."""
        data = person_covariate_data

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            person_covariates=True
        )
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            person_covariates=data["person_covariates"],
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        # Make predictions
        predictions = model.predict(data["person_ids"][:20], data["item_ids"][:20])

        assert predictions is not None
        assert len(predictions) == 20


class TestPredictionEdgeCases:
    """Tests for edge cases in prediction."""

    def test_predict_all_same_person(self, small_binary_data):
        """Test predictions for same person across different items."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        # Predict for same person, different items
        person_id = 0
        n_items = min(10, data["n_items"])
        pred_person_ids = np.repeat(person_id, n_items)
        pred_item_ids = np.arange(n_items)

        predictions = model.predict(pred_person_ids, pred_item_ids)

        assert predictions is not None
        assert len(predictions) == n_items
        # Predictions should vary across items
        assert np.std(predictions) > 0

    def test_predict_all_same_item(self, small_binary_data):
        """Test predictions for same item across different persons."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        # Predict for same item, different persons
        item_id = 0
        n_persons = min(10, data["n_persons"])
        pred_person_ids = np.arange(n_persons)
        pred_item_ids = np.repeat(item_id, n_persons)

        predictions = model.predict(pred_person_ids, pred_item_ids)

        assert predictions is not None
        assert len(predictions) == n_persons
        # Predictions should vary across persons
        assert np.std(predictions) > 0

    def test_predict_single_pair(self, small_binary_data):
        """Test prediction for a single person-item pair."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        # Predict for single pair
        predictions = model.predict(np.array([0]), np.array([0]))

        assert predictions is not None
        assert len(predictions) == 1
        assert 0 <= predictions[0] <= 1


class TestPredictionMultidimensional:
    """Tests for predictions with multidimensional models."""

    def test_predict_2d_model(self, multidim_binary_data):
        """Test predictions with 2-dimensional ideal points."""
        data = multidim_binary_data

        config = IdealPointConfig(n_dims=2, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        predictions = model.predict(data["person_ids"][:20], data["item_ids"][:20])

        assert predictions is not None
        assert len(predictions) == 20
        assert np.all(predictions >= 0) and np.all(predictions <= 1)
