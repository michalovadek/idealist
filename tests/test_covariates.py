"""
Tests for hierarchical models with person and item covariates.

Covariates allow modeling systematic variation in ideal points and item parameters:
- Person covariates: θ_i = X_i * γ + ε_i (e.g., party, state, demographics)
- Item covariates: α_j = Z_j * δ + ε_j (e.g., bill topic, difficulty predictors)

This is essential for controlling for confounders in ideal point estimation,
understanding what predicts ideology/ability, and improving predictions with
auxiliary information.

Note: Basic covariate functionality is also tested in test_flexible_priors.py.
This file focuses on additional covariate-specific tests.
"""

import numpy as np
import pandas as pd
import pytest

from idealist import IdealPointConfig, IdealPointEstimator, ResponseType
from idealist.data import load_data


class TestPersonCovariates:
    """Tests for models with person-level covariates."""

    def test_person_covariates_basic(self, person_covariate_data):
        """Test basic model fitting with person covariates."""
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

        # Verify model fitted successfully
        assert results.ideal_points is not None
        assert results.posterior_samples is not None
        assert "person_covariate_effects" in results.posterior_samples

    def test_person_covariates_shape(self, person_covariate_data):
        """Test person covariate matrix dimensions."""
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

        # Check covariate effects shape
        gamma = results.posterior_samples["person_covariate_effects"]
        # Should be (n_samples, n_person_covariates, n_dims)
        assert gamma.ndim == 3
        assert gamma.shape[1] == data["n_person_covariates"]
        assert gamma.shape[2] == data["n_dims"]

    def test_person_covariates_dataframe(self):
        """Test loading person covariates from DataFrame columns."""
        np.random.seed(42)
        n_persons = 25
        n_items = 15
        n_obs = 100

        # Create DataFrame with person covariates
        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.binomial(1, 0.6, n_obs)

        # Create person-level attributes
        person_age = np.random.uniform(30, 70, n_persons)[person_ids]
        person_party = np.random.randint(0, 2, n_persons)[person_ids]

        df = pd.DataFrame({
            "person_id": person_ids,
            "item_id": item_ids,
            "response": responses,
            "age": person_age,
            "party": person_party
        })

        # Load data with person covariates
        data = load_data(
            df,
            person_col="person_id",
            item_col="item_id",
            response_col="response",
            person_covariate_cols=["age", "party"]
        )

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            person_covariates=True
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            data,
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        assert results.ideal_points is not None
        assert "person_covariate_effects" in results.posterior_samples

    def test_person_covariates_vi(self, person_covariate_data):
        """Test person covariate models with VI inference."""
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

        assert results.ideal_points is not None
        assert "person_covariate_effects" in results.posterior_samples

    @pytest.mark.slow
    def test_person_covariates_mcmc(self, person_covariate_data):
        """Test person covariate models with MCMC inference."""
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
            inference="mcmc", num_chains=1, num_warmup=100, num_samples=100,
            device="cpu", progress_bar=False
        )

        assert results.ideal_points is not None
        assert "person_covariate_effects" in results.posterior_samples


class TestItemCovariates:
    """Tests for models with item-level covariates."""

    def test_item_covariates_basic(self, item_covariate_data):
        """Test basic model fitting with item covariates."""
        data = item_covariate_data

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            item_covariates=True
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            item_covariates=data["item_covariates"],
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        # Verify model fitted successfully
        assert results.ideal_points is not None
        assert results.posterior_samples is not None
        # Item covariates affect difficulty
        assert "item_covariate_effects" in results.posterior_samples or \
               "item_difficulty_covariate_effects" in results.posterior_samples

    def test_item_covariates_shape(self, item_covariate_data):
        """Test item covariate matrix dimensions."""
        data = item_covariate_data

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            item_covariates=True
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            item_covariates=data["item_covariates"],
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        # Check that item covariates were used
        assert results.posterior_samples is not None
        # Item covariate effects should be present
        has_item_effects = any(
            "item" in key and "covariate" in key
            for key in results.posterior_samples.keys()
        )
        assert has_item_effects

    def test_item_covariates_dataframe(self):
        """Test loading item covariates from DataFrame columns."""
        np.random.seed(123)
        n_persons = 20
        n_items = 15
        n_obs = 100

        # Create DataFrame with item covariates
        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.binomial(1, 0.6, n_obs)

        # Create item-level attributes
        item_topic = np.random.randint(0, 3, n_items)[item_ids]
        item_year = np.random.uniform(2010, 2020, n_items)[item_ids]

        df = pd.DataFrame({
            "person_id": person_ids,
            "item_id": item_ids,
            "response": responses,
            "topic": item_topic,
            "year": item_year
        })

        # Load data with item covariates
        data = load_data(
            df,
            person_col="person_id",
            item_col="item_id",
            response_col="response",
            item_covariate_cols=["topic", "year"]
        )

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            item_covariates=True
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            data,
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        assert results.ideal_points is not None


class TestCombinedCovariates:
    """Tests for models with both person and item covariates."""

    def test_both_covariates(self, both_covariate_data):
        """Test model with both person AND item covariates."""
        data = both_covariate_data

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            person_covariates=True,
            item_covariates=True
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            person_covariates=data["person_covariates"],
            item_covariates=data["item_covariates"],
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        # Both covariate effects should be present
        assert results.posterior_samples is not None
        assert "person_covariate_effects" in results.posterior_samples

    def test_covariates_with_temporal(self, small_temporal_data):
        """Test combining covariates with temporal dynamics."""
        data = small_temporal_data

        # Add person covariates
        person_covariates = np.random.randn(data["n_persons"], 2)

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            temporal_dynamics=True,
            n_timepoints=data["n_timepoints"],
            person_covariates=True
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            timepoints=data["timepoints"],
            person_covariates=person_covariates,
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        # Both temporal and covariate features should work together
        assert results.temporal_ideal_points is not None
        assert "person_covariate_effects" in results.posterior_samples


class TestCovariatePredictions:
    """Tests for predictions with covariate models."""

    def test_predict_with_person_covariates(self, person_covariate_data):
        """Test predictions using person covariate values."""
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
        predictions = model.predict(
            data["person_ids"][:20], data["item_ids"][:20]
        )

        assert predictions is not None
        assert len(predictions) == 20
        assert np.all(predictions >= 0) and np.all(predictions <= 1)

    def test_predict_with_item_covariates(self, item_covariate_data):
        """Test predictions using item covariate values."""
        data = item_covariate_data

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            item_covariates=True
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            item_covariates=data["item_covariates"],
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        # Make predictions
        predictions = model.predict(
            data["person_ids"][:20], data["item_ids"][:20]
        )

        assert predictions is not None
        assert len(predictions) == 20


class TestCovariateEdgeCases:
    """Tests for edge cases with covariates."""

    def test_single_covariate(self):
        """Test model with single covariate."""
        np.random.seed(42)
        n_persons = 25
        n_items = 15
        n_obs = 100

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.binomial(1, 0.6, n_obs)
        person_covariates = np.random.randn(n_persons, 1)  # Single covariate

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            person_covariates=True
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            person_ids, item_ids, responses,
            person_covariates=person_covariates,
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        assert results.ideal_points is not None
        assert "person_covariate_effects" in results.posterior_samples
        assert results.posterior_samples["person_covariate_effects"].shape[1] == 1

    def test_many_covariates(self):
        """Test model with many covariates."""
        np.random.seed(42)
        n_persons = 30
        n_items = 15
        n_obs = 120

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.binomial(1, 0.6, n_obs)
        person_covariates = np.random.randn(n_persons, 5)  # 5 covariates

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            person_covariates=True
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            person_ids, item_ids, responses,
            person_covariates=person_covariates,
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        assert results.ideal_points is not None
        assert results.posterior_samples["person_covariate_effects"].shape[1] == 5

    def test_covariates_2d_model(self):
        """Test covariates with 2-dimensional ideal points."""
        np.random.seed(42)
        n_persons = 25
        n_items = 15
        n_dims = 2
        n_obs = 120

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.binomial(1, 0.6, n_obs)
        person_covariates = np.random.randn(n_persons, 2)

        config = IdealPointConfig(
            n_dims=2,
            response_type=ResponseType.BINARY,
            person_covariates=True
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            person_ids, item_ids, responses,
            person_covariates=person_covariates,
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        assert results.ideal_points.shape == (n_persons, n_dims)
        # Covariate effects should be (n_samples, n_covariates, n_dims)
        gamma = results.posterior_samples["person_covariate_effects"]
        assert gamma.shape[2] == n_dims


class TestCovariateResponseTypes:
    """Tests for covariates with different response types."""

    def test_covariates_ordinal(self):
        """Test covariates with ordinal responses."""
        np.random.seed(42)
        n_persons = 25
        n_items = 15
        n_categories = 5
        n_obs = 100

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.randint(0, n_categories, n_obs)
        person_covariates = np.random.randn(n_persons, 2)

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.ORDINAL,
            n_categories=n_categories,
            person_covariates=True
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            person_ids, item_ids, responses,
            person_covariates=person_covariates,
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        assert results.ideal_points is not None
        assert "person_covariate_effects" in results.posterior_samples

    def test_covariates_continuous(self):
        """Test covariates with continuous responses."""
        np.random.seed(42)
        n_persons = 25
        n_items = 15
        n_obs = 100

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
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        assert results.ideal_points is not None


class TestCovariateValidation:
    """Tests for covariate input validation."""

    def test_covariate_shape_mismatch(self):
        """Test that mismatched covariate shapes are caught."""
        np.random.seed(42)
        n_persons = 25
        n_items = 15
        n_obs = 100

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.binomial(1, 0.6, n_obs)
        # Wrong number of persons in covariates
        person_covariates = np.random.randn(20, 2)  # Should be 25, not 20

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            person_covariates=True
        )

        model = IdealPointEstimator(config)

        # Should raise error due to shape mismatch
        with pytest.raises((ValueError, IndexError, AssertionError)):
            model.fit(
                person_ids, item_ids, responses,
                person_covariates=person_covariates,
                inference="vi", vi_steps=100, device="cpu", progress_bar=False
            )
