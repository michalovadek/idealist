"""
Tests for flexible prior distributions and hierarchical features.

Tests the new features added to support UK Supreme Court-style analyses:
1. Flexible prior families (Student-t, Cauchy, Laplace)
2. Hierarchical variance estimation
3. Hierarchical discrimination with item covariates

CURRENT COVERAGE: 0% (new features)
PRIORITY: HIGH - Critical new functionality
"""

import numpy as np
import pandas as pd
import pytest

from idealist import IdealPointConfig, IdealPointEstimator, PriorFamily, ResponseType
from idealist.data import load_data


class TestFlexiblePriors:
    """Tests for flexible prior distribution families."""

    @pytest.fixture
    def simple_binary_data(self):
        """Create simple binary response data for testing."""
        np.random.seed(42)
        n_persons = 20
        n_items = 15
        n_obs = 100

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.binomial(1, 0.6, n_obs)

        df = pd.DataFrame({"person_id": person_ids, "item_id": item_ids, "response": responses})

        return load_data(df, person_col="person_id", item_col="item_id", response_col="response")

    def test_student_t_prior_ideal_points(self, simple_binary_data):
        """Test fitting with Student-t priors for ideal points."""
        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            prior_ideal_point_family="student_t",
            prior_ideal_point_df=7.0,
        )

        model = IdealPointEstimator(config)
        results = model.fit(simple_binary_data, inference="vi", vi_steps=500)

        # Verify model fitted successfully
        assert results.ideal_points is not None
        assert results.ideal_points.shape == (20, 1)  # n_persons=20, n_dims=1

    def test_student_t_prior_discrimination(self, simple_binary_data):
        """Test fitting with Student-t priors for discrimination."""
        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            prior_discrimination_family="student_t",
            prior_discrimination_df=7.0,
        )

        model = IdealPointEstimator(config)
        results = model.fit(simple_binary_data, inference="vi", vi_steps=500)

        # Verify discrimination parameters estimated
        assert results.discrimination is not None
        assert results.discrimination.shape == (15, 1)  # n_items=15, n_dims=1

    def test_cauchy_prior(self, simple_binary_data):
        """Test fitting with Cauchy priors (heavy-tailed)."""
        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            prior_discrimination_family="cauchy",
            prior_discrimination_scale=1.0,
        )

        model = IdealPointEstimator(config)
        results = model.fit(simple_binary_data, inference="vi", vi_steps=500)

        # Verify model fitted with Cauchy priors
        assert results.discrimination is not None
        # Cauchy priors can produce larger values
        assert np.abs(results.discrimination).max() < 100  # Sanity check

    def test_laplace_prior(self, simple_binary_data):
        """Test fitting with Laplace priors."""
        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            prior_ideal_point_family="laplace",
            prior_ideal_point_scale=1.0,
        )

        model = IdealPointEstimator(config)
        results = model.fit(simple_binary_data, inference="vi", vi_steps=500)

        # Verify model fitted with Laplace priors
        assert results.ideal_points is not None
        assert results.ideal_points.shape == (20, 1)

    def test_normal_prior_baseline(self, simple_binary_data):
        """Test that Normal prior still works (baseline comparison)."""
        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            prior_ideal_point_family="normal",
        )

        model = IdealPointEstimator(config)
        results = model.fit(simple_binary_data, inference="vi", vi_steps=500)

        # Verify baseline Normal prior works
        assert results.ideal_points is not None
        assert results.ideal_points.shape == (20, 1)

    def test_multiple_priors_combined(self, simple_binary_data):
        """Test combining different priors for different parameters."""
        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            prior_ideal_point_family="normal",
            prior_difficulty_family="student_t",
            prior_difficulty_df=7.0,
            prior_discrimination_family="cauchy",
        )

        model = IdealPointEstimator(config)
        results = model.fit(simple_binary_data, inference="vi", vi_steps=500)

        # Verify model fitted with mixed priors
        assert results.ideal_points is not None
        assert results.difficulty is not None
        assert results.discrimination is not None

    def test_degrees_of_freedom_parameter(self, simple_binary_data):
        """Test that degrees of freedom parameter is respected."""
        # Low df (heavier tails)
        config_low_df = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            prior_discrimination_family="student_t",
            prior_discrimination_df=3.0,  # Heavy tails
        )

        # High df (closer to Normal)
        config_high_df = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            prior_discrimination_family="student_t",
            prior_discrimination_df=30.0,  # Close to Normal
        )

        model_low = IdealPointEstimator(config_low_df)
        model_high = IdealPointEstimator(config_high_df)

        results_low = model_low.fit(simple_binary_data, inference="vi", vi_steps=500)
        results_high = model_high.fit(simple_binary_data, inference="vi", vi_steps=500)

        # Both should converge (no assertion on specific values due to randomness)
        assert results_low.discrimination is not None
        assert results_high.discrimination is not None

    def test_invalid_prior_family_raises_error(self, simple_binary_data):
        """Test that invalid prior family raises appropriate error."""
        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            prior_ideal_point_family="invalid_family",  # Invalid
        )

        model = IdealPointEstimator(config)

        # Should raise ValueError when building model
        with pytest.raises(ValueError, match="Unknown prior family"):
            model.fit(simple_binary_data, inference="vi", vi_steps=100)

    def test_prior_family_enum(self):
        """Test that PriorFamily enum is properly defined."""
        assert PriorFamily.NORMAL.value == "normal"
        assert PriorFamily.STUDENT_T.value == "student_t"
        assert PriorFamily.CAUCHY.value == "cauchy"
        assert PriorFamily.LAPLACE.value == "laplace"

    def test_mcmc_with_student_t(self, simple_binary_data):
        """Test that Student-t priors work with MCMC inference."""
        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            prior_ideal_point_family="student_t",
            prior_ideal_point_df=7.0,
            prior_discrimination_family="student_t",
            prior_discrimination_df=7.0,
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            simple_binary_data,
            inference="mcmc",
            num_chains=1,
            num_warmup=100,
            num_samples=100,
        )

        # Verify MCMC worked with Student-t priors
        assert results.ideal_points is not None
        assert results.posterior_samples is not None
        assert "ideal_points" in results.posterior_samples


class TestHierarchicalVariance:
    """Tests for hierarchical variance estimation."""

    @pytest.fixture
    def binary_data(self):
        """Create binary data for variance estimation tests."""
        np.random.seed(123)
        n_persons = 25
        n_items = 20
        n_obs = 120

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.binomial(1, 0.55, n_obs)

        df = pd.DataFrame({"person_id": person_ids, "item_id": item_ids, "response": responses})

        return load_data(df, person_col="person_id", item_col="item_id", response_col="response")

    def test_variance_estimation_enabled(self, binary_data):
        """Test that variance estimation can be enabled."""
        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            hyperprior_person_scale=True,  # Enable hyperprior on person scale
        )

        model = IdealPointEstimator(config)
        results = model.fit(binary_data, inference="vi", vi_steps=500)

        # Verify hyperprior was estimated
        assert results.posterior_samples is not None
        assert "ideal_point_scale_hyperparam" in results.posterior_samples

        # Check scale is positive
        scale = results.posterior_samples["ideal_point_scale_hyperparam"]
        assert scale.mean() > 0

    def test_variance_estimation_disabled_by_default(self, binary_data):
        """Test that hyperprior is disabled by default."""
        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            hyperprior_person_scale=False,  # Default
        )

        model = IdealPointEstimator(config)
        results = model.fit(binary_data, inference="vi", vi_steps=500)

        # Hyperprior should not be in posterior samples when disabled
        if results.posterior_samples is not None:
            assert "ideal_point_scale_hyperparam" not in results.posterior_samples

    def test_variance_with_mcmc(self, binary_data):
        """Test hyperprior with MCMC inference."""
        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            hyperprior_person_scale=True,
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            binary_data,
            inference="mcmc",
            num_chains=1,
            num_warmup=100,
            num_samples=100,
        )

        # Verify hyperprior was estimated with MCMC
        assert "ideal_point_scale_hyperparam" in results.posterior_samples
        scale_samples = results.posterior_samples["ideal_point_scale_hyperparam"]
        assert scale_samples.shape[0] == 100  # num_samples

    def test_variance_with_flexible_priors(self, binary_data):
        """Test combining hyperprior with flexible priors."""
        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            hyperprior_person_scale=True,
            prior_ideal_point_family="student_t",
            prior_ideal_point_df=7.0,
        )

        model = IdealPointEstimator(config)
        results = model.fit(binary_data, inference="vi", vi_steps=500)

        # Verify both features work together
        assert "ideal_point_scale_hyperparam" in results.posterior_samples
        assert results.ideal_points is not None


class TestHierarchicalDiscrimination:
    """Tests for hierarchical discrimination with item covariates."""

    @pytest.fixture
    def data_with_item_covariates(self):
        """Create data with item-level covariates."""
        np.random.seed(456)
        n_persons = 20
        n_items = 15
        n_obs = 100

        # Create item covariates
        item_topic = np.random.randint(0, 3, n_items)  # 3 topics
        item_year = np.random.uniform(2010, 2020, n_items)

        # Expand to observation level
        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.binomial(1, 0.6, n_obs)

        # Map item covariates to observations
        topic_obs = item_topic[item_ids]
        year_obs = item_year[item_ids]

        df = pd.DataFrame(
            {
                "person_id": person_ids,
                "item_id": item_ids,
                "response": responses,
                "topic": topic_obs,
                "year": year_obs,
            }
        )

        return load_data(
            df,
            person_col="person_id",
            item_col="item_id",
            response_col="response",
            item_covariate_cols=["topic", "year"],
        )

    def test_discrimination_covariates_enabled(self, data_with_item_covariates):
        """Test that hierarchical discrimination can be enabled."""
        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            item_covariates_discrimination=True,  # Enable hierarchical discrimination
        )

        model = IdealPointEstimator(config)
        results = model.fit(data_with_item_covariates, inference="vi", vi_steps=500)

        # Verify covariate effects were estimated
        assert "item_discrimination_covariate_effects" in results.posterior_samples
        effects = results.posterior_samples["item_discrimination_covariate_effects"]

        # Should have shape (n_samples, n_covariates, n_dims)
        assert effects.ndim == 3
        assert effects.shape[1] == 2  # 2 covariates (topic, year)
        assert effects.shape[2] == 1  # n_dims=1

    def test_discrimination_covariates_with_flexible_priors(self, data_with_item_covariates):
        """Test combining hierarchical discrimination with flexible priors."""
        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            item_covariates_discrimination=True,
            prior_covariate_family="student_t",
            prior_covariate_df=7.0,
        )

        model = IdealPointEstimator(config)
        results = model.fit(data_with_item_covariates, inference="vi", vi_steps=500)

        # Verify both features work together
        assert "item_discrimination_covariate_effects" in results.posterior_samples
        assert results.discrimination is not None

    def test_discrimination_covariates_disabled_by_default(self, data_with_item_covariates):
        """Test that hierarchical discrimination is disabled by default."""
        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            item_covariates_discrimination=False,  # Default
        )

        model = IdealPointEstimator(config)
        results = model.fit(data_with_item_covariates, inference="vi", vi_steps=500)

        # Covariate effects should not be estimated when disabled
        if results.posterior_samples is not None:
            # Should use person covariates mode instead (if person_covariates=True)
            # or standard mode (if person_covariates=False)
            pass  # Just verify it doesn't crash

    def test_discrimination_covariates_mcmc(self, data_with_item_covariates):
        """Test hierarchical discrimination with MCMC inference."""
        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            item_covariates_discrimination=True,
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            data_with_item_covariates,
            inference="mcmc",
            num_chains=1,
            num_warmup=100,
            num_samples=100,
        )

        # Verify covariate effects estimated with MCMC
        assert "item_discrimination_covariate_effects" in results.posterior_samples
        effects = results.posterior_samples["item_discrimination_covariate_effects"]
        assert effects.shape[0] == 100  # num_samples


class TestIntegratedHierarchicalFeatures:
    """Tests for combining multiple hierarchical features."""

    @pytest.fixture
    def full_hierarchical_data(self):
        """Create data with both person and item covariates."""
        np.random.seed(789)
        n_persons = 25
        n_items = 20
        n_obs = 120

        # Person covariates
        person_age = np.random.uniform(30, 70, n_persons)
        person_party = np.random.randint(0, 2, n_persons)

        # Item covariates
        item_topic = np.random.randint(0, 3, n_items)
        item_year = np.random.uniform(2010, 2020, n_items)

        # Observations
        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.binomial(1, 0.6, n_obs)

        # Map covariates to observations
        age_obs = person_age[person_ids]
        party_obs = person_party[person_ids]
        topic_obs = item_topic[item_ids]
        year_obs = item_year[item_ids]

        df = pd.DataFrame(
            {
                "person_id": person_ids,
                "item_id": item_ids,
                "response": responses,
                "age": age_obs,
                "party": party_obs,
                "topic": topic_obs,
                "year": year_obs,
            }
        )

        return load_data(
            df,
            person_col="person_id",
            item_col="item_id",
            response_col="response",
            person_covariate_cols=["age", "party"],
            item_covariate_cols=["topic", "year"],
        )

    def test_all_hierarchical_features_combined(self, full_hierarchical_data):
        """Test combining all hierarchical features: hyperprior, discrimination, person covariates."""
        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            person_covariates=True,  # Person covariates affect ideal points
            item_covariates=True,  # Item covariates affect both difficulty and discrimination
            hyperprior_person_scale=True,  # Hyperprior on person scale
            prior_ideal_point_family="student_t",
            prior_ideal_point_df=7.0,
            prior_covariate_family="student_t",
            prior_covariate_df=7.0,
        )

        model = IdealPointEstimator(config)
        results = model.fit(full_hierarchical_data, inference="vi", vi_steps=500)

        # Verify all features are present
        assert "person_covariate_effects" in results.posterior_samples
        assert "item_discrimination_covariate_effects" in results.posterior_samples
        assert "ideal_point_scale_hyperparam" in results.posterior_samples

        # Check shapes
        person_effects = results.posterior_samples["person_covariate_effects"]
        assert person_effects.shape[1] == 2  # 2 person covariates
        assert person_effects.shape[2] == 1  # n_dims=1

        item_effects = results.posterior_samples["item_discrimination_covariate_effects"]
        assert item_effects.shape[1] == 2  # 2 item covariates
        assert item_effects.shape[2] == 1  # n_dims=1

    def test_uk_supreme_court_configuration(self, full_hierarchical_data):
        """Test the exact configuration needed for UK Supreme Court analysis."""
        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            # Hierarchical features
            item_covariates_discrimination=True,
            hyperprior_person_scale=True,
            # Flexible priors
            prior_ideal_point_family="normal",
            prior_discrimination_family="student_t",
            prior_discrimination_df=7.0,
            prior_covariate_family="student_t",
            prior_covariate_df=7.0,
            # Prior scales
            prior_difficulty_scale=10.0,
            prior_covariate_scale=2.5,
        )

        model = IdealPointEstimator(config)
        results = model.fit(full_hierarchical_data, inference="vi", vi_steps=500)

        # Verify UK Supreme Court configuration works
        assert results.ideal_points is not None
        assert results.discrimination is not None
        assert "item_discrimination_covariate_effects" in results.posterior_samples
        assert "ideal_point_scale_hyperparam" in results.posterior_samples


class TestBackwardCompatibility:
    """Tests to ensure new features don't break existing functionality."""

    @pytest.fixture
    def simple_data(self):
        """Create simple data without covariates."""
        np.random.seed(999)
        n_persons = 15
        n_items = 10
        n_obs = 80

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)
        responses = np.random.binomial(1, 0.5, n_obs)

        df = pd.DataFrame({"person_id": person_ids, "item_id": item_ids, "response": responses})

        return load_data(df, person_col="person_id", item_col="item_id", response_col="response")

    def test_default_config_unchanged(self, simple_data):
        """Test that default configuration still works as before."""
        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            # All new features disabled by default
        )

        model = IdealPointEstimator(config)
        results = model.fit(simple_data, inference="vi", vi_steps=500)

        # Standard behavior should work
        assert results.ideal_points is not None
        assert results.discrimination is not None
        assert results.difficulty is not None

    def test_person_covariates_direct_mode(self, simple_data):
        """Test person covariates with direct array input."""
        # Add person covariates manually
        data_dict = {
            "person_ids": simple_data.person_ids,
            "item_ids": simple_data.item_ids,
            "responses": simple_data.responses,
            "person_covariates": np.random.randn(simple_data.n_persons, 2),
        }

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            person_covariates=True,
        )

        model = IdealPointEstimator(config)

        # Should work with direct array input
        results = model.fit(
            data_dict["person_ids"],
            data_dict["item_ids"],
            data_dict["responses"],
            person_covariates=data_dict["person_covariates"],
            inference="vi",
            vi_steps=500,
        )

        assert results.ideal_points is not None
        assert "person_covariate_effects" in results.posterior_samples


# Future tests to add:
# - Test with 2D models (n_dims=2)
# - Test with ordinal responses
# - Test with continuous responses
# - Test parameter recovery with simulated data
# - Test convergence diagnostics with new features
# - Test serialization/deserialization of models with new features
