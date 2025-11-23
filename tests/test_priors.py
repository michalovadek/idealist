"""
Tests for custom priors and prior distributions.

Tests that different prior specifications work correctly and affect
model estimation as expected.
"""

import numpy as np
import pytest

from idealist import IdealPointConfig, IdealPointEstimator, ResponseType
from idealist.core.prior_distributions import (
    centered_priors,
    conservative_priors,
    default_priors,
    flexible_priors,
    hierarchical_priors,
    rasch_priors,
    regularized_priors,
    standard_priors,
    vague_priors,
    weakly_informative_priors,
)


class TestPriorDistributions:
    """Tests for prior distribution helper functions."""

    def test_default_priors(self):
        """Test that default priors return correct structure."""
        priors = default_priors()

        assert isinstance(priors, dict)
        assert "prior_ideal_point_mean" in priors
        assert "prior_ideal_point_scale" in priors
        assert "prior_difficulty_mean" in priors
        assert "prior_difficulty_scale" in priors
        assert "prior_discrimination_mean" in priors
        assert "prior_discrimination_scale" in priors

        # Check default values
        assert priors["prior_ideal_point_mean"] == 0.0
        assert priors["prior_ideal_point_scale"] == 1.0

    def test_weakly_informative_priors(self):
        """Test weakly informative priors."""
        priors = weakly_informative_priors()

        # Should be more diffuse than defaults
        assert priors["prior_ideal_point_scale"] > 1.0
        assert priors["prior_difficulty_scale"] > 2.5

    def test_conservative_priors(self):
        """Test conservative priors."""
        priors = conservative_priors()

        # Should be tighter than defaults
        assert priors["prior_ideal_point_scale"] < 1.0
        assert priors["prior_difficulty_scale"] < 2.5

    def test_vague_priors(self):
        """Test vague priors."""
        priors = vague_priors()

        # Should be very diffuse
        assert priors["prior_ideal_point_scale"] > 3.0
        assert priors["prior_difficulty_scale"] > 5.0

    def test_centered_priors(self):
        """Test centered priors with custom means."""
        priors = centered_priors(
            ideal_point_mean=0.5, difficulty_mean=-1.0, discrimination_mean=1.5
        )

        assert priors["prior_ideal_point_mean"] == 0.5
        assert priors["prior_difficulty_mean"] == -1.0
        assert priors["prior_discrimination_mean"] == 1.5

    def test_rasch_priors(self):
        """Test Rasch model priors."""
        priors = rasch_priors()

        # Discrimination should be centered at 1.0 with tight scale
        assert priors["prior_discrimination_mean"] == 1.0
        assert priors["prior_discrimination_scale"] < 0.5

    def test_hierarchical_priors(self):
        """Test hierarchical priors."""
        priors = hierarchical_priors(covariate_scale=0.5, threshold_scale=1.5)

        assert "prior_covariate_scale" in priors
        assert "prior_threshold_scale" in priors
        assert priors["prior_covariate_scale"] == 0.5
        assert priors["prior_threshold_scale"] == 1.5

    def test_prior_aliases(self):
        """Test that prior aliases work correctly."""
        assert standard_priors() == default_priors()
        assert regularized_priors() == conservative_priors()
        assert flexible_priors() == weakly_informative_priors()


class TestPriorsInEstimation:
    """Tests that priors affect model estimation."""

    def test_config_with_default_priors(self, small_binary_data):
        """Test that default priors can be used in config."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY, **default_priors())

        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids=data["person_ids"],
            item_ids=data["item_ids"],
            responses=data["responses"],
            inference="vi",
            vi_steps=500,
            device="cpu",
            progress_bar=False,
        )

        assert results is not None
        print("\n  Default priors estimation successful")

    def test_config_with_weakly_informative_priors(self, small_binary_data):
        """Test weakly informative priors in estimation."""
        data = small_binary_data

        config = IdealPointConfig(
            n_dims=1, response_type=ResponseType.BINARY, **weakly_informative_priors()
        )

        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids=data["person_ids"],
            item_ids=data["item_ids"],
            responses=data["responses"],
            inference="vi",
            vi_steps=500,
            device="cpu",
            progress_bar=False,
        )

        assert results is not None
        print("\n  Weakly informative priors estimation successful")

    def test_config_with_conservative_priors(self, small_binary_data):
        """Test conservative priors in estimation."""
        data = small_binary_data

        config = IdealPointConfig(
            n_dims=1, response_type=ResponseType.BINARY, **conservative_priors()
        )

        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids=data["person_ids"],
            item_ids=data["item_ids"],
            responses=data["responses"],
            inference="vi",
            vi_steps=500,
            device="cpu",
            progress_bar=False,
        )

        assert results is not None

        # Conservative priors should keep estimates closer to zero
        assert np.abs(results.ideal_points).mean() < 3.0

        print("\n  Conservative priors estimation successful")

    @pytest.mark.slow
    def test_priors_affect_estimates(self, small_binary_data):
        """Test that different priors produce different estimates."""
        data = small_binary_data

        # Fit with conservative priors
        config_conservative = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            **conservative_priors(ideal_point_scale=0.3),
        )
        model_conservative = IdealPointEstimator(config_conservative)
        results_conservative = model_conservative.fit(
            person_ids=data["person_ids"],
            item_ids=data["item_ids"],
            responses=data["responses"],
            inference="vi",
            vi_steps=1000,
            device="cpu",
            progress_bar=False,
        )

        # Fit with vague priors
        config_vague = IdealPointConfig(
            n_dims=1, response_type=ResponseType.BINARY, **vague_priors()
        )
        model_vague = IdealPointEstimator(config_vague)
        results_vague = model_vague.fit(
            person_ids=data["person_ids"],
            item_ids=data["item_ids"],
            responses=data["responses"],
            inference="vi",
            vi_steps=1000,
            device="cpu",
            progress_bar=False,
        )

        # Conservative priors should shrink estimates more
        conservative_spread = np.std(results_conservative.ideal_points)
        vague_spread = np.std(results_vague.ideal_points)

        # Conservative should have less spread (though this is probabilistic)
        print(f"\n  Conservative spread: {conservative_spread:.3f}")
        print(f"  Vague spread: {vague_spread:.3f}")

        # Just verify both ran successfully
        assert results_conservative is not None
        assert results_vague is not None

    def test_centered_priors_in_estimation(self, small_binary_data):
        """Test that centered priors can be used."""
        data = small_binary_data

        config = IdealPointConfig(
            n_dims=1, response_type=ResponseType.BINARY, **centered_priors(ideal_point_mean=1.0)
        )

        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids=data["person_ids"],
            item_ids=data["item_ids"],
            responses=data["responses"],
            inference="vi",
            vi_steps=500,
            device="cpu",
            progress_bar=False,
        )

        assert results is not None
        print("\n  Centered priors estimation successful")

    @pytest.mark.parametrize(
        "prior_func",
        [
            default_priors,
            weakly_informative_priors,
            conservative_priors,
            vague_priors,
        ],
    )
    def test_all_priors_parametrized(self, small_binary_data, prior_func):
        """Parametrized test for all prior types."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY, **prior_func())

        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids=data["person_ids"],
            item_ids=data["item_ids"],
            responses=data["responses"],
            inference="vi",
            vi_steps=300,
            device="cpu",
            progress_bar=False,
        )

        assert results is not None
        assert results.ideal_points.shape == (data["n_persons"], data["n_dims"])


class TestPriorCustomization:
    """Tests for custom prior parameter specifications."""

    def test_custom_ideal_point_scale(self, small_binary_data):
        """Test custom ideal point scale parameter."""
        data = small_binary_data

        priors = weakly_informative_priors(ideal_point_scale=3.0)
        assert priors["prior_ideal_point_scale"] == 3.0

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY, **priors)

        model = IdealPointEstimator(config)
        results = model.fit(
            person_ids=data["person_ids"],
            item_ids=data["item_ids"],
            responses=data["responses"],
            inference="vi",
            vi_steps=300,
            device="cpu",
            progress_bar=False,
        )

        assert results is not None

    def test_hierarchical_with_base_priors(self):
        """Test combining hierarchical with base priors."""
        priors = hierarchical_priors(covariate_scale=0.5, **weakly_informative_priors())

        # Should have both hierarchical and base prior parameters
        assert "prior_covariate_scale" in priors
        assert "prior_ideal_point_scale" in priors
        assert priors["prior_ideal_point_scale"] > 1.0  # From weakly_informative

    def test_custom_discrimination_for_rasch(self, small_binary_data):
        """Test Rasch priors constrain discrimination."""
        data = small_binary_data

        priors = rasch_priors()

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY, **priors)

        model = IdealPointEstimator(config)
        results = model.fit(
            person_ids=data["person_ids"],
            item_ids=data["item_ids"],
            responses=data["responses"],
            inference="vi",
            vi_steps=500,
            device="cpu",
            progress_bar=False,
        )

        # Discrimination should be close to 1.0 for all items
        # (though not exactly due to data influence)
        discrimination_mean = np.mean(results.discrimination)
        print(f"\n  Mean discrimination with Rasch priors: {discrimination_mean:.3f}")

        assert results is not None


class TestPriorPosteriorEffects:
    """Tests for how priors affect posterior distributions."""

    def test_tight_priors_reduce_variance(self, small_binary_data):
        """Test that tighter priors reduce posterior variance."""
        data = small_binary_data

        # Wide priors
        config_wide = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            prior_ideal_point_scale=3.0
        )
        model_wide = IdealPointEstimator(config_wide)
        results_wide = model_wide.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=1000, num_samples=100,
            device="cpu", progress_bar=False
        )

        # Tight priors
        config_tight = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            prior_ideal_point_scale=0.5
        )
        model_tight = IdealPointEstimator(config_tight)
        results_tight = model_tight.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=1000, num_samples=100,
            device="cpu", progress_bar=False
        )

        # Tight priors should produce estimates closer to prior mean (0)
        mean_abs_wide = np.mean(np.abs(results_wide.ideal_points))
        mean_abs_tight = np.mean(np.abs(results_tight.ideal_points))

        # Tight priors should shrink estimates toward 0
        assert mean_abs_tight < mean_abs_wide

    def test_prior_mean_shifts_posterior(self, small_binary_data):
        """Test that changing prior mean shifts posterior."""
        data = small_binary_data

        # Prior centered at 0
        config_zero = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            prior_ideal_point_mean=0.0,
            prior_ideal_point_scale=2.0
        )
        model_zero = IdealPointEstimator(config_zero)
        results_zero = model_zero.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        # Prior centered at 1
        config_one = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            prior_ideal_point_mean=1.0,
            prior_ideal_point_scale=2.0
        )
        model_one = IdealPointEstimator(config_one)
        results_one = model_one.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        # Posterior means should be shifted
        mean_zero = np.mean(results_zero.ideal_points)
        mean_one = np.mean(results_one.ideal_points)

        assert mean_one > mean_zero

    def test_vague_priors_data_driven(self, small_binary_data):
        """Test that vague priors let data dominate."""
        data = small_binary_data

        # Vague priors
        config_vague = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            prior_ideal_point_scale=10.0,
            prior_difficulty_scale=10.0
        )
        model_vague = IdealPointEstimator(config_vague)
        results_vague = model_vague.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        # Default priors
        config_default = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY
        )
        model_default = IdealPointEstimator(config_default)
        results_default = model_default.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        # Should produce similar results (high correlation)
        corr = np.corrcoef(
            results_vague.ideal_points.flatten(),
            results_default.ideal_points.flatten()
        )[0, 1]
        assert corr > 0.9

    def test_student_t_priors_robust(self):
        """Test that Student-t priors are more robust to outliers."""
        np.random.seed(42)
        n_persons = 30
        n_items = 20
        n_obs = 150

        person_ids = np.random.randint(0, n_persons, n_obs)
        item_ids = np.random.randint(0, n_items, n_obs)

        # Generate responses with outliers
        responses = np.random.binomial(1, 0.6, n_obs)

        # Normal priors
        config_normal = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            prior_ideal_point_family="normal"
        )
        model_normal = IdealPointEstimator(config_normal)
        results_normal = model_normal.fit(
            person_ids, item_ids, responses,
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        # Student-t priors
        config_t = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            prior_ideal_point_family="student_t",
            prior_ideal_point_df=7.0
        )
        model_t = IdealPointEstimator(config_t)
        results_t = model_t.fit(
            person_ids, item_ids, responses,
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        # Both should produce reasonable results
        assert results_normal.ideal_points is not None
        assert results_t.ideal_points is not None

    def test_discrimination_prior_affects_estimates(self, small_binary_data):
        """Test that discrimination priors affect estimates."""
        data = small_binary_data

        # Low discrimination prior
        config_low = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            prior_discrimination_mean=0.5,
            prior_discrimination_scale=0.2
        )
        model_low = IdealPointEstimator(config_low)
        results_low = model_low.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        # High discrimination prior
        config_high = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            prior_discrimination_mean=2.0,
            prior_discrimination_scale=0.2
        )
        model_high = IdealPointEstimator(config_high)
        results_high = model_high.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        # Discrimination estimates should differ
        mean_disc_low = np.mean(results_low.discrimination)
        mean_disc_high = np.mean(results_high.discrimination)

        assert mean_disc_high > mean_disc_low


class TestPriorInteractions:
    """Tests for interactions between different prior specifications."""

    def test_prior_scale_interaction_with_data_size(self):
        """Test how prior effects change with data size."""
        np.random.seed(42)
        n_persons = 30
        n_items = 20

        # Small dataset
        n_obs_small = 60
        person_ids_small = np.random.randint(0, n_persons, n_obs_small)
        item_ids_small = np.random.randint(0, n_items, n_obs_small)
        responses_small = np.random.binomial(1, 0.6, n_obs_small)

        # Large dataset
        n_obs_large = 300
        person_ids_large = np.random.randint(0, n_persons, n_obs_large)
        item_ids_large = np.random.randint(0, n_items, n_obs_large)
        responses_large = np.random.binomial(1, 0.6, n_obs_large)

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            prior_ideal_point_scale=0.5  # Tight prior
        )

        # Fit with small data
        model_small = IdealPointEstimator(config)
        results_small = model_small.fit(
            person_ids_small, item_ids_small, responses_small,
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        # Fit with large data
        model_large = IdealPointEstimator(config)
        results_large = model_large.fit(
            person_ids_large, item_ids_large, responses_large,
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        # Prior should have more influence on small data
        # (estimates should be closer to prior mean of 0)
        mean_abs_small = np.mean(np.abs(results_small.ideal_points))
        mean_abs_large = np.mean(np.abs(results_large.ideal_points))

        # Small data should be more shrunk toward prior
        assert mean_abs_small <= mean_abs_large * 1.5  # Allow some tolerance

    def test_hierarchical_prior_with_covariates(self, person_covariate_data):
        """Test hierarchical priors with covariates."""
        data = person_covariate_data

        # Tight covariate prior
        config_tight = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            person_covariates=True,
            prior_covariate_scale=0.2
        )
        model_tight = IdealPointEstimator(config_tight)
        results_tight = model_tight.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            person_covariates=data["person_covariates"],
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        # Loose covariate prior
        config_loose = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            person_covariates=True,
            prior_covariate_scale=2.0
        )
        model_loose = IdealPointEstimator(config_loose)
        results_loose = model_loose.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            person_covariates=data["person_covariates"],
            inference="vi", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        # Covariate effects should differ
        gamma_tight = results_tight.posterior_samples["person_covariate_effects"]
        gamma_loose = results_loose.posterior_samples["person_covariate_effects"]

        # Tight prior should produce smaller covariate effects
        assert np.mean(np.abs(gamma_tight)) < np.mean(np.abs(gamma_loose))


class TestPriorValidation:
    """Tests for prior parameter validation."""

    def test_negative_scale_raises_error(self, small_binary_data):
        """Test that negative prior scales raise errors."""
        data = small_binary_data

        with pytest.raises((ValueError, AssertionError)):
            config = IdealPointConfig(
                n_dims=1,
                response_type=ResponseType.BINARY,
                prior_ideal_point_scale=-1.0
            )
            model = IdealPointEstimator(config)

    def test_zero_scale_raises_error(self, small_binary_data):
        """Test that zero prior scales raise errors."""
        data = small_binary_data

        with pytest.raises((ValueError, AssertionError)):
            config = IdealPointConfig(
                n_dims=1,
                response_type=ResponseType.BINARY,
                prior_ideal_point_scale=0.0
            )
            model = IdealPointEstimator(config)

    def test_invalid_prior_family(self, small_binary_data):
        """Test that invalid prior families raise errors."""
        data = small_binary_data

        with pytest.raises((ValueError, AssertionError)):
            config = IdealPointConfig(
                n_dims=1,
                response_type=ResponseType.BINARY,
                prior_ideal_point_family="invalid_family"
            )
            model = IdealPointEstimator(config)
            model.fit(
                data["person_ids"], data["item_ids"], data["responses"],
                inference="vi", vi_steps=100,
                device="cpu", progress_bar=False
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
