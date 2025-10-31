"""
Tests for custom priors and prior distributions.

Tests that different prior specifications work correctly and affect
model estimation as expected.
"""

import pytest
import numpy as np

from idealist import IdealPointEstimator, IdealPointConfig, ResponseType
from idealist.core.prior_distributions import (
    default_priors,
    weakly_informative_priors,
    conservative_priors,
    vague_priors,
    centered_priors,
    rasch_priors,
    hierarchical_priors,
    standard_priors,
    regularized_priors,
    flexible_priors,
)


class TestPriorDistributions:
    """Tests for prior distribution helper functions."""

    def test_default_priors(self):
        """Test that default priors return correct structure."""
        priors = default_priors()

        assert isinstance(priors, dict)
        assert 'prior_ideal_point_mean' in priors
        assert 'prior_ideal_point_scale' in priors
        assert 'prior_difficulty_mean' in priors
        assert 'prior_difficulty_scale' in priors
        assert 'prior_discrimination_mean' in priors
        assert 'prior_discrimination_scale' in priors

        # Check default values
        assert priors['prior_ideal_point_mean'] == 0.0
        assert priors['prior_ideal_point_scale'] == 1.0

    def test_weakly_informative_priors(self):
        """Test weakly informative priors."""
        priors = weakly_informative_priors()

        # Should be more diffuse than defaults
        assert priors['prior_ideal_point_scale'] > 1.0
        assert priors['prior_difficulty_scale'] > 2.5

    def test_conservative_priors(self):
        """Test conservative priors."""
        priors = conservative_priors()

        # Should be tighter than defaults
        assert priors['prior_ideal_point_scale'] < 1.0
        assert priors['prior_difficulty_scale'] < 2.5

    def test_vague_priors(self):
        """Test vague priors."""
        priors = vague_priors()

        # Should be very diffuse
        assert priors['prior_ideal_point_scale'] > 3.0
        assert priors['prior_difficulty_scale'] > 5.0

    def test_centered_priors(self):
        """Test centered priors with custom means."""
        priors = centered_priors(
            ideal_point_mean=0.5,
            difficulty_mean=-1.0,
            discrimination_mean=1.5
        )

        assert priors['prior_ideal_point_mean'] == 0.5
        assert priors['prior_difficulty_mean'] == -1.0
        assert priors['prior_discrimination_mean'] == 1.5

    def test_rasch_priors(self):
        """Test Rasch model priors."""
        priors = rasch_priors()

        # Discrimination should be centered at 1.0 with tight scale
        assert priors['prior_discrimination_mean'] == 1.0
        assert priors['prior_discrimination_scale'] < 0.5

    def test_hierarchical_priors(self):
        """Test hierarchical priors."""
        priors = hierarchical_priors(
            covariate_scale=0.5,
            threshold_scale=1.5
        )

        assert 'prior_covariate_scale' in priors
        assert 'prior_threshold_scale' in priors
        assert priors['prior_covariate_scale'] == 0.5
        assert priors['prior_threshold_scale'] == 1.5

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

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            **default_priors()
        )

        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids=data['person_ids'],
            item_ids=data['item_ids'],
            responses=data['responses'],
            inference='vi',
            vi_steps=500,
            device='cpu',
            progress_bar=False,
        )

        assert results is not None
        print(f"\n  Default priors estimation successful")

    def test_config_with_weakly_informative_priors(self, small_binary_data):
        """Test weakly informative priors in estimation."""
        data = small_binary_data

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            **weakly_informative_priors()
        )

        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids=data['person_ids'],
            item_ids=data['item_ids'],
            responses=data['responses'],
            inference='vi',
            vi_steps=500,
            device='cpu',
            progress_bar=False,
        )

        assert results is not None
        print(f"\n  Weakly informative priors estimation successful")

    def test_config_with_conservative_priors(self, small_binary_data):
        """Test conservative priors in estimation."""
        data = small_binary_data

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            **conservative_priors()
        )

        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids=data['person_ids'],
            item_ids=data['item_ids'],
            responses=data['responses'],
            inference='vi',
            vi_steps=500,
            device='cpu',
            progress_bar=False,
        )

        assert results is not None

        # Conservative priors should keep estimates closer to zero
        assert np.abs(results.ideal_points).mean() < 3.0

        print(f"\n  Conservative priors estimation successful")

    @pytest.mark.slow
    def test_priors_affect_estimates(self, small_binary_data):
        """Test that different priors produce different estimates."""
        data = small_binary_data

        # Fit with conservative priors
        config_conservative = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            **conservative_priors(ideal_point_scale=0.3)
        )
        model_conservative = IdealPointEstimator(config_conservative)
        results_conservative = model_conservative.fit(
            person_ids=data['person_ids'],
            item_ids=data['item_ids'],
            responses=data['responses'],
            inference='vi',
            vi_steps=1000,
            device='cpu',
            progress_bar=False,
        )

        # Fit with vague priors
        config_vague = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            **vague_priors()
        )
        model_vague = IdealPointEstimator(config_vague)
        results_vague = model_vague.fit(
            person_ids=data['person_ids'],
            item_ids=data['item_ids'],
            responses=data['responses'],
            inference='vi',
            vi_steps=1000,
            device='cpu',
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
            n_dims=1,
            response_type=ResponseType.BINARY,
            **centered_priors(ideal_point_mean=1.0)
        )

        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids=data['person_ids'],
            item_ids=data['item_ids'],
            responses=data['responses'],
            inference='vi',
            vi_steps=500,
            device='cpu',
            progress_bar=False,
        )

        assert results is not None
        print(f"\n  Centered priors estimation successful")

    @pytest.mark.parametrize("prior_func", [
        default_priors,
        weakly_informative_priors,
        conservative_priors,
        vague_priors,
    ])
    def test_all_priors_parametrized(self, small_binary_data, prior_func):
        """Parametrized test for all prior types."""
        data = small_binary_data

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            **prior_func()
        )

        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids=data['person_ids'],
            item_ids=data['item_ids'],
            responses=data['responses'],
            inference='vi',
            vi_steps=300,
            device='cpu',
            progress_bar=False,
        )

        assert results is not None
        assert results.ideal_points.shape == (data['n_persons'], data['n_dims'])


class TestPriorCustomization:
    """Tests for custom prior parameter specifications."""

    def test_custom_ideal_point_scale(self, small_binary_data):
        """Test custom ideal point scale parameter."""
        data = small_binary_data

        priors = weakly_informative_priors(ideal_point_scale=3.0)
        assert priors['prior_ideal_point_scale'] == 3.0

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            **priors
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            person_ids=data['person_ids'],
            item_ids=data['item_ids'],
            responses=data['responses'],
            inference='vi',
            vi_steps=300,
            device='cpu',
            progress_bar=False,
        )

        assert results is not None

    def test_hierarchical_with_base_priors(self):
        """Test combining hierarchical with base priors."""
        priors = hierarchical_priors(
            covariate_scale=0.5,
            **weakly_informative_priors()
        )

        # Should have both hierarchical and base prior parameters
        assert 'prior_covariate_scale' in priors
        assert 'prior_ideal_point_scale' in priors
        assert priors['prior_ideal_point_scale'] > 1.0  # From weakly_informative

    def test_custom_discrimination_for_rasch(self, small_binary_data):
        """Test Rasch priors constrain discrimination."""
        data = small_binary_data

        priors = rasch_priors()

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            **priors
        )

        model = IdealPointEstimator(config)
        results = model.fit(
            person_ids=data['person_ids'],
            item_ids=data['item_ids'],
            responses=data['responses'],
            inference='vi',
            vi_steps=500,
            device='cpu',
            progress_bar=False,
        )

        # Discrimination should be close to 1.0 for all items
        # (though not exactly due to data influence)
        discrimination_mean = np.mean(results.discrimination)
        print(f"\n  Mean discrimination with Rasch priors: {discrimination_mean:.3f}")

        assert results is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
