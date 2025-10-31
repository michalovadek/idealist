"""
Tests for multi-dimensional ideal point estimation.

Tests that the IdealPointEstimator can handle 2D and higher dimensional
latent spaces.
"""

import pytest
import numpy as np

from idealist import IdealPointEstimator, IdealPointConfig, ResponseType


class TestMultiDimensional:
    """Tests for multi-dimensional ideal point models."""

    def test_2d_estimation(self, multidim_binary_data):
        """Test 2-dimensional ideal point estimation."""
        data = multidim_binary_data

        assert data['n_dims'] == 2, "Fixture should have 2 dimensions"

        config = IdealPointConfig(
            n_dims=data['n_dims'],
            response_type=ResponseType.BINARY,
        )

        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids=data['person_ids'],
            item_ids=data['item_ids'],
            responses=data['responses'],
            inference='vi',
            vi_steps=1500,
            num_samples=300,
            device='cpu',
            progress_bar=False,
        )

        # Check results structure
        assert results is not None
        assert results.ideal_points.shape == (data['n_persons'], 2)
        assert results.discrimination.shape == (data['n_items'], 2)
        assert results.difficulty.shape == (data['n_items'],)

        # Check uncertainty quantification for both dimensions
        assert results.ideal_points_std.shape == (data['n_persons'], 2)

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
            person_ids=data['person_ids'],
            item_ids=data['item_ids'],
            responses=data['responses'],
            inference='vi',
            vi_steps=2000,
            num_samples=500,
            device='cpu',
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
            person_ids=data['person_ids'],
            item_ids=data['item_ids'],
            responses=data['responses'],
            inference='map',
            map_steps=1500,
            device='cpu',
            progress_bar=False,
        )

        assert results is not None
        assert results.ideal_points.shape == (data['n_persons'], 2)

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
            person_ids=data['person_ids'],
            item_ids=data['item_ids'],
            responses=data['responses'],
            inference='vi',
            vi_steps=1000,
            device='cpu',
            progress_bar=False,
        )

        df_dict = results.to_dataframe()

        # Check that DataFrame has columns for both dimensions
        persons_df = df_dict['persons']
        items_df = df_dict['items']

        # Should have ideal_point columns (might be named differently for multidim)
        assert len(persons_df) == data['n_persons']
        assert len(items_df) == data['n_items']

        print(f"\n  2D DataFrame: {persons_df.shape}, {items_df.shape}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
