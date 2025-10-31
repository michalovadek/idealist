"""
Basic tests for ideal point estimation.

Tests the core functionality of the IdealPointEstimator class
with different inference methods (VI, MAP, MCMC).
"""

import pytest
import numpy as np
from scipy.stats import pearsonr

from idealist import IdealPointEstimator, IdealPointConfig, ResponseType


class TestBasicEstimation:
    """Tests for basic ideal point estimation."""

    def test_vi_inference(self, small_binary_data):
        """Test that VI inference runs and produces reasonable results."""
        data = small_binary_data

        config = IdealPointConfig(
            n_dims=data['n_dims'],
            response_type=ResponseType.BINARY,
        )

        model = IdealPointEstimator(config)

        # Fit with VI
        results = model.fit(
            person_ids=data['person_ids'],
            item_ids=data['item_ids'],
            responses=data['responses'],
            inference='vi',
            vi_steps=1000,
            vi_lr=0.01,
            num_samples=200,
            device='cpu',
            progress_bar=False,
        )

        # Check results structure
        assert results is not None
        assert hasattr(results, 'ideal_points')
        assert hasattr(results, 'difficulty')
        assert hasattr(results, 'discrimination')

        # Check shapes
        assert results.ideal_points.shape == (data['n_persons'], data['n_dims'])
        assert results.difficulty.shape == (data['n_items'],)
        assert results.discrimination.shape == (data['n_items'], data['n_dims'])

        # Check uncertainty quantification
        assert results.ideal_points_std is not None
        assert results.ideal_points_ci_lower is not None
        assert results.ideal_points_ci_upper is not None

        # Parameters should be in reasonable range
        assert np.abs(results.ideal_points).max() < 10.0
        assert np.abs(results.difficulty).max() < 10.0

        print(f"\n  VI completed in {results.computation_time:.2f}s")

    def test_map_inference(self, small_binary_data):
        """Test that MAP inference runs."""
        data = small_binary_data

        config = IdealPointConfig(
            n_dims=data['n_dims'],
            response_type=ResponseType.BINARY,
        )

        model = IdealPointEstimator(config)

        # Fit with MAP
        results = model.fit(
            person_ids=data['person_ids'],
            item_ids=data['item_ids'],
            responses=data['responses'],
            inference='map',
            map_steps=1000,
            map_lr=0.01,
            num_samples=200,
            device='cpu',
            progress_bar=False,
        )

        # Check results
        assert results is not None
        assert results.ideal_points.shape == (data['n_persons'], data['n_dims'])
        assert results.difficulty.shape == (data['n_items'],)

        print(f"\n  MAP completed in {results.computation_time:.2f}s")

    @pytest.mark.slow
    def test_mcmc_inference(self, small_binary_data):
        """Test that MCMC inference runs."""
        data = small_binary_data

        config = IdealPointConfig(
            n_dims=data['n_dims'],
            response_type=ResponseType.BINARY,
        )

        model = IdealPointEstimator(config)

        # Fit with MCMC (minimal for speed)
        results = model.fit(
            person_ids=data['person_ids'],
            item_ids=data['item_ids'],
            responses=data['responses'],
            inference='mcmc',
            num_warmup=50,
            num_samples=100,
            num_chains=1,
            device='cpu',
            progress_bar=False,
        )

        # Check results
        assert results is not None
        assert results.ideal_points.shape == (data['n_persons'], data['n_dims'])

        # Check MCMC-specific output
        assert results.ideal_points_samples is not None
        assert results.convergence_info['method'] == 'MCMC'

        print(f"\n  MCMC completed in {results.computation_time:.2f}s")

    @pytest.mark.slow
    def test_parameter_recovery(self, small_binary_data):
        """Test that we can recover known parameters reasonably well."""
        data = small_binary_data

        config = IdealPointConfig(
            n_dims=data['n_dims'],
            response_type=ResponseType.BINARY,
        )

        model = IdealPointEstimator(config)

        # Fit with VI (faster than MCMC)
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

        # Check correlation with true parameters
        # Note: IRT models have reflection invariance, so we check absolute correlation
        ideal_point_corr = np.corrcoef(
            results.ideal_points[:, 0],
            data['true_ideal_points'][:, 0]
        )[0, 1]

        abs_corr = np.abs(ideal_point_corr)
        print(f"\n  Ideal point recovery: r = {abs_corr:.3f}")

        # Should recover at least moderate correlation
        assert abs_corr > 0.5, f"Expected r > 0.5, got {abs_corr:.3f}"

    @pytest.mark.parametrize("inference_method", ["vi", "map"])
    def test_inference_methods_parametrized(self, small_binary_data, inference_method):
        """Parametrized test for different inference methods."""
        data = small_binary_data

        config = IdealPointConfig(
            n_dims=data['n_dims'],
            response_type=ResponseType.BINARY,
        )

        model = IdealPointEstimator(config)

        # Fit with specified method
        fit_kwargs = {
            'person_ids': data['person_ids'],
            'item_ids': data['item_ids'],
            'responses': data['responses'],
            'inference': inference_method,
            'device': 'cpu',
            'progress_bar': False,
        }

        if inference_method == 'vi':
            fit_kwargs.update({'vi_steps': 500, 'num_samples': 100})
        elif inference_method == 'map':
            fit_kwargs.update({'map_steps': 500, 'num_samples': 100})

        results = model.fit(**fit_kwargs)

        # Check basic results
        assert results is not None
        assert results.ideal_points.shape == (data['n_persons'], data['n_dims'])
        assert results.difficulty.shape == (data['n_items'],)

        print(f"\n  {inference_method.upper()} parametrized test completed")

    def test_to_dataframe(self, small_binary_data):
        """Test conversion to pandas DataFrame."""
        data = small_binary_data

        model = IdealPointEstimator(
            IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        )

        results = model.fit(
            person_ids=data['person_ids'],
            item_ids=data['item_ids'],
            responses=data['responses'],
            inference='vi',
            vi_steps=500,
            device='cpu',
            progress_bar=False,
        )

        # Convert to DataFrame
        df_dict = results.to_dataframe()

        assert 'persons' in df_dict
        assert 'items' in df_dict

        persons_df = df_dict['persons']
        items_df = df_dict['items']

        # Check structure
        assert len(persons_df) == data['n_persons']
        assert len(items_df) == data['n_items']

        # Check columns
        assert 'person' in persons_df.columns
        assert 'ideal_point' in persons_df.columns
        assert 'ideal_point_se' in persons_df.columns

        assert 'item' in items_df.columns
        assert 'difficulty' in items_df.columns
        assert 'discrimination' in items_df.columns

        print(f"\n  DataFrames created: {len(persons_df)} persons, {len(items_df)} items")


class TestDataLoaders:
    """Tests for data loading utilities."""

    def test_load_from_arrays(self, small_binary_data):
        """Test loading data from numpy arrays."""
        data = small_binary_data

        model = IdealPointEstimator(
            IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        )

        # Should work with direct arrays
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
        assert results.ideal_points.shape[0] == data['n_persons']

    def test_IdealPointData_loader(self):
        """Test loading data with IdealPointData class."""
        import pandas as pd
        from idealist.data import load_data

        # Create sample data
        df = pd.DataFrame({
            'person': ['Person_A', 'Person_A', 'Person_B', 'Person_B', 'Person_C'],
            'item': ['Item_1', 'Item_2', 'Item_1', 'Item_2', 'Item_1'],
            'response': [1, 0, 1, 1, 0],
        })

        # Load using data loader
        data = load_data(
            df,
            person_col='person',
            item_col='item',
            response_col='response',
        )

        assert data.n_persons == 3
        assert data.n_items == 2
        assert data.n_observations == 5

        # Should be able to fit model
        model = IdealPointEstimator(
            IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        )

        results = model.fit(
            data,  # Pass IdealPointData directly
            inference='vi',
            vi_steps=500,
            device='cpu',
            progress_bar=False,
        )

        assert results is not None

        # Check that names are preserved
        df_dict = results.to_dataframe()
        assert 'Person_A' in df_dict['persons']['person'].values
        assert 'Item_1' in df_dict['items']['item'].values


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
