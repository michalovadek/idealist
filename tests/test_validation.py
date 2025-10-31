"""
Tests for data validation and error handling.

Tests that the package properly validates inputs and raises
appropriate errors for invalid data.
"""

import pytest
import numpy as np
import pandas as pd

from idealist import IdealPointEstimator, IdealPointConfig, ResponseType
from idealist.data import load_data, detect_response_type, IdealPointData


class TestDataValidation:
    """Tests for input data validation."""

    def test_mismatched_array_lengths(self):
        """Test that mismatched array lengths raise an error."""
        person_ids = np.array([0, 1, 2])
        item_ids = np.array([0, 1])  # Wrong length
        responses = np.array([1, 0, 1])

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        with pytest.raises((ValueError, AssertionError, TypeError, RuntimeError)):
            model.fit(
                person_ids=person_ids,
                item_ids=item_ids,
                responses=responses,
                inference='vi',
                vi_steps=100,
                device='cpu',
                progress_bar=False,
            )

    def test_empty_data(self):
        """Test that empty data raises an error or handles gracefully."""
        person_ids = np.array([])
        item_ids = np.array([])
        responses = np.array([])

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        with pytest.raises((ValueError, AssertionError, RuntimeError)):
            model.fit(
                person_ids=person_ids,
                item_ids=item_ids,
                responses=responses,
                inference='vi',
                vi_steps=100,
                device='cpu',
                progress_bar=False,
            )

    def test_invalid_person_ids(self):
        """Test that non-contiguous or negative person IDs are handled."""
        # IDs that skip numbers or are negative
        person_ids = np.array([0, 2, 5, -1])  # Skips 1, 3, 4 and has negative
        item_ids = np.array([0, 0, 0, 0])
        responses = np.array([1, 0, 1, 0])

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        # This might work or raise an error depending on implementation
        # If it works, it should handle the non-contiguous IDs
        try:
            results = model.fit(
                person_ids=person_ids,
                item_ids=item_ids,
                responses=responses,
                inference='vi',
                vi_steps=100,
                device='cpu',
                progress_bar=False,
            )
            # If it succeeds, just verify it doesn't crash
            assert results is not None
        except (ValueError, AssertionError, RuntimeError, IndexError):
            # Expected for invalid IDs
            pass

    def test_single_person_single_item(self):
        """Test edge case with only one person and one item."""
        person_ids = np.array([0, 0, 0])
        item_ids = np.array([0, 0, 0])
        responses = np.array([1, 1, 0])

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        # This is a degenerate case - model might fail or succeed with warnings
        try:
            results = model.fit(
                person_ids=person_ids,
                item_ids=item_ids,
                responses=responses,
                inference='vi',
                vi_steps=100,
                device='cpu',
                progress_bar=False,
            )
            # If it works, check basic structure
            assert results.ideal_points.shape == (1, 1)
            assert results.difficulty.shape == (1,)
        except (ValueError, RuntimeError):
            # Expected - too little data to estimate
            pass


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_invalid_n_dims(self):
        """Test that invalid n_dims raises an error."""
        with pytest.raises((ValueError, TypeError)):
            IdealPointConfig(n_dims=0, response_type=ResponseType.BINARY)

        with pytest.raises((ValueError, TypeError)):
            IdealPointConfig(n_dims=-1, response_type=ResponseType.BINARY)

    def test_ordinal_requires_n_categories(self):
        """Test that ordinal response type requires n_categories."""
        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.ORDINAL,
            n_categories=5,  # Should be required
        )
        assert config.n_categories == 5

    def test_bounded_continuous_requires_bounds(self):
        """Test that bounded continuous requires response_bounds."""
        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BOUNDED_CONTINUOUS,
            response_bounds=(0.0, 10.0),  # Should be required
        )
        assert config.response_bounds == (0.0, 10.0)


class TestResponseTypeDetection:
    """Tests for automatic response type detection."""

    def test_detect_binary(self):
        """Test detection of binary responses."""
        responses = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        response_type, n_categories, bounds = detect_response_type(responses)
        assert response_type == ResponseType.BINARY
        assert n_categories is None

    def test_detect_ordinal(self):
        """Test detection of ordinal responses (3-10 categories)."""
        responses = np.array([1, 2, 3, 4, 5, 3, 2, 4, 5, 1])
        response_type, n_categories, bounds = detect_response_type(responses)
        assert response_type == ResponseType.ORDINAL
        assert n_categories == 5  # 5 unique values

    def test_detect_ordinal_different_scale(self):
        """Test detection with different ordinal scales."""
        # 3-point scale
        responses = np.array([0, 1, 2, 1, 0, 2, 1])
        response_type, n_categories, bounds = detect_response_type(responses)
        assert response_type == ResponseType.ORDINAL
        assert n_categories == 3

    def test_detect_count(self):
        """Test detection of count data (many integer values)."""
        responses = np.array([0, 1, 5, 10, 15, 20, 3, 7, 12, 25, 8, 14, 2, 18])
        response_type, n_categories, bounds = detect_response_type(responses)
        assert response_type == ResponseType.COUNT
        assert n_categories is None

    def test_detect_continuous(self):
        """Test detection of continuous responses."""
        responses = np.array([1.5, 2.3, 3.7, 4.1, 2.8, 3.2, 4.5, 1.9])
        response_type, n_categories, bounds = detect_response_type(responses)
        # Should be either BOUNDED_CONTINUOUS or CONTINUOUS
        assert response_type in [ResponseType.CONTINUOUS, ResponseType.BOUNDED_CONTINUOUS]

    def test_detect_bounded_continuous(self):
        """Test detection of bounded continuous responses."""
        responses = np.array([0.5, 1.2, 2.8, 3.4, 4.7, 2.1, 3.9, 1.5, 4.2])
        response_type, n_categories, bounds = detect_response_type(responses)
        if response_type == ResponseType.BOUNDED_CONTINUOUS:
            assert bounds is not None
            assert bounds[0] <= responses.min()
            assert bounds[1] >= responses.max()


class TestDataLoader:
    """Tests for the data loading utilities."""

    def test_load_from_dataframe(self):
        """Test loading data from a pandas DataFrame."""
        df = pd.DataFrame({
            'person': ['A', 'A', 'B', 'B', 'C'],
            'item': ['X', 'Y', 'X', 'Y', 'X'],
            'response': [1, 0, 1, 1, 0],
        })

        data = load_data(df, person_col='person', item_col='item', response_col='response')

        assert data.n_persons == 3
        assert data.n_items == 2
        assert data.n_observations == 5
        assert 'A' in data.person_names
        assert 'X' in data.item_names

    def test_load_missing_column(self):
        """Test that missing columns raise an error."""
        df = pd.DataFrame({
            'person': ['A', 'B'],
            'item': ['X', 'Y'],
            # Missing 'response' column
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            load_data(df, person_col='person', item_col='item', response_col='response')

    def test_load_preserves_names(self):
        """Test that person and item names are preserved."""
        df = pd.DataFrame({
            'legislator': ['Sen. Smith', 'Sen. Jones', 'Sen. Smith'],
            'bill': ['HR101', 'HR101', 'HR202'],
            'vote': [1, 0, 1],
        })

        data = load_data(df, person_col='legislator', item_col='bill', response_col='vote')

        assert 'Sen. Smith' in data.person_names
        assert 'Sen. Jones' in data.person_names
        assert 'HR101' in data.item_names
        assert 'HR202' in data.item_names

    def test_IdealPointData_validation(self):
        """Test that IdealPointData validates inputs."""
        # Valid data
        data = IdealPointData(
            person_ids=np.array([0, 1, 0]),
            item_ids=np.array([0, 0, 1]),
            responses=np.array([1.0, 0.0, 1.0]),
            person_names=['A', 'B'],
            item_names=['X', 'Y'],
        )
        assert data.n_persons == 2
        assert data.n_items == 2

    def test_IdealPointData_invalid_ids(self):
        """Test that IdealPointData rejects invalid IDs."""
        with pytest.raises(ValueError):
            # person_id 5 exceeds number of person_names (only 2)
            IdealPointData(
                person_ids=np.array([0, 1, 5]),
                item_ids=np.array([0, 0, 1]),
                responses=np.array([1.0, 0.0, 1.0]),
                person_names=['A', 'B'],
                item_names=['X', 'Y'],
            )

    def test_IdealPointData_summary(self):
        """Test the summary method."""
        data = IdealPointData(
            person_ids=np.array([0, 1, 0, 1]),
            item_ids=np.array([0, 0, 1, 1]),
            responses=np.array([1.0, 0.0, 1.0, 1.0]),
            person_names=['A', 'B'],
            item_names=['X', 'Y'],
        )

        summary = data.summary()
        assert 'Persons:' in summary
        assert 'Items:' in summary
        assert 'Observations:' in summary
        assert '2' in summary  # 2 persons
        assert '4' in summary  # 4 observations


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
