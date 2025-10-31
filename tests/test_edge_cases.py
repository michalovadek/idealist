"""
Tests for edge cases and corner scenarios.

Tests unusual but valid inputs and boundary conditions to ensure
robust behavior.
"""

import pytest
import numpy as np
import pandas as pd

from idealist import IdealPointEstimator, IdealPointConfig, ResponseType
from idealist.data import load_data


class TestDataEdgeCases:
    """Tests for edge cases in data handling."""

    def test_all_same_responses(self):
        """Test data where all responses are identical."""
        person_ids = np.array([0, 0, 1, 1, 2, 2])
        item_ids = np.array([0, 1, 0, 1, 0, 1])
        responses = np.array([1, 1, 1, 1, 1, 1])  # All ones

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        # Should handle gracefully - might converge to default or warn
        try:
            results = model.fit(
                person_ids=person_ids,
                item_ids=item_ids,
                responses=responses,
                inference='vi',
                vi_steps=300,
                device='cpu',
                progress_bar=False,
            )
            # If it succeeds, results should be valid
            assert results is not None
            assert not np.any(np.isnan(results.ideal_points))
        except (ValueError, RuntimeError):
            # Expected - no variation in data
            pass

    def test_single_observation_per_person(self):
        """Test when each person has only one observation."""
        person_ids = np.array([0, 1, 2, 3, 4])
        item_ids = np.array([0, 0, 0, 0, 0])  # Same item for all
        responses = np.array([1, 0, 1, 1, 0])

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        try:
            results = model.fit(
                person_ids=person_ids,
                item_ids=item_ids,
                responses=responses,
                inference='vi',
                vi_steps=300,
                device='cpu',
                progress_bar=False,
            )
            assert results is not None
        except (ValueError, RuntimeError):
            # Expected - minimal data
            pass

    def test_single_observation_per_item(self):
        """Test when each item has only one observation."""
        person_ids = np.array([0, 0, 0, 0, 0])  # Same person
        item_ids = np.array([0, 1, 2, 3, 4])
        responses = np.array([1, 0, 1, 1, 0])

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        try:
            results = model.fit(
                person_ids=person_ids,
                item_ids=item_ids,
                responses=responses,
                inference='vi',
                vi_steps=300,
                device='cpu',
                progress_bar=False,
            )
            assert results is not None
        except (ValueError, RuntimeError):
            # Expected - minimal data
            pass

    def test_extremely_sparse_data(self):
        """Test with very sparse observation matrix (< 5% observed)."""
        np.random.seed(42)
        n_persons = 50
        n_items = 50

        # Generate very sparse data
        person_ids = []
        item_ids = []
        responses = []

        for i in range(n_persons):
            for j in range(n_items):
                if np.random.rand() > 0.97:  # Only 3% observed
                    person_ids.append(i)
                    item_ids.append(j)
                    responses.append(np.random.choice([0, 1]))

        # Only proceed if we have some data
        if len(responses) > 10:
            config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
            model = IdealPointEstimator(config)

            try:
                results = model.fit(
                    person_ids=np.array(person_ids),
                    item_ids=np.array(item_ids),
                    responses=np.array(responses),
                    inference='vi',
                    vi_steps=500,
                    device='cpu',
                    progress_bar=False,
                )
                assert results is not None
                print(f"\n  Sparse data test: {len(responses)} observations successful")
            except (ValueError, RuntimeError):
                # Expected with very sparse data
                pass

    def test_extreme_response_values(self):
        """Test with extreme continuous response values."""
        person_ids = np.array([0, 0, 1, 1, 2, 2] * 5)
        item_ids = np.array([0, 1, 0, 1, 0, 1] * 5)
        responses = np.array([-1000, 1000, -500, 500, 0, 100] * 5).astype(float)

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.CONTINUOUS)
        model = IdealPointEstimator(config)

        try:
            results = model.fit(
                person_ids=person_ids,
                item_ids=item_ids,
                responses=responses,
                inference='vi',
                vi_steps=300,
                device='cpu',
                progress_bar=False,
            )
            assert results is not None
        except (ValueError, RuntimeError, OverflowError):
            # Expected with extreme values
            pass

    def test_many_persons_few_items(self):
        """Test with many persons but very few items."""
        np.random.seed(42)
        n_persons = 100
        n_items = 2

        person_ids = []
        item_ids = []
        responses = []

        for i in range(n_persons):
            for j in range(n_items):
                if np.random.rand() > 0.3:
                    person_ids.append(i)
                    item_ids.append(j)
                    responses.append(np.random.choice([0, 1]))

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids=np.array(person_ids),
            item_ids=np.array(item_ids),
            responses=np.array(responses),
            inference='vi',
            vi_steps=500,
            device='cpu',
            progress_bar=False,
        )

        assert results is not None
        assert results.ideal_points.shape[0] == n_persons
        print(f"\n  Many persons, few items test successful")

    def test_few_persons_many_items(self):
        """Test with few persons but many items."""
        np.random.seed(42)
        n_persons = 2
        n_items = 100

        person_ids = []
        item_ids = []
        responses = []

        for i in range(n_persons):
            for j in range(n_items):
                if np.random.rand() > 0.3:
                    person_ids.append(i)
                    item_ids.append(j)
                    responses.append(np.random.choice([0, 1]))

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids=np.array(person_ids),
            item_ids=np.array(item_ids),
            responses=np.array(responses),
            inference='vi',
            vi_steps=500,
            device='cpu',
            progress_bar=False,
        )

        assert results is not None
        assert results.difficulty.shape[0] == n_items
        print(f"\n  Few persons, many items test successful")

    def test_person_with_no_variation(self):
        """Test when a person responds the same way to all items."""
        person_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        item_ids = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        responses = np.array([1, 1, 1, 0, 1, 0, 1, 0, 1])  # Person 0 all 1s

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        try:
            results = model.fit(
                person_ids=person_ids,
                item_ids=item_ids,
                responses=responses,
                inference='vi',
                vi_steps=300,
                device='cpu',
                progress_bar=False,
            )
            assert results is not None
        except (ValueError, RuntimeError):
            pass

    def test_item_with_no_variation(self):
        """Test when an item receives the same response from all persons."""
        person_ids = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        item_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        responses = np.array([1, 1, 1, 0, 1, 0, 1, 0, 1])  # Item 0 all 1s

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        try:
            results = model.fit(
                person_ids=person_ids,
                item_ids=item_ids,
                responses=responses,
                inference='vi',
                vi_steps=300,
                device='cpu',
                progress_bar=False,
            )
            assert results is not None
        except (ValueError, RuntimeError):
            pass


class TestConfigurationEdgeCases:
    """Tests for edge cases in model configuration."""

    def test_very_high_dimensionality(self):
        """Test with unusually high number of dimensions."""
        # Try 3+ dimensions (note: some implementations may limit to 1-2)
        person_ids = np.array([0, 0, 1, 1, 2, 2] * 20)
        item_ids = np.array([0, 1, 0, 1, 0, 1] * 20)
        responses = np.array([1, 0, 1, 1, 0, 1] * 20)

        # Some implementations limit dimensionality
        with pytest.raises((ValueError, RuntimeError)):
            config = IdealPointConfig(n_dims=5, response_type=ResponseType.BINARY)
            model = IdealPointEstimator(config)

            model.fit(
                person_ids=person_ids,
                item_ids=item_ids,
                responses=responses,
                inference='vi',
                vi_steps=500,
                device='cpu',
                progress_bar=False,
            )

    def test_very_few_vi_steps(self):
        """Test with minimal VI steps."""
        person_ids = np.array([0, 0, 1, 1, 2, 2] * 10)
        item_ids = np.array([0, 1, 0, 1, 0, 1] * 10)
        responses = np.array([1, 0, 1, 1, 0, 1] * 10)

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        # Very few steps - should still complete but might not converge well
        results = model.fit(
            person_ids=person_ids,
            item_ids=item_ids,
            responses=responses,
            inference='vi',
            vi_steps=10,  # Very few
            device='cpu',
            progress_bar=False,
        )

        assert results is not None
        print(f"\n  Minimal VI steps test successful")

    def test_ordinal_with_many_categories(self):
        """Test ordinal response with many categories (10-point scale)."""
        person_ids = np.array([0, 0, 1, 1, 2, 2] * 10)
        item_ids = np.array([0, 1, 0, 1, 0, 1] * 10)
        # 10-point scale
        responses = np.array([0, 5, 2, 8, 3, 9] * 10)

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.ORDINAL,
            n_categories=10
        )
        model = IdealPointEstimator(config)

        try:
            results = model.fit(
                person_ids=person_ids,
                item_ids=item_ids,
                responses=responses,
                inference='vi',
                vi_steps=500,
                device='cpu',
                progress_bar=False,
            )
            assert results is not None
        except (ValueError, RuntimeError):
            # Expected - many categories can be challenging
            pass

    def test_count_with_zeros(self):
        """Test count data with many zeros."""
        person_ids = np.array([0, 0, 1, 1, 2, 2] * 10)
        item_ids = np.array([0, 1, 0, 1, 0, 1] * 10)
        responses = np.array([0, 0, 0, 1, 0, 2] * 10)  # Many zeros

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.COUNT)
        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids=person_ids,
            item_ids=item_ids,
            responses=responses,
            inference='vi',
            vi_steps=500,
            device='cpu',
            progress_bar=False,
        )

        assert results is not None
        print(f"\n  Count data with zeros test successful")


class TestNumericEdgeCases:
    """Tests for numerical edge cases."""

    def test_very_small_prior_scales(self):
        """Test with very tight priors (small scales)."""
        person_ids = np.array([0, 0, 1, 1, 2, 2] * 10)
        item_ids = np.array([0, 1, 0, 1, 0, 1] * 10)
        responses = np.array([1, 0, 1, 1, 0, 1] * 10)

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            prior_ideal_point_scale=0.01,  # Very tight
            prior_difficulty_scale=0.01,
            prior_discrimination_scale=0.01,
        )

        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids=person_ids,
            item_ids=item_ids,
            responses=responses,
            inference='vi',
            vi_steps=500,
            device='cpu',
            progress_bar=False,
        )

        # Should strongly regularize toward zero
        assert results is not None
        assert np.abs(results.ideal_points).mean() < 1.0

        print(f"\n  Tight priors test successful")

    def test_responses_with_nan(self):
        """Test that NaN responses are rejected or handled."""
        person_ids = np.array([0, 1, 2])
        item_ids = np.array([0, 0, 0])
        responses = np.array([1.0, np.nan, 0.0])

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        # Should raise an error or handle gracefully
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            model.fit(
                person_ids=person_ids,
                item_ids=item_ids,
                responses=responses,
                inference='vi',
                vi_steps=100,
                device='cpu',
                progress_bar=False,
            )

    def test_responses_with_inf(self):
        """Test that infinite responses are rejected."""
        person_ids = np.array([0, 1, 2])
        item_ids = np.array([0, 0, 0])
        responses = np.array([1.0, np.inf, 0.0])

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.CONTINUOUS)
        model = IdealPointEstimator(config)

        # Should raise an error
        with pytest.raises((ValueError, RuntimeError, TypeError, OverflowError)):
            model.fit(
                person_ids=person_ids,
                item_ids=item_ids,
                responses=responses,
                inference='vi',
                vi_steps=100,
                device='cpu',
                progress_bar=False,
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
