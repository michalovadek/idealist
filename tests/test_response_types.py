"""
Tests for different response types (ordinal, continuous, count).

Tests that the IdealPointEstimator can handle various types of response data
beyond simple binary responses.
"""

import pytest
import numpy as np

from idealist import IdealPointEstimator, IdealPointConfig, ResponseType


class TestOrdinalResponses:
    """Tests for ordinal response data (e.g., Likert scales)."""

    def test_ordinal_inference(self, small_ordinal_data):
        """Test that ordinal data can be fit with VI inference."""
        data = small_ordinal_data

        config = IdealPointConfig(
            n_dims=data["n_dims"],
            response_type=ResponseType.ORDINAL,
            n_categories=data["n_categories"],
        )

        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids=data["person_ids"],
            item_ids=data["item_ids"],
            responses=data["responses"],
            inference="vi",
            vi_steps=1000,
            num_samples=200,
            device="cpu",
            progress_bar=False,
        )

        # Check results structure
        assert results is not None
        assert results.ideal_points.shape == (data["n_persons"], data["n_dims"])
        assert results.difficulty.shape == (data["n_items"],)

        # Check that responses are in expected range
        assert data["responses"].min() >= 0
        assert data["responses"].max() < data["n_categories"]

        print(f"\n  Ordinal VI completed in {results.computation_time:.2f}s")


class TestContinuousResponses:
    """Tests for continuous response data."""

    def test_continuous_inference(self, small_continuous_data):
        """Test that continuous data can be fit with VI inference."""
        data = small_continuous_data

        config = IdealPointConfig(
            n_dims=data["n_dims"],
            response_type=ResponseType.CONTINUOUS,
        )

        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids=data["person_ids"],
            item_ids=data["item_ids"],
            responses=data["responses"],
            inference="vi",
            vi_steps=1000,
            num_samples=200,
            device="cpu",
            progress_bar=False,
        )

        # Check results
        assert results is not None
        assert results.ideal_points.shape == (data["n_persons"], data["n_dims"])
        assert results.difficulty.shape == (data["n_items"],)

        print(f"\n  Continuous VI completed in {results.computation_time:.2f}s")

    def test_bounded_continuous_inference(self, small_bounded_continuous_data):
        """Test that bounded continuous data can be fit."""
        data = small_bounded_continuous_data

        config = IdealPointConfig(
            n_dims=data["n_dims"],
            response_type=ResponseType.BOUNDED_CONTINUOUS,
            response_bounds=(0.0, 10.0),
        )

        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids=data["person_ids"],
            item_ids=data["item_ids"],
            responses=data["responses"],
            inference="vi",
            vi_steps=1000,
            num_samples=200,
            device="cpu",
            progress_bar=False,
        )

        # Check results
        assert results is not None
        assert results.ideal_points.shape == (data["n_persons"], data["n_dims"])

        # Verify responses are bounded
        assert data["responses"].min() >= 0.0
        assert data["responses"].max() <= 10.0

        print(f"\n  Bounded continuous VI completed in {results.computation_time:.2f}s")


class TestCountResponses:
    """Tests for count response data."""

    def test_count_inference(self, small_count_data):
        """Test that count data can be fit with VI inference."""
        data = small_count_data

        config = IdealPointConfig(
            n_dims=data["n_dims"],
            response_type=ResponseType.COUNT,
        )

        model = IdealPointEstimator(config)

        results = model.fit(
            person_ids=data["person_ids"],
            item_ids=data["item_ids"],
            responses=data["responses"],
            inference="vi",
            vi_steps=1000,
            num_samples=200,
            device="cpu",
            progress_bar=False,
        )

        # Check results
        assert results is not None
        assert results.ideal_points.shape == (data["n_persons"], data["n_dims"])
        assert results.difficulty.shape == (data["n_items"],)

        # Verify responses are non-negative integers
        assert data["responses"].min() >= 0
        assert np.allclose(data["responses"], np.round(data["responses"]))

        print(f"\n  Count VI completed in {results.computation_time:.2f}s")


@pytest.mark.parametrize(
    "response_type,fixture_name",
    [
        (ResponseType.BINARY, "small_binary_data"),
        (ResponseType.CONTINUOUS, "small_continuous_data"),
        (ResponseType.COUNT, "small_count_data"),
    ],
)
def test_response_type_parametrized(response_type, fixture_name, request):
    """Parametrized test to verify all response types work with VI inference."""
    data = request.getfixturevalue(fixture_name)

    config_kwargs = {"n_dims": data["n_dims"], "response_type": response_type}

    # Add special parameters for ordinal
    if response_type == ResponseType.ORDINAL and "n_categories" in data:
        config_kwargs["n_categories"] = data["n_categories"]

    config = IdealPointConfig(**config_kwargs)
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
    assert results.ideal_points.shape == (data["n_persons"], data["n_dims"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
