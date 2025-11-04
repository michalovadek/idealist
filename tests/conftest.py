"""Pytest fixtures for testing ideal point estimation."""

import pytest
import numpy as np


def _generate_binary_data(n_persons, n_items, n_dims, sparsity, seed):
    """
    Generate synthetic binary response data using IRT model.

    Parameters
    ----------
    n_persons : int
        Number of persons
    n_items : int
        Number of items
    n_dims : int
        Number of dimensions
    sparsity : float
        Probability threshold for missing data (higher = more sparse)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary containing generated data and true parameters
    """
    np.random.seed(seed)

    # Generate true parameters
    true_ideal_points = np.random.normal(0, 1, (n_persons, n_dims))
    true_difficulty = np.random.normal(0, 1, n_items)
    true_discrimination = np.abs(np.random.normal(1, 0.5, (n_items, n_dims)))

    # Generate responses
    person_ids = []
    item_ids = []
    responses = []

    for i in range(n_persons):
        for j in range(n_items):
            # Missing data (sparse)
            if np.random.rand() > sparsity:
                continue

            # Compute probability
            linear_pred = true_difficulty[j] + np.sum(true_discrimination[j] * true_ideal_points[i])
            prob = 1 / (1 + np.exp(-linear_pred))

            # Generate response
            response = int(np.random.rand() < prob)

            person_ids.append(i)
            item_ids.append(j)
            responses.append(response)

    return {
        "person_ids": np.array(person_ids),
        "item_ids": np.array(item_ids),
        "responses": np.array(responses),
        "n_persons": n_persons,
        "n_items": n_items,
        "n_dims": n_dims,
        "true_ideal_points": true_ideal_points,
        "true_difficulty": true_difficulty,
        "true_discrimination": true_discrimination,
    }


def _generate_ordinal_data(n_persons, n_items, n_dims, n_categories, sparsity, seed):
    """Generate synthetic ordinal response data (e.g., Likert scales)."""
    np.random.seed(seed)

    # Generate true parameters
    true_ideal_points = np.random.normal(0, 1, (n_persons, n_dims))
    true_difficulty = np.random.normal(0, 1, n_items)
    true_discrimination = np.abs(np.random.normal(1, 0.5, (n_items, n_dims)))

    # Generate threshold parameters for ordinal categories
    true_thresholds = np.sort(np.random.normal(0, 1, n_categories - 1))

    # Generate responses
    person_ids = []
    item_ids = []
    responses = []

    for i in range(n_persons):
        for j in range(n_items):
            if np.random.rand() > sparsity:
                continue

            # Compute linear predictor
            linear_pred = true_difficulty[j] + np.sum(true_discrimination[j] * true_ideal_points[i])

            # Convert to ordinal category using cumulative probabilities
            cumulative_probs = 1 / (1 + np.exp(-(linear_pred - true_thresholds)))
            probs = np.concatenate([[0], cumulative_probs, [1]])
            category_probs = np.diff(probs)

            # Ensure probabilities are valid (small floating point errors can cause issues)
            category_probs = np.abs(category_probs)
            category_probs = category_probs / category_probs.sum()

            # Sample category
            response = int(np.random.choice(n_categories, p=category_probs))

            person_ids.append(i)
            item_ids.append(j)
            responses.append(response)

    return {
        "person_ids": np.array(person_ids),
        "item_ids": np.array(item_ids),
        "responses": np.array(responses),
        "n_persons": n_persons,
        "n_items": n_items,
        "n_dims": n_dims,
        "n_categories": n_categories,
        "true_ideal_points": true_ideal_points,
        "true_difficulty": true_difficulty,
        "true_discrimination": true_discrimination,
        "true_thresholds": true_thresholds,
    }


def _generate_continuous_data(n_persons, n_items, n_dims, sparsity, seed, bounded=False):
    """Generate synthetic continuous response data."""
    np.random.seed(seed)

    # Generate true parameters
    true_ideal_points = np.random.normal(0, 1, (n_persons, n_dims))
    true_difficulty = np.random.normal(0, 1, n_items)
    true_discrimination = np.abs(np.random.normal(1, 0.5, (n_items, n_dims)))

    # Generate responses
    person_ids = []
    item_ids = []
    responses = []

    for i in range(n_persons):
        for j in range(n_items):
            if np.random.rand() > sparsity:
                continue

            # Compute linear predictor
            linear_pred = true_difficulty[j] + np.sum(true_discrimination[j] * true_ideal_points[i])

            # Add noise
            response = linear_pred + np.random.normal(0, 0.5)

            # Bound if needed
            if bounded:
                response = np.clip(response, 0, 10)

            person_ids.append(i)
            item_ids.append(j)
            responses.append(response)

    return {
        "person_ids": np.array(person_ids),
        "item_ids": np.array(item_ids),
        "responses": np.array(responses),
        "n_persons": n_persons,
        "n_items": n_items,
        "n_dims": n_dims,
        "true_ideal_points": true_ideal_points,
        "true_difficulty": true_difficulty,
        "true_discrimination": true_discrimination,
    }


def _generate_count_data(n_persons, n_items, n_dims, sparsity, seed):
    """Generate synthetic count response data (Poisson-like)."""
    np.random.seed(seed)

    # Generate true parameters
    true_ideal_points = np.random.normal(0, 1, (n_persons, n_dims))
    true_difficulty = np.random.normal(0, 0.5, n_items)
    true_discrimination = np.abs(np.random.normal(0.5, 0.2, (n_items, n_dims)))

    # Generate responses
    person_ids = []
    item_ids = []
    responses = []

    for i in range(n_persons):
        for j in range(n_items):
            if np.random.rand() > sparsity:
                continue

            # Compute rate parameter
            linear_pred = true_difficulty[j] + np.sum(true_discrimination[j] * true_ideal_points[i])
            rate = np.exp(linear_pred)

            # Generate count
            response = int(np.random.poisson(rate))

            person_ids.append(i)
            item_ids.append(j)
            responses.append(response)

    return {
        "person_ids": np.array(person_ids),
        "item_ids": np.array(item_ids),
        "responses": np.array(responses),
        "n_persons": n_persons,
        "n_items": n_items,
        "n_dims": n_dims,
        "true_ideal_points": true_ideal_points,
        "true_difficulty": true_difficulty,
        "true_discrimination": true_discrimination,
    }


@pytest.fixture
def small_binary_data():
    """Small synthetic binary response data for quick testing."""
    return _generate_binary_data(n_persons=30, n_items=15, n_dims=1, sparsity=0.8, seed=42)


@pytest.fixture
def multidim_binary_data():
    """Binary data with multiple dimensions."""
    return _generate_binary_data(n_persons=40, n_items=20, n_dims=2, sparsity=0.75, seed=999)


@pytest.fixture
def small_ordinal_data():
    """Small synthetic ordinal response data (5-point scale)."""
    return _generate_ordinal_data(
        n_persons=30, n_items=15, n_dims=1, n_categories=5, sparsity=0.8, seed=42
    )


@pytest.fixture
def small_continuous_data():
    """Small synthetic continuous response data."""
    return _generate_continuous_data(
        n_persons=30, n_items=15, n_dims=1, sparsity=0.8, seed=42, bounded=False
    )


@pytest.fixture
def small_bounded_continuous_data():
    """Small synthetic bounded continuous response data."""
    return _generate_continuous_data(
        n_persons=30, n_items=15, n_dims=1, sparsity=0.8, seed=42, bounded=True
    )


@pytest.fixture
def small_count_data():
    """Small synthetic count response data."""
    return _generate_count_data(n_persons=30, n_items=15, n_dims=1, sparsity=0.8, seed=42)
