"""Pytest fixtures for testing ideal point estimation."""

import numpy as np
import pytest


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


def _generate_temporal_data(n_persons, n_items, n_dims, n_timepoints, sparsity, seed):
    """Generate synthetic temporal data with time-varying ideal points."""
    np.random.seed(seed)

    # Generate initial ideal points
    theta_initial = np.random.normal(0, 1, (n_persons, n_dims))

    # Generate temporal evolution (random walk)
    temporal_variance = 0.1
    temporal_ideal_points = np.zeros((n_timepoints, n_persons, n_dims))
    temporal_ideal_points[0] = theta_initial

    for t in range(1, n_timepoints):
        innovations = np.random.normal(0, temporal_variance, (n_persons, n_dims))
        temporal_ideal_points[t] = temporal_ideal_points[t-1] + innovations

    # Generate item parameters
    true_difficulty = np.random.normal(0, 1, n_items)
    true_discrimination = np.abs(np.random.normal(1, 0.5, (n_items, n_dims)))

    # Generate responses across time
    person_ids = []
    item_ids = []
    timepoints = []
    responses = []

    for t in range(n_timepoints):
        for i in range(n_persons):
            for j in range(n_items):
                if np.random.rand() > sparsity:
                    continue

                # Compute probability using ideal point at time t
                linear_pred = true_difficulty[j] + np.sum(
                    true_discrimination[j] * temporal_ideal_points[t, i]
                )
                prob = 1 / (1 + np.exp(-linear_pred))

                # Generate response
                response = int(np.random.rand() < prob)

                person_ids.append(i)
                item_ids.append(j)
                timepoints.append(t)
                responses.append(response)

    return {
        "person_ids": np.array(person_ids),
        "item_ids": np.array(item_ids),
        "timepoints": np.array(timepoints),
        "responses": np.array(responses),
        "n_persons": n_persons,
        "n_items": n_items,
        "n_dims": n_dims,
        "n_timepoints": n_timepoints,
        "true_temporal_ideal_points": temporal_ideal_points,
        "true_difficulty": true_difficulty,
        "true_discrimination": true_discrimination,
    }


def _generate_covariate_data(n_persons, n_items, n_dims, n_person_covariates, n_item_covariates, sparsity, seed):
    """Generate synthetic data with person and item covariates."""
    np.random.seed(seed)

    # Generate person covariates
    person_covariates = np.random.randn(n_persons, n_person_covariates)

    # Generate item covariates
    item_covariates = np.random.randn(n_items, n_item_covariates)

    # Generate covariate effects
    gamma_person = np.random.normal(0.5, 0.2, (n_person_covariates, n_dims))  # Person covariate effects
    delta_item = np.random.normal(0.3, 0.1, n_item_covariates)  # Item covariate effects on difficulty

    # Generate ideal points from person covariates
    true_ideal_points = person_covariates @ gamma_person + np.random.normal(0, 0.3, (n_persons, n_dims))

    # Generate item parameters from item covariates
    true_difficulty = item_covariates @ delta_item + np.random.normal(0, 0.5, n_items)
    true_discrimination = np.abs(np.random.normal(1, 0.5, (n_items, n_dims)))

    # Generate responses
    person_ids = []
    item_ids = []
    responses = []

    for i in range(n_persons):
        for j in range(n_items):
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
        "person_covariates": person_covariates,
        "item_covariates": item_covariates,
        "n_persons": n_persons,
        "n_items": n_items,
        "n_dims": n_dims,
        "n_person_covariates": n_person_covariates,
        "n_item_covariates": n_item_covariates,
        "true_ideal_points": true_ideal_points,
        "true_difficulty": true_difficulty,
        "true_discrimination": true_discrimination,
        "true_gamma_person": gamma_person,
        "true_delta_item": delta_item,
    }


@pytest.fixture
def small_temporal_data():
    """Small synthetic temporal data for testing temporal dynamics."""
    return _generate_temporal_data(
        n_persons=20, n_items=12, n_dims=1, n_timepoints=3, sparsity=0.8, seed=42
    )


@pytest.fixture
def medium_temporal_data():
    """Medium synthetic temporal data with more timepoints."""
    return _generate_temporal_data(
        n_persons=25, n_items=15, n_dims=1, n_timepoints=5, sparsity=0.75, seed=123
    )


@pytest.fixture
def person_covariate_data():
    """Data with person-level covariates."""
    return _generate_covariate_data(
        n_persons=25, n_items=15, n_dims=1,
        n_person_covariates=2, n_item_covariates=0,
        sparsity=0.8, seed=42
    )


@pytest.fixture
def item_covariate_data():
    """Data with item-level covariates."""
    return _generate_covariate_data(
        n_persons=25, n_items=15, n_dims=1,
        n_person_covariates=0, n_item_covariates=2,
        sparsity=0.8, seed=123
    )


@pytest.fixture
def both_covariate_data():
    """Data with both person and item covariates."""
    return _generate_covariate_data(
        n_persons=25, n_items=15, n_dims=1,
        n_person_covariates=2, n_item_covariates=2,
        sparsity=0.8, seed=456
    )
