"""
Helper functions for common prior distributions in ideal point estimation models.

This module provides convenient functions for specifying priors with different
distribution families beyond the default Normal distribution. These can be used
to customize the prior specification in ideal point estimation models.

Examples
--------
>>> from idealist.core.prior_distributions import weakly_informative_priors
>>> config = IdealPointConfig(
...     response_type=ResponseType.BINARY,
...     **weakly_informative_priors()
... )

>>> from idealist.core.prior_distributions import conservative_priors
>>> config = IdealPointConfig(
...     response_type=ResponseType.BINARY,
...     **conservative_priors()
... )
"""

from typing import Dict


def default_priors() -> Dict[str, float]:
    """
    Default prior configuration (current defaults).

    Returns
    -------
    dict
        Dictionary of prior parameters suitable for unpacking into IdealPointConfig.

    Notes
    -----
    - Ideal points: N(0, 1) - standard normal
    - Difficulty: N(0, 2.5) - weakly informative
    - Discrimination: N(0, 2.5) - weakly informative

    Examples
    --------
    >>> config = IdealPointConfig(response_type=ResponseType.BINARY, **default_priors())
    """
    return {
        "prior_ideal_point_mean": 0.0,
        "prior_ideal_point_scale": 1.0,
        "prior_difficulty_mean": 0.0,
        "prior_difficulty_scale": 2.5,
        "prior_discrimination_mean": 0.0,
        "prior_discrimination_scale": 2.5,
    }


def weakly_informative_priors(
    ideal_point_scale: float = 2.0,
    difficulty_scale: float = 3.0,
    discrimination_scale: float = 3.0,
) -> Dict[str, float]:
    """
    Weakly informative priors suitable for most applications.

    These priors are more diffuse than the defaults, allowing the data to
    have more influence while still providing some regularization.

    Parameters
    ----------
    ideal_point_scale : float, default=2.0
        Scale for ideal point prior (default is 1.0)
    difficulty_scale : float, default=3.0
        Scale for difficulty prior (default is 2.5)
    discrimination_scale : float, default=3.0
        Scale for discrimination prior (default is 2.5)

    Returns
    -------
    dict
        Dictionary of prior parameters suitable for unpacking into IdealPointConfig.

    Notes
    -----
    - Ideal points: N(0, 2) - allows wider range of abilities
    - Difficulty: N(0, 3) - accommodates very easy/hard items
    - Discrimination: N(0, 3) - allows highly discriminating items

    Examples
    --------
    >>> config = IdealPointConfig(
    ...     response_type=ResponseType.BINARY,
    ...     **weakly_informative_priors()
    ... )

    >>> config = IdealPointConfig(
    ...     response_type=ResponseType.BINARY,
    ...     **weakly_informative_priors(ideal_point_scale=1.5)
    ... )
    """
    return {
        "prior_ideal_point_mean": 0.0,
        "prior_ideal_point_scale": ideal_point_scale,
        "prior_difficulty_mean": 0.0,
        "prior_difficulty_scale": difficulty_scale,
        "prior_discrimination_mean": 0.0,
        "prior_discrimination_scale": discrimination_scale,
    }


def conservative_priors(
    ideal_point_scale: float = 0.5,
    difficulty_scale: float = 1.0,
    discrimination_scale: float = 1.0,
) -> Dict[str, float]:
    """
    Conservative (more informative) priors for regularization.

    These priors provide stronger regularization, useful when you have:
    - Limited data
    - Noisy observations
    - Need to prevent overfitting
    - Want to keep parameters near zero

    Parameters
    ----------
    ideal_point_scale : float, default=0.5
        Scale for ideal point prior (default is 1.0)
    difficulty_scale : float, default=1.0
        Scale for difficulty prior (default is 2.5)
    discrimination_scale : float, default=1.0
        Scale for discrimination prior (default is 2.5)

    Returns
    -------
    dict
        Dictionary of prior parameters suitable for unpacking into IdealPointConfig.

    Notes
    -----
    - Ideal points: N(0, 0.5) - keeps abilities near zero
    - Difficulty: N(0, 1) - moderate difficulty range
    - Discrimination: N(0, 1) - moderate discrimination

    Examples
    --------
    >>> config = IdealPointConfig(
    ...     response_type=ResponseType.BINARY,
    ...     **conservative_priors()
    ... )
    """
    return {
        "prior_ideal_point_mean": 0.0,
        "prior_ideal_point_scale": ideal_point_scale,
        "prior_difficulty_mean": 0.0,
        "prior_difficulty_scale": difficulty_scale,
        "prior_discrimination_mean": 0.0,
        "prior_discrimination_scale": discrimination_scale,
    }


def vague_priors(
    ideal_point_scale: float = 5.0,
    difficulty_scale: float = 10.0,
    discrimination_scale: float = 10.0,
) -> Dict[str, float]:
    """
    Vague (non-informative) priors that let data dominate.

    These priors provide minimal regularization, useful when you have:
    - Lots of data
    - High-quality observations
    - Want to let the data speak for itself
    - Exploring without prior assumptions

    Parameters
    ----------
    ideal_point_scale : float, default=5.0
        Scale for ideal point prior (default is 1.0)
    difficulty_scale : float, default=10.0
        Scale for difficulty prior (default is 2.5)
    discrimination_scale : float, default=10.0
        Scale for discrimination prior (default is 2.5)

    Returns
    -------
    dict
        Dictionary of prior parameters suitable for unpacking into IdealPointConfig.

    Notes
    -----
    - Ideal points: N(0, 5) - very wide range
    - Difficulty: N(0, 10) - almost no constraint
    - Discrimination: N(0, 10) - almost no constraint

    Warning
    -------
    Very diffuse priors can lead to identification issues and slow convergence.
    Use with caution, especially with limited data.

    Examples
    --------
    >>> config = IdealPointConfig(
    ...     response_type=ResponseType.BINARY,
    ...     **vague_priors()
    ... )
    """
    return {
        "prior_ideal_point_mean": 0.0,
        "prior_ideal_point_scale": ideal_point_scale,
        "prior_difficulty_mean": 0.0,
        "prior_difficulty_scale": difficulty_scale,
        "prior_discrimination_mean": 0.0,
        "prior_discrimination_scale": discrimination_scale,
    }


def centered_priors(
    ideal_point_mean: float,
    difficulty_mean: float = 0.0,
    discrimination_mean: float = 0.0,
    ideal_point_scale: float = 1.0,
    difficulty_scale: float = 2.5,
    discrimination_scale: float = 2.5,
) -> Dict[str, float]:
    """
    Priors centered at non-zero values.

    Useful when you have prior information about parameter locations,
    such as from previous studies or domain knowledge.

    Parameters
    ----------
    ideal_point_mean : float
        Center for ideal point prior
    difficulty_mean : float, default=0.0
        Center for difficulty prior
    discrimination_mean : float, default=0.0
        Center for discrimination prior
    ideal_point_scale : float, default=1.0
        Scale for ideal point prior
    difficulty_scale : float, default=2.5
        Scale for difficulty prior
    discrimination_scale : float, default=2.5
        Scale for discrimination prior

    Returns
    -------
    dict
        Dictionary of prior parameters suitable for unpacking into IdealPointConfig.

    Examples
    --------
    >>> # Center ideal points at 0.5 (e.g., if you expect mostly above-average ability)
    >>> config = IdealPointConfig(
    ...     response_type=ResponseType.BINARY,
    ...     **centered_priors(ideal_point_mean=0.5)
    ... )

    >>> # If items are generally difficult (negative difficulty in IRT parameterization)
    >>> config = IdealPointConfig(
    ...     response_type=ResponseType.BINARY,
    ...     **centered_priors(ideal_point_mean=0.0, difficulty_mean=-1.0)
    ... )
    """
    return {
        "prior_ideal_point_mean": ideal_point_mean,
        "prior_ideal_point_scale": ideal_point_scale,
        "prior_difficulty_mean": difficulty_mean,
        "prior_difficulty_scale": difficulty_scale,
        "prior_discrimination_mean": discrimination_mean,
        "prior_discrimination_scale": discrimination_scale,
    }


def rasch_priors(
    ideal_point_scale: float = 1.0,
    difficulty_scale: float = 2.5,
) -> Dict[str, float]:
    """
    Priors suitable for Rasch models (1PL: equal discrimination).

    Sets discrimination prior to be very tight around a positive value,
    effectively implementing equal discrimination constraints.

    Parameters
    ----------
    ideal_point_scale : float, default=1.0
        Scale for ideal point prior
    difficulty_scale : float, default=2.5
        Scale for difficulty prior

    Returns
    -------
    dict
        Dictionary of prior parameters suitable for unpacking into IdealPointConfig.

    Notes
    -----
    - Ideal points: N(0, 1) - standard normal
    - Difficulty: N(0, 2.5) - weakly informative
    - Discrimination: N(1, 0.1) - tightly centered at 1.0

    This approximates a Rasch model by constraining discrimination to be
    near 1.0 for all items, rather than freely estimated.

    Examples
    --------
    >>> config = IdealPointConfig(
    ...     response_type=ResponseType.BINARY,
    ...     **rasch_priors()
    ... )
    """
    return {
        "prior_ideal_point_mean": 0.0,
        "prior_ideal_point_scale": ideal_point_scale,
        "prior_difficulty_mean": 0.0,
        "prior_difficulty_scale": difficulty_scale,
        "prior_discrimination_mean": 1.0,  # Center at 1.0
        "prior_discrimination_scale": 0.1,  # Tight constraint
    }


def hierarchical_priors(
    covariate_scale: float = 1.0,
    threshold_scale: float = 2.0,
    residual_scale: float = 1.0,
    temporal_variance: float = 0.1,
    **base_priors,
) -> Dict[str, float]:
    """
    Complete prior specification including hierarchical components.

    This function allows you to specify both base priors (for ideal points,
    difficulty, discrimination) and hierarchical priors (for covariates,
    thresholds, residuals) in one call.

    Parameters
    ----------
    covariate_scale : float, default=1.0
        Prior scale for covariate effects
    threshold_scale : float, default=2.0
        Prior scale for ordinal thresholds
    residual_scale : float, default=1.0
        Prior scale for continuous response residuals
    temporal_variance : float, default=0.1
        Variance for temporal random walk
    **base_priors
        Additional base prior parameters (e.g., from other helper functions)

    Returns
    -------
    dict
        Dictionary of prior parameters suitable for unpacking into IdealPointConfig.

    Examples
    --------
    >>> # Use with default base priors
    >>> config = IdealPointConfig(
    ...     response_type=ResponseType.ORDINAL,
    ...     person_covariates=True,
    ...     **hierarchical_priors(covariate_scale=0.5, threshold_scale=1.5)
    ... )

    >>> # Combine with weakly_informative_priors
    >>> config = IdealPointConfig(
    ...     response_type=ResponseType.ORDINAL,
    ...     person_covariates=True,
    ...     **hierarchical_priors(
    ...         covariate_scale=0.5,
    ...         **weakly_informative_priors()
    ...     )
    ... )
    """
    # Start with base priors if provided, otherwise use defaults
    if not base_priors:
        priors = default_priors()
    else:
        priors = base_priors.copy()

    # Add hierarchical components
    priors.update(
        {
            "prior_covariate_scale": covariate_scale,
            "prior_threshold_scale": threshold_scale,
            "prior_residual_scale": residual_scale,
            "temporal_variance": temporal_variance,
        }
    )

    return priors


# Convenience aliases for common use cases
def standard_priors() -> Dict[str, float]:
    """Alias for default_priors()."""
    return default_priors()


def regularized_priors() -> Dict[str, float]:
    """Alias for conservative_priors()."""
    return conservative_priors()


def flexible_priors() -> Dict[str, float]:
    """Alias for weakly_informative_priors()."""
    return weakly_informative_priors()


# Export all functions
__all__ = [
    "default_priors",
    "weakly_informative_priors",
    "conservative_priors",
    "vague_priors",
    "centered_priors",
    "rasch_priors",
    "hierarchical_priors",
    "standard_priors",
    "regularized_priors",
    "flexible_priors",
]
