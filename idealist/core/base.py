"""
Base classes and enums for ideal point estimation.

This module provides the foundational components for ideal point models, which are
used to estimate latent positions of persons and items based on observed response data.

Ideal point models are a class of latent variable models that place persons and
items in a low-dimensional space. The probability of a positive response depends on
the distance between the person's ideal point and the item's position. These models
apply to any binary, ordinal, or continuous response data where positions matter.

Key concepts:
- Ideal points: Latent positions of persons (θ, theta)
- Difficulty: Item intercept parameters (α, alpha) representing baseline popularity
- Discrimination: Item slope parameters (β, beta) representing polarity/salience
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Union, Tuple
import numpy as np


class ResponseType(Enum):
    """Type of response data."""
    BINARY = "binary"  # Dichotomous responses (0/1)
    ORDINAL = "ordinal"  # Ordered categorical (0, 1, 2, ...)
    CONTINUOUS = "continuous"  # Continuous responses (e.g., test scores)
    COUNT = "count"  # Count data (Poisson-like)
    BOUNDED_CONTINUOUS = "bounded_continuous"  # Bounded continuous (e.g., 1-5 ratings)


class IdentificationConstraint(Enum):
    """Constraint for resolving model identifiability."""
    NONE = "none"  # No constraint (may have reflection/rotation invariance)
    FIX_FIRST_PERSON = "fix_first_person"  # Fix first person's ideal point to 0
    FIX_FIRST_TWO_PERSONS = "fix_first_two_persons"  # Fix first two persons (2D)
    FIX_VARIANCE = "fix_variance"  # Fix ideal point variance to 1
    REFERENCE_SCORES = "reference_scores"  # Align to external scores (e.g., DW-NOMINATE)


@dataclass
class IdealPointConfig:
    """
    Configuration for ideal point estimation models.

    This class defines the MODEL STRUCTURE - what the model is, not how to estimate it.
    Computational/runtime settings belong in the fit() method.

    Parameters
    ----------
    n_dims : int, default=1
        Number of latent dimensions for ideal points (1 or 2)
    response_type : ResponseType, default=BINARY
        Type of response variable (BINARY, ORDINAL, CONTINUOUS, COUNT, BOUNDED_CONTINUOUS)
    n_categories : int, optional
        Number of categories for ORDINAL responses (required if response_type=ORDINAL)
    response_bounds : tuple, optional
        (lower, upper) bounds for BOUNDED_CONTINUOUS (default: (1.0, 5.0))
    missing_data_model : str, default="ignore"
        How to handle missing data: "ignore", "impute", "model"
    identification : IdentificationConstraint, default=REFERENCE_SCORES
        Method for resolving identification
    reference_scores : np.ndarray, optional
        Reference scores for identification (if using REFERENCE_SCORES)

    Model Extensions
    ----------------
    hierarchical : bool, default=False
        Include hierarchical structure for person/item covariates
    temporal_dynamics : bool, default=False
        Model ideal points as evolving over time
    temporal_model : str, default="random_walk"
        Type of temporal model: "random_walk", "ar1"

    Priors
    ------
    prior_ideal_point_mean : float, default=0.0
        Prior mean for ideal points (μ_θ)
    prior_difficulty_mean : float, default=0.0
        Prior mean for difficulty (μ_α)
    prior_discrimination_mean : float, default=1.0
        Prior mean for discrimination (μ_β)
    prior_ideal_point_scale : float, default=1.0
        Prior scale for ideal points (σ_θ)
    prior_difficulty_scale : float, default=2.5
        Prior scale for difficulty (σ_α)
    prior_discrimination_scale : float, default=2.5
        Prior scale for discrimination (σ_β)
    prior_covariate_scale : float, default=1.0
        Prior scale for covariate effects (for hierarchical models)
    prior_threshold_scale : float, default=2.0
        Prior scale for ordinal thresholds (for ORDINAL responses)
    prior_residual_scale : float, default=1.0
        Prior scale for residuals (for CONTINUOUS responses)
    prior_temporal_variance : float, default=0.1
        Prior variance for temporal evolution (for temporal models)
    prior_precision_shape : float, default=2.0
        Shape parameter for precision prior (for BOUNDED_CONTINUOUS)
    prior_precision_rate : float, default=0.1
        Rate parameter for precision prior (for BOUNDED_CONTINUOUS)
    """

    # Model structure
    n_dims: int = 1
    response_type: Optional[ResponseType] = None  # Auto-detected if None
    n_categories: Optional[int] = None  # Required for ORDINAL (auto-detected if None)
    response_bounds: Optional[tuple] = None  # For BOUNDED_CONTINUOUS (auto-detected if None)
    missing_data_model: str = "ignore"
    identification: IdentificationConstraint = IdentificationConstraint.REFERENCE_SCORES

    # Model extensions
    hierarchical: bool = False
    temporal_dynamics: bool = False
    temporal_model: str = "random_walk"

    # Priors - means (location parameters)
    prior_ideal_point_mean: float = 0.0
    prior_difficulty_mean: float = 0.0
    prior_discrimination_mean: float = 1.0

    # Priors - scales (dispersion parameters)
    prior_ideal_point_scale: float = 1.0
    prior_difficulty_scale: float = 2.5
    prior_discrimination_scale: float = 2.5
    prior_covariate_scale: float = 1.0
    prior_threshold_scale: float = 2.0
    prior_residual_scale: float = 1.0
    prior_temporal_variance: float = 0.1
    prior_precision_shape: float = 2.0
    prior_precision_rate: float = 0.1

    # Reference for identification
    reference_scores: Optional[np.ndarray] = None

    # Internal (set during fitting)
    n_persons: Optional[int] = None
    n_items: Optional[int] = None
    n_timepoints: Optional[int] = None  # For temporal models

    def __post_init__(self):
        """Validate configuration."""
        # Validate dimensionality
        if self.n_dims not in [1, 2]:
            raise ValueError("n_dims must be 1 or 2")

        # Convert string enums to enum types
        if isinstance(self.response_type, str):
            self.response_type = ResponseType(self.response_type)

        if isinstance(self.identification, str):
            self.identification = IdentificationConstraint(self.identification)

        # Validate response type specific requirements (only if response_type is specified)
        if self.response_type == ResponseType.ORDINAL:
            if self.n_categories is None:
                raise ValueError("n_categories must be specified for ORDINAL responses")
            if self.n_categories < 3:
                raise ValueError("n_categories must be at least 3 for ORDINAL responses")

        if self.response_type == ResponseType.BOUNDED_CONTINUOUS:
            if self.response_bounds is None:
                raise ValueError("response_bounds must be specified for BOUNDED_CONTINUOUS responses")
            if not isinstance(self.response_bounds, tuple) or len(self.response_bounds) != 2:
                raise ValueError("response_bounds must be a tuple of (lower, upper) for BOUNDED_CONTINUOUS")
            if self.response_bounds[0] >= self.response_bounds[1]:
                raise ValueError("response_bounds lower must be < upper")

        # Validate temporal settings
        if self.temporal_dynamics and self.temporal_model not in ["random_walk", "ar1"]:
            raise ValueError(f"temporal_model must be 'random_walk' or 'ar1', got '{self.temporal_model}'")

    @property
    def response_lower_bound(self) -> float:
        """Lower bound for bounded continuous responses."""
        return self.response_bounds[0]

    @property
    def response_upper_bound(self) -> float:
        """Upper bound for bounded continuous responses."""
        return self.response_bounds[1]

    @classmethod
    def preset(cls, preset_name: str = "binary", **kwargs) -> 'IdealPointConfig':
        """
        Create IdealPointConfig with smart defaults for common use cases.

        Parameters
        ----------
        preset_name : str
            Preset configuration name. Options:
            - "binary": Binary response data (e.g., yes/no, correct/incorrect)
            - "test_scores": Binary test responses (ability estimation)
            - "survey": Ordinal survey responses (e.g., Likert scales)
            - "ratings": Bounded continuous ratings (e.g., 1-5 stars)
        **kwargs
            Override any default values

        Returns
        -------
        config : IdealPointConfig
            Configuration with preset defaults

        Examples
        --------
        >>> config = IdealPointConfig.preset("binary")  # 1D binary ideal points
        >>> config = IdealPointConfig.preset("survey", n_categories=5)  # 5-point Likert
        """
        presets = {
            "binary": {
                "n_dims": 1,
                "response_type": ResponseType.BINARY,
                "prior_difficulty_scale": 2.5,
                "prior_discrimination_scale": 2.5,
            },
            "test_scores": {
                "n_dims": 1,
                "response_type": ResponseType.BINARY,
                "prior_difficulty_scale": 2.0,
                "prior_discrimination_scale": 1.5,
            },
            "survey": {
                "n_dims": 1,
                "response_type": ResponseType.ORDINAL,
                "n_categories": 5,  # Override with actual number
                "prior_threshold_scale": 1.5,
            },
            "ratings": {
                "n_dims": 1,
                "response_type": ResponseType.BOUNDED_CONTINUOUS,
                "response_bounds": (1.0, 5.0),
            },
        }

        if preset_name not in presets:
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Available presets: {list(presets.keys())}"
            )

        # Start with preset defaults, then override with user kwargs
        config_kwargs = presets[preset_name].copy()
        config_kwargs.update(kwargs)

        return cls(**config_kwargs)


@dataclass
class IdealPointResults:
    """
    Results from fitted ideal point estimation model.

    Standardized container for model outputs with uncertainty quantification.

    Parameters
    ----------
    ideal_points : np.ndarray
        Posterior mean ideal points, shape (n_persons, n_dims)
    ideal_points_std : np.ndarray
        Posterior std of ideal points, shape (n_persons, n_dims)
    ideal_points_samples : Optional[np.ndarray]
        Posterior samples, shape (n_samples, n_persons, n_dims)
    difficulty : np.ndarray
        Difficulty parameters (α), shape (n_items,)
    difficulty_std : np.ndarray
        Std of difficulty, shape (n_items,)
    discrimination : np.ndarray
        Discrimination parameters (β), shape (n_items, n_dims)
    discrimination_std : np.ndarray
        Std of discrimination, shape (n_items, n_dims)
    convergence_info : Dict[str, Any]
        Convergence diagnostics
    computation_time : float
        Fitting time in seconds
    """

    # Person parameters (ideal points / positions) - REQUIRED
    ideal_points: np.ndarray  # (n_persons, n_dims)
    # Item parameters - REQUIRED
    difficulty: np.ndarray  # (n_items,) - α parameters
    discrimination: np.ndarray  # (n_items, n_dims) - β parameters

    # Optional uncertainty estimates
    ideal_points_std: Optional[np.ndarray] = None
    ideal_points_samples: Optional[np.ndarray] = None  # (n_samples, n_persons, n_dims)
    ideal_points_ci_lower: Optional[np.ndarray] = None
    ideal_points_ci_upper: Optional[np.ndarray] = None
    difficulty_std: Optional[np.ndarray] = None
    discrimination_std: Optional[np.ndarray] = None

    # Ordinal response thresholds (if applicable)
    thresholds: Optional[np.ndarray] = None  # (n_categories - 1,)

    # Hierarchical components (if applicable)
    person_covariate_effects: Optional[np.ndarray] = None
    item_covariate_effects: Optional[np.ndarray] = None

    # Temporal parameters (if applicable)
    temporal_ideal_points: Optional[np.ndarray] = None  # (n_timepoints, n_persons, n_dims)

    # Diagnostics
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    computation_time: float = 0.0
    log_likelihood: Optional[float] = None

    # Internal: Original names (if loaded from data)
    _person_names: Optional[list] = field(default=None, repr=False)
    _item_names: Optional[list] = field(default=None, repr=False)

    def get_person_parameters(self, person_id: int, with_uncertainty: bool = False) -> Dict[str, Any]:
        """
        Get parameters for a specific person.

        Parameters
        ----------
        person_id : int
            Index of the person
        with_uncertainty : bool, default=False
            Include uncertainty estimates (std, confidence intervals)

        Returns
        -------
        parameters : Dict[str, Any]
            Dictionary containing ideal point and optional uncertainty
        """
        result = {
            'ideal_point': self.ideal_points[person_id],
        }
        if with_uncertainty and self.ideal_points_std is not None:
            result['std'] = self.ideal_points_std[person_id]
            if self.ideal_points_ci_lower is not None:
                result['ci_lower'] = self.ideal_points_ci_lower[person_id]
                result['ci_upper'] = self.ideal_points_ci_upper[person_id]
        return result

    def get_item_parameters(self, item_id: int, with_uncertainty: bool = False) -> Dict[str, Any]:
        """
        Get parameters for a specific item.

        Parameters
        ----------
        item_id : int
            Index of the item
        with_uncertainty : bool, default=False
            Include uncertainty estimates (std, confidence intervals)

        Returns
        -------
        parameters : Dict[str, Any]
            Dictionary containing difficulty, discrimination, and optional uncertainty
        """
        result = {
            'difficulty': self.difficulty[item_id],
            'discrimination': self.discrimination[item_id],
        }
        if with_uncertainty:
            if self.difficulty_std is not None:
                result['difficulty_std'] = self.difficulty_std[item_id]
            if self.discrimination_std is not None:
                result['discrimination_std'] = self.discrimination_std[item_id]
        return result

    def to_dataframe(
        self,
        person_names: Optional[list] = None,
        item_names: Optional[list] = None,
        include_uncertainty: bool = True,
    ):
        """
        Convert results to pandas DataFrames with original names.

        Parameters
        ----------
        person_names : Optional[list]
            Original person identifiers.
            If None, uses names from data if available, otherwise generates defaults.
        item_names : Optional[list]
            Original item identifiers.
            If None, uses names from data if available, otherwise generates defaults.
        include_uncertainty : bool, default=True
            Include standard errors and confidence intervals if available

        Returns
        -------
        results : Dict[str, pd.DataFrame]
            Dictionary with keys:
            - 'persons': DataFrame with person parameters (ideal points)
            - 'items': DataFrame with item parameters (difficulty, discrimination)

        Examples
        --------
        >>> results_df = results.to_dataframe(
        ...     person_names=['Person_A', 'Person_B', ...],
        ...     item_names=['Item_1', 'Item_2', ...]
        ... )
        >>> print(results_df['persons'].head())
        >>> print(results_df['items'].head())
        """
        import pandas as pd

        n_persons = self.ideal_points.shape[0]
        n_items = self.difficulty.shape[0]
        n_dims = self.ideal_points.shape[1]

        # Use stored names if available, otherwise generate defaults
        if person_names is None:
            person_names = self._person_names if self._person_names is not None else [f"Person_{i}" for i in range(n_persons)]
        if item_names is None:
            item_names = self._item_names if self._item_names is not None else [f"Item_{i}" for i in range(n_items)]

        # Validate names
        if len(person_names) != n_persons:
            raise ValueError(
                f"person_names length ({len(person_names)}) doesn't match "
                f"number of persons ({n_persons})"
            )
        if len(item_names) != n_items:
            raise ValueError(
                f"item_names length ({len(item_names)}) doesn't match "
                f"number of items ({n_items})"
            )

        # Build person DataFrame
        person_data = {'person': person_names}

        # Add ideal points (one column per dimension)
        if n_dims == 1:
            person_data['ideal_point'] = self.ideal_points[:, 0]
            if include_uncertainty and self.ideal_points_std is not None:
                person_data['ideal_point_se'] = self.ideal_points_std[:, 0]
            if include_uncertainty and self.ideal_points_ci_lower is not None:
                person_data['ideal_point_ci_lower'] = self.ideal_points_ci_lower[:, 0]
                person_data['ideal_point_ci_upper'] = self.ideal_points_ci_upper[:, 0]
        else:
            for dim in range(n_dims):
                person_data[f'ideal_point_dim{dim+1}'] = self.ideal_points[:, dim]
                if include_uncertainty and self.ideal_points_std is not None:
                    person_data[f'ideal_point_dim{dim+1}_se'] = self.ideal_points_std[:, dim]
                if include_uncertainty and self.ideal_points_ci_lower is not None:
                    person_data[f'ideal_point_dim{dim+1}_ci_lower'] = self.ideal_points_ci_lower[:, dim]
                    person_data[f'ideal_point_dim{dim+1}_ci_upper'] = self.ideal_points_ci_upper[:, dim]

        persons_df = pd.DataFrame(person_data)

        # Build item DataFrame
        item_data = {'item': item_names}
        item_data['difficulty'] = self.difficulty

        # Add discrimination (one column per dimension)
        if n_dims == 1:
            item_data['discrimination'] = self.discrimination[:, 0]
            if include_uncertainty and self.discrimination_std is not None:
                item_data['discrimination_se'] = self.discrimination_std[:, 0]
        else:
            for dim in range(n_dims):
                item_data[f'discrimination_dim{dim+1}'] = self.discrimination[:, dim]
                if include_uncertainty and self.discrimination_std is not None:
                    item_data[f'discrimination_dim{dim+1}_se'] = self.discrimination_std[:, dim]

        if include_uncertainty and self.difficulty_std is not None:
            item_data['difficulty_se'] = self.difficulty_std

        items_df = pd.DataFrame(item_data)

        return {
            'persons': persons_df,
            'items': items_df,
        }


class BaseIdealPointModel(ABC):
    """
    Abstract base class for all ideal point estimation models.

    All concrete implementations (NumPyro, JAX-MAP, etc.) inherit from this
    and implement the abstract methods.

    Parameters
    ----------
    config : IdealPointConfig
        Model configuration
    """

    def __init__(self, config: Optional[IdealPointConfig] = None):
        self.config = config or IdealPointConfig()
        self.results: Optional[IdealPointResults] = None
        self._is_fitted = False

    @abstractmethod
    def fit(
        self,
        person_ids: np.ndarray,
        item_ids: np.ndarray,
        responses: np.ndarray,
        person_covariates: Optional[np.ndarray] = None,
        item_covariates: Optional[np.ndarray] = None,
        timepoints: Optional[np.ndarray] = None,
        **kwargs
    ) -> IdealPointResults:
        """
        Fit the ideal point estimation model.

        Parameters
        ----------
        person_ids : np.ndarray, shape (n_observations,)
            Integer IDs for persons/respondents (0 to n_persons-1)
        item_ids : np.ndarray, shape (n_observations,)
            Integer IDs for items/questions (0 to n_items-1)
        responses : np.ndarray, shape (n_observations,)
            Observed responses (type depends on config.response_type)
        person_covariates : Optional[np.ndarray], shape (n_persons, n_person_covariates)
            Person-level covariates
        item_covariates : Optional[np.ndarray], shape (n_items, n_item_covariates)
            Item-level covariates
        timepoints : Optional[np.ndarray], shape (n_observations,)
            Time index for temporal models
        **kwargs
            Implementation-specific parameters

        Returns
        -------
        results : IdealPointResults
            Fitted model results
        """
        pass

    @abstractmethod
    def predict(
        self,
        person_ids: np.ndarray,
        item_ids: np.ndarray,
        timepoints: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict responses for person-item pairs.

        Parameters
        ----------
        person_ids : np.ndarray
            Person IDs
        item_ids : np.ndarray
            Item IDs
        timepoints : Optional[np.ndarray]
            Time indices (for temporal models)

        Returns
        -------
        predictions : np.ndarray
            Predicted response probabilities or values
        """
        pass

    def save(self, filepath: str):
        """Save fitted model to disk."""
        from .persistence import ModelIO
        if not self._is_fitted:
            raise ValueError("Model must be fitted before saving")
        ModelIO.save(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'BaseIdealPointModel':
        """Load fitted model from disk."""
        from .persistence import ModelIO
        return ModelIO.load(filepath)

    def get_ideal_points(
        self,
        person_ids: Optional[np.ndarray] = None,
        with_uncertainty: bool = False,
        timepoint: Optional[int] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Get ideal point estimates.

        Parameters
        ----------
        person_ids : Optional[np.ndarray]
            Specific persons to retrieve (None = all)
        with_uncertainty : bool
            Include uncertainty estimates
        timepoint : Optional[int]
            Time index for temporal models

        Returns
        -------
        ideal_points : np.ndarray or Dict
            If with_uncertainty=False: array of shape (n_persons, n_dims)
            If with_uncertainty=True: dict with 'mean', 'std', 'ci_lower', 'ci_upper'
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")

        if timepoint is not None and self.results.temporal_ideal_points is not None:
            ideal_points = self.results.temporal_ideal_points[timepoint]
        else:
            ideal_points = self.results.ideal_points

        if person_ids is not None:
            ideal_points = ideal_points[person_ids]

        if not with_uncertainty:
            return ideal_points

        result = {'mean': ideal_points}
        if self.results.ideal_points_std is not None:
            result['std'] = self.results.ideal_points_std[person_ids] if person_ids is not None else self.results.ideal_points_std
        if self.results.ideal_points_ci_lower is not None:
            result['ci_lower'] = self.results.ideal_points_ci_lower[person_ids] if person_ids is not None else self.results.ideal_points_ci_lower
            result['ci_upper'] = self.results.ideal_points_ci_upper[person_ids] if person_ids is not None else self.results.ideal_points_ci_upper

        return result

    def get_item_parameters(
        self,
        item_ids: Optional[np.ndarray] = None,
        with_uncertainty: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Get item parameter estimates.

        Parameters
        ----------
        item_ids : Optional[np.ndarray]
            Specific items to retrieve (None = all)
        with_uncertainty : bool
            Include uncertainty estimates

        Returns
        -------
        parameters : Dict[str, np.ndarray]
            Dictionary with 'difficulty' and 'discrimination' (and std if requested)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")

        result = {
            'difficulty': self.results.difficulty,
            'discrimination': self.results.discrimination,
        }

        if item_ids is not None:
            result = {k: v[item_ids] for k, v in result.items()}

        if with_uncertainty:
            if self.results.difficulty_std is not None:
                result['difficulty_std'] = self.results.difficulty_std[item_ids] if item_ids is not None else self.results.difficulty_std
            if self.results.discrimination_std is not None:
                result['discrimination_std'] = self.results.discrimination_std[item_ids] if item_ids is not None else self.results.discrimination_std

        return result

    def summary(self) -> str:
        """Get model summary statistics."""
        if not self._is_fitted:
            return "Model not fitted"

        lines = [
            f"Ideal Point Estimation Model Summary",
            f"=" * 50,
            f"Model type: {self.__class__.__name__}",
            f"Latent dimensions: {self.config.n_dims}",
            f"Response type: {self.config.response_type.value}",
            f"",
            f"Sample size:",
            f"  Persons: {self.results.ideal_points.shape[0]}",
            f"  Items: {self.results.difficulty.shape[0]}",
            f"",
            f"Computation time: {self.results.computation_time:.2f}s",
        ]

        if self.results.log_likelihood is not None:
            lines.append(f"Log-likelihood: {self.results.log_likelihood:.2f}")

        if self.results.convergence_info:
            lines.append(f"\nConvergence info:")
            for key, val in self.results.convergence_info.items():
                lines.append(f"  {key}: {val}")

        return "\n".join(lines)
