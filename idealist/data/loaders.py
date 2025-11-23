"""
Data loading utilities for ideal point estimation models.

Loads person-item response data in standard format (one row per observation):
- CSV files with person, item, response columns
- Pandas DataFrames with one observation per row
- Optional person and item covariates

Ideal point models estimate latent positions of persons and items on a low-dimensional
space based on binary, ordinal, or continuous response data. Applications include:
- Political science: legislative voting, judicial decisions
- Psychometrics: test responses, ability estimation
- Marketing: product preferences, ratings
- Social science: survey responses, attitude measurement
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..core.base import ResponseType


@dataclass
class IdealPointData:
    """
    Container for ideal point estimation data with both numeric arrays and original labels.

    This data structure stores person-item response data for ideal point estimation across
    various applications including political science, psychometrics, marketing, and social science.

    Attributes
    ----------
    person_ids : np.ndarray
        Integer IDs for persons (0 to n_persons-1)
    item_ids : np.ndarray
        Integer IDs for items (0 to n_items-1)
    responses : np.ndarray
        Response values (binary, ordinal, continuous, or count data)
    person_names : List[str]
        Original person identifiers/names
    item_names : List[str]
        Original item identifiers/names
    person_covariates : Optional[pd.DataFrame]
        Person-level covariates (if provided)
    item_covariates : Optional[pd.DataFrame]
        Item-level covariates (if provided)
    metadata : Dict
        Additional metadata about the dataset including summary statistics
    """

    person_ids: np.ndarray
    item_ids: np.ndarray
    responses: np.ndarray
    person_names: List[str]
    item_names: List[str]
    person_covariates: Optional[pd.DataFrame] = None
    item_covariates: Optional[pd.DataFrame] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        """Validate data after initialization."""
        # Check array shapes
        if not (len(self.person_ids) == len(self.item_ids) == len(self.responses)):
            raise ValueError(
                f"Arrays must have same length: person_ids={len(self.person_ids)}, "
                f"item_ids={len(self.item_ids)}, responses={len(self.responses)}"
            )

        # Check ID ranges
        if self.person_ids.max() >= len(self.person_names):
            raise ValueError(
                f"person_ids max ({self.person_ids.max()}) exceeds number of person_names "
                f"({len(self.person_names)})"
            )

        if self.item_ids.max() >= len(self.item_names):
            raise ValueError(
                f"item_ids max ({self.item_ids.max()}) exceeds number of item_names "
                f"({len(self.item_names)})"
            )

        # Initialize metadata if not provided
        if self.metadata is None:
            self.metadata = self._compute_metadata()

    def _compute_metadata(self) -> Dict:
        """Compute dataset statistics."""
        return {
            "n_persons": len(self.person_names),
            "n_items": len(self.item_names),
            "n_observations": len(self.responses),
            "response_rate": float(np.mean(self.responses)),
            "obs_per_person_mean": len(self.responses) / len(self.person_names),
            "obs_per_item_mean": len(self.responses) / len(self.item_names),
            "response_min": float(np.min(self.responses)),
            "response_max": float(np.max(self.responses)),
        }

    @property
    def n_persons(self) -> int:
        """Number of unique persons."""
        return len(self.person_names)

    @property
    def n_items(self) -> int:
        """Number of unique items."""
        return len(self.item_names)

    @property
    def n_observations(self) -> int:
        """Total number of observations."""
        return len(self.responses)

    def summary(self) -> str:
        """Return a summary of the dataset."""
        lines = [
            "Ideal Point Data Summary",
            "=" * 60,
            f"Persons:      {self.n_persons:,}",
            f"Items:        {self.n_items:,}",
            f"Observations: {self.n_observations:,}",
            f"Obs/person:   {self.metadata['obs_per_person_mean']:.1f} (mean)",
            f"Obs/item:     {self.metadata['obs_per_item_mean']:.1f} (mean)",
            "",
            (
                f"Response range: [{self.metadata['response_min']:.0f}, "
                f"{self.metadata['response_max']:.0f}]"
            ),
            f"Response rate:  {self.metadata['response_rate']:.1%}",
        ]

        if self.person_covariates is not None:
            lines.append(f"\nPerson covariates: {self.person_covariates.shape[1]} variables")

        if self.item_covariates is not None:
            lines.append(f"Item covariates: {self.item_covariates.shape[1]} variables")

        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"IdealPointData(n_persons={self.n_persons}, n_items={self.n_items}, "
            f"n_observations={self.n_observations})"
        )


def validate_response_data(
    responses: np.ndarray,
    response_type: ResponseType,
    n_categories: Optional[int] = None,
    response_bounds: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Validate that response data is compatible with the specified response type.

    Parameters
    ----------
    responses : np.ndarray
        Array of response values
    response_type : ResponseType
        The specified response type
    n_categories : Optional[int]
        Number of categories (required for ORDINAL)
    response_bounds : Optional[Tuple[float, float]]
        Response bounds (required for BOUNDED_CONTINUOUS)

    Raises
    ------
    ValueError
        If the response data is incompatible with the specified response type

    Notes
    -----
    Validation checks:
    - BINARY: Must have exactly 2 unique values
    - ORDINAL: Must have integer values in range [0, n_categories-1]
    - COUNT: Must have non-negative integer values
    - BOUNDED_CONTINUOUS: Must have values within specified bounds
    - CONTINUOUS: No specific restrictions (accepts any numeric values)
    """
    unique_values = np.unique(responses)
    n_unique = len(unique_values)
    response_min = float(np.min(responses))
    response_max = float(np.max(responses))

    # Check if all values are integers
    is_integer = np.allclose(responses, np.round(responses))

    if response_type == ResponseType.BINARY:
        if n_unique != 2:
            raise ValueError(
                f"response_type='binary' requires exactly 2 unique categories, "
                f"but found {n_unique} unique values: {unique_values.tolist()}"
            )

    elif response_type == ResponseType.ORDINAL:
        if n_categories is None:
            raise ValueError("n_categories must be specified for response_type='ordinal'")

        if not is_integer:
            raise ValueError(
                f"response_type='ordinal' requires integer values, "
                f"but found non-integer values (min={response_min:.4f}, max={response_max:.4f})"
            )

        # Check that all values are in the valid range [0, n_categories-1]
        if response_min < 0 or response_max >= n_categories:
            raise ValueError(
                f"response_type='ordinal' with n_categories={n_categories} requires "
                f"values in range [0, {n_categories-1}], but found values in range "
                f"[{int(response_min)}, {int(response_max)}]"
            )

        # Check that we actually have the expected number of categories
        if n_unique != n_categories:
            raise ValueError(
                f"response_type='ordinal' with n_categories={n_categories} expects "
                f"{n_categories} unique values, but found {n_unique} unique values: "
                f"{unique_values.tolist()}"
            )

    elif response_type == ResponseType.COUNT:
        if not is_integer:
            raise ValueError(
                f"response_type='count' requires integer values, "
                f"but found non-integer values (min={response_min:.4f}, max={response_max:.4f})"
            )

        if response_min < 0:
            raise ValueError(
                f"response_type='count' requires non-negative integer values, "
                f"but found negative values (min={int(response_min)})"
            )

    elif response_type == ResponseType.BOUNDED_CONTINUOUS:
        if response_bounds is None:
            raise ValueError(
                "response_bounds must be specified for response_type='bounded_continuous'"
            )

        lower_bound, upper_bound = response_bounds

        if response_min < lower_bound or response_max > upper_bound:
            raise ValueError(
                f"response_type='bounded_continuous' with bounds={response_bounds} requires "
                f"all values in range [{lower_bound}, {upper_bound}], but found values "
                f"in range [{response_min:.4f}, {response_max:.4f}]"
            )

    elif response_type == ResponseType.CONTINUOUS:
        # Continuous accepts any numeric values - no validation needed
        pass


def detect_response_type(
    responses: np.ndarray,
) -> Tuple[ResponseType, Optional[int], Optional[Tuple[float, float]]]:
    """
    Automatically detect the appropriate response type from the data.

    Parameters
    ----------
    responses : np.ndarray
        Array of response values

    Returns
    -------
    response_type : ResponseType
        Detected response type
    n_categories : Optional[int]
        Number of categories (for ordinal responses)
    response_bounds : Optional[Tuple[float, float]]
        Response bounds (for bounded continuous responses)

    Notes
    -----
    Detection logic:
    - BINARY: Exactly 2 unique values (typically 0/1 or 1/2)
    - ORDINAL: 3-10 integer values (e.g., Likert scales 1-5)
    - COUNT: Non-negative integers with range > 10
    - BOUNDED_CONTINUOUS: Continuous values within a finite range
    - CONTINUOUS: Continuous values with unbounded or wide range
    """
    unique_values = np.unique(responses)
    n_unique = len(unique_values)
    response_min = float(np.min(responses))
    response_max = float(np.max(responses))
    response_range = response_max - response_min

    # Check if all values are integers
    is_integer = np.allclose(responses, np.round(responses))

    # Binary: exactly 2 unique values
    if n_unique == 2:
        return ResponseType.BINARY, None, None

    # Ordinal: 3-10 integer categories
    if is_integer and 3 <= n_unique <= 10:
        # Always return n_unique as n_categories
        # (the model expects 0-indexed categories from 0 to n_categories-1)
        n_categories = n_unique
        return ResponseType.ORDINAL, n_categories, None

    # Count: non-negative integers with many unique values
    if is_integer and response_min >= 0 and n_unique > 10:
        return ResponseType.COUNT, None, None

    # Continuous data (not integer)
    if not is_integer:
        # Bounded continuous: finite range that's reasonably small
        # (e.g., ratings on a scale like 1.0 to 5.0)
        if response_range <= 20:  # Heuristic threshold
            bounds = (response_min, response_max)
            return ResponseType.BOUNDED_CONTINUOUS, None, bounds
        else:
            # Unbounded continuous
            return ResponseType.CONTINUOUS, None, None

    # Default to continuous for any other case
    return ResponseType.CONTINUOUS, None, None


def load_data(
    data: Union[str, Path, pd.DataFrame],
    person_col: str,
    item_col: str,
    response_col: str,
    person_covariate_cols: Optional[List[str]] = None,
    item_covariate_cols: Optional[List[str]] = None,
) -> IdealPointData:
    """
    Load ideal point estimation data.

    This function loads person-item response data in standard format where each row
    represents a single observation (one person responding to one item).

    Data format (one row per observation):
        person | item | response | person_group | item_category
        ------ | ---- | -------- | ------------ | -------------
        P001   | I001 | 1        | A            | Type1
        P001   | I002 | 0        | A            | Type2
        P002   | I001 | 1        | B            | Type1
        ...

    Parameters
    ----------
    data : str, Path, or pd.DataFrame
        CSV file path or DataFrame with observations as rows (one row per person-item pair)
    person_col : str
        Column name containing person identifiers
    item_col : str
        Column name containing item identifiers
    response_col : str
        Column name containing responses (can be binary, ordinal, continuous, or count)
    person_covariate_cols : List[str], optional
        Column names for person-level covariates (e.g., demographics, groups)
    item_covariate_cols : List[str], optional
        Column names for item-level covariates (e.g., categories, difficulty)

    Returns
    -------
    IdealPointData
        Data object ready for ideal point model fitting

    Examples
    --------
    >>> from idealist.data import load_data
    >>> # Basic usage
    >>> data = load_data('responses.csv',
    ...                  person_col='person',
    ...                  item_col='item',
    ...                  response_col='response')
    >>> print(data.summary())

    >>> # With covariates
    >>> data = load_data('responses.csv',
    ...                  person_col='person',
    ...                  item_col='item',
    ...                  response_col='response',
    ...                  person_covariate_cols=['group', 'age'],
    ...                  item_covariate_cols=['category'])
    """
    # Load data if it's a file
    if isinstance(data, (str, Path)):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    # Validate required columns
    required_cols = [person_col, item_col, response_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Extract unique persons and items (preserve order)
    person_names = df[person_col].unique().tolist()
    item_names = df[item_col].unique().tolist()

    # Create ID mappings
    person_to_id = {name: idx for idx, name in enumerate(person_names)}
    item_to_id = {name: idx for idx, name in enumerate(item_names)}

    # Convert to integer IDs
    person_ids = df[person_col].map(person_to_id).values.astype(np.int32)
    item_ids = df[item_col].map(item_to_id).values.astype(np.int32)
    responses = df[response_col].values.astype(np.float32)

    # Handle covariates
    person_covariates = None
    item_covariates = None

    if person_covariate_cols:
        # Extract person-level covariates (one row per person)
        person_cov_df = df[[person_col] + person_covariate_cols].drop_duplicates(
            subset=[person_col]
        )
        person_cov_df = person_cov_df.set_index(person_col).loc[person_names]
        person_covariates = person_cov_df[person_covariate_cols]

    if item_covariate_cols:
        # Extract item-level covariates (one row per item)
        item_cov_df = df[[item_col] + item_covariate_cols].drop_duplicates(subset=[item_col])
        item_cov_df = item_cov_df.set_index(item_col).loc[item_names]
        item_covariates = item_cov_df[item_covariate_cols]

    return IdealPointData(
        person_ids=person_ids,
        item_ids=item_ids,
        responses=responses,
        person_names=person_names,
        item_names=item_names,
        person_covariates=person_covariates,
        item_covariates=item_covariates,
    )
