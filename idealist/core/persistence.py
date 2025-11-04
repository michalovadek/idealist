"""
Model persistence - save and load fitted ideal point estimation models.

Handles serialization of model parameters, configuration, and results.
"""

from __future__ import annotations

import pickle
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .base import IdealPointConfig

if TYPE_CHECKING:
    from .base import BaseIdealPointModel


class ModelIO:
    """
    Save and load ideal point estimation models to/from disk.

    Supports multiple formats:
    - .pkl: Full Python pickle (fastest, Python-only)
    - .npz: NumPy arrays + JSON metadata (portable)
    - .json: JSON format (human-readable, limited precision)
    """

    @staticmethod
    def save(model: "BaseIdealPointModel", filepath: str, format: str = "auto"):
        """
        Save fitted model to disk.

        Parameters
        ----------
        model : BaseIdealPointModel
            Fitted model instance
        filepath : str
            Path to save file
        format : str
            'auto', 'pickle', 'npz', or 'json'
        """
        filepath = Path(filepath)

        # Auto-detect format
        if format == "auto":
            suffix = filepath.suffix.lower()
            if suffix == ".pkl":
                format = "pickle"
            elif suffix == ".npz":
                format = "npz"
            elif suffix == ".json":
                format = "json"
            else:
                format = "pickle"
                filepath = filepath.with_suffix(".pkl")

        if format == "pickle":
            ModelIO._save_pickle(model, filepath)
        elif format == "npz":
            ModelIO._save_npz(model, filepath)
        elif format == "json":
            ModelIO._save_json(model, filepath)
        else:
            raise ValueError(f"Unknown format: {format}")

    @staticmethod
    def load(filepath: str, format: str = "auto") -> "BaseIdealPointModel":
        """
        Load fitted model from disk.

        Parameters
        ----------
        filepath : str
            Path to saved file
        format : str
            'auto', 'pickle', 'npz', or 'json'

        Returns
        -------
        model : BaseIdealPointModel
            Loaded model instance
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Auto-detect format
        if format == "auto":
            suffix = filepath.suffix.lower()
            if suffix == ".pkl":
                format = "pickle"
            elif suffix == ".npz":
                format = "npz"
            elif suffix == ".json":
                format = "json"
            else:
                raise ValueError(f"Cannot auto-detect format from: {suffix}")

        if format == "pickle":
            return ModelIO._load_pickle(filepath)
        elif format == "npz":
            return ModelIO._load_npz(filepath)
        elif format == "json":
            return ModelIO._load_json(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")

    @staticmethod
    def _save_pickle(model: "BaseIdealPointModel", filepath: Path):
        """Save using pickle."""
        with open(filepath, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load_pickle(filepath: Path) -> "BaseIdealPointModel":
        """Load using pickle."""
        with open(filepath, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _save_npz(model: "BaseIdealPointModel", filepath: Path):
        """Save using NumPy .npz format + JSON metadata."""
        # Extract numpy arrays
        arrays = {
            "ideal_points": model.results.ideal_points,
            "difficulty": model.results.difficulty,
            "discrimination": model.results.discrimination,
        }

        if model.results.ideal_points_std is not None:
            arrays["ideal_points_std"] = model.results.ideal_points_std
        if model.results.difficulty_std is not None:
            arrays["difficulty_std"] = model.results.difficulty_std
        if model.results.discrimination_std is not None:
            arrays["discrimination_std"] = model.results.discrimination_std
        if model.results.ideal_points_samples is not None:
            arrays["ideal_points_samples"] = model.results.ideal_points_samples
        if model.results.temporal_ideal_points is not None:
            arrays["temporal_ideal_points"] = model.results.temporal_ideal_points

        # Save arrays
        np.savez_compressed(filepath, **arrays)

        # Save metadata as JSON
        metadata = {
            "model_class": model.__class__.__name__,
            "config": ModelIO._config_to_dict(model.config),
            "convergence_info": model.results.convergence_info,
            "computation_time": model.results.computation_time,
            "log_likelihood": model.results.log_likelihood,
        }

        metadata_path = filepath.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def _load_npz(filepath: Path) -> "BaseIdealPointModel":
        """Load from NumPy .npz format + JSON metadata."""
        # Load arrays
        data = np.load(filepath)

        # Load metadata
        metadata_path = filepath.with_suffix(".json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Reconstruct config
        from .base import IdealPointResults

        config = ModelIO._dict_to_config(metadata["config"])

        # Reconstruct results
        results = IdealPointResults(
            ideal_points=data["ideal_points"],
            difficulty=data["difficulty"],
            discrimination=data["discrimination"],
            ideal_points_std=data.get("ideal_points_std"),
            difficulty_std=data.get("difficulty_std"),
            discrimination_std=data.get("discrimination_std"),
            ideal_points_samples=data.get("ideal_points_samples"),
            temporal_ideal_points=data.get("temporal_ideal_points"),
            convergence_info=metadata["convergence_info"],
            computation_time=metadata["computation_time"],
            log_likelihood=metadata["log_likelihood"],
        )

        # Reconstruct model
        model_class_name = metadata["model_class"]
        if model_class_name == "IdealPointEstimator":
            from ..models.ideal_point import IdealPointEstimator

            model = IdealPointEstimator(config)
        else:
            raise ValueError(f"Unknown model class: {model_class_name}")

        model.results = results
        model._is_fitted = True

        return model

    @staticmethod
    def _save_json(model: "BaseIdealPointModel", filepath: Path):
        """Save as JSON (human-readable but larger and less precise)."""
        data = {
            "model_class": model.__class__.__name__,
            "config": ModelIO._config_to_dict(model.config),
            "results": {
                "ideal_points": model.results.ideal_points.tolist(),
                "difficulty": model.results.difficulty.tolist(),
                "discrimination": model.results.discrimination.tolist(),
                "computation_time": model.results.computation_time,
                "log_likelihood": model.results.log_likelihood,
                "convergence_info": model.results.convergence_info,
            },
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _load_json(filepath: Path) -> "BaseIdealPointModel":
        """Load from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)

        from .base import IdealPointResults

        config = ModelIO._dict_to_config(data["config"])

        results = IdealPointResults(
            ideal_points=np.array(data["results"]["ideal_points"]),
            difficulty=np.array(data["results"]["difficulty"]),
            discrimination=np.array(data["results"]["discrimination"]),
            computation_time=data["results"]["computation_time"],
            log_likelihood=data["results"]["log_likelihood"],
            convergence_info=data["results"]["convergence_info"],
        )

        # Reconstruct model
        model_class_name = data["model_class"]
        if model_class_name == "IdealPointEstimator":
            from ..models.ideal_point import IdealPointEstimator

            model = IdealPointEstimator(config)
        else:
            raise ValueError(f"Unknown model class: {model_class_name}")

        model.results = results
        model._is_fitted = True

        return model

    @staticmethod
    def _config_to_dict(config: Any) -> dict:
        """Convert IdealPointConfig to dictionary."""
        from .base import ResponseType, IdentificationConstraint

        d = config.__dict__.copy()
        # Convert enums to strings
        if isinstance(d["response_type"], ResponseType):
            d["response_type"] = d["response_type"].value
        if isinstance(d["identification"], IdentificationConstraint):
            d["identification"] = d["identification"].value
        # Convert numpy arrays to lists
        if d.get("reference_scores") is not None:
            d["reference_scores"] = d["reference_scores"].tolist()
        return d

    @staticmethod
    def _dict_to_config(d: dict) -> "IdealPointConfig":
        """Convert dictionary to IdealPointConfig."""
        from .base import IdealPointConfig, ResponseType, IdentificationConstraint

        # Convert strings to enums
        if "response_type" in d:
            d["response_type"] = ResponseType(d["response_type"])
        if "identification" in d:
            d["identification"] = IdentificationConstraint(d["identification"])
        # Convert lists to numpy arrays
        if d.get("reference_scores") is not None:
            d["reference_scores"] = np.array(d["reference_scores"])
        return IdealPointConfig(**d)
