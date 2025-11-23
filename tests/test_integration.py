"""
Integration tests for complete workflows.

Tests end-to-end workflows that a user would typically perform,
including data loading, model fitting, results extraction, and persistence.
"""

import numpy as np
import pandas as pd
import pytest

from idealist import IdealPointConfig, IdealPointEstimator, ResponseType
from idealist.core.persistence import ModelIO
from idealist.data import detect_response_type, load_data


class TestEndToEndWorkflows:
    """Integration tests for complete user workflows."""

    @pytest.mark.integration
    def test_complete_binary_workflow(self, tmp_path):
        """Test complete workflow: data creation -> fit -> save -> load -> predict."""
        # Create sample data
        df = pd.DataFrame(
            {
                "senator": ["Smith", "Smith", "Jones", "Jones", "Brown", "Brown"] * 10,
                "bill": ["HR101", "HR202", "HR101", "HR202", "HR101", "HR202"] * 10,
                "vote": [1, 0, 1, 1, 0, 0] * 10,
            }
        )

        # Load data
        data = load_data(df, person_col="senator", item_col="bill", response_col="vote")

        assert data.n_persons == 3
        assert data.n_items == 2

        # Fit model
        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(data, inference="vi", vi_steps=500, device="cpu", progress_bar=False)

        assert results is not None
        assert results.ideal_points.shape == (3, 1)

        # Convert to DataFrame
        df_results = results.to_dataframe()
        assert "persons" in df_results
        assert "Smith" in df_results["persons"]["person"].values

        # Save model (use npz format due to JAX serialization issues with pickle)
        model_path = tmp_path / "senator_model.npz"
        ModelIO.save(model, str(model_path))

        # Load model
        loaded_model = ModelIO.load(str(model_path))
        assert loaded_model.results is not None

        print("\n  Complete binary workflow successful")

    @pytest.mark.integration
    def test_ordinal_data_workflow(self):
        """Test workflow with ordinal (Likert scale) data."""
        # Create Likert scale survey data (0-indexed: 0-4)
        df = pd.DataFrame(
            {
                "respondent": ["R1", "R1", "R2", "R2", "R3", "R3"] * 5,
                "question": ["Q1", "Q2", "Q1", "Q2", "Q1", "Q2"] * 5,
                "rating": [4, 3, 2, 2, 1, 0] * 5,  # 0-4 scale (5 categories)
            }
        )

        # Auto-detect response type
        response_type, n_categories, bounds = detect_response_type(df["rating"].values)

        assert response_type == ResponseType.ORDINAL
        assert n_categories == 5

        # Load and fit
        data = load_data(df, person_col="respondent", item_col="question", response_col="rating")

        config = IdealPointConfig(
            n_dims=1, response_type=ResponseType.ORDINAL, n_categories=n_categories
        )

        model = IdealPointEstimator(config)
        results = model.fit(data, inference="vi", vi_steps=500, device="cpu", progress_bar=False)

        assert results is not None
        assert results.ideal_points.shape[0] == 3

        print("\n  Ordinal workflow successful")

    @pytest.mark.integration
    def test_continuous_data_workflow(self):
        """Test workflow with continuous response data."""
        # Create test score data
        np.random.seed(42)
        n_obs = 100

        df = pd.DataFrame(
            {
                "student": [f"S{i}" for i in np.random.randint(0, 20, n_obs)],
                "test": [f"T{i}" for i in np.random.randint(0, 10, n_obs)],
                "score": np.random.uniform(0, 10, n_obs),
            }
        )

        # Detect type
        response_type, n_categories, bounds = detect_response_type(df["score"].values)
        assert response_type in [ResponseType.CONTINUOUS, ResponseType.BOUNDED_CONTINUOUS]

        # Load and fit
        data = load_data(df, person_col="student", item_col="test", response_col="score")

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.CONTINUOUS)
        model = IdealPointEstimator(config)

        results = model.fit(data, inference="vi", vi_steps=500, device="cpu", progress_bar=False)

        assert results is not None

        print("\n  Continuous workflow successful")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_multidimensional_workflow(self):
        """Test workflow with 2D ideal points."""
        # Create data with two underlying dimensions
        np.random.seed(123)

        # Generate legislators with two ideological dimensions
        legislators = [f"L{i}" for i in range(30)]
        bills = [f"B{i}" for i in range(20)]

        # Create votes
        data_rows = []
        for leg in legislators:
            for bill in bills:
                if np.random.rand() > 0.3:  # 70% observed
                    vote = np.random.choice([0, 1])
                    data_rows.append({"legislator": leg, "bill": bill, "vote": vote})

        df = pd.DataFrame(data_rows)

        # Load data
        data = load_data(df, person_col="legislator", item_col="bill", response_col="vote")

        # Fit 2D model
        config = IdealPointConfig(n_dims=2, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(data, inference="vi", vi_steps=1000, device="cpu", progress_bar=False)

        assert results is not None
        assert results.ideal_points.shape[1] == 2

        # Extract results
        df_results = results.to_dataframe()
        assert len(df_results["persons"]) == 30

        print("\n  2D workflow successful")

    @pytest.mark.integration
    def test_data_summary_workflow(self):
        """Test workflow focusing on data summary and exploration."""
        df = pd.DataFrame(
            {
                "person": ["A", "A", "B", "B", "C", "C", "D"] * 10,
                "item": ["X", "Y", "X", "Y", "X", "Y", "X"] * 10,
                "response": [1, 0, 1, 1, 0, 1, 1] * 10,
            }
        )

        # Load data
        data = load_data(df, person_col="person", item_col="item", response_col="response")

        # Check summary
        summary = data.summary()
        assert "Persons:" in summary
        assert "Items:" in summary
        assert "4" in summary  # 4 persons
        assert "2" in summary  # 2 items

        # Check metadata
        assert data.metadata["n_persons"] == 4
        assert data.metadata["n_items"] == 2
        assert data.metadata["n_observations"] == 70

        print("\n  Data summary workflow successful")

    @pytest.mark.integration
    def test_compare_inference_methods(self, small_binary_data):
        """Test comparing different inference methods on same data."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)

        # Fit with VI
        model_vi = IdealPointEstimator(config)
        results_vi = model_vi.fit(
            person_ids=data["person_ids"],
            item_ids=data["item_ids"],
            responses=data["responses"],
            inference="vi",
            vi_steps=500,
            device="cpu",
            progress_bar=False,
        )

        # Fit with MAP
        model_map = IdealPointEstimator(config)
        results_map = model_map.fit(
            person_ids=data["person_ids"],
            item_ids=data["item_ids"],
            responses=data["responses"],
            inference="map",
            map_steps=500,
            device="cpu",
            progress_bar=False,
        )

        # Both should produce results
        assert results_vi is not None
        assert results_map is not None

        # Results should be similar (not identical due to different methods)
        correlation = np.corrcoef(
            results_vi.ideal_points.flatten(), results_map.ideal_points.flatten()
        )[0, 1]

        print(f"\n  VI vs MAP correlation: {correlation:.3f}")

        # Should be reasonably correlated
        assert abs(correlation) > 0.5

    @pytest.mark.integration
    def test_model_persistence_workflow(self, small_binary_data, tmp_path):
        """Test complete persistence workflow."""
        data = small_binary_data

        # Fit original model
        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
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

        original_ideal_points = results.ideal_points.copy()

        # Save in multiple formats (skip pickle due to JAX serialization)
        npz_path = tmp_path / "model.npz"
        json_path = tmp_path / "model.json"

        ModelIO.save(model, str(npz_path), format="npz")
        ModelIO.save(model, str(json_path), format="json")

        # Load and verify
        loaded_npz = ModelIO.load(str(npz_path))
        loaded_json = ModelIO.load(str(json_path))

        np.testing.assert_array_almost_equal(loaded_npz.results.ideal_points, original_ideal_points)
        np.testing.assert_array_almost_equal(
            loaded_json.results.ideal_points,
            original_ideal_points,
            decimal=5,  # JSON has less precision
        )

        print("\n  Persistence workflow successful")

    @pytest.mark.integration
    def test_error_recovery_workflow(self):
        """Test workflow with error recovery and validation."""
        # Create data with potential issues
        df = pd.DataFrame(
            {
                "person": ["A", "B", "C"],
                "item": ["X", "Y", "Z"],
                "response": [1, 0, 1],
            }
        )

        # Load data
        data = load_data(df, person_col="person", item_col="item", response_col="response")

        # This is minimal data - might have issues
        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        # Try to fit - might succeed or fail gracefully
        try:
            results = model.fit(
                data,
                inference="vi",
                vi_steps=300,
                device="cpu",
                progress_bar=False,
            )
            # If it succeeds, check results are valid
            assert results is not None
            assert not np.any(np.isnan(results.ideal_points))
        except (ValueError, RuntimeError) as e:
            # If it fails, should be with a clear error message
            assert len(str(e)) > 0

        print("\n  Error recovery workflow tested")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
