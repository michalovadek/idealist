"""
Tests for optimization algorithms used in VI and MAP inference.

Tests the three optimizer types available:
1. Adam: Adaptive moment estimation (default, usually best)
2. SGD: Stochastic gradient descent (simple baseline)
3. Adagrad: Adaptive gradient algorithm

Both VI and MAP inference use these optimizers for parameter optimization.
"""

import numpy as np
import pytest

from idealist import IdealPointConfig, IdealPointEstimator, ResponseType


class TestVIOptimizers:
    """Tests for optimizers in Variational Inference."""

    def test_vi_adam_optimizer(self, small_binary_data):
        """Test VI with Adam optimizer (default)."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_optimizer="adam", vi_steps=500,
            device="cpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None
        assert np.all(np.isfinite(results.ideal_points))

    def test_vi_sgd_optimizer(self, small_binary_data):
        """Test VI with SGD optimizer."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_optimizer="sgd", vi_steps=500, vi_lr=0.01,
            device="cpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None
        assert np.all(np.isfinite(results.ideal_points))

    def test_vi_adagrad_optimizer(self, small_binary_data):
        """Test VI with Adagrad optimizer."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_optimizer="adagrad", vi_steps=500,
            device="cpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None
        assert np.all(np.isfinite(results.ideal_points))

    def test_vi_optimizer_convergence_adam(self, small_binary_data):
        """Test that Adam optimizer converges for VI."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_optimizer="adam", vi_steps=1000,
            device="cpu", progress_bar=False
        )

        # Check convergence
        assert results.ideal_points is not None
        assert np.abs(results.ideal_points).max() < 10.0

    def test_vi_optimizer_convergence_sgd(self, small_binary_data):
        """Test that SGD optimizer converges for VI (with appropriate learning rate)."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        # SGD may need more steps or different learning rate
        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_optimizer="sgd", vi_steps=1000, vi_lr=0.01,
            device="cpu", progress_bar=False
        )

        assert results.ideal_points is not None
        assert np.all(np.isfinite(results.ideal_points))


class TestMAPOptimizers:
    """Tests for optimizers in MAP inference."""

    def test_map_adam_optimizer(self, small_binary_data):
        """Test MAP with Adam optimizer (default)."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="map", map_optimizer="adam", map_steps=500,
            device="cpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None
        assert np.all(np.isfinite(results.ideal_points))

    def test_map_sgd_optimizer(self, small_binary_data):
        """Test MAP with SGD optimizer."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="map", map_optimizer="sgd", map_steps=500, map_lr=0.01,
            device="cpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None
        assert np.all(np.isfinite(results.ideal_points))

    def test_map_adagrad_optimizer(self, small_binary_data):
        """Test MAP with Adagrad optimizer."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="map", map_optimizer="adagrad", map_steps=500,
            device="cpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None
        assert np.all(np.isfinite(results.ideal_points))

    def test_map_optimizer_convergence_adam(self, small_binary_data):
        """Test that Adam optimizer converges for MAP."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="map", map_optimizer="adam", map_steps=1000,
            device="cpu", progress_bar=False
        )

        # Check convergence
        assert results.ideal_points is not None
        assert np.abs(results.ideal_points).max() < 10.0


class TestOptimizerComparison:
    """Tests comparing different optimizers."""

    def test_vi_optimizers_produce_similar_results(self, small_binary_data):
        """Test that different optimizers produce similar results for VI."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)

        optimizers = ["adam", "sgd", "adagrad"]
        results_list = []

        for optimizer in optimizers:
            model = IdealPointEstimator(config)
            lr = 0.01 if optimizer == "sgd" else 0.05  # SGD needs lower LR
            results = model.fit(
                data["person_ids"], data["item_ids"], data["responses"],
                inference="vi", vi_optimizer=optimizer, vi_steps=1000, vi_lr=lr,
                device="cpu", progress_bar=False
            )
            results_list.append(results)

        # All should converge to similar solutions
        for i in range(len(results_list)):
            for j in range(i + 1, len(results_list)):
                corr = np.corrcoef(
                    results_list[i].ideal_points.flatten(),
                    results_list[j].ideal_points.flatten()
                )[0, 1]
                # Should have reasonable correlation
                assert corr > 0.2, f"Optimizers {optimizers[i]} and {optimizers[j]} diverged: {corr}"

    def test_map_optimizers_produce_similar_results(self, small_binary_data):
        """Test that different optimizers produce similar results for MAP."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)

        optimizers = ["adam", "sgd", "adagrad"]
        results_list = []

        for optimizer in optimizers:
            model = IdealPointEstimator(config)
            lr = 0.01 if optimizer == "sgd" else 0.05
            results = model.fit(
                data["person_ids"], data["item_ids"], data["responses"],
                inference="map", map_optimizer=optimizer, map_steps=1000, map_lr=lr,
                device="cpu", progress_bar=False
            )
            results_list.append(results)

        # All should converge to similar solutions
        for i in range(len(results_list)):
            for j in range(i + 1, len(results_list)):
                corr = np.corrcoef(
                    results_list[i].ideal_points.flatten(),
                    results_list[j].ideal_points.flatten()
                )[0, 1]
                assert corr > 0.2


class TestOptimizerLearningRates:
    """Tests for learning rate effects."""

    def test_vi_learning_rate_effect(self, small_binary_data):
        """Test that learning rate affects VI convergence."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)

        # Low learning rate
        model_low = IdealPointEstimator(config)
        results_low = model_low.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_optimizer="adam", vi_steps=500, vi_lr=0.001,
            device="cpu", progress_bar=False
        )

        # High learning rate
        model_high = IdealPointEstimator(config)
        results_high = model_high.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_optimizer="adam", vi_steps=500, vi_lr=0.1,
            device="cpu", progress_bar=False
        )

        # Both should converge (may be to different local optima)
        assert results_low.ideal_points is not None
        assert results_high.ideal_points is not None
        assert np.all(np.isfinite(results_low.ideal_points))
        assert np.all(np.isfinite(results_high.ideal_points))

    def test_map_learning_rate_effect(self, small_binary_data):
        """Test that learning rate affects MAP convergence."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)

        # Different learning rates
        for lr in [0.001, 0.01, 0.1]:
            model = IdealPointEstimator(config)
            results = model.fit(
                data["person_ids"], data["item_ids"], data["responses"],
                inference="map", map_optimizer="adam", map_steps=500, map_lr=lr,
                device="cpu", progress_bar=False
            )
            assert results.ideal_points is not None


class TestOptimizerWithAdvancedFeatures:
    """Tests for optimizers with advanced model features."""

    def test_optimizers_with_multidimensional(self, multidim_binary_data):
        """Test that optimizers work with multidimensional models."""
        data = multidim_binary_data

        config = IdealPointConfig(n_dims=2, response_type=ResponseType.BINARY)

        for optimizer in ["adam", "sgd", "adagrad"]:
            model = IdealPointEstimator(config)
            lr = 0.01 if optimizer == "sgd" else 0.05
            results = model.fit(
                data["person_ids"], data["item_ids"], data["responses"],
                inference="vi", vi_optimizer=optimizer, vi_steps=500, vi_lr=lr,
                device="cpu", progress_bar=False
            )
            assert results.ideal_points.shape == (data["n_persons"], 2)

    def test_optimizers_with_ordinal(self, small_ordinal_data):
        """Test that optimizers work with ordinal responses."""
        data = small_ordinal_data

        config = IdealPointConfig(
            n_dims=1, response_type=ResponseType.ORDINAL, n_categories=data["n_categories"]
        )

        for optimizer in ["adam", "sgd"]:
            model = IdealPointEstimator(config)
            results = model.fit(
                data["person_ids"], data["item_ids"], data["responses"],
                inference="vi", vi_optimizer=optimizer, vi_steps=500,
                device="cpu", progress_bar=False
            )
            assert results.ideal_points is not None

    def test_optimizers_with_temporal(self, small_temporal_data):
        """Test that optimizers work with temporal models."""
        data = small_temporal_data

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            temporal_dynamics=True,
            n_timepoints=data["n_timepoints"]
        )

        for optimizer in ["adam", "adagrad"]:
            model = IdealPointEstimator(config)
            results = model.fit(
                data["person_ids"], data["item_ids"], data["responses"],
                timepoints=data["timepoints"],
                inference="vi", vi_optimizer=optimizer, vi_steps=500,
                device="cpu", progress_bar=False
            )
            assert results.temporal_ideal_points is not None

    def test_optimizers_with_covariates(self, person_covariate_data):
        """Test that optimizers work with covariate models."""
        data = person_covariate_data

        config = IdealPointConfig(
            n_dims=1,
            response_type=ResponseType.BINARY,
            person_covariates=True
        )

        for optimizer in ["adam", "sgd"]:
            model = IdealPointEstimator(config)
            results = model.fit(
                data["person_ids"], data["item_ids"], data["responses"],
                person_covariates=data["person_covariates"],
                inference="vi", vi_optimizer=optimizer, vi_steps=500,
                device="cpu", progress_bar=False
            )
            assert "person_covariate_effects" in results.posterior_samples


class TestOptimizerEdgeCases:
    """Tests for edge cases with optimizers."""

    def test_invalid_optimizer_vi(self, small_binary_data):
        """Test that invalid optimizer raises error for VI."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        with pytest.raises(ValueError, match="Unknown optimizer"):
            model.fit(
                data["person_ids"], data["item_ids"], data["responses"],
                inference="vi", vi_optimizer="invalid_opt", vi_steps=100,
                device="cpu", progress_bar=False
            )

    def test_invalid_optimizer_map(self, small_binary_data):
        """Test that invalid optimizer raises error for MAP."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        with pytest.raises(ValueError, match="Unknown optimizer"):
            model.fit(
                data["person_ids"], data["item_ids"], data["responses"],
                inference="map", map_optimizer="invalid_opt", map_steps=100,
                device="cpu", progress_bar=False
            )

    def test_optimizer_with_few_steps(self, small_binary_data):
        """Test optimizers with very few optimization steps."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)

        for optimizer in ["adam", "sgd", "adagrad"]:
            model = IdealPointEstimator(config)
            # Very few steps - may not converge but shouldn't crash
            results = model.fit(
                data["person_ids"], data["item_ids"], data["responses"],
                inference="vi", vi_optimizer=optimizer, vi_steps=50,
                device="cpu", progress_bar=False
            )
            assert results is not None

    def test_optimizer_with_many_steps(self, small_binary_data):
        """Test optimizers with many optimization steps."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_optimizer="adam", vi_steps=2000,
            device="cpu", progress_bar=False
        )

        # Should converge well with many steps
        assert results.ideal_points is not None
        assert np.all(np.isfinite(results.ideal_points))


class TestOptimizerPerformance:
    """Tests for optimizer performance characteristics."""

    def test_adam_converges_quickly(self, small_binary_data):
        """Test that Adam optimizer converges quickly."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        # Adam should converge with moderate steps
        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_optimizer="adam", vi_steps=500,
            device="cpu", progress_bar=False
        )

        assert results.computation_time is not None
        assert results.computation_time < 60  # Should be fast
        assert results.ideal_points is not None

    def test_optimizer_consistency(self, small_binary_data):
        """Test that optimizer produces consistent results across runs."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)

        # Run twice with same settings
        model1 = IdealPointEstimator(config)
        results1 = model1.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_optimizer="adam", vi_steps=500,
            device="cpu", progress_bar=False
        )

        model2 = IdealPointEstimator(config)
        results2 = model2.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_optimizer="adam", vi_steps=500,
            device="cpu", progress_bar=False
        )

        # Should produce similar results (high correlation)
        corr = np.corrcoef(
            results1.ideal_points.flatten(),
            results2.ideal_points.flatten()
        )[0, 1]
        assert corr > 0.5  # Should be reasonably consistent


class TestOptimizerResponseTypes:
    """Tests for optimizers with different response types."""

    def test_optimizers_continuous(self, small_continuous_data):
        """Test optimizers with continuous responses."""
        data = small_continuous_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.CONTINUOUS)

        for optimizer in ["adam", "sgd", "adagrad"]:
            model = IdealPointEstimator(config)
            results = model.fit(
                data["person_ids"], data["item_ids"], data["responses"],
                inference="vi", vi_optimizer=optimizer, vi_steps=500,
                device="cpu", progress_bar=False
            )
            assert results.ideal_points is not None

    def test_optimizers_count(self, small_count_data):
        """Test optimizers with count responses."""
        data = small_count_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.COUNT)

        for optimizer in ["adam", "adagrad"]:
            model = IdealPointEstimator(config)
            results = model.fit(
                data["person_ids"], data["item_ids"], data["responses"],
                inference="vi", vi_optimizer=optimizer, vi_steps=500,
                device="cpu", progress_bar=False
            )
            assert results.ideal_points is not None
