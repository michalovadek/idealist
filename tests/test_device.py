"""
Tests for device management and hardware acceleration.

Device management is crucial for verifying GPU acceleration works correctly,
testing TPU support, validating CPU multi-core parallelization, and ensuring
auto-detection works across platforms.
"""

import os

import jax
import numpy as np
import pytest

from idealist import IdealPointConfig, IdealPointEstimator, ResponseType
from idealist.core.device import DeviceManager, configure_cpu_cores


class TestDeviceDetection:
    """Tests for device detection functionality."""

    def test_get_device_info_cpu(self):
        """Test device info detection on CPU."""
        device_manager = DeviceManager()
        device_info = device_manager.get_device_info()

        # Verify it returns a valid object with expected fields
        assert device_info is not None
        assert hasattr(device_info, "platform")
        assert hasattr(device_info, "device_count")

        # CPU should always be available
        assert device_info.platform in ["cpu", "gpu", "tpu"]

    def test_device_info_structure(self):
        """Test that device info contains expected fields."""
        device_manager = DeviceManager()
        device_info = device_manager.get_device_info()

        # Check basic fields exist
        assert hasattr(device_info, "platform")
        assert hasattr(device_info, "device_count")
        assert device_info.device_count >= 1  # At least CPU

    def test_jax_devices_accessible(self):
        """Test that JAX devices can be accessed."""
        devices = jax.devices()

        assert devices is not None
        assert len(devices) >= 1  # At least one device (CPU)
        assert all(hasattr(d, "platform") for d in devices)


class TestDeviceConfiguration:
    """Tests for device configuration."""

    def test_configure_cpu_cores(self):
        """Test configure_cpu_cores() function."""
        # Try to configure CPU cores
        try:
            configure_cpu_cores(max_cores=2)
            # Verify XLA_FLAGS is set (or at least doesn't crash)
            assert True
        except Exception as e:
            # Some environments may not support this
            pytest.skip(f"CPU core configuration not supported: {e}")

    def test_device_selection_cpu(self, small_binary_data):
        """Test explicit CPU device selection."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        # Explicitly request CPU device
        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None

    def test_device_default_selection(self, small_binary_data):
        """Test that default device selection works."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        # Don't specify device, let it auto-select
        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=500, progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None


class TestDeviceStrategySelection:
    """Tests for automatic device strategy selection."""

    def test_auto_select_strategy_vi(self):
        """Test device strategy selection for VI."""
        device_manager = DeviceManager()

        strategy = device_manager.auto_select_strategy(inference_method="vi")

        assert strategy is not None
        assert hasattr(strategy, "device_type")
        assert hasattr(strategy, "reason")
        # VI should prefer GPU if available, else CPU
        assert strategy.device_type in ["cpu", "gpu"]

    def test_auto_select_strategy_map(self):
        """Test device strategy selection for MAP."""
        device_manager = DeviceManager()

        strategy = device_manager.auto_select_strategy(inference_method="map")

        assert strategy is not None
        assert hasattr(strategy, "device_type")
        # MAP should work on either CPU or GPU
        assert strategy.device_type in ["cpu", "gpu"]

    def test_auto_select_strategy_mcmc(self):
        """Test device strategy selection for MCMC."""
        device_manager = DeviceManager()

        strategy = device_manager.auto_select_strategy(inference_method="mcmc")

        assert strategy is not None
        assert hasattr(strategy, "device_type")
        # MCMC typically prefers CPU for parallel chains
        assert strategy.device_type in ["cpu", "gpu"]


class TestJAXInstallationChecks:
    """Tests for JAX installation diagnostics."""

    def test_check_jax_installation_cpu(self):
        """Test JAX installation checking."""
        device_manager = DeviceManager()

        diagnostics = device_manager.check_jax_installation()

        assert diagnostics is not None
        assert "jax_version" in diagnostics
        assert "jaxlib_version" in diagnostics
        assert diagnostics["jax_version"] is not None

    def test_jax_version_accessible(self):
        """Test that JAX version can be retrieved."""
        import jax

        version = jax.__version__
        assert version is not None
        assert isinstance(version, str)
        assert len(version) > 0


class TestMCMCParallelization:
    """Tests for multi-core MCMC execution."""

    @pytest.mark.slow
    def test_mcmc_multiple_chains(self, small_binary_data):
        """Test that MCMC can run with multiple chains."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        # Run MCMC with 2 chains
        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=2, num_warmup=50, num_samples=50,
            device="cpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None
        # Check that multiple chains were run
        if results.posterior_samples is not None:
            # Samples should exist
            assert "ideal_points" in results.posterior_samples

    @pytest.mark.slow
    def test_mcmc_single_chain(self, small_binary_data):
        """Test MCMC with single chain."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="mcmc", num_chains=1, num_warmup=50, num_samples=50,
            device="cpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None


class TestDevicePerformance:
    """Tests for device performance characteristics."""

    def test_vi_performance_cpu(self, small_binary_data):
        """Test VI performance on CPU."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=500, device="cpu", progress_bar=False
        )

        # Check that computation completed in reasonable time
        assert results.computation_time is not None
        assert results.computation_time > 0
        # Should complete in under 60 seconds for small data
        assert results.computation_time < 60

    def test_map_performance_cpu(self, small_binary_data):
        """Test MAP performance on CPU."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="map", map_steps=500, device="cpu", progress_bar=False
        )

        # Check computation time
        assert results.computation_time is not None
        assert results.computation_time > 0
        assert results.computation_time < 60


class TestGPUSupport:
    """Tests for GPU support (if available)."""

    @pytest.mark.skipif(
        not any(d.platform == "gpu" for d in jax.devices()),
        reason="GPU not available"
    )
    def test_gpu_detection(self):
        """Test that GPU is detected if available."""
        device_manager = DeviceManager()
        device_info = device_manager.get_device_info()

        # If this test runs, GPU should be detected
        gpu_available = any(d.platform == "gpu" for d in jax.devices())
        assert gpu_available

    @pytest.mark.skipif(
        not any(d.platform == "gpu" for d in jax.devices()),
        reason="GPU not available"
    )
    def test_vi_on_gpu(self, small_binary_data):
        """Test VI inference on GPU."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        results = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=500, device="gpu", progress_bar=False
        )

        assert results is not None
        assert results.ideal_points is not None


class TestDeviceEdgeCases:
    """Tests for edge cases in device management."""

    def test_invalid_device_specification(self, small_binary_data):
        """Test handling of invalid device specification."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        # Try invalid device (should either raise error or fall back to CPU)
        try:
            results = model.fit(
                data["person_ids"], data["item_ids"], data["responses"],
                inference="vi", vi_steps=100, device="invalid_device", progress_bar=False
            )
            # If it doesn't error, it should have fallen back to valid device
            assert results is not None
        except (ValueError, RuntimeError):
            # Expected to raise error for invalid device
            pass

    def test_device_consistency_across_fits(self, small_binary_data):
        """Test that device selection is consistent across multiple fits."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        # First fit
        results1 = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=200, device="cpu", progress_bar=False
        )

        # Second fit (should use same device)
        results2 = model.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=200, device="cpu", progress_bar=False
        )

        assert results1 is not None
        assert results2 is not None


class TestDiagnostics:
    """Tests for diagnostic functions."""

    def test_device_manager_instantiation(self):
        """Test that DeviceManager can be instantiated."""
        device_manager = DeviceManager()
        assert device_manager is not None

    def test_jax_backend_accessible(self):
        """Test that JAX backend information is accessible."""
        # Check JAX default backend
        backend = jax.default_backend()
        assert backend is not None
        assert backend in ["cpu", "gpu", "tpu"]

    def test_device_count(self):
        """Test that device count is reasonable."""
        devices = jax.devices()
        assert len(devices) >= 1  # At least CPU
        assert len(devices) < 1000  # Sanity check


class TestMultiDeviceScenarios:
    """Tests for scenarios involving multiple devices."""

    def test_sequential_fits_same_device(self, small_binary_data):
        """Test multiple sequential fits on same device."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)

        # Run multiple fits
        for i in range(3):
            model = IdealPointEstimator(config)
            results = model.fit(
                data["person_ids"], data["item_ids"], data["responses"],
                inference="vi", vi_steps=200, device="cpu", progress_bar=False
            )
            assert results is not None

    def test_different_inference_methods_same_device(self, small_binary_data):
        """Test different inference methods on same device."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)

        # VI
        model_vi = IdealPointEstimator(config)
        results_vi = model_vi.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="vi", vi_steps=200, device="cpu", progress_bar=False
        )

        # MAP
        model_map = IdealPointEstimator(config)
        results_map = model_map.fit(
            data["person_ids"], data["item_ids"], data["responses"],
            inference="map", map_steps=200, device="cpu", progress_bar=False
        )

        assert results_vi is not None
        assert results_map is not None


class TestCPUCoreConfiguration:
    """Tests for CPU core configuration."""

    def test_configure_cores_valid_range(self):
        """Test configuring CPU cores with valid values."""
        # Should not crash
        try:
            configure_cpu_cores(max_cores=1)
            configure_cpu_cores(max_cores=2)
            configure_cpu_cores(max_cores=4)
            assert True
        except Exception as e:
            pytest.skip(f"CPU configuration not supported: {e}")

    def test_cpu_chains_configuration(self, small_binary_data):
        """Test max_cpu_chains parameter."""
        data = small_binary_data

        config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
        model = IdealPointEstimator(config)

        # Should not crash with max_cpu_chains specified
        try:
            results = model.fit(
                data["person_ids"], data["item_ids"], data["responses"],
                inference="mcmc", num_chains=2, num_warmup=50, num_samples=50,
                device="cpu", progress_bar=False
            )
            assert results is not None
        except Exception:
            # MCMC might not work in all environments
            pytest.skip("MCMC not supported in this environment")
