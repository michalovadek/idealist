"""
Device management for CPU/GPU/TPU acceleration.

Handles device detection, selection, and distributed computing setup.
"""

import multiprocessing
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


# Apply XLA CPU core detection workaround BEFORE JAX import
# This fixes JAX/NumPyro issues with detecting available CPU cores
def _configure_xla_cpu_cores(max_cores: Optional[int] = None):
    """
    Configure XLA to recognize available CPU cores.

    JAX/NumPyro sometimes fails to detect the correct number of CPU cores,
    especially on certain platforms (particularly Windows). This function
    sets the XLA_FLAGS environment variable to force the correct core count.

    This must be called before JAX is imported to take effect.

    Parameters
    ----------
    max_cores : Optional[int]
        Maximum number of cores to use. If None, uses all available cores.
        If specified, uses min(max_cores, actual_cpu_count).

    The workaround is only applied if:
    1. XLA_FLAGS is not already configured by the user
    2. The system has multiple CPU cores available

    References
    ----------
    - JAX Issue: https://github.com/google/jax/issues/1408
    - NumPyro parallelization: https://num.pyro.ai/en/stable/utilities.html#parallelization
    """
    try:
        # Get actual CPU count
        actual_cpu_count = multiprocessing.cpu_count()

        # Determine target core count
        if max_cores is not None:
            if max_cores < 1:
                logger.warning(f"Invalid max_cores={max_cores}, using all cores")
                cpu_count = actual_cpu_count
            else:
                cpu_count = min(max_cores, actual_cpu_count)
                if max_cores > actual_cpu_count:
                    logger.warning(
                        f"Requested {max_cores} cores but only {actual_cpu_count} available. "
                        f"Using {cpu_count} cores."
                    )
        else:
            # Check environment variable for user preference
            env_max_cores = os.environ.get("IDEALIST_MAX_CPU_CORES")
            if env_max_cores is not None:
                try:
                    cpu_count = min(int(env_max_cores), actual_cpu_count)
                    logger.debug(f"Using IDEALIST_MAX_CPU_CORES={cpu_count}")
                except ValueError:
                    cpu_count = actual_cpu_count
                    logger.warning(
                        f"Invalid IDEALIST_MAX_CPU_CORES={env_max_cores}, using all cores"
                    )
            else:
                cpu_count = actual_cpu_count

        # Check if already configured
        existing_flags = os.environ.get("XLA_FLAGS", "")
        if "xla_force_host_platform_device_count" in existing_flags:
            logger.debug("XLA_FLAGS already configured, skipping auto-configuration")
            return  # Already configured, don't override

        # Only apply for multi-core systems
        if cpu_count <= 1:
            return

        # Build the XLA flags
        if existing_flags:
            # Append to existing flags
            new_flags = f"{existing_flags} --xla_force_host_platform_device_count={cpu_count}"
        else:
            # Set new flags
            new_flags = f"--xla_force_host_platform_device_count={cpu_count}"

        os.environ["XLA_FLAGS"] = new_flags

        # Use debug level to avoid cluttering user output
        # Will be visible if user enables logging
        if max_cores is not None or env_max_cores is not None:
            logger.debug(f"Configured XLA to use {cpu_count} of {actual_cpu_count} CPU cores")
        else:
            logger.debug(f"Configured XLA to recognize {cpu_count} CPU cores for parallel MCMC")

    except Exception as e:
        # Silently fail - not critical
        # Log at debug level in case it's helpful for troubleshooting
        logger.debug(f"Could not configure XLA CPU cores: {e}")


# Apply the workaround immediately when module is imported
# This ensures it happens before JAX is loaded anywhere in the package
# Users can set IDEALIST_MAX_CPU_CORES environment variable to limit cores
_configure_xla_cpu_cores()


@dataclass
class DeviceInfo:
    """Information about available compute devices."""

    has_gpu: bool = False
    has_tpu: bool = False
    gpu_count: int = 0
    gpu_names: List[str] = None
    tpu_cores: int = 0
    default_backend: str = "cpu"

    def __post_init__(self):
        if self.gpu_names is None:
            self.gpu_names = []


class DeviceManager:
    """
    Manages compute device selection and configuration.

    Handles automatic detection of GPUs/TPUs and sets up JAX accordingly.
    """

    _device_info: Optional[DeviceInfo] = None

    @classmethod
    def get_device_info(cls, backend: str = "jax") -> DeviceInfo:
        """
        Get information about available devices.

        Parameters
        ----------
        backend : str
            'jax' or 'tensorflow'

        Returns
        -------
        info : DeviceInfo
            Device information
        """
        if cls._device_info is not None:
            return cls._device_info

        info = DeviceInfo()

        if backend == "jax":
            try:
                import jax

                devices = jax.devices()

                # Check for GPUs
                gpu_devices = [d for d in devices if d.platform == "gpu"]
                info.has_gpu = len(gpu_devices) > 0
                info.gpu_count = len(gpu_devices)
                info.gpu_names = [str(d) for d in gpu_devices]

                # Check for TPUs
                tpu_devices = [d for d in devices if d.platform == "tpu"]
                info.has_tpu = len(tpu_devices) > 0
                info.tpu_cores = len(tpu_devices)

                # Determine default
                if info.has_tpu:
                    info.default_backend = "tpu"
                elif info.has_gpu:
                    info.default_backend = "gpu"
                else:
                    info.default_backend = "cpu"

            except ImportError:
                pass

        cls._device_info = info
        return info

    @classmethod
    def check_jax_installation(cls) -> Dict[str, Any]:
        """
        Check JAX installation and provide diagnostics.

        Returns
        -------
        dict
            Diagnostic information about JAX installation including:
            - jax_version: JAX version string
            - jaxlib_version: JAXlib version string
            - has_cuda: Whether CUDA support is available
            - cuda_version: CUDA version if available
            - devices: List of available devices
            - warnings: List of warning messages
            - recommendations: List of installation recommendations
        """
        diagnostics = {
            "jax_version": None,
            "jaxlib_version": None,
            "has_cuda": False,
            "cuda_version": None,
            "devices": [],
            "warnings": [],
            "recommendations": [],
        }

        try:
            import jax
            import jaxlib

            diagnostics["jax_version"] = jax.__version__
            diagnostics["jaxlib_version"] = jaxlib.__version__

            # Get available devices
            devices = jax.devices()
            diagnostics["devices"] = [f"{d.platform}:{d.id}" for d in devices]

            # Check for CUDA
            try:
                from jax.lib import xla_bridge

                backend = xla_bridge.get_backend()
                platform = backend.platform

                if platform == "gpu":
                    diagnostics["has_cuda"] = True
                    # Try to get CUDA version
                    try:
                        cuda_version = backend.platform_version
                        diagnostics["cuda_version"] = cuda_version
                    except Exception:  # noqa: S110
                        pass

            except Exception:
                pass

            # Generate warnings and recommendations
            gpu_devices = [d for d in devices if d.platform == "gpu"]
            cpu_only = len(gpu_devices) == 0

            # Check if user might have GPU hardware but CPU-only JAX
            if cpu_only and os.environ.get("CUDA_VISIBLE_DEVICES") is None:
                try:
                    # Try to detect NVIDIA GPU presence
                    import subprocess

                    result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=2)
                    if result.returncode == 0:
                        diagnostics["warnings"].append(
                            "NVIDIA GPU detected but JAX is using CPU-only mode"
                        )
                        diagnostics["recommendations"].append(
                            "Install JAX with CUDA support for GPU acceleration:\n"
                            "  pip install --upgrade jax[cuda12]  # For CUDA 12.x\n"
                            "  pip install --upgrade jax[cuda11]  # For CUDA 11.x\n"
                            "Or install idealist with GPU support:\n"
                            "  pip install idealist[cuda12]"
                        )
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass  # nvidia-smi not available, no GPU

            # Check for version mismatches
            if diagnostics["jax_version"] != diagnostics["jaxlib_version"]:
                diagnostics["warnings"].append(
                    f"JAX version ({diagnostics['jax_version']}) differs from "
                    f"JAXlib version ({diagnostics['jaxlib_version']})"
                )
                diagnostics["recommendations"].append(
                    "Reinstall JAX with matching versions:\n" "  pip install --upgrade jax jaxlib"
                )

        except ImportError as e:
            diagnostics["warnings"].append(f"Failed to import JAX: {e}")
            diagnostics["recommendations"].append("Install JAX:\n" "  pip install jax jaxlib")

        return diagnostics

    @classmethod
    def setup_device(
        cls,
        use_gpu: bool = True,
        use_tpu: bool = False,
        distributed: bool = False,
        device_id: Optional[int] = None,
    ) -> str:
        """
        Setup compute device for JAX and NumPyro.

        Parameters
        ----------
        use_gpu : bool
            Try to use GPU if available
        use_tpu : bool
            Try to use TPU if available (overrides use_gpu)
        distributed : bool
            Setup for distributed training across devices
        device_id : Optional[int]
            Specific device ID to use (None = all available)

        Returns
        -------
        device : str
            Selected device type: 'cpu', 'gpu', or 'tpu'
        """
        try:
            import jax  # noqa: F401
            import numpyro
        except ImportError:
            return "cpu"

        info = cls.get_device_info("jax")

        # Priority: TPU > GPU > CPU
        if use_tpu and info.has_tpu:
            # TPU setup
            numpyro.set_platform("tpu")
            if distributed:
                # JAX automatically handles TPU distribution
                logger.info(f"Using {info.tpu_cores} TPU cores (distributed)")
            return "tpu"

        elif use_gpu and info.has_gpu:
            # GPU setup
            numpyro.set_platform("gpu")
            if device_id is not None:
                # Use specific GPU
                os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
                logger.info(f"Using GPU {device_id}")
            elif distributed and info.gpu_count > 1:
                # Multi-GPU
                logger.info(f"Using {info.gpu_count} GPUs (distributed)")
            else:
                logger.info("Using GPU (single device)")
            return "gpu"

        elif use_gpu and not info.has_gpu:
            # User requested GPU but none available
            logger.warning(
                "GPU requested but no GPU devices found. Falling back to CPU.\n"
                "To use GPU acceleration, install JAX with CUDA support:\n"
                "  pip install --upgrade 'jax[cuda12]'  # For CUDA 12.x\n"
                "  pip install --upgrade 'jax[cuda11]'  # For CUDA 11.x\n"
                "Or reinstall idealist with GPU support:\n"
                "  pip install 'idealist[cuda12]'"
            )
            # Fallback to CPU
            os.environ["JAX_PLATFORMS"] = "cpu"
            numpyro.set_platform("cpu")
            return "cpu"

        else:
            # CPU fallback
            os.environ["JAX_PLATFORMS"] = "cpu"
            numpyro.set_platform("cpu")
            logger.info("Using CPU")
            return "cpu"

    @classmethod
    def set_precision(cls, precision: str = "float32"):
        """
        Set default floating point precision.

        Parameters
        ----------
        precision : str
            'float32' or 'float64'
        """
        try:
            import jax

            if precision == "float64":
                jax.config.update("jax_enable_x64", True)
            else:
                jax.config.update("jax_enable_x64", False)
        except ImportError:
            pass

    @classmethod
    def enable_jit(cls, enable: bool = True):
        """Enable or disable JIT compilation."""
        try:
            import jax

            jax.config.update("jax_disable_jit", not enable)
        except ImportError:
            pass

    @classmethod
    def get_memory_info(cls) -> Dict[str, Any]:
        """Get memory usage information (if available)."""
        info = {"available": False}

        try:
            import jax

            # Try to get memory info from first GPU
            devices = jax.devices("gpu")
            if devices:
                # This is a simplified version - actual memory tracking
                # requires device-specific APIs
                info["available"] = True
                info["device_count"] = len(devices)
        except Exception:
            pass

        return info

    @classmethod
    def auto_select_strategy(
        cls,
        inference_method: str,
        n_persons: int,
        n_items: int,
        n_obs: int,
        use_device: str = "auto",
        max_cpu_chains: int = 4,
    ) -> Dict[str, Any]:
        """
        Automatically select best device and parallelization strategy.

        Parameters
        ----------
        inference_method : str
            'mcmc', 'vi', or 'map'
        n_persons : int
            Number of persons/respondents
        n_items : int
            Number of items
        n_obs : int
            Number of observations
        use_device : str
            'auto', 'cpu', 'gpu', or 'tpu'
        max_cpu_chains : int
            Maximum number of chains for CPU MCMC

        Returns
        -------
        strategy : dict
            {
                'device': str,           # 'cpu', 'gpu', or 'tpu'
                'num_chains': int or None,  # For MCMC only
                'cpu_cores': int or None,   # For CPU parallelization
                'setup_commands': List[str], # Setup commands to execute
                'reason': str,           # Explanation
            }
        """
        info = cls.get_device_info("jax")

        # Manual override
        if use_device != "auto":
            return cls._manual_strategy(use_device, inference_method, info, max_cpu_chains)

        # Auto selection
        model_size = {"n_persons": n_persons, "n_items": n_items, "n_obs": n_obs}
        return cls._auto_strategy(inference_method, model_size, info, max_cpu_chains)

    @classmethod
    def _auto_strategy(
        cls,
        inference_method: str,
        model_size: Dict[str, int],
        info: DeviceInfo,
        max_cpu_chains: int,
    ) -> Dict[str, Any]:
        """Implement auto-selection logic."""
        import os

        has_gpu = info.has_gpu
        cpu_cores = os.cpu_count() or 4

        is_large = (
            model_size["n_persons"] > 500
            or model_size["n_items"] > 200
            or model_size["n_obs"] > 10000
        )

        if inference_method == "mcmc":
            # MCMC: CPU multi-chain (usually best)
            num_chains = min(max_cpu_chains, max(2, cpu_cores // 2))

            if is_large and has_gpu:
                return {
                    "device": "gpu",
                    "num_chains": num_chains,
                    "cpu_cores": num_chains,
                    "setup_commands": [
                        f"numpyro.set_host_device_count({num_chains})",
                    ],
                    "reason": f"Large model: GPU with {num_chains} parallel chains",
                }
            else:
                return {
                    "device": "cpu",
                    "num_chains": num_chains,
                    "cpu_cores": num_chains,
                    "setup_commands": [
                        f"numpyro.set_host_device_count({num_chains})",
                    ],
                    "reason": (
                        f"CPU with {num_chains} parallel MCMC chains " f"(1.6-2x speedup expected)"
                    ),
                }

        elif inference_method in ["vi", "map"]:
            if has_gpu:
                speedup = "5-10x" if inference_method == "vi" else "3-5x"
                return {
                    "device": "gpu",
                    "num_chains": None,
                    "cpu_cores": None,
                    "setup_commands": [],
                    "reason": f"GPU for {inference_method.upper()} ({speedup} speedup expected)",
                }
            else:
                return {
                    "device": "cpu",
                    "num_chains": None,
                    "cpu_cores": None,
                    "setup_commands": [],
                    "reason": f"CPU for {inference_method.upper()}",
                }

        else:
            raise ValueError(f"Unknown inference method: {inference_method}")

    @classmethod
    def _manual_strategy(
        cls,
        use_device: str,
        inference_method: str,
        info: DeviceInfo,
        max_cpu_chains: int,
    ) -> Dict[str, Any]:
        """Handle manual device selection."""
        import os

        cpu_cores = os.cpu_count() or 4

        if use_device == "cpu" and inference_method == "mcmc":
            num_chains = min(max_cpu_chains, max(2, cpu_cores // 2))
            return {
                "device": "cpu",
                "num_chains": num_chains,
                "cpu_cores": num_chains,
                "setup_commands": [f"numpyro.set_host_device_count({num_chains})"],
                "reason": f"Manual selection: CPU with {num_chains} chains",
            }
        else:
            return {
                "device": use_device,
                "num_chains": None,
                "cpu_cores": None,
                "setup_commands": [],
                "reason": f"Manual selection: {use_device}",
            }

    @classmethod
    def apply_strategy(cls, strategy: Dict[str, Any]) -> None:
        """
        Apply the selected strategy (execute setup commands).

        This includes setting the platform and configuring parallelization.

        Parameters
        ----------
        strategy : dict
            Strategy dict from auto_select_strategy()
        """
        import re

        try:
            import jax
            import numpyro

            # Set platform based on device selection
            device = strategy.get("device", "cpu")
            numpyro.set_platform(device)

            # Execute additional setup commands
            for cmd in strategy["setup_commands"]:
                # Execute setup command
                if "numpyro.set_host_device_count" in cmd:
                    match = re.search(r"set_host_device_count\((\d+)\)", cmd)
                    if match:
                        num_cores = int(match.group(1))
                        numpyro.set_host_device_count(num_cores)
                        logger.debug(f"Set NumPyro host device count to {num_cores}")

            # Log final device count for verification
            device_count = jax.local_device_count()
            logger.debug(f"JAX reports {device_count} local device(s) available")

        except ImportError:
            logger.warning("JAX/NumPyro not available - skipping device configuration")
        except Exception as e:
            logger.warning(f"Failed to apply device strategy: {e}")


def get_device_info(backend: str = "jax") -> DeviceInfo:
    """
    Convenience function to get device information.

    Parameters
    ----------
    backend : str
        'jax' or 'tensorflow'

    Returns
    -------
    info : DeviceInfo
        Device information
    """
    return DeviceManager.get_device_info(backend)


def auto_select_device(prefer_gpu: bool = True, prefer_tpu: bool = False) -> str:
    """
    Automatically select best available device.

    Parameters
    ----------
    prefer_gpu : bool
        Prefer GPU over CPU if available
    prefer_tpu : bool
        Prefer TPU over GPU/CPU if available

    Returns
    -------
    device : str
        Selected device: 'cpu', 'gpu', or 'tpu'
    """
    return DeviceManager.setup_device(
        use_gpu=prefer_gpu,
        use_tpu=prefer_tpu,
        distributed=False,
    )


def configure_cpu_cores(max_cores: Optional[int] = None) -> None:
    """
    Configure the maximum number of CPU cores to use.

    This function must be called BEFORE importing any idealist models or
    JAX/NumPyro, otherwise it will have no effect.

    Parameters
    ----------
    max_cores : Optional[int]
        Maximum number of CPU cores to use. If None, uses all available cores.
        If specified, uses min(max_cores, actual_cpu_count).

    Examples
    --------
    Limit to 4 cores (call before importing models)::

        import os
        os.environ['IDEALIST_MAX_CPU_CORES'] = '4'
        from idealist import IdealPointEstimator

    Or use the function directly::

        from idealist.core.device import configure_cpu_cores
        configure_cpu_cores(max_cores=4)
        from idealist import IdealPointEstimator

    Notes
    -----
    The recommended approach is to set the IDEALIST_MAX_CPU_CORES environment
    variable before running your script:

    .. code-block:: bash

        export IDEALIST_MAX_CPU_CORES=4
        python your_script.py

    This function reconfigures XLA_FLAGS, so it will only work if called
    before JAX is imported for the first time.
    """
    # Check if JAX is already imported
    if "jax" in globals() or "jax" in locals():
        logger.warning(
            "configure_cpu_cores() called after JAX was already imported. "
            "This will have no effect. Call this function before importing "
            "any idealist models or JAX."
        )
        return

    # Reconfigure XLA
    _configure_xla_cpu_cores(max_cores=max_cores)
