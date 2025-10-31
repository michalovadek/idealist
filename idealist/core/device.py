"""
Device management for CPU/GPU/TPU acceleration.

Handles device detection, selection, and distributed computing setup.
"""

import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from ..utils.logging import get_logger

logger = get_logger(__name__)


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
                gpu_devices = [d for d in devices if d.platform == 'gpu']
                info.has_gpu = len(gpu_devices) > 0
                info.gpu_count = len(gpu_devices)
                info.gpu_names = [str(d) for d in gpu_devices]

                # Check for TPUs
                tpu_devices = [d for d in devices if d.platform == 'tpu']
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
    def setup_device(
        cls,
        use_gpu: bool = True,
        use_tpu: bool = False,
        distributed: bool = False,
        device_id: Optional[int] = None,
    ) -> str:
        """
        Setup compute device for JAX.

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
            import jax
        except ImportError:
            return "cpu"

        info = cls.get_device_info("jax")

        # Priority: TPU > GPU > CPU
        if use_tpu and info.has_tpu:
            # TPU setup
            if distributed:
                # JAX automatically handles TPU distribution
                logger.info(f"Using {info.tpu_cores} TPU cores (distributed)")
            return "tpu"

        elif use_gpu and info.has_gpu:
            # GPU setup
            if device_id is not None:
                # Use specific GPU
                os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
                logger.info(f"Using GPU {device_id}")
            elif distributed and info.gpu_count > 1:
                # Multi-GPU
                logger.info(f"Using {info.gpu_count} GPUs (distributed)")
            else:
                logger.info(f"Using GPU (single device)")
            return "gpu"

        else:
            # CPU fallback
            os.environ['JAX_PLATFORMS'] = 'cpu'
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
        model_size = {'n_persons': n_persons, 'n_items': n_items, 'n_obs': n_obs}
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

        is_large = (model_size['n_persons'] > 500 or
                    model_size['n_items'] > 200 or
                    model_size['n_obs'] > 10000)

        if inference_method == 'mcmc':
            # MCMC: CPU multi-chain (usually best)
            num_chains = min(max_cpu_chains, max(2, cpu_cores // 2))

            if is_large and has_gpu:
                return {
                    'device': 'gpu',
                    'num_chains': num_chains,
                    'cpu_cores': num_chains,
                    'setup_commands': [
                        f'numpyro.set_host_device_count({num_chains})',
                    ],
                    'reason': f'Large model: GPU with {num_chains} parallel chains',
                }
            else:
                return {
                    'device': 'cpu',
                    'num_chains': num_chains,
                    'cpu_cores': num_chains,
                    'setup_commands': [
                        f'numpyro.set_host_device_count({num_chains})',
                    ],
                    'reason': f'CPU with {num_chains} parallel MCMC chains (1.6-2x speedup expected)',
                }

        elif inference_method in ['vi', 'map']:
            if has_gpu:
                speedup = "5-10x" if inference_method == 'vi' else "3-5x"
                return {
                    'device': 'gpu',
                    'num_chains': None,
                    'cpu_cores': None,
                    'setup_commands': [],
                    'reason': f'GPU for {inference_method.upper()} ({speedup} speedup expected)',
                }
            else:
                return {
                    'device': 'cpu',
                    'num_chains': None,
                    'cpu_cores': None,
                    'setup_commands': [],
                    'reason': f'CPU for {inference_method.upper()}',
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

        if use_device == "cpu" and inference_method == 'mcmc':
            num_chains = min(max_cpu_chains, max(2, cpu_cores // 2))
            return {
                'device': 'cpu',
                'num_chains': num_chains,
                'cpu_cores': num_chains,
                'setup_commands': [f'numpyro.set_host_device_count({num_chains})'],
                'reason': f'Manual selection: CPU with {num_chains} chains',
            }
        else:
            return {
                'device': use_device,
                'num_chains': None,
                'cpu_cores': None,
                'setup_commands': [],
                'reason': f'Manual selection: {use_device}',
            }

    @classmethod
    def apply_strategy(cls, strategy: Dict[str, Any]) -> None:
        """
        Apply the selected strategy (execute setup commands).

        Parameters
        ----------
        strategy : dict
            Strategy dict from auto_select_strategy()
        """
        import re

        for cmd in strategy['setup_commands']:
            # Execute setup command
            if 'numpyro.set_host_device_count' in cmd:
                try:
                    import numpyro
                    match = re.search(r'set_host_device_count\((\d+)\)', cmd)
                    if match:
                        num_cores = int(match.group(1))
                        numpyro.set_host_device_count(num_cores)
                except ImportError:
                    pass  # NumPyro not available


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
