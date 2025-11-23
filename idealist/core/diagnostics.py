"""
Diagnostics utilities for checking JAX installation and device configuration.
"""

from typing import Any, Dict

from ..utils.logging import get_logger
from .device import DeviceManager

logger = get_logger(__name__)


def check_installation(verbose: bool = True) -> Dict[str, Any]:
    """
    Check idealist installation and provide diagnostics.

    This function checks:
    - JAX and JAXlib versions
    - Available devices (CPU/GPU/TPU)
    - CUDA support and version
    - Installation recommendations

    Parameters
    ----------
    verbose : bool, default=True
        If True, print diagnostic information to console

    Returns
    -------
    dict
        Diagnostic information including:
        - jax_version: JAX version string
        - jaxlib_version: JAXlib version string
        - has_cuda: Whether CUDA support is available
        - cuda_version: CUDA version if available
        - devices: List of available devices
        - warnings: List of warning messages
        - recommendations: List of installation recommendations

    Examples
    --------
    Check your installation::

        from idealist.core.diagnostics import check_installation
        check_installation()

    Get diagnostics programmatically::

        diagnostics = check_installation(verbose=False)
        if diagnostics['warnings']:
            print("Installation issues detected!")
    """
    diagnostics = DeviceManager.check_jax_installation()

    if verbose:
        print("=" * 70)
        print("Idealist Installation Diagnostics")
        print("=" * 70)
        print()

        # Version information
        print("JAX Installation:")
        print(f"  JAX version:    {diagnostics['jax_version']}")
        print(f"  JAXlib version: {diagnostics['jaxlib_version']}")
        print(f"  CUDA support:   {diagnostics['has_cuda']}")
        if diagnostics["cuda_version"]:
            print(f"  CUDA version:   {diagnostics['cuda_version']}")
        print()

        # Device information
        print("Available Devices:")
        if diagnostics["devices"]:
            for device in diagnostics["devices"]:
                print(f"  - {device}")
        else:
            print("  (No devices detected)")
        print()

        # Warnings
        if diagnostics["warnings"]:
            print("Warnings:")
            for warning in diagnostics["warnings"]:
                print(f"  ! {warning}")
            print()

        # Recommendations
        if diagnostics["recommendations"]:
            print("Recommendations:")
            for rec in diagnostics["recommendations"]:
                # Indent multi-line recommendations
                lines = rec.split("\n")
                for i, line in enumerate(lines):
                    if i == 0:
                        print(f"  - {line}")
                    else:
                        print(f"    {line}")
            print()

        # Summary
        print("=" * 70)
        if not diagnostics["warnings"]:
            print("SUCCESS: Installation looks good!")
        else:
            print("NOTICE: Installation issues detected - see recommendations above")
        print("=" * 70)

    return diagnostics


def print_device_info():
    """
    Print information about available compute devices.

    This is a convenience function that displays:
    - Number of available devices
    - Device types (CPU/GPU/TPU)
    - Device names
    - Recommended device for different inference methods

    Examples
    --------
    ::

        from idealist.core.diagnostics import print_device_info
        print_device_info()
    """
    info = DeviceManager.get_device_info("jax")

    print("=" * 70)
    print("Device Information")
    print("=" * 70)
    print()

    print("Available Devices:")
    print("  CPUs:  Available (multi-core parallelization supported)")
    print(f"  GPUs:  {'Yes' if info.has_gpu else 'No'}", end="")
    if info.has_gpu:
        print(f" ({info.gpu_count} GPU{'s' if info.gpu_count > 1 else ''})")
    else:
        print()
    print(f"  TPUs:  {'Yes' if info.has_tpu else 'No'}", end="")
    if info.has_tpu:
        print(f" ({info.tpu_cores} cores)")
    else:
        print()
    print()

    print(f"Default backend: {info.default_backend.upper()}")
    print()

    print("Recommendations:")
    print("  • MCMC:     CPU with parallel chains (automatically configured)")
    print("  • VI/MAP:   GPU if available (5-10x speedup)")
    print("  • Small models: CPU is often faster due to transfer overhead")
    print()

    print("=" * 70)


if __name__ == "__main__":
    # Allow running as: python -m idealist.core.diagnostics
    check_installation(verbose=True)
