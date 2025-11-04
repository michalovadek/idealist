# Installation Guide

## CPU Installation (Default)

For most users:

```bash
pip install idealist
```

This installs JAX with CPU-only support, which works on all platforms and is sufficient for small to medium models.

## GPU Installation (CUDA)

For GPU acceleration on NVIDIA GPUs, install JAX with CUDA support. **Important:** JAX's CPU and GPU versions conflict, so use a fresh environment.

### Step 1: Create fresh environment

```bash
# Using conda
conda create -n idealist-gpu python=3.11
conda activate idealist-gpu

# Or using venv
python -m venv idealist-gpu
source idealist-gpu/bin/activate  # Windows: idealist-gpu\Scripts\activate
```

### Step 2: Install with GPU support

For CUDA 12.x:
```bash
pip install idealist[cuda12]
```

For CUDA 11.x:
```bash
pip install idealist[cuda11]
```

### Step 3: Verify installation

```python
import jax
print(f"JAX devices: {jax.devices()}")
# Should show: [cuda(id=0)] or similar
```

## TPU Installation (Google Cloud)

For TPU support:
```bash
pip install idealist
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## Check Your Installation

```python
from idealist import check_installation
check_installation()
```

See [Troubleshooting](troubleshooting.md) for common issues.
