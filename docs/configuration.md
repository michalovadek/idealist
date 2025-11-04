# Configuration Guide

## Device Configuration

### Automatic (Recommended)

```python
results = estimator.fit(data, device='auto', inference='mcmc')
```

The package automatically:
- Detects available devices (CPU/GPU/TPU)
- Selects optimal device for inference method
- Configures CPU parallelization

### Manual Device Selection

```python
# Use CPU explicitly
results = estimator.fit(data, device='cpu', inference='mcmc')

# Use GPU explicitly
results = estimator.fit(data, device='gpu', inference='vi')
```

## CPU Core Configuration

idealist automatically detects and uses all available CPU cores. You can limit this:

### Method 1: Runtime parameter (recommended)

```python
results = estimator.fit(
    data,
    inference='mcmc',
    device='cpu',
    max_cpu_chains=4  # Use only 4 cores
)
```

### Method 2: Environment variable

Before running Python:
```bash
export IDEALIST_MAX_CPU_CORES=4
python your_script.py
```

Or in Python:
```python
import os
os.environ['IDEALIST_MAX_CPU_CORES'] = '4'
from idealist import IdealPointEstimator
```

### Method 3: Advanced function

```python
from idealist import configure_cpu_cores
configure_cpu_cores(max_cores=4)
from idealist import IdealPointEstimator
```

### Method 4: Manual XLA configuration

```python
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
from idealist import IdealPointEstimator
```

## Performance Tips

### MCMC
- **Best on:** CPU with multiple parallel chains
- **Automatically configured:** Yes
- **Typical speedup:** 1.6-2x with 4 cores

### VI/MAP
- **Best on:** GPU for large models
- **GPU speedup:** 5-10x
- **Small models:** CPU often faster (no transfer overhead)

### Model Size Guidelines

- **Small** (<100 persons, <50 items): CPU is fine
- **Medium** (100-500 persons, 50-200 items): CPU or GPU
- **Large** (>500 persons, >200 items): GPU recommended for VI/MAP

## Advanced Configuration

### Precision

```python
from idealist.core.device import DeviceManager
DeviceManager.set_precision('float64')  # Higher precision
```

### JIT Compilation

```python
from idealist.core.device import DeviceManager
DeviceManager.enable_jit(False)  # Disable for debugging
```

### Memory Management

```python
import os
# Reduce GPU memory pre-allocation
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
```
