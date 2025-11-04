# Troubleshooting Guide

## Diagnostic Tools

### Check your installation
```python
from idealist import check_installation
check_installation()
```

### Check available devices
```python
from idealist import print_device_info
print_device_info()
```

## Common Issues

### GPU requested but no GPU devices found

**Symptom:** Model runs on CPU despite `device='gpu'`

**Cause:** You have CPU-only JAX installed

**Solution:**
```bash
pip uninstall jax jaxlib idealist
pip install idealist[cuda12]  # or cuda11
```

Check CUDA version: `nvidia-smi`

### ImportError: DLL load failed (Windows)

**Symptom:** Import errors on Windows with GPU

**Solutions:**
1. Install Visual C++ Redistributable (2019 or later)
2. Ensure CUDA Toolkit is installed and matches JAX version
3. Try fresh conda environment

### Version mismatch between jax and jaxlib

**Symptom:** Import errors or warnings about version mismatch

**Solution:**
```bash
pip install --upgrade jax jaxlib
# Or for GPU:
pip install --upgrade 'jax[cuda12]'
```

### Out of memory errors on GPU

**Solutions:**
1. Reduce batch size or model complexity
2. Use CPU for small models (often faster due to transfer overhead)
3. Enable XLA memory optimization:
   ```python
   import os
   os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
   ```

### JAX not detecting all CPU cores

**Symptom:** Only 1 CPU device available despite multiple cores

**Solution:** idealist automatically configures this. If issues persist:
```bash
export IDEALIST_MAX_CPU_CORES=8  # Set to your core count
python your_script.py
```

See [Configuration Guide](configuration.md) for more details.
