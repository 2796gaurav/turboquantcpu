# Installation

## Requirements

- **Python**: 3.9 or higher
- **CPU**: x86_64 with AVX2 (AVX-512 recommended) or ARM64 with NEON
- **RAM**: 4GB minimum (depends on model size)
- **OS**: Linux, macOS, Windows (WSL)

## Basic Installation

Install from PyPI:

```bash
pip install turboquantcpu
```

## Optional Dependencies

### HuggingFace Transformers Support

For HuggingFace model integration:

```bash
pip install turboquantcpu[hf]
```

This installs:
- `transformers>=4.38`
- `accelerate>=0.26`
- `safetensors>=0.4`

### Development Dependencies

For running tests and benchmarks:

```bash
pip install turboquantcpu[dev]
```

### All Extras

Install everything:

```bash
pip install turboquantcpu[all]
```

## Building C Extensions (Recommended)

For maximum performance, build the SIMD-optimized C kernels from source:

```bash
# Clone repository
git clone https://github.com/2796gaurav/turboquantcpu.git
cd turboquantcpu

# Build C extensions
python setup.py build_ext --inplace

# Install in development mode
pip install -e .
```

### Compiler Requirements

- **Linux**: GCC 7+ or Clang 6+
- **macOS**: Xcode Command Line Tools
- **Windows**: Visual Studio 2019+ or MinGW-w64

## Verify Installation

Check that everything is working:

```python
from turboquantcpu import print_cpu_capabilities
print_cpu_capabilities()
```

Expected output:

```
════════════════════════════════════════════════════
  TurboQuantCPU v0.0.3 — Host CPU Capabilities
════════════════════════════════════════════════════
  CPU     : Intel(R) Core(TM) i7-...
  Arch    : x86_64  (intel)
  Cores   : 8 phys / 16 logical
  SIMD    : AVX2+FMA  (8×f32/instr)
  OpenMP  : 16 threads
  Kernel  : tqcpu_kernels v0.3.0 [avx2+fma] OpenMP=16
```

## Troubleshooting

### C Extension Not Found

If you see a warning about C extensions:

```
C extension not compiled. Falling back to NumPy implementations.
```

The package will still work, but slower. To fix:

```bash
python setup.py build_ext --inplace
```

### Import Errors

Make sure you're using Python 3.9+:

```bash
python --version
```

### SIMD Not Detected

Check your CPU flags:

```bash
# Linux
cat /proc/cpuinfo | grep flags | head -1

# macOS
sysctl -a | grep machdep.cpu.features
```

Look for `avx2` in the output. If missing, the package will fall back to NumPy implementations.
