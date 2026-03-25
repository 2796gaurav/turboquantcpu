# TurboQuantCPU

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/CPU-AVX2%2FAVX--512-green.svg" alt="CPU: AVX2/AVX-512">
</p>

**CPU-optimized KV cache quantization for LLM inference with mathematical guarantees**

TurboQuantCPU implements the TurboQuant algorithm (ICLR 2026) for provably optimal KV cache compression. Achieve **4-14× memory reduction** with mathematical guarantees on quality—enabling longer contexts and larger batches on CPU-only deployments.

## Key Features

- **:zap: High Compression**: Up to 14× memory savings with QJL 1-bit quantization
- **:white_check_mark: Mathematical Guarantees**: Only library with provably unbiased attention scores
- **:gear: CPU Optimized**: AVX2/AVX-512 SIMD kernels for maximum performance
- **:hugging_face: HuggingFace Native**: One-line integration with any Transformers model
- **:books: Research-Backed**: Based on peer-reviewed ICLR 2026 and AAAI 2025 papers

## Quick Example

```python
from turboquantcpu import TurboQuantizer, TurboMode
import numpy as np

# Create quantizer
quantizer = TurboQuantizer(
    head_dim=128,
    num_kv_heads=8,
    mode=TurboMode.PROD,
    bits=4
)

# Compress keys
keys = np.random.randn(4096, 8, 128).astype(np.float32)
state = quantizer.compress(keys)

# Check compression
orig_size = 4096 * 8 * 128 * 2  # FP16 bytes
comp_size = state.memory_bytes()
print(f"Compression: {orig_size / comp_size:.1f}×")  # 4×
```

## Where TurboQuantCPU Shines

| Use Case | Benefit |
|----------|---------|
| **Long Context** | Run 128K+ context on consumer CPUs |
| **Multi-Model Serving** | Serve 3-4× more models on same hardware |
| **Edge/On-Device** | Deploy LLMs on resource-constrained devices |
| **Provable Quality** | Mathematical guarantees on attention accuracy |

## Installation

```bash
pip install turboquantcpu
```

With HuggingFace support:

```bash
pip install turboquantcpu[hf]
```

## Documentation Structure

- **[Installation](installation.md)** — Detailed installation instructions
- **[Quick Start](quickstart.md)** — Get started in 5 minutes
- **[HuggingFace Integration](huggingface.md)** — Use with any Transformers model
- **[Algorithms](algorithms.md)** — Deep dive into the math
- **[API Reference](api.md)** — Complete API documentation
- **[Benchmarks](benchmarks.md)** — Performance comparisons

## Citation

If you use TurboQuantCPU in your research, please cite:

```bibtex
@inproceedings{zandieh2026turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  booktitle={ICLR},
  year={2026}
}
```

## License

MIT License — See [License](license.md) for details.
