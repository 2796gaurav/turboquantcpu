# TurboQuantCPU Examples

This directory contains practical examples demonstrating how to use TurboQuantCPU for KV cache compression in LLM inference.

## Quick Start

```bash
# Install TurboQuantCPU
pip install turboquantcpu

# Run the getting started example
python 01_getting_started.py
```

## Examples Overview

| Example | What You'll Learn | Time |
|---------|------------------|------|
| `01_getting_started.py` | Basic usage with HuggingFace models | 2 min |
| `02_quantization_modes.py` | Compare all 4 quantization modes | 3 min |
| `03_long_context.py` | Handle very long contexts | 2 min |
| `04_batch_processing.py` | Process multiple sequences efficiently | 2 min |
| `05_api_reference.py` | Complete API with all options | 3 min |

## Running Examples

Each example is self-contained and can be run independently:

```bash
# One-line KV cache compression
python 01_getting_started.py

# Compare quantization modes
python 02_quantization_modes.py

# Long context demonstration
python 03_long_context.py

# Batch processing
python 04_batch_processing.py

# Complete API reference
python 05_api_reference.py
```

## What is KV Cache Compression?

When LLMs generate text, they store Key-Value (KV) pairs for each token. This cache grows linearly with sequence length:

```
Memory = 2 × layers × heads × head_dim × seq_len × 2 bytes
```

For example, a 7B model at 8K context needs ~1 GB just for KV cache!

**TurboQuantCPU compresses this by 4-14×** with mathematical guarantees on quality.

## Which Example Should I Start With?

### New to TurboQuantCPU?
→ Start with `01_getting_started.py`

### Choosing the right compression mode?
→ Run `02_quantization_modes.py`

### Working with long documents?
→ See `03_long_context.py`

### Processing many sequences?
→ Check `04_batch_processing.py`

### Building custom integrations?
→ Reference `05_api_reference.py`

## Common Use Cases

### Document Q&A
```python
# Process 32K token documents on 16GB RAM
from turboquantcpu import patch_model
cache = patch_model(model, mode="prod", bits=4)
```

### API Server
```python
# Serve multiple models efficiently
from turboquantcpu import patch_model
cache = patch_model(model, mode="qjl")  # 14× compression
```

### Batch Inference
```python
# Process datasets with limited memory
from turboquantcpu import patch_model
cache = patch_model(model, mode="prod", bits=4)
# Process batch of sequences...
```

## Troubleshooting

### Out of Memory?
- Use `mode="qjl"` for maximum compression (14×)
- Reduce `max_seq_len` for sliding window
- Use `value_mode="int4"` for value compression

### Need Maximum Quality?
- Use `mode="prod"` (provably unbiased)
- Use `bits=8` for near-lossless compression
- Use `value_mode="int8"` for values

### Speed Concerns?
- Compression overhead is typically 5-15%
- Benefits increase with sequence length
- C extensions auto-enable for AVX2/AVX-512

## Next Steps

1. Run the examples in order (01 → 05)
2. Adapt the code to your use case
3. Check the [benchmarks](../benchmarks/) for detailed comparisons
4. See the main [README](../README.md) for complete documentation

## Support

- GitHub Issues: https://github.com/2796gaurav/turboquantcpu/issues
- Documentation: https://2796gaurav.github.io/turboquantcpu/
