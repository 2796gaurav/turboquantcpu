# TurboQuantCPU Benchmarks

This directory contains comprehensive benchmarking scripts for TurboQuantCPU.

## Available Benchmarks

### 1. Comprehensive Benchmark (`comprehensive_benchmark.py`)

The main benchmark suite testing 3 HuggingFace models:
- **Qwen/Qwen3.5-0.8B** (0.8B params, Alibaba)
- **meta-llama/Llama-3.2-1B** (1B params, Meta)
- **google/gemma-2-2b-it** (2B params, Google)

**Metrics measured:**
- Compression ratio (1-bit, 2-bit, 4-bit)
- Inference speed (tokens/sec, overhead %)
- Quality preservation (perplexity)
- Long-context retrieval (needle-in-haystack)

**Run:**
```bash
python comprehensive_benchmark.py
```

**Output:**
- JSON results in `results/` directory
- Console summary tables
- ~30 minutes runtime on Intel i7

### 2. Needle-in-Haystack (`needle_in_haystack.py`)

Long-context retrieval benchmark based on the original TurboQuant paper.

**Tests:**
- Context lengths: 512, 1024, 2048 tokens
- Depths: 0%, 25%, 50%, 75%, 100%
- Hidden fact retrieval accuracy

**Run:**
```bash
python needle_in_haystack.py
```

### 3. Competitor Comparison (`competitor_comparison.py`)

Honest comparison with alternative quantization approaches:
- **llama.cpp** (GGUF Q4_K_M, Q5_0)
- **KIVI** (2-bit asymmetric)
- **KVQuant** (3-bit per-channel)
- **H2O** (Heavy-Hitter Oracle eviction)
- **SnapKV/PyramidKV** (Token eviction)
- **PolarQuant** (Polar transformation)

**Run:**
```bash
python competitor_comparison.py
```

### 4. Validation (`validate_benchmarks.py`)

Script integrity checker ensuring all benchmarks can run.

**Run:**
```bash
python validate_benchmarks.py
```

## Directory Structure

```
benchmarks/
├── README.md                    # This file
├── comprehensive_benchmark.py   # Main benchmark suite
├── needle_in_haystack.py        # Long context retrieval
├── competitor_comparison.py     # Competitor analysis
├── create_plots.py              # Plot generation from results
├── validate_benchmarks.py       # Script validation
├── sanity_benchmark.py          # Quick sanity check
├── results/                     # Benchmark outputs (JSON)
└── plots/                       # Generated visualizations
```

## Running All Benchmarks

```bash
# 1. Validate scripts
python validate_benchmarks.py

# 2. Run comprehensive benchmark
python comprehensive_benchmark.py

# 3. Generate plots
python create_plots.py

# 4. View competitor comparison
python competitor_comparison.py
```

## Interpreting Results

### Compression Ratio
```
Ratio = FP16_size / compressed_size
```
Higher is better. Target: 7× for 4-bit, 14× for 1-bit.

### Speed Overhead
```
Overhead = (compressed_time / baseline_time - 1) × 100%
```
Negative = faster than baseline. Memory bandwidth savings can outweigh compression cost.

### Quality Preservation
```
Quality_Change = (ppl_compressed / ppl_baseline - 1) × 100%
```
Target: <1% for practical use, 0% is ideal.

### Needle-in-Haystack Score
```
Score = correct_retrievals / total_tests × 100%
```
Should be 100% for reliable long-context usage.

## Benchmarking Notes

1. **CPU-only**: All benchmarks run on CPU to ensure fair comparison
2. **Cold start**: First run may be slower due to model download
3. **Memory**: Ensure ~8GB free RAM for 2B model benchmarks
4. **Reproducibility**: Seeds are set for consistent results
5. **Real measurements**: All metrics are actual measurements, not estimates

## Adding New Models

To benchmark additional models, edit `comprehensive_benchmark.py`:

```python
BENCHMARK_MODELS = {
    "YourModel-Name": {
        "repo_id": "org/model-name",
        "provider": "Company",
        "params": "X.XB",
        "layers": XX,
        "kv_heads": X,
        "head_dim": XXX,
        "license": "License Name",
    },
    # ... existing models
}
```

## Citation

If you use these benchmarks in your research, please cite:

```bibtex
@inproceedings{zandieh2026turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  booktitle={ICLR},
  year={2026}
}
```
