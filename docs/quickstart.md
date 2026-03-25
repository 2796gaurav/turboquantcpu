# Quick Start

This guide will get you up and running with TurboQuantCPU in 5 minutes.

## Basic Usage

### 1. Create a Quantizer

```python
from turboquantcpu import TurboQuantizer, TurboMode

quantizer = TurboQuantizer(
    head_dim=128,        # Head dimension (e.g., 64, 128)
    num_kv_heads=8,      # Number of KV heads
    mode=TurboMode.PROD, # Unbiased inner product
    bits=4               # Bits per coordinate
)
```

### 2. Compress Keys

```python
import numpy as np

# Generate or load keys: (seq_len, num_heads, head_dim)
seq_len = 4096
keys = np.random.randn(seq_len, 8, 128).astype(np.float32)

# Compress
state = quantizer.compress(keys)

# Check compression ratio
orig_bytes = seq_len * 8 * 128 * 2  # FP16
comp_bytes = state.memory_bytes()
ratio = orig_bytes / comp_bytes
print(f"Compressed {ratio:.1f}×")
```

### 3. Compute Attention Scores

```python
# Generate query: (batch, num_heads, head_dim)
query = np.random.randn(1, 8, 128).astype(np.float32)

# Compute scores: (batch, num_heads, seq_len)
scores = quantizer.scores(query, state)
```

## Complete Example

```python
import numpy as np
from turboquantcpu import TurboQuantizer, TurboMode, QJLQuantizer

# Configuration
head_dim = 128
num_heads = 8
seq_len = 4096

# Generate synthetic KV cache
keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)

# Method 1: Turbo-PROD (Recommended)
print("=== Turbo-PROD ===")
turbo = TurboQuantizer(
    head_dim=head_dim,
    num_kv_heads=num_heads,
    mode=TurboMode.PROD,
    bits=4
)
state_turbo = turbo.compress(keys)
print(f"Compression: {seq_len * num_heads * head_dim * 2 / state_turbo.memory_bytes():.1f}×")

# Method 2: QJL (Maximum compression)
print("\n=== QJL ===")
qjl = QJLQuantizer(head_dim, num_heads, layer_idx=0)
state_qjl = qjl.compress(keys)
print(f"Compression: {seq_len * num_heads * head_dim * 2 / state_qjl.memory_bytes():.1f}×")

# Compare quality
query = np.random.randn(1, num_heads, head_dim).astype(np.float32)
true_scores = np.einsum("bhd,shd->bhs", query, keys)

turbo_scores = turbo.scores(query, state_turbo)
qjl_scores = qjl.scores(query, state_qjl)

turbo_mae = np.abs(turbo_scores - true_scores).mean()
qjl_mae = np.abs(qjl_scores - true_scores).mean()

print(f"\nTurbo-PROD MAE: {turbo_mae:.4f}")
print(f"QJL MAE: {qjl_mae:.4f}")
```

## Modes Explained

| Mode | Bits | Compression | Best For |
|------|------|-------------|----------|
| `TurboMode.QJL` | 1 | ~14× | Maximum compression |
| `TurboMode.MSE` | 2-8 | ~2-8× | Best reconstruction |
| `TurboMode.PROD` | 2-8 | ~2-4× | Unbiased attention |

## Next Steps

- **[HuggingFace Integration](huggingface.md)** — Use with real models
- **[Algorithms](algorithms.md)** — Understand the math
- **[API Reference](api.md)** — Complete API documentation
