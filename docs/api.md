# API Reference

Complete reference for all TurboQuantCPU public APIs.

## High-Level API (Recommended)

### `patch_model`

```python
patch_model(model, mode="prod", bits=4, **kwargs)
```

Patches a HuggingFace model to use compressed KV cache.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | PreTrainedModel | required | HuggingFace model to patch |
| `mode` | str | `"prod"` | Quantization mode: `"prod"`, `"mse"`, `"qjl"`, `"polar"` |
| `bits` | int | `4` | Bits per coordinate (1, 2, 4, 8) |
| `max_seq_len` | int | `None` | Maximum sequence length |
| `value_mode` | str | `"int8"` | Value compression: `"int8"`, `"int4"`, `"fp16"` |

**Returns:** `CompressedKVCache` instance

**Example:**

```python
from turboquantcpu import patch_model

# Recommended: PROD mode with 4-bit
patch_model(model, mode="prod", bits=4)

# Maximum compression: QJL 1-bit
patch_model(model, mode="qjl")

# Long context with custom value compression
patch_model(model, mode="prod", bits=4, max_seq_len=32768, value_mode="int4")
```

---

### `unpatch_model`

```python
unpatch_model(model)
```

Restores a patched model to its original state.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | PreTrainedModel | Model previously patched with `patch_model()` |

**Returns:** `None`

**Example:**

```python
from turboquantcpu import patch_model, unpatch_model

cache = patch_model(model, mode="prod")
# ... use model ...
unpatch_model(model)  # Restore original
```

---

### `PatchConfig`

Configuration class for advanced patching options.

```python
from turboquantcpu import PatchConfig

config = PatchConfig(
    mode="prod",
    bits=4,
    max_seq_len=32768,
    value_mode="int8",
    h2o_budget=None,  # Heavy-Hitter budget (experimental)
    group_size=None,  # Group size for grouped quantization
)

cache = patch_model(model, cfg=config)
```

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | TurboMode/str | `TurboMode.PROD` | Quantization mode |
| `bits` | int | 4 | Bits per coordinate |
| `max_seq_len` | int | 32768 | Maximum sequence length |
| `value_mode` | str | `"int8"` | Value compression mode |
| `h2o_budget` | int/None | None | Heavy-Hitter Oracle budget |
| `group_size` | int/None | None | Group size for grouped quantization |

---

## Quantization Modes

### `TurboMode`

Enum for quantization modes.

```python
from turboquantcpu import TurboMode

TurboMode.PROD   # Unbiased attention (recommended)
TurboMode.MSE    # Best reconstruction quality
TurboMode.QJL    # 1-bit maximum compression
TurboMode.POLAR  # Outlier-resistant
```

**Mode Details:**

| Mode | Bits | Compression | Unbiased | Best For |
|------|------|-------------|----------|----------|
| `PROD` | 4 | 7.3× | ✅ Yes | General use, quality critical |
| `MSE` | 4 | 7.3× | ❌ No | Best reconstruction |
| `QJL` | 1 | 14× | ✅ Yes | Extreme memory constraints |
| `POLAR` | 4 | 7.3× | ❌ No | Outlier-heavy models |

---

## Low-Level API

### `TurboQuantizer`

Direct access to TurboQuant quantization.

```python
from turboquantcpu import TurboQuantizer, TurboMode

quantizer = TurboQuantizer(
    head_dim=128,
    num_kv_heads=8,
    mode=TurboMode.PROD,
    bits=4,
)

# Compress keys
state = quantizer.compress(keys)

# Compute scores
scores = quantizer.scores(query, state)
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `head_dim` | int | required | Dimension of each attention head |
| `num_kv_heads` | int | required | Number of KV heads |
| `mode` | TurboMode | `TurboMode.PROD` | Quantization mode |
| `bits` | int | 4 | Bits per coordinate |
| `codebook_type` | str | `"lloyd"` | Codebook type: `"lloyd"`, `"uniform"` |

**Methods:**

#### `compress(keys)`

Compress key tensors.

**Parameters:**
- `keys`: Tensor of shape `(batch, heads, seq_len, head_dim)`

**Returns:** `TurboState` compressed state

#### `scores(query, state)`

Compute attention scores.

**Parameters:**
- `query`: Query tensor `(batch, heads, 1, head_dim)`
- `state`: Compressed key state from `compress()`

**Returns:** Attention scores `(batch, heads, seq_len)`

#### `reconstruct(state)`

Reconstruct keys from compressed state.

**Parameters:**
- `state`: Compressed state

**Returns:** Reconstructed keys

---

### `QJLQuantizer`

1-bit quantization via QJL (Johnson-Lindenstrauss transform).

```python
from turboquantcpu import QJLQuantizer, QJLState

quantizer = QJLQuantizer(
    head_dim=128,
    num_hashs=1,  # Number of JL projections
)

state = quantizer.compress(keys)
scores = quantizer.scores(query, state)
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `head_dim` | int | required | Dimension of each attention head |
| `num_hashs` | int | 1 | Number of JL hash functions |

---

### `PolarQuantizer`

Polar transformation-based quantization for outlier resistance.

```python
from turboquantcpu import PolarQuantizer, PolarState

quantizer = PolarQuantizer(
    head_dim=128,
    bits=4,
)

state = quantizer.compress(keys)
scores = quantizer.scores(query, state)
```

---

## Cache Management

### `CompressedKVCache`

KV cache with compression support.

```python
from turboquantcpu import CacheConfig, LayerCache, CompressedKVCache

config = CacheConfig(
    num_layers=32,
    num_kv_heads=8,
    head_dim=128,
    max_seq_len=32768,
    kv_quantizer=quantizer,
)

cache = CompressedKVCache(config)
```

**Methods:**

#### `update(layer_idx, key_states, value_states)`

Update cache with new key-value states.

#### `get(layer_idx)`

Retrieve cached states for a layer.

#### `memory_report()`

Get memory usage statistics.

**Returns:**

```python
{
    "original_fp16_MB": 25.2,      # Original FP16 size
    "compressed_MB": 3.5,          # Compressed size
    "compression_ratio": 7.3,      # Compression ratio
}
```

---

## Utility Functions

### `fwht`

Fast Walsh-Hadamard Transform (in-place).

```python
from turboquantcpu import fwht

import torch
x = torch.randn(128)
fwht(x)  # In-place transform
```

### `cpu_features`

Get CPU feature detection information.

```python
from turboquantcpu import cpu_features

info = cpu_features()
print(info)
# {
#     "has_avx2": True,
#     "has_avx512": False,
#     "has_neon": False,
#     "has_fma": True,
#     "optimal_batch_size": 64,
# }
```

---

## Type Hints

Complete type hints for all public APIs:

```python
from typing import Optional, Literal, Union
from torch import Tensor

# Mode type
TurboModeType = Literal["prod", "mse", "qjl", "polar"]

# Value mode type
ValueModeType = Literal["int8", "int4", "fp16"]

# Configuration
def patch_model(
    model: PreTrainedModel,
    mode: Union[TurboModeType, TurboMode] = "prod",
    bits: Literal[1, 2, 4, 8] = 4,
    max_seq_len: Optional[int] = None,
    value_mode: ValueModeType = "int8",
    **kwargs
) -> CompressedKVCache: ...
```

---

## Error Handling

TurboQuantCPU raises specific exceptions for different error conditions:

| Exception | When Raised | How to Handle |
|-----------|-------------|---------------|
| `ValueError` | Invalid mode, bits, or configuration | Check parameters match allowed values |
| `RuntimeError` | Model not supported or C extension failed | Check model compatibility, rebuild extensions |
| `ImportError` | Missing dependencies | Install required packages |

**Example:**

```python
from turboquantcpu import patch_model, TurboMode

try:
    cache = patch_model(model, mode="invalid_mode")
except ValueError as e:
    print(f"Invalid mode: {e}")
    # Use valid mode instead
    cache = patch_model(model, mode="prod")
```

---

## Supported Models

TurboQuantCPU supports all HuggingFace CausalLM models:

| Architecture | Examples | Notes |
|--------------|----------|-------|
| Llama | Llama-2, Llama-3, Mistral | Full support |
| Qwen | Qwen2, Qwen2.5 | Full support |
| Gemma | Gemma, Gemma-2 | Full support |
| Phi | Phi-2, Phi-3 | Full support |
| Falcon | Falcon-7B, Falcon-40B | Full support |
| GPT-NeoX | Pythia, GPT-NeoX | Full support |

Any model using standard HuggingFace `Cache` interface is supported.
