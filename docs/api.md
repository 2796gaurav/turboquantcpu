# API Reference

## Core Quantizers

### TurboQuantizer

```python
class TurboQuantizer(
    head_dim: int,
    num_kv_heads: int,
    num_q_heads: Optional[int] = None,
    layer_idx: int = 0,
    mode: Union[TurboMode, str] = TurboMode.PROD,
    bits: int = 4,
    outlier_frac: float = 0.0,
    n_threads: int = 0,
)
```

Main quantizer implementing TurboQuant algorithms.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `head_dim` | int | — | Head vector dimension |
| `num_kv_heads` | int | — | Number of KV heads (GQA) |
| `num_q_heads` | int | None | Number of query heads |
| `layer_idx` | int | 0 | Layer index (for deterministic rotations) |
| `mode` | TurboMode/str | PROD | Quantization mode |
| `bits` | int | 4 | Bits per coordinate |
| `outlier_frac` | float | 0.0 | Fraction of outlier channels |
| `n_threads` | int | 0 | Thread pool size (0=auto) |

**Methods:**

#### compress(keys)

Compress key vectors.

```python
def compress(
    keys: Union[np.ndarray, torch.Tensor]
) -> TurboState
```

**Parameters:**
- `keys`: (seq_len, num_kv_heads, head_dim) float32/float16

**Returns:**
- `TurboState`: Compressed representation

#### scores(query, state)

Compute attention scores.

```python
def scores(
    query: Union[np.ndarray, torch.Tensor],
    state: TurboState
) -> np.ndarray
```

**Parameters:**
- `query`: (batch, num_q_heads, head_dim)
- `state`: TurboState from compress()

**Returns:**
- Scores: (batch, num_q_heads, seq_len)

### QJLQuantizer

```python
class QJLQuantizer(
    head_dim: int,
    num_kv_heads: int,
    num_q_heads: Optional[int] = None,
    layer_idx: int = 0,
    seed_offset: int = 0,
    normalise_keys: bool = True,
    n_threads: int = 0,
)
```

1-bit QJL quantizer for maximum compression.

### PolarQuantizer

```python
class PolarQuantizer(
    head_dim: int,
    num_kv_heads: int,
    num_q_heads: Optional[int] = None,
    layer_idx: int = 0,
    n_r_bits: int = 2,
    n_theta_bits: int = 2,
    n_threads: int = 0,
)
```

Polar quantization for outlier resistance.

## Enums

### TurboMode

```python
class TurboMode(str, Enum):
    QJL = "qjl"      # 1-bit maximum compression
    MSE = "mse"      # MSE-optimal quantization
    PROD = "prod"    # Unbiased inner product (recommended)
```

## KV Cache Management

### CompressedKVCache

```python
class CompressedKVCache(
    layers: List[LayerCache],
    config: CacheConfig
)
```

Full model compressed KV cache.

**Class Methods:**

#### from_config(config)

```python
@classmethod
def from_config(cls, config: CacheConfig) -> CompressedKVCache
```

Create cache from configuration.

### CacheConfig

```python
class CacheConfig(
    num_layers: int,
    num_kv_heads: int,
    num_q_heads: int,
    head_dim: int,
    max_seq_len: int = 32768,
    mode: str = "prod",
    bits: int = 4,
    value_mode: str = "int8",
    value_group_size: int = 32,
    outlier_frac: float = 0.0,
    h2o_config: Optional[H2OConfig] = None,
    memory_budget_gb: float = 4.0,
)
```

## HuggingFace Integration

### patch_model

```python
def patch_model(
    model: nn.Module,
    mode: str = "prod",
    bits: int = 4,
    max_seq_len: int = 32768,
    verbose: bool = True,
    cfg: Optional[PatchConfig] = None,
) -> CompressedKVCache
```

Patch a HuggingFace model for KV cache compression.

**Parameters:**
- `model`: HuggingFace AutoModelForCausalLM
- `mode`: Quantization mode ("qjl", "mse", "prod", "polar")
- `bits`: Bits per coordinate
- `max_seq_len`: Maximum sequence length
- `verbose`: Print patching info
- `cfg`: Detailed configuration (optional)

**Returns:**
- `CompressedKVCache`: Cache instance attached to model

### unpatch_model

```python
def unpatch_model(model: nn.Module) -> int
```

Remove all TurboQuantCPU patches.

**Returns:**
- Number of hooks removed

## Utilities

### print_cpu_capabilities

```python
def print_cpu_capabilities() -> None
```

Print system CPU capabilities and detected features.

### cpu_info

```python
@functools.lru_cache(maxsize=1)
def cpu_info() -> CPUInfo
```

Get CPU information (cached).

**Returns:**
- `CPUInfo` with fields: brand, arch, simd features, etc.

## Benchmarking

### run_full_benchmark

```python
def run_full_benchmark(
    head_dim: int = 128,
    num_heads: int = 8,
    verbose: bool = True,
) -> Dict[str, dict]
```

Run comprehensive benchmark suite.

**Returns:**
- Dictionary with correctness, memory, speed benchmarks

### run_correctness_check

```python
def run_correctness_check(
    head_dim: int = 128,
    num_heads: int = 8,
    seq_len: int = 512,
    n_trials: int = 30,
    verbose: bool = True,
) -> Dict[str, float]
```

Verify mathematical guarantees.

## Exceptions

All functions may raise:

- `ValueError`: Invalid parameters
- `RuntimeError`: Operation on empty cache
- `ImportError`: Missing optional dependencies
