# Complete Parameter Reference

This guide provides exhaustive documentation for every parameter in TurboQuantCPU.

## TurboQuantizer Parameters

### `head_dim` (int, required)

**Description:** The dimension of each attention head vector.

**Common Values:**
- 64: Smaller models (Qwen 0.5B, some distilled models)
- 128: Standard size (Llama 2/3, Mistral, Qwen 7B, most 7B-13B models)
- 256: Larger models (Gemma 2 9B, some specialized architectures)

**Technical Details:**
- Internally padded to next power of 2 for FWHT efficiency
- d=128 → padded to 128 (no padding)
- d=96 → padded to 128 (32 zeros added)

**Performance Impact:**
- Higher dimensions = more computation per token
- But also better compression ratios (fixed overhead amortized)

**Example:**
```python
# For Llama-3-8B (head_dim=128)
quantizer = TurboQuantizer(head_dim=128, num_kv_heads=8, ...)

# For Phi-3-mini (head_dim=96, padded to 128)
quantizer = TurboQuantizer(head_dim=96, num_kv_heads=32, ...)
```

---

### `num_kv_heads` (int, required)

**Description:** Number of key/value heads in the model.

**Also Known As:**
- GQA (Grouped Query Attention) heads
- KV heads
- num_key_value_heads (HuggingFace config)

**Common Patterns:**

| Model Size | Q Heads | KV Heads | GQA Ratio |
|------------|---------|----------|-----------|
| 0.5B-1B | 14-16 | 2-4 | 4:1 to 8:1 |
| 7B | 32 | 8 | 4:1 |
| 8B | 32 | 8 | 4:1 |
| 70B | 64 | 8 | 8:1 |

**Why It Matters:**
- Fewer KV heads = more compression
- GQA reduces memory without quality loss
- Query heads broadcast to KV heads

**Example:**
```python
# Llama-3-8B: 32 query heads, 8 KV heads
quantizer = TurboQuantizer(
    head_dim=128,
    num_kv_heads=8,      # KV heads
    num_q_heads=32,      # Query heads
    ...
)
```

---

### `num_q_heads` (int, optional)

**Description:** Number of query heads. Defaults to `num_kv_heads` if not specified.

**When to Specify:**
- Always specify for GQA models
- Can omit for MHA (Multi-Head Attention) where Q heads = KV heads

**Impact:**
- Used for broadcasting in attention computation
- Does not affect compression (only keys are compressed)

---

### `layer_idx` (int, default=0)

**Description:** Index of the transformer layer. Used for deterministic random rotations.

**Why It Matters:**
- Each layer gets different random seeds
- Ensures diversity across layers
- Required for proper attention computation

**Best Practice:**
```python
# Create one quantizer per layer
quantizers = [
    TurboQuantizer(
        head_dim=128,
        num_kv_heads=8,
        layer_idx=i,  # Unique per layer
        ...
    )
    for i in range(32)  # 32 layers
]
```

---

### `mode` (TurboMode or str, default="prod")

**Description:** Quantization algorithm selection.

**Options:**

#### `"qjl"` or `TurboMode.QJL`
- **Bits:** 1-bit per element
- **Compression:** ~14×
- **Best for:** Maximum compression, long contexts
- **Trade-off:** Higher variance in estimates

#### `"mse"` or `TurboMode.MSE`
- **Bits:** Configurable (2-8)
- **Compression:** ~2-8×
- **Best for:** Best reconstruction quality
- **Trade-off:** Biased attention scores

#### `"prod"` or `TurboMode.PROD` (Recommended)
- **Bits:** Configurable (2-8)
- **Compression:** ~2-4×
- **Best for:** Unbiased attention + good compression
- **Trade-off:** Slower than MSE (two-stage)

**Selection Guide:**
```
Need maximum compression? → "qjl"
Need best quality? → "mse"
Need unbiased + good compression? → "prod" (default)
```

---

### `bits` (int, default=4)

**Description:** Number of bits per coordinate.

**Valid Range:** 1-8

**Impact by Mode:**

| Mode | bits | Actual Bits | Compression |
|------|------|-------------|-------------|
| QJL | 1 | 1 | ~14× |
| MSE | 2 | 2 | ~2× |
| MSE | 4 | 4 | ~4× |
| PROD | 4 | 3+1 | ~1.7-4× |

**Recommendation:**
- Start with 4 bits for most use cases
- Use 2 bits for higher compression (slight quality loss)
- Use 1 bit only with QJL mode

---

### `outlier_frac` (float, default=0.0)

**Description:** Fraction of channels to allocate extra bits for outliers.

**Range:** 0.0 to 1.0

**How It Works:**
- 0.0: All channels use `bits` bits
- 0.1: Top 10% of channels use `bits+1` bits
- Creates "fractional-bit" quantization (e.g., 2.5-bit)

**When to Use:**
- Models with known outlier channels
- When some dimensions have much higher variance
- For achieving specific average bit-rates

**Example:**
```python
# 2.5-bit quantization (50% at 2 bits, 50% at 3 bits)
quantizer = TurboQuantizer(
    head_dim=128,
    num_kv_heads=8,
    bits=2,
    outlier_frac=0.5,  # Half channels get 3 bits
    ...
)
```

---

### `n_threads` (int, default=0)

**Description:** Number of threads for parallel compression.

**Special Values:**
- 0: Auto (min(num_kv_heads, 4))
- N: Use N threads explicitly

**Performance Notes:**
- Parallelism is per-head
- Diminishing returns beyond num_kv_heads
- Memory bandwidth often the bottleneck, not CPU

**Recommendation:**
- Leave at 0 for automatic tuning
- Set to 1 for deterministic/debugging

---

## QJLQuantizer Parameters

### `seed_offset` (int, default=0)

**Description:** Additional seed offset for random rotation.

**Use Case:**
- When you need different rotations than TurboQuant in same layer
- Ensures independence between QJL and Turbo stages in PROD mode

---

### `normalise_keys` (bool, default=True)

**Description:** Whether to L2-normalize keys before quantization.

**Should Always Be True:**
- Required for unbiased estimator
- Norm stored separately and restored during scoring

**When False:**
- Only for special research purposes
- Breaks mathematical guarantees

---

## PolarQuantizer Parameters

### `n_r_bits` (int, default=2)

**Description:** Bits for radius quantization.

**Impact:**
- More bits = finer radius resolution
- Radius is always positive (Rayleigh distribution)

---

### `n_theta_bits` (int, default=2)

**Description:** Bits for angle quantization.

**Impact:**
- More bits = finer angular resolution
- Angle is in [0, 2π)

**Total Bits:**
```
total_bits = n_r_bits + n_theta_bits  # per pair (2 coordinates)
bits_per_coord = total_bits / 2
```

**Common Configurations:**

| n_r_bits | n_theta_bits | Total | bits/coord | Compression |
|----------|--------------|-------|------------|-------------|
| 2 | 2 | 4 | 2 | ~4× |
| 3 | 3 | 6 | 3 | ~2.7× |
| 2 | 3 | 5 | 2.5 | ~3.2× |

---

## CacheConfig Parameters

### `max_seq_len` (int, default=32768)

**Description:** Maximum sequence length before sliding window eviction.

**Behavior:**
- Cache grows until max_seq_len
- Then oldest tokens are dropped
- Prevents unbounded memory growth

**Recommendation:**
- Set to model's max context length
- Larger = more memory but longer context

---

### `value_mode` (str, default="int8")

**Description:** Quantization mode for value cache.

**Options:**
- `"fp16"`: No compression (2 bytes/element)
- `"int8"`: 8-bit symmetric (1 byte/element)
- `"int4"`: 4-bit asymmetric (0.5 bytes/element)

**Impact:**
- Values are easier to quantize than keys
- INT8 typically has <0.1% perplexity impact
- INT4 gives 2× compression with minor quality loss

---

### `value_group_size` (int, default=32)

**Description:** Group size for value quantization scaling.

**Trade-off:**
- Smaller groups = better accuracy, more overhead
- Larger groups = worse accuracy, less overhead

**Recommendation:**
- 32 is good default
- 64 for higher compression
- 16 for better quality

---

### `h2o_config` (H2OConfig, optional)

**Description:** Configuration for H2O sparse attention.

**When Enabled:**
- Tracks attention scores across tokens
- Evicts low-attention ("light") tokens
- Keeps high-attention ("heavy hitter") tokens

**Parameters:**
- `heavy_budget`: Number of heavy hitter tokens to keep
- `recent_budget`: Number of recent tokens to always keep

---

## PatchConfig Parameters

### `skip_layers` (List[int], default=[])

**Description:** Layer indices to skip patching.

**Use Cases:**
- Skip first layer (often has different patterns)
- Skip specific layers for debugging
- Gradual deployment (patch half the layers)

---

### `verbose` (bool, default=True)

**Description:** Whether to print patching information.

**Output Includes:**
- Model architecture detected
- Number of layers patched
- Compression estimates
- Memory savings projection

---

## Complete Configuration Examples

### Example 1: Maximum Compression (Long Context)

```python
from turboquantcpu import TurboQuantizer, TurboMode

# QJL for 14× compression
quantizer = TurboQuantizer(
    head_dim=128,
    num_kv_heads=8,
    num_q_heads=32,
    mode=TurboMode.QJL,
    bits=1,
    layer_idx=0,
)

# Expected: 14× compression
# Use case: 128K+ context on limited memory
```

### Example 2: Best Quality (Accuracy Critical)

```python
# Turbo-MSE with 4 bits
quantizer = TurboQuantizer(
    head_dim=128,
    num_kv_heads=8,
    mode=TurboMode.MSE,
    bits=4,
    outlier_frac=0.1,  # Extra precision for outliers
    layer_idx=0,
)

# Expected: 4× compression, minimal quality loss
# Use case: Production serving where accuracy matters
```

### Example 3: Balanced (Recommended)

```python
# Turbo-PROD with 4 bits (default)
quantizer = TurboQuantizer(
    head_dim=128,
    num_kv_heads=8,
    num_q_heads=32,
    mode=TurboMode.PROD,  # Unbiased
    bits=4,
    layer_idx=0,
)

# Expected: 3-4× compression, unbiased attention
# Use case: General purpose, best trade-off
```

### Example 4: Model-Specific (Qwen 7B)

```python
# Qwen2.5-7B configuration
quantizer = TurboQuantizer(
    head_dim=128,
    num_kv_heads=4,      # GQA: 4 KV heads
    num_q_heads=28,      # 28 query heads
    mode="prod",
    bits=4,
    layer_idx=layer_idx, # Set per layer
)
```

### Example 5: Full Model Cache

```python
from turboquantcpu import CacheConfig, CompressedKVCache

config = CacheConfig(
    num_layers=32,
    num_kv_heads=8,
    num_q_heads=32,
    head_dim=128,
    max_seq_len=32768,
    mode="prod",
    bits=4,
    value_mode="int8",
    value_group_size=32,
)

cache = CompressedKVCache.from_config(config)
```

---

## Parameter Tuning Guide

### For Maximum Compression

```python
{
    "mode": "qjl",
    "bits": 1,
    "value_mode": "int4",
}
# Expected: 14× keys + 2× values = ~8× total
```

### For Maximum Quality

```python
{
    "mode": "mse",
    "bits": 4,
    "outlier_frac": 0.1,
    "value_mode": "int8",
}
# Expected: 4× compression, <0.5% quality loss
```

### For Long Context

```python
{
    "mode": "prod",
    "bits": 4,
    "max_seq_len": 131072,
    "h2o_config": H2OConfig(heavy_budget=1024, recent_budget=256),
}
# Expected: 4× compression + smart eviction
```

### For Edge Devices

```python
{
    "mode": "qjl",
    "bits": 1,
    "n_threads": 1,  # Save power
    "value_mode": "int4",
}
# Expected: Maximum memory savings, single-threaded
```

---

## Performance Impact Reference

| Parameter | Memory Impact | Speed Impact | Quality Impact |
|-----------|--------------|--------------|----------------|
| `head_dim` | Linear | Linear | None |
| `mode="qjl"` | 14× smaller | Slower | Higher variance |
| `mode="mse"` | 4× smaller | Medium | Biased |
| `mode="prod"` | 3× smaller | Slowest | Unbiased |
| `bits` | Linear | Linear | Exponential |
| `outlier_frac` | Minor | Minor | Positive |
| `n_threads` | None | Parallel speedup | None |
