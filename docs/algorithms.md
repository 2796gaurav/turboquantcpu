# Algorithms Deep Dive

TurboQuantCPU implements three core quantization algorithms, each optimized for different use cases.

## QJL: 1-Bit Quantized Johnson-Lindenstrauss

**Paper**: [QJL: 1-Bit Quantized JL Transform for KV Cache Quantization](https://arxiv.org/abs/2406.03482)

### Key Idea

Quantize each dimension to a single bit (sign only) after random rotation. The estimator is provably unbiased.

### Algorithm

```
1. Normalize: k̂ = k / ‖k‖
2. Rotate:    k̃ = H · diag(σ) · k̂  (FWHT)
3. Quantize:  Q(k) = sign(k̃) ∈ {-1, +1}^d
```

### Estimator

```
⟨q, k⟩ ≈ (π / 2d) · ‖k‖ · (Hσq) · sign(Hσk)
```

### Guarantees

| Property | Guarantee |
|----------|-----------|
| **Unbiased** | E[estimate] = ⟨q, k⟩ exactly |
| **MSE** | O(‖q‖²‖k‖² / d) |
| **Compression** | d/8 + 2 bytes per vector (~14× for d=128) |

### When to Use

- Maximum compression needed
- Can tolerate some variance in estimates
- Long contexts where memory is critical

## TurboQuant-MSE

**Paper**: [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)

### Key Idea

After random rotation, coordinates follow a concentrated distribution. Apply optimal Lloyd-Max scalar quantization per coordinate.

### Algorithm

```
1. Normalize: k̂ = k / ‖k‖
2. Rotate:    k̃ = √d · H · diag(σ) · k̂
3. Quantize:  q_i = argmin_c |k̃_i - c|  (Lloyd-Max codebook)
4. Store:     b-bit index + float16 norm
```

### Guarantees

| Property | Guarantee |
|----------|-----------|
| **MSE** | Within 2.7× of Shannon lower bound |
| **Optimality** | Optimal scalar quantization |
| **Bits** | b bits per coordinate |

### Lloyd-Max Codebooks

Precomputed optimal codebooks for N(0,1) source:

| Bits | Levels | SNR (dB) |
|------|--------|----------|
| 1 | 2 | 4.4 |
| 2 | 4 | 9.3 |
| 3 | 8 | 13.9 |
| 4 | 16 | 18.5 |

### When to Use

- Best reconstruction quality needed
- MSE distortion should be minimized
- Inner product bias is acceptable

## TurboQuant-PROD (Recommended)

### Key Idea

Combine (b-1)-bit MSE quantization with 1-bit QJL on the residual. This gives an unbiased estimator for inner products.

### Algorithm

```
Stage 1: (b-1)-bit MSE
  k̃ = TurboQuant_MSE^(b-1)(k)

Stage 2: 1-bit QJL on residual
  r = k - k̃
  Q_r = QJL(r)

Estimator:
  ⟨q, k⟩ ≈ ⟨q, k̃⟩ + (π/2d) · ‖r‖ · (Hσq) · sign(Hσr̂)
```

### Guarantees

| Property | Guarantee |
|----------|-----------|
| **Unbiased** | E[estimate] = ⟨q, k⟩ exactly |
| **Distortion** | Near-optimal for inner products |
| **Compression** | ~4× for 4-bit |

### When to Use

- Unbiased attention scores required
- Good compression needed
- Best overall trade-off

## PolarQuant

### Key Idea

After FWHT rotation, transform to polar coordinates (r, θ). The radius follows a Rayleigh distribution (bounded, no outliers) and angle is uniform.

### Algorithm

```
1. Rotate: (x, y) = FWHT(k)
2. Transform: r = √(x² + y²), θ = atan2(y, x)
3. Quantize:
   - r: Lloyd-Max for Rayleigh(sigma=1/√d)
   - θ: Uniform quantization
```

### Guarantees

| Property | Guarantee |
|----------|-----------|
| **Outlier-resistant** | Bounded radius distribution |
| **Table lookup** | Fast inner product via precomputed tables |
| **Compression** | ~4.2× for 4-bit |

### When to Use

- Data has outliers
- Fast dequantization needed
- Polar structure fits workload

## Mathematical Comparison

### Distortion Bounds

For b-bit quantization on d-dimensional unit vectors:

| Algorithm | MSE Distortion | IP Distortion | Unbiased IP |
|-----------|---------------|---------------|-------------|
| **QJL** | O(1/d) | O(1/d) | ✅ Yes |
| **Turbo-MSE** | O(4^(-b)) | Biased | ❌ No |
| **Turbo-PROD** | O(4^(-b)) | O(4^(-b)) | ✅ Yes |
| **Polar** | O(4^(-b)) | O(4^(-b)) | ❌ No |

### Memory per Vector (d=128)

| Algorithm | Bytes | vs FP16 |
|-----------|-------|---------|
| FP16 | 256 | 1.0× |
| **QJL** | **18** | **14.2×** |
| Turbo-2bit | 130 | 2.0× |
| **Turbo-4bit** | **66** | **3.9×** |
| Polar-4bit | 66 | 3.9× |

## Choosing an Algorithm

```
Need maximum compression? → QJL
Need best quality? → Turbo-MSE
Need unbiased + good compression? → Turbo-PROD (recommended)
Have outlier issues? → Polar
```
