"""
TurboQuantCPU — Ultra-fast CPU-only KV cache quantization for LLM inference.

Based on TurboQuant (ICLR 2026) - achieves near-optimal distortion rates with
provable guarantees for both MSE and inner product estimation.

Key Features:
  • QJL (1-bit): Zero-overhead quantization with unbiased inner product estimator
  • TurboQuant-MSE: Lloyd-Max optimal scalar quantization after FWHT rotation
  • TurboQuant-PROD: Unbiased inner product quantizer (MSE + QJL residual)
  • PolarQuant: Polar transformation for outlier-resistant quantization
  • SIMD-optimized: AVX-512, AVX2, AMX, NEON support
  • Memory efficient: 4-14× compression with <1% quality loss

Example:
    >>> from turboquantcpu import TurboQuantizer, TurboMode
    >>> quantizer = TurboQuantizer(head_dim=128, num_kv_heads=8, mode="prod", bits=4)
    >>> state = quantizer.compress(keys)  # (seq, H, d) → compressed
    >>> scores = quantizer.scores(query, state)  # unbiased attention scores

Reference:
  Zandieh, Daliri, Hadian, Mirrokni (ICLR 2026)
  "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
  https://arxiv.org/abs/2504.19874
"""

__version__ = "0.0.15"
__author__ = "TurboQuantCPU Contributors"

# Core quantizers
from turboquantcpu.turbo import (
    TurboQuantizer,
    TurboState,
    TurboMode,
)

from turboquantcpu.qjl import (
    QJLQuantizer,
    QJLState,
)

from turboquantcpu.polar import (
    PolarQuantizer,
    PolarState,
)

# KV cache management
from turboquantcpu.kv_cache import (
    CacheConfig,
    LayerCache,
    CompressedKVCache,
)

from turboquantcpu.value_quant import (
    ValueQuantizer,
    ValueState,
    ValueMode,
)

# Codebooks
from turboquantcpu.codebooks import (
    Codebook,
    PolarCodebook,
    get_codebook,
    get_polar_codebook,
)

# FWHT utilities
from turboquantcpu.fwht import (
    fwht,
    fwht_numpy,
    fwht_torch,
    randomized_fwht,
    randomized_fwht_batch,
    get_signs,
    pack_bits,
    unpack_bits,
    FWHTBackend,
    set_backend,
)

# CPU features detection
from turboquantcpu.cpu_features import (
    CPUInfo,
    cpu_info,
    print_cpu_capabilities,
)

# HuggingFace integration
from turboquantcpu.patch import (
    PatchConfig,
    patch_model,
    unpatch_model,
)

# Sparse attention
from turboquantcpu.sparse_attn import (
    H2OConfig,
    AttentionTracker,
    H2OEvictionPolicy,
)

# Benchmarking
from turboquantcpu.benchmark import (
    run_correctness_check,
    benchmark_memory,
    benchmark_speed,
    run_full_benchmark,
)

__all__ = [
    # Version
    "__version__",
    # TurboQuant
    "TurboQuantizer",
    "TurboState",
    "TurboMode",
    # QJL
    "QJLQuantizer",
    "QJLState",
    # PolarQuant
    "PolarQuantizer",
    "PolarState",
    # KV Cache
    "CacheConfig",
    "LayerCache",
    "CompressedKVCache",
    # Value Quant
    "ValueQuantizer",
    "ValueState",
    "ValueMode",
    # Codebooks
    "Codebook",
    "PolarCodebook",
    "get_codebook",
    "get_polar_codebook",
    # FWHT
    "fwht",
    "fwht_numpy",
    "fwht_torch",
    "randomized_fwht",
    "randomized_fwht_batch",
    "get_signs",
    "pack_bits",
    "unpack_bits",
    "FWHTBackend",
    "set_backend",
    # CPU
    "CPUInfo",
    "cpu_info",
    "print_cpu_capabilities",
    # Patch
    "PatchConfig",
    "patch_model",
    "unpatch_model",
    # Sparse
    "H2OConfig",
    "AttentionTracker",
    "H2OEvictionPolicy",
    # Benchmark
    "run_correctness_check",
    "benchmark_memory",
    "benchmark_speed",
    "run_full_benchmark",
]


def _check_c_extension():
    """Check if C extension is compiled and warn if not."""
    import os
    import glob
    
    # Check for compiled extension
    module_dir = os.path.dirname(os.path.abspath(__file__))
    patterns = [
        "_tqcpu_kernels*.so",
        "_tqcpu_kernels*.dylib", 
        "_tqcpu_kernels*.dll",
    ]
    
    for pattern in patterns:
        if glob.glob(os.path.join(module_dir, pattern)):
            return True
    
    # Also try importing
    try:
        from turboquantcpu._tqcpu_kernels import tqcpu_version
        return True
    except ImportError:
        import warnings
        warnings.warn(
            "C extension not compiled. Falling back to NumPy implementations. "
            "For best performance, build with: python setup.py build_ext --inplace",
            RuntimeWarning,
            stacklevel=2,
        )
        return False


# Check on import
_C_EXT_AVAILABLE = _check_c_extension()
