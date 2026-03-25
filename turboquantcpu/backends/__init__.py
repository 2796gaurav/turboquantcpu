"""
Backend implementations for different CPU architectures.

Backends:
  - numpy: Pure NumPy implementation (fallback)
  - avx2: AVX2/FMA optimized kernels
  - avx512: AVX-512 optimized kernels
  - neon: ARM NEON optimized kernels
  - amx: Intel AMX tile accelerator (experimental)
"""

from enum import Enum, auto


class BackendType(Enum):
    NUMPY = auto()
    AVX2 = auto()
    AVX512 = auto()
    NEON = auto()
    AMX = auto()


def get_best_backend():
    """Return the best available backend for this CPU."""
    from ..cpu_features import cpu_info
    
    info = cpu_info()
    
    if info.has_amx_int8:
        return BackendType.AMX
    elif info.has_avx512f:
        return BackendType.AVX512
    elif info.has_avx2:
        return BackendType.AVX2
    elif info.has_neon:
        return BackendType.NEON
    else:
        return BackendType.NUMPY
