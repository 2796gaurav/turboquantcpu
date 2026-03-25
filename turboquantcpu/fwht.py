"""
fwht.py — Fast Walsh-Hadamard Transform with multi-backend dispatch.

The Randomised WHT (SRHT) is the core rotation in TurboQuant:
    x̃ = (1/√d) · H · diag(signs) · x      [O(d log d), orthogonal]

Why WHT instead of random Gaussian rotation:
  • O(d log d) vs O(d²) — critical for decode latency
  • Self-inverse: H·H = I (normalised form)
  • Cache-friendly butterfly pattern maps perfectly to SIMD
  • Zero memory allocation (in-place butterfly)
  • AVX2: 8 f32 pairs/cycle → ~6× speedup vs scalar
  • AVX-512: 16 f32 pairs/cycle → ~12× speedup vs scalar

Backends (auto-selected):
  1. C extension (AVX-512 / AVX2 / NEON) — fastest
  2. NumPy vectorised butterfly — ~3–5× slower than C
  3. PyTorch — similar to NumPy, supports GPU tensors (not used for KV cache)
"""

from __future__ import annotations
import ctypes, math, os
from enum import Enum, auto
from typing import Optional, Union

import numpy as np
import torch

__all__ = [
    "FWHTBackend", "set_backend",
    "fwht", "fwht_numpy", "fwht_torch",
    "randomized_fwht", "randomized_fwht_batch",
    "get_signs", "pack_bits", "unpack_bits",
]


# ── Backend enum ──────────────────────────────────────────────────────

class FWHTBackend(Enum):
    C_EXT = auto()
    NUMPY = auto()
    TORCH = auto()
    AUTO  = auto()


# ── C extension loader ────────────────────────────────────────────────

_lib: Optional[ctypes.CDLL] = None
_FOUND_C = False


def _try_load_c_ext() -> bool:
    global _lib, _FOUND_C
    search = [
        # Check in _kernels subdirectory first
        os.path.join(os.path.dirname(__file__), "_kernels",
                     "_tqcpu_kernels.so"),
        os.path.join(os.path.dirname(__file__), "_kernels",
                     "_tqcpu_kernels.dylib"),
        # Then check in parent directory
        os.path.join(os.path.dirname(__file__),
                     "_tqcpu_kernels.so"),
        os.path.join(os.path.dirname(__file__),
                     "_tqcpu_kernels.dylib"),
        # Also try compiled via setup.py (installed as module)
    ]
    for p in search:
        if os.path.exists(p):
            try:
                _lib = ctypes.CDLL(p)
                # Prototype functions
                F = ctypes.c_float
                I = ctypes.c_int
                U8 = ctypes.POINTER(ctypes.c_uint8)
                I8 = ctypes.POINTER(ctypes.c_int8)
                PF = ctypes.POINTER(ctypes.c_float)

                _lib.tqcpu_fwht_f32.argtypes      = [PF, I]
                _lib.tqcpu_fwht_f32.restype        = None
                _lib.tqcpu_rfwht_f32.argtypes      = [PF, PF, I]
                _lib.tqcpu_rfwht_f32.restype       = None
                _lib.tqcpu_fwht_batch_f32.argtypes = [PF, I, I]
                _lib.tqcpu_fwht_batch_f32.restype  = None
                _lib.tqcpu_rfwht_batch_f32.argtypes= [PF, PF, I, I]
                _lib.tqcpu_rfwht_batch_f32.restype = None
                _lib.tqcpu_sign_i8.argtypes        = [PF, I8, I]
                _lib.tqcpu_sign_i8.restype         = None
                _lib.tqcpu_pack_signs.argtypes     = [I8, U8, I]
                _lib.tqcpu_pack_signs.restype      = None
                _lib.tqcpu_unpack_signs.argtypes   = [U8, I8, I]
                _lib.tqcpu_unpack_signs.restype    = None
                _lib.tqcpu_dot_f32_i8.argtypes     = [PF, I8, I]
                _lib.tqcpu_dot_f32_i8.restype      = ctypes.c_float
                _lib.tqcpu_matmul_f32_i8.argtypes  = [PF, I8, PF, I, I, I]
                _lib.tqcpu_matmul_f32_i8.restype   = None
                _lib.tqcpu_softmax_f32.argtypes    = [PF, I]
                _lib.tqcpu_softmax_f32.restype     = None
                _lib.tqcpu_weighted_sum_f32.argtypes = [PF,PF,PF,I,I]
                _lib.tqcpu_weighted_sum_f32.restype  = None

                _FOUND_C = True
                return True
            except (OSError, AttributeError):
                pass

    # Try importing via Python extension (if installed via setup.py)
    try:
        from . import _tqcpu_kernels as _kern  # type: ignore
        # wrap module functions — not ideal but works if setup.py built it
        _FOUND_C = False  # use numpy fallback; module not a cdll
    except ImportError:
        pass

    return False


_try_load_c_ext()


# ── Global backend ────────────────────────────────────────────────────

_ACTIVE_BACKEND: FWHTBackend = FWHTBackend.C_EXT if _FOUND_C else FWHTBackend.NUMPY


def set_backend(backend: FWHTBackend) -> None:
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = backend


def _pick(hint: FWHTBackend) -> FWHTBackend:
    if hint != FWHTBackend.AUTO:
        return hint
    return _ACTIVE_BACKEND


# ── NumPy FWHT ────────────────────────────────────────────────────────

def fwht_numpy(x: np.ndarray, inplace: bool = False) -> np.ndarray:
    """
    Normalised FWHT along last axis. Self-inverse: fwht(fwht(x)) == x.
    Last dim must be power-of-2.
    Time: O(d log d).  Memory: O(1) extra (in-place butterfly).
    """
    n = x.shape[-1]
    if n <= 1:
        return x if inplace else x.copy()
    if n & (n-1):
        raise ValueError(f"Last dim must be power-of-2, got {n}")

    y = x if inplace else x.copy().astype(np.float32, copy=False)
    orig = y.shape
    prefix = y.shape[:-1]
    h = 1
    while h < n:
        y  = y.reshape(*prefix, n//(2*h), 2, h)
        lo = y[..., 0, :].copy()
        hi = y[..., 1, :]
        y[..., 0, :] = lo + hi
        y[..., 1, :] = lo - hi
        y = y.reshape(orig)
        h <<= 1
    y *= 1.0 / math.sqrt(n)
    return y


# ── PyTorch FWHT ──────────────────────────────────────────────────────

def fwht_torch(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """Normalised FWHT along last axis for PyTorch tensors. Self-inverse."""
    n = x.shape[-1]
    if n <= 1:
        return x.clone() if not inplace else x
    if n & (n-1):
        raise ValueError(f"Last dim must be power-of-2, got {n}")

    y = x if inplace else x.clone().float()
    h = 1
    while h < n:
        y  = y.view(*y.shape[:-1], n//(2*h), 2, h)
        lo = y[..., 0, :].clone()
        hi = y[..., 1, :]
        y[..., 0, :] = lo + hi
        y[..., 1, :] = lo - hi
        y = y.view(*x.shape)
        h <<= 1
    y = y * (1.0/math.sqrt(n))
    return y


# ── C-ext FWHT ───────────────────────────────────────────────────────

def _fwht_c_1d(x: np.ndarray) -> np.ndarray:
    """In-place FWHT via C ext on a contiguous 1-D float32 array."""
    y   = np.ascontiguousarray(x, dtype=np.float32)
    ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    _lib.tqcpu_fwht_f32(ptr, ctypes.c_int(y.shape[-1]))
    return y


def _rfwht_c_1d(x: np.ndarray, signs: np.ndarray) -> np.ndarray:
    y   = np.ascontiguousarray(x, dtype=np.float32)
    sg  = np.ascontiguousarray(signs, dtype=np.float32)
    ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    sp  = sg.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    _lib.tqcpu_rfwht_f32(ptr, sp, ctypes.c_int(y.shape[-1]))
    return y


def _rfwht_batch_c(X: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Batch randomised FWHT via C ext (OpenMP parallel)."""
    B, n = X.shape
    Y    = np.ascontiguousarray(X, dtype=np.float32)
    sg   = np.ascontiguousarray(signs, dtype=np.float32)
    ptr  = Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    sp   = sg.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    _lib.tqcpu_rfwht_batch_f32(ptr, sp, ctypes.c_int(B), ctypes.c_int(n))
    return Y


# ── Unified dispatcher ────────────────────────────────────────────────

def fwht(
    x: Union[np.ndarray, torch.Tensor],
    backend: FWHTBackend = FWHTBackend.AUTO,
    inplace: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Fast Walsh-Hadamard Transform — normalised, self-inverse.
    Dispatches to fastest available backend.
    Accepts numpy arrays or PyTorch tensors.
    Last dim must be power-of-2.
    """
    be = _pick(backend)
    if isinstance(x, torch.Tensor):
        return fwht_torch(x, inplace=inplace)
    # Numpy path
    if be == FWHTBackend.C_EXT and _FOUND_C and x.ndim == 1:
        return _fwht_c_1d(x)
    return fwht_numpy(x, inplace=inplace)


# ── Sign generation ───────────────────────────────────────────────────

def get_signs(
    d: int,
    seed: int,
    device: Optional[torch.device] = None,
    as_numpy: bool = False,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Generate a deterministic ±1 sign vector of length d.

    Same (seed, d) always produces the same signs — reproducible across
    processes, machines, and versions (uses numpy default_rng).
    """
    rng  = np.random.default_rng(seed)
    snp  = rng.choice([-1.0, 1.0], size=d).astype(np.float32)
    if as_numpy:
        return snp
    return torch.from_numpy(snp).to(device or torch.device("cpu"))


# ── Randomised FWHT ───────────────────────────────────────────────────

def _pad_to_p2(x: np.ndarray) -> Tuple[np.ndarray, int]:
    d    = x.shape[-1]
    d_p2 = 1 << max(1, (d-1).bit_length())
    if d == d_p2:
        return x, d_p2
    pad = np.zeros((*x.shape[:-1], d_p2-d), dtype=x.dtype)
    return np.concatenate([x, pad], axis=-1), d_p2


from typing import Tuple


def randomized_fwht(
    x: Union[np.ndarray, torch.Tensor],
    signs: Union[np.ndarray, torch.Tensor],
    backend: FWHTBackend = FWHTBackend.AUTO,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply the Randomised WHT: (1/√d') · H · diag(signs) · x

    This is an orthogonal JL-style map: rows are i.i.d. uniform on S^{d'-1}.
    Pads x to next power-of-2 if needed (zero-padding).

    Parameters
    ----------
    x     : (..., d) input
    signs : (d',) where d' ≥ d is a power-of-2

    Returns
    -------
    (..., d') rotated vector
    """
    is_torch = isinstance(x, torch.Tensor)
    if is_torch:
        sg = signs if isinstance(signs, torch.Tensor) else torch.from_numpy(signs)
        sg = sg.to(x.device, dtype=x.dtype)
        d, d_p2 = x.shape[-1], len(sg)
        if d < d_p2:
            pad = torch.zeros(*x.shape[:-1], d_p2-d, dtype=x.dtype, device=x.device)
            xp  = torch.cat([x, pad], dim=-1)
        else:
            xp = x
        return fwht_torch(xp * sg)

    # Numpy path
    sg = signs if isinstance(signs, np.ndarray) else signs.numpy()
    d_p2 = len(sg)
    xp, _ = _pad_to_p2(x.astype(np.float32, copy=False)) if x.shape[-1] < d_p2 else (x.copy().astype(np.float32), d_p2)

    be = _pick(backend)
    if be == FWHTBackend.C_EXT and _FOUND_C and xp.ndim == 1:
        return _rfwht_c_1d(xp, sg)
    # Vectorised numpy path for batched
    xp = xp * sg.astype(xp.dtype)
    return fwht_numpy(xp, inplace=True)


def randomized_fwht_batch(
    X: np.ndarray,      # (B, d) float32
    signs: np.ndarray,  # (d_p2,) float32
) -> np.ndarray:
    """
    Batch randomised FWHT: (B×d) → (B×d_p2).
    Uses C ext with OpenMP if available; otherwise numpy.
    """
    B, d = X.shape
    d_p2 = len(signs)

    if d < d_p2:
        pad = np.zeros((B, d_p2-d), dtype=np.float32)
        Xp  = np.concatenate([X.astype(np.float32, copy=False), pad], axis=1)
    else:
        Xp  = np.ascontiguousarray(X, dtype=np.float32)

    if _FOUND_C:
        return _rfwht_batch_c(Xp, signs)

    # Numpy batch fallback
    Xp = Xp * signs[None, :]
    return fwht_numpy(Xp)


# ── Bit packing helpers ───────────────────────────────────────────────

def pack_bits(signs_i8: np.ndarray) -> np.ndarray:
    """
    int8 {-1,+1} → 1-bit/element packed uint8 (8× memory savings).
    Encoding: bit=1 ↔ positive (+1), bit=0 ↔ negative (-1).
    Uses numpy.packbits with LSB-first ordering.
    """
    bits = (signs_i8 > 0).astype(np.uint8)
    return np.packbits(bits, axis=-1, bitorder="little")


def unpack_bits(packed: np.ndarray, d: int) -> np.ndarray:
    """
    Packed uint8 → int8 {-1,+1}.
    d: original dimension (trims padding bits).
    """
    bits = np.unpackbits(packed, axis=-1, bitorder="little")[..., :d]
    return np.where(bits, np.int8(1), np.int8(-1))


def unpack_bits_c(packed: np.ndarray, d: int) -> np.ndarray:
    """Use C kernel for unpacking (much faster for large arrays)."""
    if not _FOUND_C:
        return unpack_bits(packed, d)
    
    # packed shape: (..., d_packed)
    # output shape: (..., d)
    batch_shape = packed.shape[:-1]
    n_packed = packed.shape[-1]
    n_total = int(np.prod(batch_shape))
    
    flat = np.ascontiguousarray(packed.reshape(-1), dtype=np.uint8)
    n_bits = len(flat) * 8
    out_flat = np.empty(n_bits, dtype=np.int8)
    
    ip = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    op = out_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    _lib.tqcpu_unpack_signs(ip, op, ctypes.c_int(n_bits))
    
    # Reshape and trim to original dimension
    # Each row has d elements
    out = out_flat[:n_total * d].reshape(*batch_shape, d)
    return out


def pack_bits_c(signs_i8: np.ndarray) -> np.ndarray:
    """Use C kernel for packing (faster for large arrays)."""
    if not _FOUND_C:
        return pack_bits(signs_i8)
    flat = np.ascontiguousarray(signs_i8.ravel(), dtype=np.int8)
    n    = len(flat)
    nb   = (n+7)//8
    out  = np.empty(nb, dtype=np.uint8)
    ip   = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    op   = out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    _lib.tqcpu_pack_signs(ip, op, ctypes.c_int(n))
    return out.reshape(*signs_i8.shape[:-1], -1)


def matmul_f32_i8_c(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Matrix multiply: A (batch×n, f32) @ B (M×n, i8).T → C (batch×M, f32)
    Uses optimized C kernel with OpenMP.
    """
    if not _FOUND_C:
        # Fallback to numpy
        return A @ B.T.astype(np.float32)
    
    batch, n = A.shape
    M = B.shape[0]
    
    A_contig = np.ascontiguousarray(A, dtype=np.float32)
    B_contig = np.ascontiguousarray(B, dtype=np.int8)
    C = np.empty((batch, M), dtype=np.float32)
    
    _lib.tqcpu_matmul_f32_i8.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    _lib.tqcpu_matmul_f32_i8.restype = None
    
    _lib.tqcpu_matmul_f32_i8(
        A_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(batch),
        ctypes.c_int(M),
        ctypes.c_int(n)
    )
    
    return C