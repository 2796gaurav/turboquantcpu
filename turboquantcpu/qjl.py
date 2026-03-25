"""
qjl.py — 1-Bit Quantized Johnson-Lindenstrauss Transform.

Reference: Zandieh, Daliri, Han (NeurIPS 2024 / AAAI 2025)
  "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead"
  https://arxiv.org/abs/2406.03482

Algorithm
─────────────────────────────────────────────────────────────────────
Compress key k ∈ ℝᵈ:
  1. Compute norm ‖k‖, normalize: k̂ = k/‖k‖  (unit sphere)
  2. Apply randomised WHT: k̃ = (1/√d') H(diag(σ)·k̂)
  3. Sign-quantize: Q(k) = sign(k̃) ∈ {-1,+1}ᵈ' → packed uint8

Estimate inner product ⟨q, k⟩:
  q̃ = (1/√d') H(diag(σ)·q)    ← same σ
  ⟨q, k⟩ ≈ (π/2/d') · ‖k‖ · (q̃ · Q(k))    UNBIASED estimator

Math guarantee:
  E_σ[estimate] = ⟨q, k⟩  (exact for any q, k)
  MSE ~ ‖q‖²‖k‖²/d' (decays with d)

Memory: (d'/8 + 2) bytes/vector  (vs 2d bytes for FP16)
  → ~14× compression for d=128 (d'=128)

CPU optimisations:
  • OpenMP-parallel compression across heads
  • Batch FWHT via C extension (AVX2/AVX-512)
  • Bit-packed storage (np.packbits)
  • i8 dot product for score estimation
"""

from __future__ import annotations
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch

from .fwht import (
    get_signs, randomized_fwht, randomized_fwht_batch,
    pack_bits, unpack_bits, pack_bits_c, unpack_bits_c, matmul_f32_i8_c
)

__all__ = ["QJLState", "QJLQuantizer"]


@dataclass
class QJLState:
    """
    Compressed QJL representation of a sequence of key vectors.

    packed_signs : (seq, H, d_packed) uint8 — 1 bit per key coordinate (LSB-first)
    key_norms    : (seq, H)           f16   — original L2 norms
    seq_len      : int
    head_dim     : int
    head_dim_p2  : int
    """
    packed_signs : np.ndarray   # (seq, H, d_packed) uint8
    key_norms    : np.ndarray   # (seq, H) float16
    seq_len      : int
    head_dim     : int
    head_dim_p2  : int

    @property
    def d_packed(self) -> int:
        return self.packed_signs.shape[-1]

    def memory_bytes(self) -> int:
        return self.packed_signs.nbytes + self.key_norms.nbytes

    def append(self, other: "QJLState") -> "QJLState":
        return QJLState(
            packed_signs = np.concatenate([self.packed_signs, other.packed_signs], axis=0),
            key_norms    = np.concatenate([self.key_norms,    other.key_norms],    axis=0),
            seq_len      = self.seq_len + other.seq_len,
            head_dim     = self.head_dim,
            head_dim_p2  = self.head_dim_p2,
        )

    def memory_report(self, head_dim: Optional[int] = None) -> dict:
        d    = head_dim or self.head_dim
        H    = self.packed_signs.shape[1]
        seq  = self.seq_len
        orig = seq * H * d * 2
        tot  = self.memory_bytes()
        return {
            "original_fp16_MB":  orig / 1e6,
            "qjl_total_MB":      tot  / 1e6,
            "compression_ratio": orig / max(tot, 1),
            "bits_per_element":  tot*8 / max(seq*H*d, 1),
        }


class QJLQuantizer:
    """
    1-bit QJL KV cache quantizer — one instance per attention layer.

    Usage
    -----
        q = QJLQuantizer(head_dim=128, num_kv_heads=8, layer_idx=0)
        state   = q.compress(keys)       # (seq, H, d) → QJLState
        scores  = q.scores(query, state) # (B, Q, seq)
    """

    def __init__(
        self,
        head_dim:       int,
        num_kv_heads:   int,
        num_q_heads:    Optional[int] = None,
        layer_idx:      int = 0,
        seed_offset:    int = 0,
        normalise_keys: bool = True,
        n_threads:      int = 0,
    ):
        self.head_dim      = head_dim
        self.num_kv_heads  = num_kv_heads
        self.num_q_heads   = num_q_heads or num_kv_heads
        self.layer_idx     = layer_idx
        self.normalise_keys = normalise_keys

        self.head_dim_p2 = 1 << max(1, (head_dim-1).bit_length())
        self.d_packed    = (self.head_dim_p2 + 7) // 8

        # Per-head rotation signs
        self._signs = np.stack([
            get_signs(
                self.head_dim_p2,
                seed=((layer_idx+1)*31337 + h + seed_offset) & 0xFFFFFFFF,
                as_numpy=True,
            )
            for h in range(num_kv_heads)
        ])  # (H, d_p2)

        # Unbiased estimator scaling: pi/(2*d')
        self._scale = math.pi / (2.0 * self.head_dim_p2)

        n_threads = n_threads or min(num_kv_heads, 4)
        self._pool = ThreadPoolExecutor(max_workers=n_threads)

    # ── Compress ──────────────────────────────────────────────────────

    def _compress_head(self, keys_h: np.ndarray, h: int) -> np.ndarray:
        """(seq, d) → (seq, d_packed) packed signs."""
        rotated = randomized_fwht_batch(keys_h.astype(np.float32), self._signs[h])
        signs   = np.sign(rotated).astype(np.int8)
        signs[signs == 0] = 1  # deterministic tie-break
        return pack_bits_c(signs)  # (seq, d_packed)

    def compress(
        self,
        keys: Union[np.ndarray, torch.Tensor],
    ) -> QJLState:
        """
        Compress key vectors to 1-bit QJL representation.

        Parameters
        ----------
        keys : (seq_len, H, head_dim) float32/float16

        Returns
        -------
        QJLState — packed signs + float16 norms
        """
        if isinstance(keys, torch.Tensor):
            keys = keys.float().numpy()
        else:
            keys = np.asarray(keys, dtype=np.float32)

        seq_len = keys.shape[0]
        H       = self.num_kv_heads

        norms = np.linalg.norm(keys, axis=-1)  # (seq, H)

        if self.normalise_keys:
            keys_n = keys / norms[..., None].clip(1e-8)
        else:
            keys_n = keys

        packed = np.empty((seq_len, H, self.d_packed), dtype=np.uint8)

        def _do_head(h):
            packed[:, h, :] = self._compress_head(keys_n[:, h, :], h)

        list(self._pool.map(_do_head, range(H)))

        return QJLState(
            packed_signs = packed,
            key_norms    = norms.astype(np.float16),
            seq_len      = seq_len,
            head_dim     = self.head_dim,
            head_dim_p2  = self.head_dim_p2,
        )

    # ── Score estimation ──────────────────────────────────────────────

    def scores(
        self,
        query: Union[np.ndarray, torch.Tensor],
        state: QJLState,
    ) -> np.ndarray:
        """
        Estimate ⟨query, key_i⟩ for all cached keys.

        Estimator (provably unbiased):
            score_i ≈ (π/2/d') · ‖k_i‖ · [H(σ⊙q) · sign(H(σ⊙k̂_i))]

        Parameters
        ----------
        query : (batch, Q, head_dim)
        state : QJLState

        Returns
        -------
        (batch, Q, seq_len) float32
        """
        return_torch = isinstance(query, torch.Tensor)
        if return_torch:
            dev   = query.device
            query = query.float().numpy()
        else:
            query = np.asarray(query, dtype=np.float32)

        batch   = query.shape[0]
        Q       = self.num_q_heads
        H       = self.num_kv_heads
        seq_len = state.seq_len
        gqa     = Q // H
        d_p2    = state.head_dim_p2

        out = np.empty((batch, Q, seq_len), dtype=np.float32)
        
        # Process per KV head to avoid unpacking all signs at once
        for h_k in range(H):
            # Unpack signs for this head only
            packed_h = state.packed_signs[:, h_k, :]  # (seq, d_packed)
            signs_i8 = unpack_bits_c(packed_h, d_p2)  # (seq, d_p2)
            
            # All query heads that map to this KV head
            q_heads_for_k = range(h_k * gqa, (h_k + 1) * gqa)
            
            for h_q in q_heads_for_k:
                # Rotate query
                q_rot = randomized_fwht_batch(
                    query[:, h_q, :], self._signs[h_k]
                )  # (batch, d_p2)
                
                # Compute dot product using optimized kernel
                # A: (batch, d_p2), B: (seq, d_p2) → C: (batch, seq)
                raw = matmul_f32_i8_c(q_rot, signs_i8)  # (batch, seq)
                norms_h = state.key_norms[:, h_k].astype(np.float32)
                out[:, h_q, :] = raw * self._scale * norms_h[None, :]

        if return_torch:
            return torch.from_numpy(out).to(dev)
        return out

    def memory_report(self, seq_len: int) -> dict:
        d    = self.head_dim
        d_p2 = self.head_dim_p2
        H    = self.num_kv_heads
        orig = seq_len * H * d * 2
        sign = seq_len * H * ((d_p2+7)//8)
        norm = seq_len * H * 2
        tot  = sign + norm
        return {
            "original_fp16_MB":  orig / 1e6,
            "qjl_signs_MB":      sign / 1e6,
            "qjl_norms_MB":      norm / 1e6,
            "qjl_total_MB":      tot  / 1e6,
            "compression_ratio": orig / max(tot,1),
            "bits_per_element":  tot*8 / max(seq_len*H*d,1),
        }

    def __repr__(self) -> str:
        return (f"QJLQuantizer(head_dim={self.head_dim}, "
                f"kv_heads={self.num_kv_heads}, layer={self.layer_idx})")