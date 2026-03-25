"""
polar.py — PolarQuant: KV cache quantization via polar transformation.

References:
  Han, Kacham, Karbasi, Mirrokni, Zandieh.
  "PolarQuant: Quantizing KV Caches with Polar Transformation." AISTATS 2026.

  Wu et al.
  "PolarQuant: Leveraging Polar Transformation for Efficient Key Cache
   Quantization and Decoding Acceleration." arXiv 2502.00527.

Core insight
─────────────────────────────────────────────────────────────────────
After random preconditioning (FWHT rotation), adjacent coordinate pairs
(x_{2i}, x_{2i+1}) of a KV vector form a 2D Gaussian. The RoPE-coupled
outlier pairs, which normally resist scalar quantization, become
well-structured circular patterns when viewed in polar coordinates:
  • Radius r   ~ Rayleigh(sigma=1/sqrt(d)) — tightly bounded, no outliers
  • Angle  θ   ~ Uniform[0, 2pi) (level 1) — quantized with uniform codebook

This eliminates the need for per-block scale/zero-point storage (ZERO overhead).
The inner product <q_pair, k_pair> = r * (qx*cos(θ) + qy*sin(θ)) becomes a
fast table lookup: precompute cos/sin table indexed by θ-quantization index.

Speed advantage:
  • Dequantize-free scoring: directly look up cos(θ_i) from table
  • Table-lookup inner product: 2 multiplies + 1 add per 2D pair vs 2*d
  • Achieves 4.2× compression vs FP16 with best quality (per paper)

Memory layout:
  • r_idx:     (seq, H, d/2)  uint8  — radius quantization indices
  • theta_idx: (seq, H, d/2)  uint8  — angle quantization indices
  • norms:     (seq, H)       f16    — stored separately (needed for PROD stage)
"""

from __future__ import annotations
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Union, Tuple

import numpy as np
import torch

from .fwht     import get_signs, randomized_fwht_batch, fwht_numpy
from .codebooks import get_polar_codebook, PolarCodebook

__all__ = ["PolarState", "PolarQuantizer"]


# ── Compressed state ─────────────────────────────────────────────────

@dataclass
class PolarState:
    """
    PolarQuant compressed keys for one attention layer.

    r_idx     : (seq, H, n_pairs) uint8 — radius quantization indices
    theta_idx : (seq, H, n_pairs) uint8 — angle quantization indices
    norms     : (seq, H)          f16   — original L2 norms (for PROD mode)
    seq_len   : int
    head_dim  : int
    n_pairs   : int = head_dim_p2 // 2
    """
    r_idx     : np.ndarray   # (seq, H, n_pairs)
    theta_idx : np.ndarray   # (seq, H, n_pairs)
    norms     : np.ndarray   # (seq, H) float16
    seq_len   : int
    head_dim  : int
    head_dim_p2: int

    @property
    def n_pairs(self) -> int:
        return self.r_idx.shape[-1]

    def memory_bytes(self) -> int:
        return self.r_idx.nbytes + self.theta_idx.nbytes + self.norms.nbytes

    def append(self, other: "PolarState") -> "PolarState":
        return PolarState(
            r_idx      = np.concatenate([self.r_idx,     other.r_idx],     axis=0),
            theta_idx  = np.concatenate([self.theta_idx, other.theta_idx], axis=0),
            norms      = np.concatenate([self.norms,     other.norms],     axis=0),
            seq_len    = self.seq_len + other.seq_len,
            head_dim   = self.head_dim,
            head_dim_p2= self.head_dim_p2,
        )

    def memory_report(self) -> dict:
        orig_fp16 = self.seq_len * (self.norms.shape[1]) * self.head_dim * 2
        total     = self.memory_bytes()
        bpe       = total * 8 / max(self.seq_len * self.norms.shape[1] * self.head_dim, 1)
        return {
            "original_fp16_MB": orig_fp16 / 1e6,
            "polar_MB":         total / 1e6,
            "compression_ratio": orig_fp16 / max(total, 1),
            "bits_per_element": bpe,
        }


# ── Main quantizer ────────────────────────────────────────────────────

class PolarQuantizer:
    """
    PolarQuant KV cache quantizer — one instance per attention layer.

    Bit allocation: n_r_bits + n_theta_bits bits per pair = per-coord average.
    Common configs:
      n_r_bits=2, n_theta_bits=2 → 4 bits per pair → 2 bits/coord
      n_r_bits=3, n_theta_bits=3 → 6 bits per pair → 3 bits/coord
      n_r_bits=2, n_theta_bits=3 → 5 bits per pair → 2.5 bits/coord
    """

    def __init__(
        self,
        head_dim:     int,
        num_kv_heads: int,
        num_q_heads:  Optional[int] = None,
        layer_idx:    int = 0,
        n_r_bits:     int = 2,
        n_theta_bits: int = 2,
        n_threads:    int = 0,
    ):
        self.head_dim     = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_q_heads  = num_q_heads or num_kv_heads
        self.layer_idx    = layer_idx
        self.n_r_bits     = n_r_bits
        self.n_theta_bits = n_theta_bits
        self.bits_per_pair = n_r_bits + n_theta_bits
        self.bits_per_elem = self.bits_per_pair / 2

        self.head_dim_p2 = 1 << max(1, (head_dim-1).bit_length())
        self.n_pairs     = self.head_dim_p2 // 2

        # Polar codebook
        self._cb = get_polar_codebook(n_r_bits, n_theta_bits, self.head_dim_p2)

        # Per-head rotation signs (deterministic)
        self._signs = np.stack([
            get_signs(
                self.head_dim_p2,
                seed=((layer_idx+1)*77777 + h*1301) & 0xFFFFFFFF,
                as_numpy=True,
            )
            for h in range(num_kv_heads)
        ])  # (H, d_p2)

        n_threads = n_threads or min(num_kv_heads, 4)
        self._pool = ThreadPoolExecutor(max_workers=n_threads)

    # ─── Compress ────────────────────────────────────────────────────

    def _compress_head(self, keys_h: np.ndarray, h: int
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compress (seq, d) keys for head h → (seq, n_pairs) r_idx, theta_idx.
        """
        # Rotate: (seq, d_p2)
        rotated = randomized_fwht_batch(keys_h.astype(np.float32), self._signs[h])
        # Split into pairs
        seq = rotated.shape[0]
        x_  = rotated.reshape(seq, self.n_pairs, 2)
        xc  = x_[:, :, 0]  # (seq, n_pairs) x-coord
        yc  = x_[:, :, 1]  # (seq, n_pairs) y-coord

        # Quantize
        r_idx, t_idx = self._cb.encode_pair(xc.ravel(), yc.ravel())
        return r_idx.reshape(seq, self.n_pairs), t_idx.reshape(seq, self.n_pairs)

    def compress(
        self,
        keys: Union[np.ndarray, torch.Tensor],
    ) -> PolarState:
        """
        Compress key vectors to PolarQuant representation.

        Parameters
        ----------
        keys : (seq_len, num_kv_heads, head_dim) float32/float16

        Returns
        -------
        PolarState with r_idx, theta_idx, norms
        """
        if isinstance(keys, torch.Tensor):
            keys = keys.float().numpy()
        else:
            keys = np.asarray(keys, dtype=np.float32)

        seq_len = keys.shape[0]
        H       = self.num_kv_heads

        # Compute norms and normalise
        norms = np.linalg.norm(keys, axis=-1)           # (seq, H)
        safe  = norms[..., None].clip(1e-8)
        keys_n = keys / safe

        # Allocate outputs
        r_idx_all = np.empty((seq_len, H, self.n_pairs), dtype=np.uint8)
        t_idx_all = np.empty((seq_len, H, self.n_pairs), dtype=np.uint8)

        def _do_head(h):
            ri, ti = self._compress_head(keys_n[:, h, :], h)
            r_idx_all[:, h, :] = ri
            t_idx_all[:, h, :] = ti

        list(self._pool.map(_do_head, range(H)))

        return PolarState(
            r_idx      = r_idx_all,
            theta_idx  = t_idx_all,
            norms      = norms.astype(np.float16),
            seq_len    = seq_len,
            head_dim   = self.head_dim,
            head_dim_p2= self.head_dim_p2,
        )

    # ─── Scores ──────────────────────────────────────────────────────

    def scores(
        self,
        query: Union[np.ndarray, torch.Tensor],
        state: PolarState,
    ) -> np.ndarray:
        """
        Compute attention logits via polar table-lookup.

        Estimator (exact for quantized keys):
          <q, k_i> ≈ norm_i * sum_j r_{ij} * (qx_j*cos(θ_{ij}) + qy_j*sin(θ_{ij}))

        Parameters
        ----------
        query : (batch, num_q_heads, head_dim)
        state : PolarState

        Returns
        -------
        (batch, num_q_heads, seq_len) float32
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

        out = np.zeros((batch, Q, seq_len), dtype=np.float32)

        for h_q in range(Q):
            h_k  = h_q // gqa
            sg_h = self._signs[h_k]

            for b in range(batch):
                # Rotate query: (d_p2,)
                q_h = query[b, h_q, :]
                q_rot = randomized_fwht_batch(
                    q_h[None, :], sg_h
                )[0]  # (d_p2,)

                # Split into pairs: (n_pairs, 2)
                q_pairs = q_rot.reshape(self.n_pairs, 2)
                qx, qy  = q_pairs[:, 0], q_pairs[:, 1]

                # Table-lookup dot for all seq_len keys at once
                # r_idx:   (seq, n_pairs), theta_idx: (seq, n_pairs)
                ri  = state.r_idx[:, h_k, :]     # (seq, n_pairs)
                ti  = state.theta_idx[:, h_k, :] # (seq, n_pairs)

                r_lv = self._cb.r_levels[ri.ravel()].reshape(seq_len, self.n_pairs)
                ct   = self._cb.cos_table[ti.ravel()].reshape(seq_len, self.n_pairs)
                st   = self._cb.sin_table[ti.ravel()].reshape(seq_len, self.n_pairs)

                # (seq, n_pairs) · (n_pairs,) → (seq,)
                raw = np.sum(r_lv * (qx[None,:]*ct + qy[None,:]*st), axis=1)

                # Restore key norms
                norms_h = state.norms[:, h_k].astype(np.float32)  # (seq,)
                out[b, h_q, :] = raw * norms_h

        if return_torch:
            return torch.from_numpy(out).to(dev)
        return out

    def memory_report(self, seq_len: int) -> dict:
        H = self.num_kv_heads
        d = self.head_dim
        orig  = seq_len * H * d * 2
        total = seq_len * H * (2 * self.n_pairs + 2)  # r+theta (uint8) + norm(f16)
        return {
            "original_fp16_MB":  orig / 1e6,
            "polar_MB":          total / 1e6,
            "compression_ratio": orig / max(total, 1),
            "bits_per_element":  total*8 / max(seq_len*H*d, 1),
            "mode": f"polar-{self.n_r_bits}r+{self.n_theta_bits}θ",
        }

    def __repr__(self) -> str:
        return (f"PolarQuantizer(head_dim={self.head_dim}, "
                f"kv_heads={self.num_kv_heads}, "
                f"bits={self.n_r_bits}r+{self.n_theta_bits}θ, "
                f"layer={self.layer_idx})")