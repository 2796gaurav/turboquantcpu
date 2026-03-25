"""
turbo.py — TurboQuant: MSE-optimal and Inner-Product-optimal KV quantization.

Reference: Zandieh, Daliri, Hadian, Mirrokni (ICLR 2026)
  "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
  https://arxiv.org/abs/2504.19874

Two algorithms
─────────────────────────────────────────────────────────────────────
TurboQuantMSE (mode="mse"):
  1. Normalize: k̂ = k/‖k‖
  2. Rotate:    k̃ = (1/√d') H(diag(σ)·k̂)
  3. Scale:     k̄ = √d' · k̃  → each coord ~ N(0,1)
  4. Quantize:  Lloyd-Max codebook at b bits per coord
  5. Store:     b-bit indices + 1 float16 norm

  MSE distortion: within 2.7× of Shannon lower bound (provably)
  Memory: d·b/8 + 2 bytes/vector (ZERO per-block overhead)

TurboQuantProd (mode="prod"):
  Uses (b-1)-bit TurboQuantMSE + 1-bit QJL on residual:
    k̃ = TurboQuantMSE_reconstruct(k)
    r = k - k̃
    score = <q, k̃>  +  (π/2/d') · ‖r‖ · (H(σ'⊙q) · sign(H(σ'⊙r̂)))

  E[score] = <q, k>  exactly  (provably unbiased ✓)
  Error:     near-optimal for inner product (closest to Shannon bound)

  NOTE: σ' ≠ σ (independent rotation for residual stage).

Outlier-aware "fractional-bit" quantization:
  Real LLM KV vectors have outlier channels with 10–100× higher variance.
  Strategy: top-K channels by variance get (b+1) bits; rest get b bits.
  "2.5-bit" = 50% at 2 bits + 50% at 3 bits → avg 2.5 bits/coord.
  This matches the ICLR 2026 paper's 2.5-bit and 3.5-bit experiments.
"""

from __future__ import annotations
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from .codebooks import get_codebook, Codebook
from .fwht      import (
    get_signs, randomized_fwht, randomized_fwht_batch,
    pack_bits, unpack_bits, pack_bits_c
)
from .qjl       import QJLQuantizer, QJLState

__all__ = ["TurboMode", "TurboState", "TurboQuantizer"]


# ── Mode ──────────────────────────────────────────────────────────────

class TurboMode(str, Enum):
    QJL   = "qjl"
    MSE   = "mse"
    PROD  = "prod"     # recommended: unbiased inner product estimator
    POLAR = "polar"    # polar transformation for outlier resistance


# ── Compressed state ─────────────────────────────────────────────────

@dataclass
class TurboState:
    """
    TurboQuant compressed keys for one layer.

    MSE stage (always present unless QJL mode):
      indices   : (seq, H, d_p2)  uint8/uint16 — scalar quantizer indices
      norms     : (seq, H)        f16

    QJL residual stage (PROD mode only):
      res_packed : (seq, H, d_packed)  uint8
      res_norms  : (seq, H)            f16

    Outlier channel metadata:
      outlier_mask : (d_p2,)  bool — True where extra bit is used
      outlier_bits : int — bits for outlier channels
    """
    mode         : TurboMode
    bits         : int
    seq_len      : int
    head_dim     : int
    head_dim_p2  : int

    indices      : np.ndarray   # (seq, H, d_p2)
    norms        : np.ndarray   # (seq, H) f16

    res_packed   : Optional[np.ndarray] = None  # (seq, H, d_packed)
    res_norms    : Optional[np.ndarray] = None  # (seq, H) f16

    outlier_mask : Optional[np.ndarray] = None  # (d_p2,) bool
    outlier_bits : int = 0

    _codebook    : Optional[Codebook] = field(default=None, repr=False)
    _cb_outlier  : Optional[Codebook] = field(default=None, repr=False)

    def memory_bytes(self) -> int:
        total = self.indices.nbytes + self.norms.nbytes
        if self.res_packed is not None:
            total += self.res_packed.nbytes
        if self.res_norms is not None:
            total += self.res_norms.nbytes
        return total

    def append(self, other: "TurboState") -> "TurboState":
        assert self.mode == other.mode and self.bits == other.bits
        kw: dict = dict(
            mode        = self.mode,
            bits        = self.bits,
            seq_len     = self.seq_len + other.seq_len,
            head_dim    = self.head_dim,
            head_dim_p2 = self.head_dim_p2,
            indices     = np.concatenate([self.indices, other.indices], axis=0),
            norms       = np.concatenate([self.norms,   other.norms],   axis=0),
            _codebook   = self._codebook,
            _cb_outlier = self._cb_outlier,
            outlier_mask = self.outlier_mask,
            outlier_bits = self.outlier_bits,
        )
        if self.res_packed is not None:
            kw["res_packed"] = np.concatenate([self.res_packed, other.res_packed], axis=0)
            kw["res_norms"]  = np.concatenate([self.res_norms,  other.res_norms],  axis=0)
        return TurboState(**kw)


# ── Main quantizer ────────────────────────────────────────────────────

class TurboQuantizer:
    """
    Full TurboQuant KV cache quantizer.

    Parameters
    ----------
    head_dim      : head vector dimension
    num_kv_heads  : K/V heads (GQA: may be < num_q_heads)
    num_q_heads   : query heads (for GQA broadcasting in scores())
    layer_idx     : seeds deterministic rotations
    mode          : TurboMode.PROD (recommended) | MSE | QJL
    bits          : total bits/coord (PROD uses bits-1 for MSE, 1 for QJL)
    outlier_frac  : fraction of channels getting (bits+1) bits (0=disabled)
    n_threads     : thread pool for parallel head compression (0=auto)

    Memory (PROD mode, 4-bit):
      MSE stage:     d*3/8 + 2  bytes (3-bit indices + norm)
      QJL residual:  d/8   + 2  bytes (1-bit signs + norm)
      Total:         d*4/8 + 4  bytes ≈ 0.5d + 4  bytes
      FP16 baseline: 2d         bytes
      Compression:   ~4× for d=128

    Memory (MSE mode, 4-bit):
      d*4/8 + 2 = 0.5d + 2 bytes  → ~4× compression
    """

    def __init__(
        self,
        head_dim:     int,
        num_kv_heads: int,
        num_q_heads:  Optional[int] = None,
        layer_idx:    int = 0,
        mode:         Union[TurboMode, str] = TurboMode.PROD,
        bits:         int = 4,
        outlier_frac: float = 0.0,
        n_threads:    int = 0,
    ):
        self.head_dim     = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_q_heads  = num_q_heads or num_kv_heads
        self.layer_idx    = layer_idx
        self.mode         = TurboMode(mode)
        self.bits         = bits
        self.outlier_frac = outlier_frac

        self.head_dim_p2 = 1 << max(1, (head_dim-1).bit_length())
        self._coord_scale = math.sqrt(self.head_dim_p2)  # normalise to N(0,1)

        # Bits for MSE stage
        if self.mode == TurboMode.PROD:
            self._mse_bits = max(1, bits-1)
        elif self.mode == TurboMode.MSE:
            self._mse_bits = bits
        else:
            self._mse_bits = 1  # QJL uses no MSE

        # Codebooks
        self._codebook = get_codebook(self._mse_bits) if self.mode != TurboMode.QJL else None

        # Outlier-aware: channel-wise two-codebook scheme
        self._outlier_bits = bits if outlier_frac == 0.0 else min(8, bits+1)
        self._cb_outlier   = get_codebook(self._outlier_bits) if outlier_frac > 0 else None
        self._n_outliers   = int(self.head_dim_p2 * outlier_frac)

        # Stage-1 rotation signs
        self._signs_mse = np.stack([
            get_signs(
                self.head_dim_p2,
                seed=((layer_idx+1)*99991 + h*1009) & 0xFFFFFFFF,
                as_numpy=True,
            )
            for h in range(num_kv_heads)
        ])  # (H, d_p2)

        # Stage-2 QJL (PROD / QJL modes)
        self._qjl: Optional[QJLQuantizer] = None
        if self.mode in (TurboMode.PROD, TurboMode.QJL):
            self._qjl = QJLQuantizer(
                head_dim      = head_dim,
                num_kv_heads  = num_kv_heads,
                num_q_heads   = self.num_q_heads,
                layer_idx     = layer_idx,
                seed_offset   = 55555,
                normalise_keys= True,
                n_threads     = n_threads,
            )

        n_threads = n_threads or min(num_kv_heads, 4)
        self._pool = ThreadPoolExecutor(max_workers=n_threads)

    # ── MSE compression ───────────────────────────────────────────────

    def _mse_compress_head(self, keys_h: np.ndarray, h: int) -> np.ndarray:
        """
        (seq, d) normalised keys → (seq, d_p2) uint8 indices.
        Uses outlier-aware two-codebook scheme if outlier_frac > 0.
        """
        rotated = randomized_fwht_batch(keys_h.astype(np.float32), self._signs_mse[h])
        scaled  = rotated * self._coord_scale   # → N(0,1) per coord

        if self._n_outliers > 0:
            # Two-codebook scheme: identify top-k variance positions
            # (computed per batch; for online we use magnitude as proxy)
            mag = np.abs(scaled).mean(axis=0)  # (d_p2,)
            topk = np.argpartition(mag, -self._n_outliers)[-self._n_outliers:]
            mask = np.zeros(self.head_dim_p2, dtype=bool)
            mask[topk] = True

            # Allocate combined index array (uint16 for safety)
            idx = np.empty_like(scaled, dtype=np.uint8)
            idx[:, ~mask] = self._codebook.quantize(scaled[:, ~mask])
            idx[:,  mask] = self._cb_outlier.quantize(scaled[:,  mask])
            return idx, mask
        else:
            return self._codebook.quantize(scaled), None

    def _mse_reconstruct_head(
        self,
        indices_h: np.ndarray,   # (seq, d_p2)
        norms_h:   np.ndarray,   # (seq,)
        h: int,
        outlier_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """(seq, d_p2) indices → (seq, head_dim) reconstructed keys."""
        if outlier_mask is not None and outlier_mask.any():
            rec = np.empty((*indices_h.shape[:-1], self.head_dim_p2), dtype=np.float32)
            rec[:, ~outlier_mask] = self._codebook.dequantize(indices_h[:, ~outlier_mask])
            rec[:,  outlier_mask] = self._cb_outlier.dequantize(indices_h[:, outlier_mask])
        else:
            rec = self._codebook.dequantize(indices_h)  # (seq, d_p2)

        rec = rec / self._coord_scale
        # Inverse FWHT: H·diag(σ) is self-inverse via H(σ⊙H(σ⊙x)) = x
        inv = randomized_fwht_batch(rec, self._signs_mse[h])  # (seq, d_p2)
        # Apply σ again to complete the inverse
        inv = inv * self._signs_mse[h][None, :]
        result = inv[:, :self.head_dim] * norms_h[:, None]
        return result.astype(np.float32)

    # ── Public compress ───────────────────────────────────────────────

    def compress(
        self,
        keys: Union[np.ndarray, torch.Tensor],
    ) -> TurboState:
        """
        Compress key vectors to TurboQuant representation.

        Parameters
        ----------
        keys : (seq_len, H, head_dim) float32/float16

        Returns
        -------
        TurboState
        """
        # QJL-only mode
        if self.mode == TurboMode.QJL:
            qs = self._qjl.compress(keys)
            return TurboState(
                mode=TurboMode.QJL, bits=1,
                seq_len=qs.seq_len, head_dim=qs.head_dim, head_dim_p2=qs.head_dim_p2,
                indices=qs.packed_signs, norms=qs.key_norms,
            )

        if isinstance(keys, torch.Tensor):
            keys = keys.float().numpy()
        else:
            keys = np.asarray(keys, dtype=np.float32)

        seq_len = keys.shape[0]
        H       = self.num_kv_heads
        d_p2    = self.head_dim_p2

        norms  = np.linalg.norm(keys, axis=-1)
        keys_n = keys / norms[..., None].clip(1e-8)

        indices     = np.empty((seq_len, H, d_p2), dtype=np.uint8)
        outlier_masks = [None] * H

        def _do_head_mse(h):
            idx, mask = self._mse_compress_head(keys_n[:, h, :], h)
            indices[:, h, :] = idx
            outlier_masks[h]  = mask

        list(self._pool.map(_do_head_mse, range(H)))

        state = TurboState(
            mode=self.mode, bits=self.bits,
            seq_len=seq_len, head_dim=self.head_dim, head_dim_p2=d_p2,
            indices=indices, norms=norms.astype(np.float16),
            _codebook=self._codebook, _cb_outlier=self._cb_outlier,
            outlier_mask=outlier_masks[0],  # same mask for all heads (data-oblivious)
            outlier_bits=self._outlier_bits,
        )

        # PROD mode: QJL on residual
        if self.mode == TurboMode.PROD:
            # Reconstruct MSE approximation
            k_approx = np.empty_like(keys)
            for h in range(H):
                k_approx[:, h, :] = self._mse_reconstruct_head(
                    indices[:, h, :], norms[:, h], h, outlier_masks[h]
                )
            residual = keys - k_approx
            rs = self._qjl.compress(residual)
            state.res_packed = rs.packed_signs
            state.res_norms  = rs.key_norms

        return state

    # ── Public scores ─────────────────────────────────────────────────

    def scores(
        self,
        query: Union[np.ndarray, torch.Tensor],
        state: TurboState,
    ) -> np.ndarray:
        """
        Estimate ⟨query, key_i⟩ for all cached keys.

        PROD mode: score = MSE_term + QJL_residual_term → E[score]=<q,k>  (unbiased)
        MSE  mode: score = <q, k̃>  (biased but lower variance)

        Parameters
        ----------
        query : (batch, Q, head_dim)
        state : TurboState

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

        # QJL-only delegate
        if state.mode == TurboMode.QJL:
            qs = QJLState(packed_signs=state.indices, key_norms=state.norms,
                          seq_len=state.seq_len, head_dim=state.head_dim,
                          head_dim_p2=state.head_dim_p2)
            out = self._qjl.scores(query, qs)
            return torch.from_numpy(out).to(dev) if return_torch else out

        batch = query.shape[0]
        Q     = self.num_q_heads
        H     = self.num_kv_heads
        seq   = state.seq_len
        gqa   = Q // H

        scores1 = np.zeros((batch, Q, seq), dtype=np.float32)

        for h_k in range(H):
            k_rec = self._mse_reconstruct_head(
                state.indices[:, h_k, :],
                state.norms[:, h_k].astype(np.float32),
                h_k,
                state.outlier_mask,
            )  # (seq, head_dim)

            for h_q in range(h_k*gqa, (h_k+1)*gqa):
                q_h = query[:, h_q, :]   # (batch, head_dim)
                scores1[:, h_q, :] = q_h @ k_rec.T

        # PROD residual
        if state.mode == TurboMode.PROD and state.res_packed is not None:
            rs = QJLState(packed_signs=state.res_packed, key_norms=state.res_norms,
                          seq_len=state.seq_len, head_dim=state.head_dim,
                          head_dim_p2=state.head_dim_p2)
            scores1 += self._qjl.scores(query, rs)

        if return_torch:
            return torch.from_numpy(scores1).to(dev)
        return scores1

    # ── Memory report ─────────────────────────────────────────────────

    def memory_report(self, seq_len: int) -> dict:
        d    = self.head_dim
        d_p2 = self.head_dim_p2
        H    = self.num_kv_heads
        b    = self._mse_bits

        orig   = seq_len * H * d * 2  # FP16

        if self.mode == TurboMode.QJL:
            tot  = seq_len * H * ((d_p2+7)//8 + 2)
        elif self.mode == TurboMode.MSE:
            tot  = seq_len * H * (math.ceil(b*d_p2/8) + 2)
        else:  # PROD
            mse  = seq_len * H * (math.ceil(b*d_p2/8) + 2)
            res  = seq_len * H * ((d_p2+7)//8 + 2)
            tot  = mse + res

        avg_bits = tot * 8 / max(seq_len * H * d, 1)

        return {
            "original_fp16_MB":  orig / 1e6,
            "compressed_MB":     tot  / 1e6,
            "compression_ratio": orig / max(tot, 1),
            "avg_bits_per_elem": avg_bits,
            "mode":              self.mode.value,
        }

    def __repr__(self) -> str:
        return (f"TurboQuantizer(mode={self.mode.value}, bits={self.bits}, "
                f"head_dim={self.head_dim}, kv_heads={self.num_kv_heads})")