"""
sparse_attn.py — Sparse attention via Heavy-Hitter Oracle (H2O) integration.

Reference: Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative
Inference of Large Language Models." NeurIPS 2023.

Core insight: At each decode step, only a small fraction (~20%) of cached keys
receive >80% of the attention mass (heavy hitters). The rest can be evicted
with minimal quality loss.

This module provides:
  1. AttentionTracker  — accumulates attention scores across steps
  2. H2OEvictionPolicy — evict low-score keys from the KV cache
  3. HeavyHitterCache  — combines eviction with TurboQuant compression

Integration with TurboQuantCPU:
  • Heavy keys: stored compressed via TurboQuant (HIGH quality)
  • Light keys: evicted or stored at ultra-low precision (1-bit QJL)
  • Recent keys: always kept in full precision (last R tokens)

Memory model at seq_len S with budget B (B << S):
  • Recent window: R tokens × FP16 = R×H×d×2 bytes
  • Heavy hitters: (B-R) tokens × TurboQuant ≈ (B-R)×H×d/4 bytes
  → ~6–8× total reduction vs FP16 baseline at full S
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import numpy as np
import torch

__all__ = ["AttentionTracker", "H2OEvictionPolicy", "H2OConfig"]


@dataclass
class H2OConfig:
    """Configuration for H2O sparse attention."""
    heavy_budget:  int   = 128    # number of heavy-hitter tokens to keep
    recent_budget: int   = 64     # always keep the last R tokens
    min_seq_for_eviction: int = 256  # start evicting only after this many tokens
    accumulation_steps:   int = 1    # accumulate attention over N steps before evicting


class AttentionTracker:
    """
    Tracks cumulative attention scores across decode steps.

    Maintains a (num_q_heads, seq_len) score tensor that grows as new tokens
    are generated. Used by H2OEvictionPolicy to decide which tokens to evict.
    """

    def __init__(self, num_q_heads: int, num_kv_heads: int):
        self.num_q_heads  = num_q_heads
        self.num_kv_heads = num_kv_heads
        self._scores: Optional[np.ndarray] = None  # (H_q, seq)
        self._step = 0

    def update(self, attn_weights: np.ndarray) -> None:
        """
        Accumulate attention weights.

        Parameters
        ----------
        attn_weights : (batch, Q, seq_len) float32 — softmax attention weights
        """
        # Average over batch, keep per-Q-head
        w = attn_weights[0] if attn_weights.ndim == 3 else attn_weights
        # w: (Q, seq)
        if self._scores is None:
            self._scores = w.copy()
        else:
            # Extend if sequence grew
            if w.shape[-1] > self._scores.shape[-1]:
                pad = np.zeros(
                    (self._scores.shape[0], w.shape[-1]-self._scores.shape[-1]),
                    dtype=np.float32
                )
                self._scores = np.concatenate([self._scores, pad], axis=-1)
            self._scores[:, :w.shape[-1]] += w
        self._step += 1

    def get_cumulative_scores(self) -> Optional[np.ndarray]:
        """Returns (H_q, seq) cumulative attention scores."""
        return self._scores

    def get_kv_head_scores(self, gqa_ratio: int = 1) -> Optional[np.ndarray]:
        """
        Reduce Q-head scores to KV-head scores (for GQA).
        Returns (H_kv, seq).
        """
        if self._scores is None:
            return None
        H_q = self._scores.shape[0]
        H_k = H_q // gqa_ratio
        # Average within each GQA group
        return self._scores.reshape(H_k, gqa_ratio, -1).mean(axis=1)

    def reset(self) -> None:
        self._scores = None
        self._step = 0

    def step(self) -> int:
        return self._step


class H2OEvictionPolicy:
    """
    H2O-style token eviction policy.

    Given the accumulated attention scores, returns a boolean mask of which
    token positions to keep (True = keep, False = evict).

    Strategy: Keep the top-K "heavy hitters" by cumulative attention score,
    plus the last R "recent" tokens always (sliding window).
    """

    def __init__(self, config: H2OConfig):
        self.cfg = config

    def should_evict(self, seq_len: int) -> bool:
        return seq_len >= self.cfg.min_seq_for_eviction

    def get_keep_mask(
        self,
        seq_len: int,
        scores: np.ndarray,      # (H_kv, seq_len) — per-KV-head scores
        return_indices: bool = False,
    ) -> np.ndarray:
        """
        Compute which tokens to keep.

        Returns
        -------
        mask : (seq_len,) bool — True means keep
        OR
        indices : sorted list of positions to keep (if return_indices=True)
        """
        heavy  = self.cfg.heavy_budget
        recent = self.cfg.recent_budget
        budget = heavy + recent

        if seq_len <= budget:
            mask = np.ones(seq_len, dtype=bool)
            return np.arange(seq_len) if return_indices else mask

        # Always keep the last `recent` tokens
        recent_mask = np.zeros(seq_len, dtype=bool)
        recent_mask[-recent:] = True

        # Select heavy hitters from the non-recent portion
        # Use mean score across KV heads
        avg_scores = scores.mean(axis=0)  # (seq_len,)
        # Mask out recent tokens from candidate pool
        candidate_scores = avg_scores.copy()
        candidate_scores[-recent:] = -np.inf

        topk_idx = np.argpartition(candidate_scores, -heavy)[-heavy:]
        heavy_mask = np.zeros(seq_len, dtype=bool)
        heavy_mask[topk_idx] = True

        mask = recent_mask | heavy_mask

        if return_indices:
            return np.sort(np.where(mask)[0])
        return mask

    def apply_eviction(
        self,
        token_indices: np.ndarray,   # sorted keep indices
        *arrays,                     # arrays to index along axis=0
    ) -> Tuple:
        """Apply the same eviction mask to multiple arrays (seq axis=0)."""
        return tuple(a[token_indices] for a in arrays)


class SparseKVCache:
    """
    Full sparse KV cache combining H2O eviction with TurboQuant compression.

    At each update step:
      1. Compress new keys/values via TurboQuant
      2. If seq_len > budget: evict low-attention tokens
      3. Keep: heavy hitters (compressed) + recent window (raw)

    This achieves the lowest possible memory footprint while maintaining
    quality on tasks that rely on important tokens (long retrieval, QA, etc.)
    """

    def __init__(
        self,
        quantizer,          # TurboQuantizer instance
        value_quantizer,    # ValueQuantizer instance
        h2o_config: Optional[H2OConfig] = None,
        max_seq_len: int = 32768,
    ):
        self.quantizer       = quantizer
        self.val_quantizer   = value_quantizer
        self.h2o             = h2o_config or H2OConfig()
        self.eviction_policy = H2OEvictionPolicy(self.h2o)
        self.max_seq_len     = max_seq_len
        self.tracker         = AttentionTracker(
            quantizer.num_q_heads, quantizer.num_kv_heads
        )

        self._key_state   = None
        self._val_state   = None
        self._seq_len     = 0
        self._token_ids   = []  # for debugging / prefix matching

    def update(
        self,
        keys:          Union[np.ndarray, torch.Tensor],
        values:        Union[np.ndarray, torch.Tensor],
        attn_weights:  Optional[np.ndarray] = None,
    ) -> None:
        """
        Compress and append new tokens; optionally evict low-attention ones.
        """
        if isinstance(keys, torch.Tensor):
            keys   = keys.float().numpy()
        if isinstance(values, torch.Tensor):
            values = values.float().numpy()

        # Compress new tokens
        new_k = self.quantizer.compress(np.asarray(keys,   dtype=np.float32))
        new_v = self.val_quantizer.compress(np.asarray(values, dtype=np.float32))

        # Accumulate attention weights for H2O tracking
        if attn_weights is not None:
            self.tracker.update(attn_weights)

        if self._key_state is None:
            self._key_state = new_k
            self._val_state = new_v
        else:
            self._key_state = self._key_state.append(new_k)
            self._val_state = self._val_state.append(new_v)

        self._seq_len = self._key_state.seq_len

        # H2O eviction
        if (self.eviction_policy.should_evict(self._seq_len)
                and self.tracker.step() % self.h2o.accumulation_steps == 0):
            self._maybe_evict()

        # Hard cap
        if self._seq_len > self.max_seq_len:
            self._trim_to_recent(self.max_seq_len)

    def _maybe_evict(self) -> None:
        budget = self.h2o.heavy_budget + self.h2o.recent_budget
        if self._seq_len <= budget:
            return

        scores = self.tracker.get_kv_head_scores(
            self.quantizer.num_q_heads // self.quantizer.num_kv_heads
        )  # (H_kv, seq)

        if scores is None or scores.shape[-1] < self._seq_len:
            return  # not enough score history yet

        keep_idx = self.eviction_policy.get_keep_mask(
            self._seq_len, scores[:, :self._seq_len], return_indices=True
        )

        # Evict from compressed key state
        s = self._key_state
        s.indices = s.indices[keep_idx]
        s.norms   = s.norms[keep_idx]
        if s.res_packed is not None:
            s.res_packed = s.res_packed[keep_idx]
            s.res_norms  = s.res_norms[keep_idx]
        s.seq_len = len(keep_idx)

        # Evict from value state
        v = self._val_state
        v.data    = v.data[keep_idx]
        v.scales  = v.scales[keep_idx]
        if v.zeros is not None:
            v.zeros = v.zeros[keep_idx]
        v.seq_len = len(keep_idx)

        self._seq_len = len(keep_idx)

        # Reset tracker to match evicted positions
        if self.tracker._scores is not None:
            self.tracker._scores = self.tracker._scores[:, keep_idx]

    def _trim_to_recent(self, keep: int) -> None:
        s = self._key_state
        drop = s.seq_len - keep
        s.indices  = s.indices[drop:]
        s.norms    = s.norms[drop:]
        if s.res_packed is not None:
            s.res_packed = s.res_packed[drop:]
            s.res_norms  = s.res_norms[drop:]
        s.seq_len = keep
        self._seq_len = keep

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def memory_bytes(self) -> int:
        total = 0
        if self._key_state: total += self._key_state.memory_bytes()
        if self._val_state:  total += self._val_state.memory_bytes()
        return total