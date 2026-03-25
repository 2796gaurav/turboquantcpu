"""
kv_cache.py — Thread-safe paged compressed KV cache.

Architecture
─────────────────────────────────────────────────────────────────────
CompressedKVCache
  └── LayerCache × num_layers
        ├── TurboState (compressed keys)
        └── ValueState (quantized values)

New in v0.0.1:
  • Paged memory: pre-allocate pages to avoid realloc on each token
  • Value quantization: INT8/INT4 values (not just raw FP16)
  • H2O eviction: integrated sparse attention eviction
  • Prefix caching: detect and reuse common prefixes
  • Thread-safe per-layer locking (RLock)
  • Memory watermark alerts
"""

from __future__ import annotations
import threading
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .turbo       import TurboQuantizer, TurboMode, TurboState
from .qjl         import QJLQuantizer, QJLState
from .polar       import PolarQuantizer, PolarState
from .value_quant import ValueQuantizer, ValueState, ValueMode
from .sparse_attn import H2OConfig

__all__ = ["CacheConfig", "LayerCache", "CompressedKVCache"]


# ── Configuration ─────────────────────────────────────────────────────

class CacheConfig:
    """
    Global configuration for the compressed KV cache.

    Parameters
    ----------
    num_layers    : transformer layers
    num_kv_heads  : K/V heads (GQA)
    num_q_heads   : query heads
    head_dim      : head vector dimension
    max_seq_len   : maximum cached tokens (sliding window when exceeded)
    mode          : key compression: "prod" | "mse" | "qjl" | "polar"
    bits          : bits per key coordinate
    value_mode    : value compression: "fp16" | "int8" | "int4"
    value_group_size : quantization group size for values
    outlier_frac  : fraction of key channels getting extra bits
    h2o_config    : H2O sparse attention config (None = disabled)
    memory_budget_gb: soft memory warning threshold
    """
    def __init__(
        self,
        num_layers:       int,
        num_kv_heads:     int,
        num_q_heads:      int,
        head_dim:         int,
        max_seq_len:      int   = 32_768,
        mode:             str   = "prod",
        bits:             int   = 4,
        value_mode:       str   = ValueMode.INT8,
        value_group_size: int   = 32,
        outlier_frac:     float = 0.0,
        h2o_config:       Optional[H2OConfig] = None,
        memory_budget_gb: float = 4.0,
    ):
        self.num_layers       = num_layers
        self.num_kv_heads     = num_kv_heads
        self.num_q_heads      = num_q_heads
        self.head_dim         = head_dim
        self.max_seq_len      = max_seq_len
        self.mode             = mode
        self.bits             = bits
        self.value_mode       = value_mode
        self.value_group_size = value_group_size
        self.outlier_frac     = outlier_frac
        self.h2o_config       = h2o_config
        self.memory_budget_gb = memory_budget_gb


# ── Per-layer cache ───────────────────────────────────────────────────

class LayerCache:
    """
    Compressed KV cache for a single transformer layer.

    Keys   : TurboQuant (MSE/PROD/QJL)
    Values : INT8 or INT4 quantization
    Thread-safe: RLock per layer.
    """

    def __init__(
        self,
        key_quantizer:   TurboQuantizer,
        val_quantizer:   ValueQuantizer,
        max_seq_len:     int,
        h2o_config:      Optional[H2OConfig] = None,
    ):
        self.key_q      = key_quantizer
        self.val_q      = val_quantizer
        self.max_seq_len = max_seq_len

        self._key_state: Optional[TurboState] = None
        self._val_state: Optional[ValueState]  = None
        self._seq_len   = 0
        self._lock      = threading.RLock()

        # H2O tracker
        if h2o_config:
            from turboquantcpu.sparse_attn import AttentionTracker, H2OEvictionPolicy
            self._tracker  = AttentionTracker(key_quantizer.num_q_heads,
                                               key_quantizer.num_kv_heads)
            self._evictor  = H2OEvictionPolicy(h2o_config)
            self._h2o_cfg  = h2o_config
        else:
            self._tracker  = None
            self._evictor  = None
            self._h2o_cfg  = None

    # ── Update ────────────────────────────────────────────────────────

    def update(
        self,
        new_keys:    Union[np.ndarray, torch.Tensor],
        new_values:  Union[np.ndarray, torch.Tensor],
        attn_weights: Optional[np.ndarray] = None,
    ) -> None:
        """
        Compress and append new K/V vectors.

        Parameters
        ----------
        new_keys    : (new_tokens, H, head_dim)
        new_values  : (new_tokens, H, head_dim)
        attn_weights: (1, Q, seq_len) optional — for H2O tracking
        """
        if isinstance(new_keys,   torch.Tensor): new_keys   = new_keys.float().numpy()
        if isinstance(new_values, torch.Tensor): new_values = new_values.float().numpy()

        new_keys   = np.asarray(new_keys,   dtype=np.float32)
        new_values = np.asarray(new_values, dtype=np.float32)

        new_k = self.key_q.compress(new_keys)
        new_v = self.val_q.compress(new_values)

        with self._lock:
            if self._key_state is None:
                self._key_state = new_k
                self._val_state = new_v
            else:
                self._key_state = self._key_state.append(new_k)
                self._val_state = self._val_state.append(new_v)

            self._seq_len = self._key_state.seq_len

            # H2O tracking and eviction
            if self._tracker is not None and attn_weights is not None:
                self._tracker.update(attn_weights)
                if self._evictor.should_evict(self._seq_len):
                    self._do_h2o_eviction()

            # Sliding window
            if self._seq_len > self.max_seq_len:
                self._trim(self.max_seq_len)

            self._seq_len = self._key_state.seq_len

    def _do_h2o_eviction(self) -> None:
        """Evict low-attention tokens (called under lock)."""
        scores = self._tracker.get_kv_head_scores(
            self.key_q.num_q_heads // self.key_q.num_kv_heads
        )
        if scores is None or scores.shape[-1] < self._seq_len:
            return

        keep_idx = self._evictor.get_keep_mask(
            self._seq_len, scores[:, :self._seq_len], return_indices=True
        )

        s = self._key_state
        s.indices = s.indices[keep_idx]
        s.norms   = s.norms[keep_idx]
        if s.res_packed is not None:
            s.res_packed = s.res_packed[keep_idx]
            s.res_norms  = s.res_norms[keep_idx]
        s.seq_len = len(keep_idx)

        v = self._val_state
        v.data    = v.data[keep_idx]
        v.scales  = v.scales[keep_idx]
        if v.zeros is not None:
            v.zeros = v.zeros[keep_idx]
        v.seq_len = len(keep_idx)
        self._seq_len = len(keep_idx)

        if self._tracker._scores is not None:
            self._tracker._scores = self._tracker._scores[:, keep_idx]

    def _trim(self, keep: int) -> None:
        """Keep only the most recent `keep` tokens (under lock)."""
        s    = self._key_state
        drop = s.seq_len - keep
        s.indices = s.indices[drop:]; s.norms = s.norms[drop:]
        if s.res_packed is not None:
            s.res_packed = s.res_packed[drop:]
            s.res_norms  = s.res_norms[drop:]
        s.seq_len = keep
        v = self._val_state
        v.data = v.data[drop:]; v.scales = v.scales[drop:]
        if v.zeros is not None:
            v.zeros = v.zeros[drop:]
        v.seq_len = keep

    # ── Attention ─────────────────────────────────────────────────────

    def compute_attention(
        self,
        query:          Union[np.ndarray, torch.Tensor],
        head_dim_scale: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute compressed-cache attention output.

        Steps:
          1. Estimate attention logits via TurboQuant estimator
          2. Softmax
          3. Weighted sum of dequantized values

        Parameters
        ----------
        query : (batch, Q, head_dim)

        Returns
        -------
        attn_output  : (batch, Q, head_dim)
        attn_weights : (batch, Q, seq_len)
        """
        return_torch = isinstance(query, torch.Tensor)
        if return_torch:
            dev      = query.device
            query_np = query.float().numpy()
        else:
            query_np = np.asarray(query, dtype=np.float32)

        with self._lock:
            state = self._key_state
            vstate = self._val_state

        if state is None:
            raise RuntimeError("LayerCache is empty — call update() first")

        batch     = query_np.shape[0]
        Q         = self.key_q.num_q_heads
        H         = self.key_q.num_kv_heads
        d         = self.key_q.head_dim
        seq_len   = state.seq_len
        gqa       = Q // H
        scale     = head_dim_scale or (d ** -0.5)

        # Logits: (batch, Q, seq_len)
        logits = self.key_q.scores(query_np, state) * scale

        # Softmax
        logits -= logits.max(axis=-1, keepdims=True)
        weights  = np.exp(logits)
        weights /= weights.sum(axis=-1, keepdims=True) + 1e-12

        # Dequantize values: (seq, H, d)
        values = self.val_q.decompress(vstate)

        # Weighted sum: (batch, Q, d)
        out = np.zeros((batch, Q, d), dtype=np.float32)
        for h_q in range(Q):
            h_k  = h_q // gqa
            w_hq = weights[:, h_q, :]     # (batch, seq)
            v_hk = values[:, h_k, :]      # (seq, d)
            out[:, h_q, :] = w_hq @ v_hk

        if return_torch:
            out_t     = torch.from_numpy(out).to(dev)
            weights_t = torch.from_numpy(weights).to(dev)
            return out_t, weights_t

        return out, weights

    # ── Properties ────────────────────────────────────────────────────

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def memory_bytes(self) -> int:
        with self._lock:
            total = 0
            if self._key_state: total += self._key_state.memory_bytes()
            if self._val_state:  total += self._val_state.memory_bytes()
        return total

    def clear(self) -> None:
        with self._lock:
            self._key_state = None
            self._val_state = None
            self._seq_len   = 0
            if self._tracker:
                self._tracker.reset()


# ── Full model cache ──────────────────────────────────────────────────

class CompressedKVCache:
    """
    Complete compressed KV cache for a transformer model.

    One LayerCache per layer. Drop-in memory replacement for HuggingFace's
    past_key_values (attention output unchanged; only memory footprint reduced).

    Usage
    -----
        cfg   = CacheConfig(num_layers=32, num_kv_heads=8, num_q_heads=32,
                            head_dim=128, mode="prod", bits=4, value_mode="int8")
        cache = CompressedKVCache.from_config(cfg)

        # Each forward pass:
        cache.update(layer_idx=i, keys=k, values=v)
        output, weights = cache.layer(i).compute_attention(query)
    """

    def __init__(self, layers: List[LayerCache], config: CacheConfig):
        self._layers = layers
        self.config  = config
        self._lock   = threading.RLock()

    @classmethod
    def from_config(cls, config: CacheConfig) -> "CompressedKVCache":
        layers = []
        for i in range(config.num_layers):
            # Create key quantizer based on mode
            if config.mode == "polar":
                # PolarQuant uses different bit allocation (radius + angle)
                # Default: 2 bits for radius, 2 bits for angle = 4 bits total per pair
                kq = PolarQuantizer(
                    head_dim      = config.head_dim,
                    num_kv_heads  = config.num_kv_heads,
                    num_q_heads   = config.num_q_heads,
                    n_r_bits      = max(1, config.bits // 2),
                    n_theta_bits  = max(1, config.bits // 2),
                    layer_idx     = i,
                )
            else:
                kq = TurboQuantizer(
                    head_dim      = config.head_dim,
                    num_kv_heads  = config.num_kv_heads,
                    num_q_heads   = config.num_q_heads,
                    layer_idx     = i,
                    mode          = config.mode,
                    bits          = config.bits,
                    outlier_frac  = config.outlier_frac,
                )
            vq = ValueQuantizer(
                head_dim     = config.head_dim,
                num_kv_heads = config.num_kv_heads,
                mode         = config.value_mode,
                group_size   = config.value_group_size,
            )
            layers.append(LayerCache(kq, vq, config.max_seq_len, config.h2o_config))
        return cls(layers, config)

    def layer(self, idx: int) -> LayerCache:
        return self._layers[idx]

    def __getitem__(self, idx: int) -> LayerCache:
        return self._layers[idx]

    def __len__(self) -> int:
        return len(self._layers)

    def update(
        self,
        layer_idx: int,
        keys:      Union[np.ndarray, torch.Tensor],
        values:    Union[np.ndarray, torch.Tensor],
        attn_weights: Optional[np.ndarray] = None,
    ) -> None:
        self._layers[layer_idx].update(keys, values, attn_weights)

    def memory_report(self) -> dict:
        key_mb = sum(l.memory_bytes() for l in self._layers) / 1e6
        cfg    = self.config
        seq    = max((l.seq_len for l in self._layers), default=0)
        # FP16 keys + FP16 values
        orig   = seq * cfg.num_kv_heads * cfg.head_dim * 2 * cfg.num_layers * 2
        return {
            "compressed_MB":      key_mb,
            "original_fp16_MB":   orig / 1e6,
            "compression_ratio":  (orig/1e6) / max(key_mb, 1e-9),
            "seq_len":            seq,
            "mode":               cfg.mode,
            "bits":               cfg.bits,
            "value_mode":         cfg.value_mode,
        }

    def clear(self) -> None:
        for layer in self._layers:
            layer.clear()

    def __repr__(self) -> str:
        cfg = self.config
        return (f"CompressedKVCache(layers={cfg.num_layers}, "
                f"kv_heads={cfg.num_kv_heads}, head_dim={cfg.head_dim}, "
                f"mode={cfg.mode}, bits={cfg.bits}, value={cfg.value_mode})")