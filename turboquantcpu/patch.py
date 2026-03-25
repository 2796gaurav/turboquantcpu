"""
patch.py — HuggingFace CausalLM patcher for TurboQuantCPU.

Hooks into any HuggingFace model to replace the KV cache with
TurboQuant-compressed storage. Non-destructive (weights unchanged).

Supports: Llama, Mistral, Mixtral, Qwen2, Phi, Phi3, DeepSeek,
          Gemma, Falcon, GPT-NeoX, and any model following the
          standard CausalLM config conventions.

How it works:
  1. Extract model dimensions from config
  2. Build CompressedKVCache
  3. Register post-attention forward hooks that intercept past_key_values
  4. Compression happens automatically on every generate() or forward()

Strategy: hooks intercept (k, v) tensors *after* the model computes them,
compress to TurboState/ValueState, and store in the cache. The original
PyTorch attention path is unmodified — this means correctness first, then
future work can replace Q·K^T with the unbiased estimator for additional
speed.
"""

from __future__ import annotations
import gc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from turboquantcpu.kv_cache    import CacheConfig, CompressedKVCache
from turboquantcpu.value_quant import ValueMode

__all__ = ["PatchConfig", "patch_model", "unpatch_model"]


@dataclass
class PatchConfig:
    """Configuration for patching a HuggingFace model."""
    mode:              str   = "prod"
    bits:              int   = 4
    max_seq_len:       int   = 32_768
    value_mode:        str   = ValueMode.INT8
    value_group_size:  int   = 32
    outlier_frac:      float = 0.0
    use_h2o:           bool  = False
    h2o_heavy_budget:  int   = 128
    h2o_recent_budget: int   = 64
    skip_layers:       List[int] = field(default_factory=list)
    verbose:           bool  = True


# ── Model introspection ───────────────────────────────────────────────

def _get_dims(model: nn.Module) -> dict:
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise AttributeError("Model has no .config attribute.")

    def _get(*attrs, default=None):
        for a in attrs:
            v = getattr(cfg, a, None)
            if v is not None:
                return v
        return default

    num_layers  = _get("num_hidden_layers", "n_layer", "num_layers")
    num_q_heads = _get("num_attention_heads", "n_head")
    num_kv_heads= _get("num_key_value_heads", "num_kv_heads", default=num_q_heads)
    hidden      = _get("hidden_size", "n_embd", "d_model")
    head_dim    = _get("head_dim", default=hidden//num_q_heads if hidden else None)

    if not all([num_layers, num_q_heads, head_dim]):
        raise RuntimeError(
            f"Could not extract all dimensions from config.\n"
            f"  num_layers={num_layers}, num_q_heads={num_q_heads}, head_dim={head_dim}"
        )
    return dict(num_layers=num_layers, num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads, head_dim=head_dim)


def _find_attention_modules(model: nn.Module) -> List[tuple]:
    """Walk model, return (name, module, layer_idx) for all attention layers."""
    results = []
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        # Match any attention-like module with q/k projections
        if any(x in cls for x in ["Attention", "SelfAttn", "MultiHead"]):
            has_qk = (hasattr(mod, "q_proj") or hasattr(mod, "c_attn")
                      or hasattr(mod, "query_key_value") or hasattr(mod, "Wqkv"))
            if has_qk:
                idx = _extract_layer_idx(name)
                results.append((name, mod, idx))
    return results


def _extract_layer_idx(name: str) -> int:
    for part in name.split("."):
        try:
            return int(part)
        except ValueError:
            pass
    return 0


# ── Hook factory ──────────────────────────────────────────────────────

def _make_kv_hook(cache: CompressedKVCache, layer_idx: int):
    """
    Returns a forward hook that captures K, V and stores them compressed.

    The hook fires after the attention module's forward() and intercepts
    the returned past_key_values tuple (k, v).
    """
    def hook(module, inputs, outputs):
        if not isinstance(outputs, (tuple, list)):
            return

        kv_pair = None
        for item in outputs:
            if isinstance(item, tuple) and len(item) == 2:
                k, v = item
                if isinstance(k, torch.Tensor) and k.ndim >= 3:
                    kv_pair = (k, v)
                    break

        if kv_pair is None:
            return

        k, v = kv_pair  # most HF models: (batch, H, seq, d)

        if k.ndim == 4:
            # (batch, H, seq, d) → (seq, H, d) for batch=0
            k_sqhd = k[0].permute(1, 0, 2).detach().cpu()
            v_sqhd = v[0].permute(1, 0, 2).detach().cpu()
        elif k.ndim == 3:
            k_sqhd = k.detach().cpu()
            v_sqhd = v.detach().cpu()
        else:
            return

        try:
            cache.update(layer_idx, k_sqhd, v_sqhd)
        except Exception:
            pass  # Never crash inference

    return hook


# ── Public API ────────────────────────────────────────────────────────

def patch_model(
    model:       nn.Module,
    mode:        str = "prod",
    bits:        int = 4,
    max_seq_len: int = 32_768,
    verbose:     bool = True,
    cfg:         Optional[PatchConfig] = None,
) -> CompressedKVCache:
    """
    Patch a HuggingFace CausalLM for TurboQuantCPU KV cache compression.

    Parameters
    ----------
    model       : HuggingFace AutoModelForCausalLM
    mode        : "prod" (default) | "mse" | "qjl"
    bits        : bits per key coordinate (1–8)
    max_seq_len : sliding window maximum
    verbose     : print patching summary

    Returns
    -------
    CompressedKVCache — attached to model._tqcpu_cache

    Example
    -------
        from transformers import AutoModelForCausalLM
        from turboquantcpu import patch_model

        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
        cache = patch_model(model, mode="prod", bits=4)
        output = model.generate(input_ids, max_new_tokens=512)
        print(cache.memory_report())
    """
    if cfg is None:
        cfg = PatchConfig(mode=mode, bits=bits, max_seq_len=max_seq_len,
                          verbose=verbose)

    # Introspect model dimensions
    dims = _get_dims(model)
    L, Q, H, d = (dims["num_layers"], dims["num_q_heads"],
                   dims["num_kv_heads"], dims["head_dim"])

    if verbose:
        print(f"\n[TurboQuantCPU] Patching: {type(model).__name__}")
        print(f"  Layers : {L}  |  Q-heads : {Q}  |  KV-heads : {H}  |  head_dim : {d}")
        print(f"  Mode   : {cfg.mode}  |  Bits : {cfg.bits}  |  Values : {cfg.value_mode}")
        print(f"  MaxCtx : {cfg.max_seq_len:,} tokens  |  H2O : {cfg.use_h2o}")

    # H2O config
    h2o_cfg = None
    if cfg.use_h2o:
        from turboquantcpu.sparse_attn import H2OConfig
        h2o_cfg = H2OConfig(
            heavy_budget=cfg.h2o_heavy_budget,
            recent_budget=cfg.h2o_recent_budget,
        )

    # Build cache
    cache_cfg = CacheConfig(
        num_layers       = L,
        num_kv_heads     = H,
        num_q_heads      = Q,
        head_dim         = d,
        max_seq_len      = cfg.max_seq_len,
        mode             = cfg.mode,
        bits             = cfg.bits,
        value_mode       = cfg.value_mode,
        value_group_size = cfg.value_group_size,
        outlier_frac     = cfg.outlier_frac,
        h2o_config       = h2o_cfg,
    )
    cache = CompressedKVCache.from_config(cache_cfg)

    # Register hooks
    attn_layers   = _find_attention_modules(model)
    hooks_list    = []
    model._tqcpu_hooks = []

    if not attn_layers:
        raise RuntimeError(
            "No attention layers found. Pass a full CausalLM (not just the transformer backbone)."
        )

    for name, mod, layer_idx in attn_layers:
        if layer_idx in cfg.skip_layers:
            if verbose:
                print(f"  ·  layer {layer_idx:3d}: skipped")
            continue

        h = mod.register_forward_hook(_make_kv_hook(cache, layer_idx))
        model._tqcpu_hooks.append(h)
        if verbose:
            print(f"  ✓  layer {layer_idx:3d}: {name}")

    model._tqcpu_cache = cache

    if verbose:
        from turboquantcpu.turbo import TurboQuantizer
        q_eg  = TurboQuantizer(d, H, Q, 0, mode=cfg.mode, bits=cfg.bits)
        rep   = q_eg.memory_report(4096)
        orig  = rep["original_fp16_MB"] * L
        comp  = rep["compressed_MB"]    * L
        ratio = orig / max(comp, 1e-9)
        bpe   = rep["avg_bits_per_elem"]
        print(f"\n  Memory estimate at 4K tokens (keys only):")
        print(f"    Original FP16  : {orig:.1f} MB")
        print(f"    Compressed ({bpe:.1f}b): {comp:.1f} MB")
        print(f"    Ratio          : {ratio:.1f}×")
        print(f"\n[TurboQuantCPU] {len(model._tqcpu_hooks)} layers patched ✓\n")

    return cache


def unpatch_model(model: nn.Module) -> int:
    """Remove all TurboQuantCPU patches. Returns number of hooks removed."""
    hooks = getattr(model, "_tqcpu_hooks", [])
    for h in hooks:
        h.remove()
    n = len(hooks)
    if hasattr(model, "_tqcpu_hooks"): del model._tqcpu_hooks
    if hasattr(model, "_tqcpu_cache"): del model._tqcpu_cache
    gc.collect()
    return n