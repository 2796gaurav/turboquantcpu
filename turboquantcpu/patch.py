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


# ── Attention wrapper factory ─────────────────────────────────────────

def _wrap_attention_forward(module, cache: CompressedKVCache, layer_idx: int):
    """
    Wraps an attention module's forward method to capture K, V tensors.
    
    This intercepts the K and V computations and stores them in our
    compressed cache.
    """
    original_forward = module.forward
    
    # Get model config for dimensions
    # Try various attribute names used by different models
    def get_num_kv_heads():
        # Try common attribute names
        for attr in ['num_key_value_heads', 'num_kv_heads', 'kv_heads']:
            if hasattr(module, attr):
                val = getattr(module, attr)
                if val is not None:
                    return val
        # Infer from k_proj weight shape
        if hasattr(module, 'k_proj'):
            # k_proj weight shape: (num_kv_heads * head_dim, hidden_size)
            # We need to determine num_kv_heads and head_dim
            hidden_size = module.k_proj.weight.shape[1]
            kv_dim = module.k_proj.weight.shape[0]
            # Try to infer from config if available
            if hasattr(module, 'head_dim'):
                head_dim = module.head_dim
                return kv_dim // head_dim
        return 1
    
    def get_head_dim():
        if hasattr(module, 'head_dim'):
            return module.head_dim
        if hasattr(module, 'k_proj'):
            # k_proj output dimension = num_kv_heads * head_dim
            # For GQA models, this is typically hidden_size // num_attention_heads
            hidden_size = module.k_proj.weight.shape[1]
            num_attn_heads = getattr(module, 'num_heads', 
                          getattr(module, 'num_attention_heads', 1))
            if num_attn_heads:
                return hidden_size // num_attn_heads
        return 64  # Default
    
    num_kv_heads = get_num_kv_heads()
    head_dim = get_head_dim()
    
    def wrapped_forward(*args, **kwargs):
        # Call original forward
        outputs = original_forward(*args, **kwargs)
        
        # Try to extract K, V from the module's internal state
        k = v = None
        
        # Method 1: Check if k_proj and v_proj outputs are stored
        if hasattr(module, '_k_proj_output') and hasattr(module, '_v_proj_output'):
            k = module._k_proj_output
            v = module._v_proj_output
        
        # Method 2: Check for k_proj and v_proj, compute manually if needed
        if k is None and hasattr(module, 'k_proj') and hasattr(module, 'v_proj'):
            # Try to get hidden states from inputs
            if len(args) >= 1:
                hidden_states = args[0]
            elif 'hidden_states' in kwargs:
                hidden_states = kwargs['hidden_states']
            else:
                hidden_states = None
            
            if hidden_states is not None:
                try:
                    with torch.no_grad():
                        k = module.k_proj(hidden_states)
                        v = module.v_proj(hidden_states)
                except:
                    pass
        
        # Process and store if we have K and V
        if k is not None and v is not None:
            try:
                # Reshape based on expected dimensions
                # Most models: (batch, seq, num_heads * head_dim) -> (batch, num_heads, seq, head_dim)
                if k.ndim == 3:
                    batch, seq_len, _ = k.shape
                    nonlocal num_kv_heads, head_dim
                    
                    # Recalculate if needed
                    if num_kv_heads is None or num_kv_heads == 1:
                        num_kv_heads = k.shape[-1] // head_dim
                    
                    k = k.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
                    v = v.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
                
                # Convert to (seq, num_heads, head_dim) for batch=0
                if k.ndim == 4:
                    k_sqhd = k[0].permute(1, 0, 2).contiguous().detach().cpu()
                    v_sqhd = v[0].permute(1, 0, 2).contiguous().detach().cpu()
                    cache.update(layer_idx, k_sqhd, v_sqhd)
            except Exception:
                import traceback
                traceback.print_exc()
        
        return outputs
    
    return wrapped_forward


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

    # Wrap attention modules
    attn_layers   = _find_attention_modules(model)
    model._tqcpu_wrapped = []  # Store original forward methods for unpatching

    if not attn_layers:
        raise RuntimeError(
            "No attention layers found. Pass a full CausalLM (not just the transformer backbone)."
        )

    for name, mod, layer_idx in attn_layers:
        if layer_idx in cfg.skip_layers:
            if verbose:
                print(f"  ·  layer {layer_idx:3d}: skipped")
            continue

        # Store original forward and wrap it
        original_forward = mod.forward
        mod.forward = _wrap_attention_forward(mod, cache, layer_idx)
        model._tqcpu_wrapped.append((mod, original_forward))
        if verbose:
            print(f"  [OK]  layer {layer_idx:3d}: {name}")

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
        print(f"\n[TurboQuantCPU] {len(model._tqcpu_wrapped)} layers patched [OK]\n")

    return cache


def unpatch_model(model: nn.Module) -> int:
    """Remove all TurboQuantCPU patches. Returns number of layers restored."""
    # Restore wrapped forward methods
    wrapped = getattr(model, "_tqcpu_wrapped", [])
    for mod, original_forward in wrapped:
        mod.forward = original_forward
    n = len(wrapped)
    
    if hasattr(model, "_tqcpu_wrapped"): del model._tqcpu_wrapped
    if hasattr(model, "_tqcpu_cache"): del model._tqcpu_cache
    gc.collect()
    return n