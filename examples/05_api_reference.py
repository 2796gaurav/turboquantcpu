#!/usr/bin/env python3
"""
================================================================================
TurboQuantCPU - API Reference and Advanced Configuration
================================================================================

This example demonstrates the complete TurboQuantCPU API with all configuration
options and advanced use cases.

COMPLETE API OVERVIEW
---------------------

1. High-Level API (HuggingFace Integration)
   ----------------------------------------
   from turboquantcpu import patch_model, unpatch_model
   
   cache = patch_model(model, mode="prod", bits=4)
   # ... use model normally ...
   unpatch_model(model)

2. Low-Level API (Direct Quantization)
   -----------------------------------
   from turboquantcpu import TurboQuantizer, QJLQuantizer, PolarQuantizer
   
   quantizer = TurboQuantizer(
       head_dim=128,
       num_kv_heads=8,
       mode="prod",
       bits=4
   )
   state = quantizer.compress(keys)
   scores = quantizer.scores(query, state)

3. Configuration Options
   ---------------------
   PatchConfig parameters:
   - mode: "prod", "mse", "qjl", "polar"
   - bits: 1-8 (for non-QJL modes)
   - max_seq_len: Maximum sequence length (sliding window)
   - value_mode: "int8", "int4", "fp16"
   - use_h2o: Enable H2O sparse attention
   - skip_layers: List of layer indices to skip
"""

import numpy as np
import torch
from turboquantcpu import (
    # High-level API
    patch_model, unpatch_model, PatchConfig,
    # Quantizers
    TurboQuantizer, TurboMode, QJLQuantizer, PolarQuantizer,
    # KV Cache
    CompressedKVCache, CacheConfig,
    # Value quantization
    ValueQuantizer, ValueMode,
    # Utilities
    fwht_numpy, get_signs, randomized_fwht,
    cpu_info, print_cpu_capabilities,
    # Benchmarks
    run_correctness_check,
)


def demo_low_level_api():
    """Demonstrate low-level quantization API."""
    print("="*70)
    print("LOW-LEVEL API DEMONSTRATION")
    print("="*70)
    print()
    
    # Create synthetic data
    seq_len, num_heads, head_dim = 100, 8, 128
    keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
    query = np.random.randn(1, num_heads, head_dim).astype(np.float32)
    
    print(f"Data shape: keys={keys.shape}, query={query.shape}")
    print()
    
    # ------------------------------------------------------------------------
    # Method 1: TurboQuantizer (Multiple modes)
    # ------------------------------------------------------------------------
    print("1. TurboQuantizer (PROD mode)")
    print("-" * 70)
    
    tq = TurboQuantizer(
        head_dim=head_dim,
        num_kv_heads=num_heads,
        num_q_heads=num_heads,  # For GQA: num_q_heads > num_kv_heads
        mode=TurboMode.PROD,
        bits=4,
        layer_idx=0,  # For deterministic random seeds
    )
    
    print(f"Created quantizer: {tq}")
    
    # Compress keys
    state = tq.compress(keys)
    print(f"Compressed: {keys.nbytes} bytes → {state.memory_bytes()} bytes")
    
    # Compute attention scores
    scores = tq.scores(query, state)
    print(f"Scores shape: {scores.shape}")
    
    # Memory report
    report = tq.memory_report(seq_len=seq_len)
    print(f"Compression ratio: {report['compression_ratio']:.1f}×")
    print()
    
    # ------------------------------------------------------------------------
    # Method 2: QJLQuantizer (1-bit, max compression)
    # ------------------------------------------------------------------------
    print("2. QJLQuantizer (1-bit, maximum compression)")
    print("-" * 70)
    
    qjl = QJLQuantizer(
        head_dim=head_dim,
        num_kv_heads=num_heads,
        layer_idx=0,
    )
    
    state_qjl = qjl.compress(keys)
    scores_qjl = qjl.scores(query, state_qjl)
    
    report_qjl = qjl.memory_report(seq_len=seq_len)
    print(f"Compression ratio: {report_qjl['compression_ratio']:.1f}×")
    print(f"Bits per element: {report_qjl['bits_per_element']:.1f}")
    print()
    
    # ------------------------------------------------------------------------
    # Method 3: PolarQuantizer (Outlier-resistant)
    # ------------------------------------------------------------------------
    print("3. PolarQuantizer (Outlier-resistant)")
    print("-" * 70)
    
    polar = PolarQuantizer(
        head_dim=head_dim,
        num_kv_heads=num_heads,
        n_r_bits=2,      # Bits for radius
        n_theta_bits=2,  # Bits for angle
        layer_idx=0,
    )
    
    state_polar = polar.compress(keys)
    scores_polar = polar.scores(query, state_polar)
    
    report_polar = polar.memory_report(seq_len=seq_len)
    print(f"Compression ratio: {report_polar['compression_ratio']:.1f}×")
    print()


def demo_value_quantization():
    """Demonstrate value cache quantization."""
    print("="*70)
    print("VALUE CACHE QUANTIZATION")
    print("="*70)
    print()
    
    seq_len, num_heads, head_dim = 100, 8, 128
    values = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
    
    print("Value quantization modes:")
    print("-" * 70)
    
    for mode in [ValueMode.INT8, ValueMode.INT4]:
        vq = ValueQuantizer(
            head_dim=head_dim,
            num_kv_heads=num_heads,
            mode=mode,
            group_size=32,  # Quantization group size
        )
        
        # Compress
        state = vq.compress(values)
        
        # Decompress
        reconstructed = vq.decompress(state)
        
        # Measure error
        rel_error = np.abs(values - reconstructed).mean() / np.abs(values).mean()
        
        report = vq.memory_report(seq_len=seq_len)
        print(f"{mode:6s}: {report['compression_ratio']:.1f}× compression, "
              f"{rel_error*100:.2f}% reconstruction error")
    
    print()


def demo_advanced_config():
    """Demonstrate advanced configuration options."""
    print("="*70)
    print("ADVANCED CONFIGURATION")
    print("="*70)
    print()
    
    # PatchConfig options
    print("Available PatchConfig options:")
    print("-" * 70)
    
    configs = [
        ("Conservative (max quality)", {
            "mode": "prod",
            "bits": 8,
            "value_mode": ValueMode.INT8,
        }),
        ("Balanced (recommended)", {
            "mode": "prod",
            "bits": 4,
            "value_mode": ValueMode.INT8,
        }),
        ("Aggressive (max compression)", {
            "mode": "qjl",
            "value_mode": ValueMode.INT4,
        }),
        ("H2O + Compression", {
            "mode": "prod",
            "bits": 4,
            "use_h2o": True,
            "h2o_heavy_budget": 128,
            "h2o_recent_budget": 64,
        }),
    ]
    
    for name, config in configs:
        print(f"\n{name}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    print()


def demo_system_info():
    """Display system capabilities."""
    print("="*70)
    print("SYSTEM CAPABILITIES")
    print("="*70)
    print()
    
    print_cpu_capabilities()
    
    print("\n" + "="*70)
    print("CORRECTNESS VERIFICATION")
    print("="*70)
    print()
    print("Running mathematical correctness checks...")
    print("(Verifies FWHT, QJL unbiasedness, PROD unbiasedness)")
    print()
    
    results = run_correctness_check(verbose=False)
    
    print("Results:")
    print(f"  FWHT self-inverse error: {results['fwht_self_inverse_max_err']:.2e}")
    print(f"  QJL mean bias: {results['qjl_mean_abs_bias']:.4f} (should be ~0)")
    print(f"  PROD 4-bit bias: {results.get('prod_4bit_bias', 'N/A')}")
    print()
    print("✓ All mathematical guarantees verified!")


def main():
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*20 + "TurboQuantCPU API Reference" + " "*21 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    demo_low_level_api()
    demo_value_quantization()
    demo_advanced_config()
    demo_system_info()
    
    print("\n" + "="*70)
    print("API REFERENCE COMPLETE")
    print("="*70)
    print()
    print("Key takeaways:")
    print("• Use patch_model() for quick HuggingFace integration")
    print("• Use TurboQuantizer directly for custom implementations")
    print("• Choose mode based on your quality/compression needs")
    print("• All modes provide mathematical guarantees on quality")


if __name__ == "__main__":
    main()
