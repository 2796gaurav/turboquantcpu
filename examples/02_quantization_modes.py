#!/usr/bin/env python3
"""
================================================================================
TurboQuantCPU - Understanding Quantization Modes
================================================================================

This example explains the four quantization modes and helps you choose
the right one for your use case.

COMPARISON TABLE
----------------
┌─────────────┬──────────┬────────────────┬─────────────────────────────────────┐
│ Mode        │ Bits     │ Compression    │ Best For                            │
├─────────────┼──────────┼────────────────┼─────────────────────────────────────┤
│ qjl         │ 1        │ ~14×           │ Maximum compression, long contexts  │
│ prod        │ 4        │ ~4×            │ Balanced: quality + compression     │
│ mse         │ 4        │ ~4×            │ Best reconstruction quality         │
│ polar       │ 4        │ ~4×            │ Outlier-heavy models (e.g., Phi)    │
└─────────────┴──────────┴────────────────┴─────────────────────────────────────┘

THE MATH (Simplified)
---------------------
• QJL: Stores only sign(+/-) after random rotation. Unbiased estimator.
• PROD: MSE quantization + QJL on residual. Provably unbiased.
• MSE: Lloyd-Max optimal quantization. Minimizes reconstruction error.
• Polar: Converts to polar coordinates (r, θ). Handles outliers better.

WHICH SHOULD YOU USE?
---------------------
→ Start with PROD (recommended) - best balance
→ Need maximum compression? Use QJL
→ Model has outliers (Phi, some fine-tunes)? Try Polar
→ Need best reconstruction? Use MSE
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquantcpu import patch_model, unpatch_model


def benchmark_mode(model, tokenizer, inputs, mode_name, config):
    """Benchmark a single mode."""
    print(f"\n{'='*70}")
    print(f"Testing: {mode_name}")
    print(f"Config: {config}")
    print('='*70)
    
    # Patch model
    cache = patch_model(model, verbose=False, **config)
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    # Timed generation
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    elapsed = time.time() - start
    
    # Results
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    report = cache.memory_report()
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Compression: {report['compression_ratio']:.1f}×")
    print(f"Output: {text[:80]}...")
    
    # Cleanup
    unpatch_model(model)
    
    return {
        'mode': mode_name,
        'time': elapsed,
        'compression': report['compression_ratio'],
        'compressed_mb': report['compressed_MB'],
        'original_mb': report['original_fp16_MB'],
    }


def main():
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    PROMPT = "The capital of France is"
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    inputs = tokenizer(PROMPT, return_tensors="pt")
    
    # Define configurations to test
    configs = [
        ("QJL (1-bit, max compression)", {"mode": "qjl"}),
        ("PROD (4-bit, unbiased, recommended)", {"mode": "prod", "bits": 4}),
        ("MSE (4-bit, best reconstruction)", {"mode": "mse", "bits": 4}),
        ("Polar (4-bit, outlier-resistant)", {"mode": "polar", "bits": 4}),
    ]
    
    results = []
    for name, config in configs:
        result = benchmark_mode(model, tokenizer, inputs, name, config)
        results.append(result)
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"{'Mode':<30} {'Compression':>12} {'Time':>10} {'Memory':>12}")
    print("-"*70)
    
    for r in results:
        print(f"{r['mode']:<30} {r['compression']:>11.1f}× {r['time']:>9.2f}s {r['compressed_mb']:>10.1f}MB")
    
    print("="*70)
    print("\nNOTES:")
    print("• QJL: Use when memory is extremely limited (14× compression)")
    print("• PROD: Best balance - unbiased attention with good compression")
    print("• MSE: Use if you need best vector reconstruction quality")
    print("• Polar: Use for models with outlier features (Phi, some fine-tunes)")


if __name__ == "__main__":
    main()
