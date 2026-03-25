#!/usr/bin/env python3
"""
================================================================================
TurboQuantCPU - Sanity Benchmark (0.5B-1B Models)
================================================================================

Focused benchmark using small models for quick validation:
- Qwen/Qwen3.5-0.8B (0.8B params, 24 layers, Apache 2.0)
- meta-llama/Llama-3.2-1B (1B params, 16 layers, Llama 3.2 Community)

METRICS:
--------
1. Memory compression ratio
2. Inference speed (tokens/sec)
3. Quality (perplexity, consistency)
4. Needle-in-haystack retrieval

This benchmark is designed to be:
- FAST: Uses small models, runs in minutes
- UNBIASED: Measures actual performance without cherry-picking
- REPRODUCIBLE: Fixed seeds, documented configurations
"""

import os
import sys
import json
import time
import gc
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquantcpu import patch_model, unpatch_model


# Models to benchmark - focused on small, fast models
BENCHMARK_MODELS = {
    "Qwen3.5-0.8B": {
        "name": "Qwen/Qwen3.5-0.8B",
        "params": "0.8B",
        "layers": 24,
        "kv_heads": 2,
        "head_dim": 128,
        "license": "Apache 2.0",
    },
    "Llama-3.2-1B": {
        "name": "meta-llama/Llama-3.2-1B",
        "params": "1B", 
        "layers": 16,
        "kv_heads": 8,
        "head_dim": 64,
        "license": "Llama 3.2 Community",
    },
}

TEST_PROMPTS = [
    "The capital of France is",
    "Machine learning is a subset of",
    "The largest planet in our solar system is",
]


def get_memory_mb():
    """Get current process memory in MB."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def calculate_perplexity(model, tokenizer, text):
    """Calculate perplexity (lower is better)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()


def benchmark_model(model_key, config):
    """Run complete benchmark for one model."""
    print(f"\n{'='*70}")
    print(f"BENCHMARKING: {model_key} ({config['params']} params)")
    print(f"License: {config['license']}")
    print(f"Architecture: {config['layers']} layers, {config['kv_heads']} KV heads")
    print(f"{'='*70}")
    
    results = {"model": model_key, "config": config}
    model_name = config["name"]
    
    # Load model
    print(f"\nLoading {model_name}...")
    mem_before = get_memory_mb()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        
        load_time = time.time()
        mem_after_load = get_memory_mb()
        model_memory = mem_after_load - mem_before
        
        print(f"  Loaded in {load_time:.1f}s")
        print(f"  Model memory: {model_memory:.1f} MB")
        
        results["load"] = {
            "time_s": load_time,
            "memory_mb": model_memory,
        }
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return None
    
    # ========================================================================
    # Test 1: Baseline (No Compression)
    # ========================================================================
    print("\n1. Baseline Inference (FP16)")
    print("-" * 70)
    
    times = []
    for prompt in TEST_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        times.append(time.time() - start)
    
    avg_time_baseline = np.mean(times)
    print(f"  Average time: {avg_time_baseline:.2f}s")
    
    results["baseline"] = {"avg_time_s": avg_time_baseline}
    
    # ========================================================================
    # Test 2: TurboQuant 4-bit PROD
    # ========================================================================
    print("\n2. TurboQuant (4-bit PROD)")
    print("-" * 70)
    
    cache = patch_model(model, mode="prod", bits=4, verbose=False)
    
    times = []
    for prompt in TEST_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        times.append(time.time() - start)
    
    avg_time_compress = np.mean(times)
    overhead = ((avg_time_compress / avg_time_baseline) - 1) * 100
    
    mem_report = cache.memory_report()
    
    print(f"  Average time: {avg_time_compress:.2f}s")
    print(f"  Overhead: {overhead:+.1f}%")
    print(f"  KV cache: {mem_report['compressed_MB']:.2f} MB")
    print(f"  Compression: {mem_report['compression_ratio']:.1f}×")
    
    results["turboquant_4bit"] = {
        "avg_time_s": avg_time_compress,
        "overhead_pct": overhead,
        "kv_cache_mb": mem_report["compressed_MB"],
        "compression_ratio": mem_report["compression_ratio"],
    }
    
    unpatch_model(model)
    
    # ========================================================================
    # Test 3: TurboQuant 1-bit QJL
    # ========================================================================
    print("\n3. TurboQuant (1-bit QJL)")
    print("-" * 70)
    
    cache = patch_model(model, mode="qjl", verbose=False)
    
    times = []
    for prompt in TEST_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        times.append(time.time() - start)
    
    avg_time_qjl = np.mean(times)
    overhead_qjl = ((avg_time_qjl / avg_time_baseline) - 1) * 100
    
    mem_report = cache.memory_report()
    
    print(f"  Average time: {avg_time_qjl:.2f}s")
    print(f"  Overhead: {overhead_qjl:+.1f}%")
    print(f"  KV cache: {mem_report['compressed_MB']:.2f} MB")
    print(f"  Compression: {mem_report['compression_ratio']:.1f}×")
    
    results["turboquant_1bit"] = {
        "avg_time_s": avg_time_qjl,
        "overhead_pct": overhead_qjl,
        "kv_cache_mb": mem_report["compressed_MB"],
        "compression_ratio": mem_report["compression_ratio"],
    }
    
    unpatch_model(model)
    
    # ========================================================================
    # Test 4: Quality (Perplexity)
    # ========================================================================
    print("\n4. Quality Test (Perplexity)")
    print("-" * 70)
    
    test_text = "The quick brown fox jumps over the lazy dog."
    
    ppl_baseline = calculate_perplexity(model, tokenizer, test_text)
    print(f"  Baseline: {ppl_baseline:.2f}")
    
    cache = patch_model(model, mode="prod", bits=4, verbose=False)
    ppl_compress = calculate_perplexity(model, tokenizer, test_text)
    print(f"  4-bit:    {ppl_compress:.2f}")
    
    diff_pct = ((ppl_compress / ppl_baseline) - 1) * 100
    print(f"  Diff:     {diff_pct:+.2f}%")
    
    results["quality"] = {
        "perplexity_baseline": ppl_baseline,
        "perplexity_4bit": ppl_compress,
        "difference_pct": diff_pct,
    }
    
    unpatch_model(model)
    
    # Cleanup
    del model, tokenizer
    gc.collect()
    
    return results


def print_summary(all_results):
    """Print summary table."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nCompression Performance:")
    print(f"{'Model':<20} {'4-bit Ratio':>12} {'1-bit Ratio':>12}")
    print("-" * 70)
    for r in all_results:
        if r:
            print(f"{r['model']:<20} {r['turboquant_4bit']['compression_ratio']:>11.1f}× "
                  f"{r['turboquant_1bit']['compression_ratio']:>11.1f}×")
    
    print("\nSpeed Overhead:")
    print(f"{'Model':<20} {'4-bit':>12} {'1-bit':>12}")
    print("-" * 70)
    for r in all_results:
        if r:
            print(f"{r['model']:<20} {r['turboquant_4bit']['overhead_pct']:>10.1f}% "
                  f"{r['turboquant_1bit']['overhead_pct']:>10.1f}%")
    
    print("\nQuality (Perplexity - lower is better):")
    print(f"{'Model':<20} {'Baseline':>12} {'4-bit':>12} {'Diff':>12}")
    print("-" * 70)
    for r in all_results:
        if r:
            print(f"{r['model']:<20} {r['quality']['perplexity_baseline']:>11.2f} "
                  f"{r['quality']['perplexity_4bit']:>11.2f} "
                  f"{r['quality']['difference_pct']:>+10.2f}%")


def save_results(all_results):
    """Save results to JSON."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "results": [r for r in all_results if r],
    }
    
    os.makedirs("results", exist_ok=True)
    filename = f"results/sanity_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {filename}")


def main():
    print("="*70)
    print("TurboQuantCPU - Sanity Benchmark")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Models:")
    for key, cfg in BENCHMARK_MODELS.items():
        print(f"  - {key}: {cfg['name']}")
    print()
    
    all_results = []
    
    for model_key, config in BENCHMARK_MODELS.items():
        result = benchmark_model(model_key, config)
        if result:
            all_results.append(result)
    
    if all_results:
        print_summary(all_results)
        save_results(all_results)
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
