#!/usr/bin/env python3
"""
Real Model Benchmark for TurboQuantCPU.

Tests TurboQuantCPU with actual HuggingFace models:
- Memory usage comparison
- Generation quality
- Throughput measurements
"""

import time
import gc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquantcpu import patch_model, unpatch_model


def get_memory_usage():
    """Get current memory usage in MB."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def benchmark_model(model_name: str, prompt: str, max_new_tokens: int = 50):
    """Benchmark a model with and without TurboQuantCPU."""
    
    print(f"\n{'='*70}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*70}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model (FP32)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=5, do_sample=False)
    
    # Baseline generation
    print("\n--- Baseline (No Compression) ---")
    gc.collect()
    baseline_memory = get_memory_usage()
    
    start_time = time.time()
    with torch.no_grad():
        baseline_output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    baseline_time = time.time() - start_time
    baseline_memory_end = get_memory_usage()
    
    baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
    baseline_kv_memory = baseline_memory_end - baseline_memory
    
    print(f"Generation time: {baseline_time:.2f}s")
    print(f"Memory delta: {baseline_kv_memory:.1f} MB")
    print(f"Output: {baseline_text[:100]}...")
    
    # Patch with TurboQuantCPU
    print("\n--- With TurboQuantCPU (PROD-4bit) ---")
    cache = patch_model(model, mode="prod", bits=4, verbose=True)
    
    gc.collect()
    compressed_memory = get_memory_usage()
    
    start_time = time.time()
    with torch.no_grad():
        compressed_output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    compressed_time = time.time() - start_time
    compressed_memory_end = get_memory_usage()
    
    compressed_text = tokenizer.decode(compressed_output[0], skip_special_tokens=True)
    compressed_kv_memory = compressed_memory_end - compressed_memory
    
    print(f"\nGeneration time: {compressed_time:.2f}s")
    print(f"Memory delta: {compressed_kv_memory:.1f} MB")
    print(f"Output: {compressed_text[:100]}...")
    
    # Memory report from cache
    report = cache.memory_report()
    print(f"\n--- Memory Report from Cache ---")
    print(f"Original FP16 (keys+values): {report['original_fp16_MB']:.1f} MB")
    print(f"Compressed: {report['compressed_MB']:.1f} MB")
    print(f"Compression ratio: {report['compression_ratio']:.1f}×")
    
    # Comparison
    print(f"\n--- Comparison ---")
    if compressed_kv_memory > 0:
        print(f"Measured memory ratio: {baseline_kv_memory / compressed_kv_memory:.1f}×")
    print(f"Speed overhead: {(compressed_time / baseline_time - 1) * 100:.1f}%")
    
    # Text similarity check
    baseline_words = set(baseline_text.split())
    compressed_words = set(compressed_text.split())
    common_words = baseline_words & compressed_words
    similarity = len(common_words) / max(len(baseline_words), len(compressed_words))
    print(f"Output similarity: {similarity*100:.1f}%")
    
    # Cleanup
    unpatch_model(model)
    del model
    gc.collect()
    
    return {
        "model": model_name,
        "baseline_time": baseline_time,
        "compressed_time": compressed_time,
        "baseline_memory_mb": baseline_kv_memory,
        "compressed_memory_mb": compressed_kv_memory,
        "compression_ratio": report['compression_ratio'],
        "similarity": similarity,
    }


def main():
    """Run real model benchmarks."""
    
    # Check for psutil
    try:
        import psutil
    except ImportError:
        print("Installing psutil for memory tracking...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil
    
    print("\n" + "="*70)
    print("TurboQuantCPU Real Model Benchmark")
    print("="*70)
    
    # Small model for testing
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    prompt = "Explain quantum computing in simple terms:"
    
    try:
        results = benchmark_model(model_name, prompt, max_new_tokens=30)
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Model: {results['model']}")
        print(f"Compression ratio: {results['compression_ratio']:.1f}×")
        print(f"Speed overhead: {(results['compressed_time'] / results['baseline_time'] - 1) * 100:.1f}%")
        print(f"Output similarity: {results['similarity']*100:.1f}%")
        print("="*70)
        
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
