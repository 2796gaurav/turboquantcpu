#!/usr/bin/env python3
"""
================================================================================
TurboQuantCPU - Batch Processing Example
================================================================================

This example shows how TurboQuantCPU enables processing multiple long
sequences efficiently - perfect for:
  • Document processing pipelines
  • Batch inference APIs
  • Dataset preprocessing
  • Evaluation on benchmarks

WHY BATCH PROCESSING BENEFITS FROM KV CACHE COMPRESSION
--------------------------------------------------------
Without compression:
  Batch of 10 × 8K context = 10 × 2 GB = 20 GB KV cache!

With TurboQuantCPU (4× compression):
  Batch of 10 × 8K context = 10 × 0.5 GB = 5 GB KV cache

That's the difference between needing a high-end GPU vs. a laptop!

TIPS FOR BATCH PROCESSING
--------------------------
1. Use padding='longest' to align sequences
2. Process similar-length sequences together
3. Enable mixed precision (torch_dtype=torch.bfloat16) if supported
4. Monitor memory with cache.memory_report()
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquantcpu import patch_model, unpatch_model


def main():
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print("="*70)
    print("BATCH PROCESSING DEMONSTRATION")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print()
    
    # Sample documents to process
    documents = [
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        "Photosynthesis is the process by which plants convert light energy into chemical energy.",
        "The Great Wall of China is a series of fortifications built across the northern borders of China.",
        "Quantum mechanics is a fundamental theory in physics describing nature at the smallest scales.",
    ]
    
    instruction = "Summarize the following text in one sentence:"
    
    print(f"Processing {len(documents)} documents...")
    print(f"Instruction: {instruction}")
    print()
    
    # Load model
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
    
    # ========================================================================
    # WITHOUT COMPRESSION
    # ========================================================================
    print("WITHOUT TurboQuantCPU:")
    print("-" * 70)
    
    inputs = tokenizer(
        [f"{instruction}\n\n{text}" for text in documents],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
        )
    elapsed_no_compress = time.time() - start
    
    print(f"Time: {elapsed_no_compress:.2f}s")
    print(f"Memory per sequence: Full FP16 (no tracking without compression)")
    print()
    
    # Clear cache
    if hasattr(model, 'past_key_values'):
        model.past_key_values = None
    
    # ========================================================================
    # WITH COMPRESSION
    # ========================================================================
    print("WITH TurboQuantCPU (4-bit PROD):")
    print("-" * 70)
    
    cache = patch_model(model, mode="prod", bits=4)
    
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
        )
    elapsed_compress = time.time() - start
    
    # Decode outputs
    summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    print(f"Time: {elapsed_compress:.2f}s")
    
    # Memory report
    report = cache.memory_report()
    print(f"Total compressed KV cache: {report['compressed_MB']:.2f} MB")
    print(f"Estimated FP16 KV cache: {report['original_fp16_MB']:.2f} MB")
    print(f"Compression ratio: {report['compression_ratio']:.1f}×")
    print()
    
    # Show results
    print("="*70)
    print("SUMMARIES")
    print("="*70)
    for i, (doc, summary) in enumerate(zip(documents, summaries), 1):
        # Extract just the generated part
        prompt = f"{instruction}\n\n{doc}"
        generated = summary[len(prompt):].strip()
        print(f"{i}. {generated}")
    
    print()
    print("="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print(f"Without compression: {elapsed_no_compress:.2f}s")
    print(f"With compression:    {elapsed_compress:.2f}s")
    print(f"Overhead:            {((elapsed_compress/elapsed_no_compress - 1) * 100):+.1f}%")
    print()
    print(f"Memory saved:        {report['original_fp16_MB'] - report['compressed_MB']:.1f} MB")
    print("="*70)
    
    unpatch_model(model)
    
    print("\n✓ Batch processing demo completed!")
    print("\nNote: Overhead is typically 5-15% for the benefit of 4× memory savings.")


if __name__ == "__main__":
    main()
