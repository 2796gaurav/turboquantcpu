#!/usr/bin/env python3
"""
================================================================================
TurboQuantCPU - Getting Started Example
================================================================================

This example shows the simplest way to use TurboQuantCPU to compress KV cache
and reduce memory usage during LLM inference.

WHAT IS KV CACHE COMPRESSION?
------------------------------
When LLMs generate text, they store "Key-Value" (KV) pairs for each token to
avoid recomputing attention. This cache grows with sequence length:

  Memory = 2 × layers × heads × head_dim × seq_len × 2 bytes (FP16)

For a 7B model at 8K context: ~1 GB just for KV cache!

TurboQuantCPU compresses this by 4-14× with minimal quality loss.

INSTALLATION
------------
    pip install turboquantcpu

That's it! TurboQuantCPU includes everything - no separate "[hf]" extras needed.

WHEN TO USE TURBOQUANTCPU
--------------------------
✓ Running long contexts (4K+ tokens) on limited RAM
✓ Serving multiple models on same hardware  
✓ Edge/on-device deployment
✓ Batch processing many sequences
✓ You need mathematical guarantees on quality

WHEN NOT TO USE
----------------
✗ Very short contexts (<1K tokens) - overhead not worth it
✗ Maximum raw speed is the only priority - use llama.cpp
✗ You have plenty of GPU memory - use vLLM instead
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquantcpu import patch_model, unpatch_model


def main():
    # ========================================================================
    # STEP 1: Choose a model
    # ========================================================================
    # We use Qwen2.5-0.5B as it's small but demonstrates the concept.
    # Works with ANY HuggingFace model: Llama, Mistral, Phi, Gemma, etc.
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"Loading {MODEL_NAME}...")
    print("(This is a small model for demonstration - works with any size)")
    
    # ========================================================================
    # STEP 2: Load tokenizer and model
    # ========================================================================
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load in float32 for CPU inference
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    # ========================================================================
    # STEP 3: Enable TurboQuantCPU compression (ONE LINE!)
    # ========================================================================
    # mode="prod"  - Recommended: Unbiased attention scores (mathematically proven)
    # bits=4       - 4 bits per coordinate = ~4× compression
    #
    # Other options:
    #   mode="qjl"   - 1-bit, 14× compression, fastest, slightly lower quality
    #   mode="mse"   - 4-bit, optimized for reconstruction quality
    #   mode="polar" - 4-bit, best for outlier-heavy models
    #
    print("\n" + "="*70)
    print("ENABLING KV CACHE COMPRESSION")
    print("="*70)
    print("Mode: PROD (provably unbiased attention)")
    print("Bits: 4 bits per coordinate")
    print("Expected compression: ~4× for keys, ~2× for values")
    print("="*70)
    
    cache = patch_model(model, mode="prod", bits=4)
    
    # ========================================================================
    # STEP 4: Generate text as usual
    # ========================================================================
    prompt = "Explain quantum computing in simple terms:"
    print(f"\nPrompt: {prompt}")
    print("-" * 70)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response:\n{response}\n")
    
    # ========================================================================
    # STEP 5: Check memory savings
    # ========================================================================
    report = cache.memory_report()
    
    print("="*70)
    print("MEMORY SAVINGS REPORT")
    print("="*70)
    print(f"Original FP16 KV cache:  {report['original_fp16_MB']:.2f} MB")
    print(f"Compressed KV cache:     {report['compressed_MB']:.2f} MB")
    print(f"Compression ratio:       {report['compression_ratio']:.1f}×")
    print(f"Memory saved:            {report['original_fp16_MB'] - report['compressed_MB']:.2f} MB")
    print("="*70)
    
    # ========================================================================
    # STEP 6: Cleanup (optional - restores original model)
    # ========================================================================
    unpatch_model(model)
    print("\nModel restored to original state.")
    print("\n✓ Example completed successfully!")


if __name__ == "__main__":
    main()
