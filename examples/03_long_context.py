#!/usr/bin/env python3
"""
================================================================================
TurboQuantCPU - Long Context Demo (The "Killer Feature")
================================================================================

This example demonstrates TurboQuantCPU's main strength: enabling LLMs to
handle VERY LONG contexts that would normally run out of memory.

THE PROBLEM
-----------
A 7B model at 32K context needs ~4 GB for KV cache (FP16).
On a 16 GB RAM laptop, that's 25% of available memory just for cache!

THE SOLUTION
------------
TurboQuantCPU compresses KV cache by 4-14×, reducing 4 GB → 1 GB or less.
This enables:
  • Processing entire books/documents
  • Long-form conversation history
  • Code analysis of large repositories
  • Multi-document RAG

CONTEXT LENGTH VS MEMORY (Llama-3.1-8B example)
-----------------------------------------------
Context    FP16 Memory    Turbo-4bit    Savings
-------    -----------    ----------    -------
4K         268 MB         71 MB         197 MB
16K        1,074 MB       285 MB        789 MB
32K        2,147 MB       570 MB        1,577 MB
128K       8,589 MB       2,281 MB      6,308 MB

At 128K context, you save over 6 GB of RAM!
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquantcpu import patch_model, unpatch_model


def create_long_prompt(base_prompt, repeat_text, target_tokens, tokenizer):
    """Create a prompt of approximately target_tokens length."""
    # Estimate tokens per repeat
    repeat_tokens = len(tokenizer.encode(repeat_text))
    num_repeats = target_tokens // repeat_tokens
    
    long_context = (repeat_text + " ") * num_repeats
    full_prompt = f"Context: {long_context}\n\nQuestion: {base_prompt}\nAnswer:"
    
    actual_tokens = len(tokenizer.encode(full_prompt))
    return full_prompt, actual_tokens


def main():
    # Using a small model for demo (works with any size)
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print("="*70)
    print("LONG CONTEXT DEMONSTRATION")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print("Note: Using 0.5B model for demo speed. Same principle applies to 70B+\n")
    
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
    # SCENARIO: Document Question Answering
    # ========================================================================
    print("Scenario: Question answering on a ~2,000 token document")
    print("-" * 70)
    
    # Simulate a long document
    document_paragraph = """
    The history of artificial intelligence (AI) began in antiquity, with myths, 
    stories and rumors of artificial beings endowed with intelligence or consciousness 
    by master craftsmen. The field of AI research was founded at a workshop held on 
    the campus of Dartmouth College during the summer of 1956. Those who attended 
    would become the leaders of AI research for decades. Many of them predicted that 
    a machine as intelligent as a human being would exist in no more than a generation 
    and they were given millions of dollars to make this vision come true. 
    """
    
    question = "Where was the field of AI research founded?"
    
    long_prompt, token_count = create_long_prompt(
        question, document_paragraph, 2000, tokenizer
    )
    
    print(f"Document length: ~{token_count} tokens")
    print(f"Question: {question}")
    print()
    
    # ========================================================================
    # WITHOUT COMPRESSION (show memory estimate)
    # ========================================================================
    print("WITHOUT TurboQuantCPU:")
    
    # Calculate memory for this model
    num_layers = 24  # Qwen2.5-0.5B
    num_kv_heads = 2
    head_dim = 64
    
    fp16_memory = token_count * num_layers * num_kv_heads * head_dim * 2 * 2  # K + V
    fp16_memory_mb = fp16_memory / (1024 * 1024)
    
    print(f"  KV cache memory: ~{fp16_memory_mb:.1f} MB (FP16)")
    print(f"  Model weights: ~1,000 MB")
    print(f"  Total: ~{fp16_memory_mb + 1000:.1f} MB")
    print()
    
    # ========================================================================
    # WITH COMPRESSION
    # ========================================================================
    print("WITH TurboQuantCPU (4-bit PROD mode):")
    
    # Estimate compressed memory
    compressed_memory_mb = fp16_memory_mb / 3.8  # ~3.8x compression
    print(f"  KV cache memory: ~{compressed_memory_mb:.1f} MB (compressed)")
    print(f"  Model weights: ~1,000 MB")
    print(f"  Total: ~{compressed_memory_mb + 1000:.1f} MB")
    print(f"  SAVINGS: ~{fp16_memory_mb - compressed_memory_mb:.1f} MB")
    print()
    
    # Enable compression
    print("Enabling compression and generating answer...")
    cache = patch_model(model, mode="prod", bits=4)
    
    inputs = tokenizer(long_prompt, return_tensors="pt", truncation=True, max_length=4096)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the answer part
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    
    print(f"\nAnswer: {answer}")
    
    # Report actual savings
    report = cache.memory_report()
    print("\n" + "="*70)
    print("ACTUAL MEMORY USAGE")
    print("="*70)
    print(f"Original KV cache (extrapolated): {fp16_memory_mb:.1f} MB")
    print(f"Compressed KV cache (actual):     {report['compressed_MB']:.1f} MB")
    print(f"Actual compression:               {report['compression_ratio']:.1f}×")
    print("="*70)
    
    unpatch_model(model)
    
    print("\n✓ Long context demo completed!")
    print("\nTry increasing the target token count to see how far you can push it!")


if __name__ == "__main__":
    main()
