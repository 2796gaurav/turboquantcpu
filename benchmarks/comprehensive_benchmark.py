#!/usr/bin/env python3
"""
================================================================================
TurboQuantCPU - Comprehensive Benchmark (3 Models)
================================================================================

Thorough benchmarking on 3 core HuggingFace models:
1. Qwen/Qwen3.5-0.8B (0.8B params, Alibaba) - Apache 2.0
2. meta-llama/Llama-3.2-1B (1B params, Meta) - Llama 3.2 License
3. google/gemma-2-2b-it (2B params, Google) - Gemma Terms

BENCHMARKS:
-----------
- Memory compression ratio (1-bit, 2-bit, 4-bit)
- Inference speed (tokens/sec, overhead %)
- Quality (perplexity, needle-in-haystack retrieval)
- Long context scaling

METRICS:
--------
- Compression Ratio = FP16 size / compressed size
- Speed Overhead = (compressed_time / baseline_time - 1) × 100%
- Quality Loss = (ppl_compressed / ppl_baseline - 1) × 100%
- Needle Retrieval = % correct at various depths/lengths

IMPORTANT:
----------
- All measurements are ACTUAL (not simulated)
- All models are HuggingFace Transformers (no GGUF)
- CPU-only inference
================================================================================
"""

import os
import sys
import json
import time
import gc
import psutil
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquantcpu import patch_model, unpatch_model

# Configuration
BENCHMARK_MODELS = {
    "Qwen3.5-0.8B": {
        "repo_id": "Qwen/Qwen3.5-0.8B",
        "provider": "Alibaba",
        "params": "0.8B",
        "layers": 24,
        "kv_heads": 2,
        "head_dim": 128,
        "license": "Apache 2.0",
    },
    "Llama-3.2-1B": {
        "repo_id": "meta-llama/Llama-3.2-1B",
        "provider": "Meta",
        "params": "1B",
        "layers": 16,
        "kv_heads": 8,
        "head_dim": 64,
        "license": "Llama 3.2",
    },
    "Gemma-2-2B": {
        "repo_id": "google/gemma-2-2b-it",
        "provider": "Google",
        "params": "2B",
        "layers": 18,
        "kv_heads": 4,
        "head_dim": 128,
        "license": "Gemma",
    },
}

TEST_PROMPTS = [
    "The capital of France is Paris. The capital of Germany is",
    "Machine learning is a subset of artificial intelligence that",
    "The largest planet in our solar system is Jupiter, which",
    "Water boils at 100 degrees Celsius at standard atmospheric",
    "The speed of light in vacuum is approximately 299,792",
]

NEEDLE_PROMPTS = [
    "The secret code is BLUE-7829. Remember this code.",
    "The magic word is ALOHOMORA. This is important.",
    "The hidden key is XJ-900-KL. Store this safely.",
]


def get_memory_mb() -> float:
    """Get current process memory in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def calculate_perplexity(model, tokenizer, text: str, device: str = "cpu") -> float:
    """Calculate perplexity (lower is better)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()


def load_model(repo_id: str):
    """Load HuggingFace model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    return model, tokenizer


def benchmark_compression(
    model,
    tokenizer,
    test_prompts: List[str],
    bits: int,
    mode: str = "prod"
) -> Dict:
    """Benchmark compression at specific bit-width."""
    # Warmup
    inputs = tokenizer(test_prompts[0], return_tensors="pt")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    # Baseline timing
    times_baseline = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        start = time.time()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        times_baseline.append(time.time() - start)
    
    baseline_time = np.mean(times_baseline)
    
    # Apply TurboQuant
    cache = patch_model(model, mode=mode, bits=bits, verbose=False)
    
    # Compressed timing
    times_compressed = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        start = time.time()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        times_compressed.append(time.time() - start)
    
    compressed_time = np.mean(times_compressed)
    overhead_pct = ((compressed_time / baseline_time) - 1) * 100
    
    mem_report = cache.memory_report()
    
    unpatch_model(model)
    
    return {
        "bits": bits,
        "mode": mode,
        "baseline_time_ms": baseline_time * 1000,
        "compressed_time_ms": compressed_time * 1000,
        "overhead_pct": overhead_pct,
        "compression_ratio": mem_report["compression_ratio"],
        "compressed_mb": mem_report["compressed_MB"],
        "original_mb": mem_report["original_fp16_MB"],
        "tokens_per_sec": 20 / compressed_time,
    }


def benchmark_quality(
    model,
    tokenizer,
    test_texts: List[str],
    bits: int,
    mode: str = "prod"
) -> Dict:
    """Benchmark quality preservation via perplexity."""
    # Baseline perplexity
    ppls_baseline = []
    for text in test_texts:
        ppl = calculate_perplexity(model, tokenizer, text)
        ppls_baseline.append(ppl)
    
    ppl_baseline = np.mean(ppls_baseline)
    
    # Compressed perplexity
    cache = patch_model(model, mode=mode, bits=bits, verbose=False)
    
    ppls_compressed = []
    for text in test_texts:
        ppl = calculate_perplexity(model, tokenizer, text)
        ppls_compressed.append(ppl)
    
    ppl_compressed = np.mean(ppls_compressed)
    
    unpatch_model(model)
    
    quality_change_pct = ((ppl_compressed / ppl_baseline) - 1) * 100
    
    return {
        "bits": bits,
        "mode": mode,
        "ppl_baseline": ppl_baseline,
        "ppl_compressed": ppl_compressed,
        "quality_change_pct": quality_change_pct,
    }


def benchmark_needle_in_haystack(
    model,
    tokenizer,
    context_lengths: List[int] = None,
    depths: List[float] = None,
    num_trials: int = 2
) -> Dict:
    """
    Needle-in-haystack benchmark for long context retrieval.
    
    Tests if model can retrieve hidden information at various
    context lengths and depths.
    """
    if context_lengths is None:
        context_lengths = [512, 1024, 2048]
    if depths is None:
        depths = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Filler text for haystack
    filler_sentences = [
        "The weather today is quite pleasant with clear skies.",
        "Economic indicators suggest steady growth this quarter.",
        "Recent research has shown promising results in the field.",
        "The project timeline has been adjusted to accommodate delays.",
        "Scientists continue to explore new frontiers in technology.",
        "Education remains a fundamental pillar of modern society.",
        "Climate patterns have shifted significantly over recent decades.",
        "Urban planning requires careful consideration of many factors.",
        "Healthcare systems face increasing demands and challenges.",
        "Transportation infrastructure requires ongoing maintenance.",
    ] * 20  # Repeat to ensure enough content
    
    results = {
        "baseline": {"total": 0, "correct": 0, "by_length": {}, "by_depth": {}},
        "turboquant": {"total": 0, "correct": 0, "by_length": {}, "by_depth": {}},
    }
    
    def create_haystack(length: int, needle: str, depth: float) -> str:
        """Create haystack with needle at specific depth."""
        # Estimate tokens needed
        sample = " ".join(filler_sentences[:10])
        tokens_per_sentence = len(tokenizer.encode(sample)) / 10
        
        num_sentences = int((length - len(tokenizer.encode(needle))) / tokens_per_sentence)
        sentences = filler_sentences[:num_sentences]
        
        insert_pos = int(len(sentences) * depth)
        sentences.insert(insert_pos, needle)
        
        return " ".join(sentences)
    
    def test_retrieval(haystack: str, question: str, answer: str) -> bool:
        """Test if model retrieves correct answer."""
        prompt = f"Context: {haystack}\n\nQuestion: {question}\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=15, do_sample=False)
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.lower() in generated.lower()
    
    # Test baseline
    print("  Testing baseline (FP16)...")
    for length in context_lengths:
        results["baseline"]["by_length"][length] = {"total": 0, "correct": 0}
    for depth in depths:
        results["baseline"]["by_depth"][depth] = {"total": 0, "correct": 0}
    
    for length in context_lengths:
        for depth in depths:
            for trial in range(num_trials):
                needle_info = NEEDLE_PROMPTS[trial % len(NEEDLE_PROMPTS)]
                haystack = create_haystack(length, needle_info, depth)
                
                # Extract question and answer from needle
                parts = needle_info.replace("The ", "").split(" is ")
                question = f"What is the {parts[0]}?"
                answer = parts[1].rstrip(".")
                
                correct = test_retrieval(haystack, question, answer)
                
                results["baseline"]["total"] += 1
                results["baseline"]["correct"] += 1 if correct else 0
                results["baseline"]["by_length"][length]["total"] += 1
                results["baseline"]["by_length"][length]["correct"] += 1 if correct else 0
                results["baseline"]["by_depth"][depth]["total"] += 1
                results["baseline"]["by_depth"][depth]["correct"] += 1 if correct else 0
    
    # Test TurboQuant 4-bit
    print("  Testing TurboQuant 4-bit PROD...")
    cache = patch_model(model, mode="prod", bits=4, verbose=False)
    
    for length in context_lengths:
        results["turboquant"]["by_length"][length] = {"total": 0, "correct": 0}
    for depth in depths:
        results["turboquant"]["by_depth"][depth] = {"total": 0, "correct": 0}
    
    for length in context_lengths:
        for depth in depths:
            for trial in range(num_trials):
                needle_info = NEEDLE_PROMPTS[trial % len(NEEDLE_PROMPTS)]
                haystack = create_haystack(length, needle_info, depth)
                
                parts = needle_info.replace("The ", "").split(" is ")
                question = f"What is the {parts[0]}?"
                answer = parts[1].rstrip(".")
                
                correct = test_retrieval(haystack, question, answer)
                
                results["turboquant"]["total"] += 1
                results["turboquant"]["correct"] += 1 if correct else 0
                results["turboquant"]["by_length"][length]["total"] += 1
                results["turboquant"]["by_length"][length]["correct"] += 1 if correct else 0
                results["turboquant"]["by_depth"][depth]["total"] += 1
                results["turboquant"]["by_depth"][depth]["correct"] += 1 if correct else 0
    
    unpatch_model(model)
    
    # Calculate scores
    results["baseline"]["score_pct"] = (
        results["baseline"]["correct"] / results["baseline"]["total"] * 100
    )
    results["turboquant"]["score_pct"] = (
        results["turboquant"]["correct"] / results["turboquant"]["total"] * 100
    )
    
    return results


def benchmark_model(model_name: str, config: Dict) -> Optional[Dict]:
    """Run full benchmark suite on a model."""
    print(f"\n{'='*70}")
    print(f"BENCHMARKING: {model_name}")
    print(f"Repo: {config['repo_id']}")
    print(f"Provider: {config['provider']} | Params: {config['params']}")
    print(f"{'='*70}")
    
    results = {
        "model": model_name,
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }
    
    try:
        # Load model
        print(f"\n[1/4] Loading model...")
        mem_before = get_memory_mb()
        model, tokenizer = load_model(config["repo_id"])
        mem_after = get_memory_mb()
        
        results["load_memory_mb"] = mem_after - mem_before
        print(f"  Model memory: {results['load_memory_mb']:.1f} MB")
        
        # Compression benchmarks
        print(f"\n[2/4] Compression benchmarks...")
        results["compression"] = {}
        
        for bits in [4, 2, 1]:
            print(f"  Testing {bits}-bit PROD...")
            comp_result = benchmark_compression(model, tokenizer, TEST_PROMPTS, bits, "prod")
            results["compression"][f"{bits}bit"] = comp_result
            print(f"    Ratio: {comp_result['compression_ratio']:.1f}×, "
                  f"Overhead: {comp_result['overhead_pct']:+.1f}%")
        
        # Quality benchmarks
        print(f"\n[3/4] Quality benchmarks...")
        results["quality"] = {}
        
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning algorithms improve with more data.",
            "Climate change affects global weather patterns significantly.",
        ]
        
        for bits in [4, 2]:
            print(f"  Testing {bits}-bit quality...")
            qual_result = benchmark_quality(model, tokenizer, test_texts, bits, "prod")
            results["quality"][f"{bits}bit"] = qual_result
            print(f"    PPL: {qual_result['ppl_baseline']:.2f} → {qual_result['ppl_compressed']:.2f} "
                  f"({qual_result['quality_change_pct']:+.2f}%)")
        
        # Needle-in-haystack
        print(f"\n[4/4] Needle-in-haystack (long context)...")
        needle_results = benchmark_needle_in_haystack(
            model, tokenizer,
            context_lengths=[512, 1024],
            depths=[0.0, 0.5, 1.0],
            num_trials=2
        )
        results["needle_in_haystack"] = needle_results
        print(f"    Baseline: {needle_results['baseline']['score_pct']:.1f}%, "
              f"4-bit: {needle_results['turboquant']['score_pct']:.1f}%")
        
        # Cleanup
        del model, tokenizer
        gc.collect()
        
        print(f"\n{'='*70}")
        print(f"✓ {model_name} benchmark complete!")
        print(f"{'='*70}")
        
        return results
        
    except Exception as e:
        print(f"\n✗ ERROR benchmarking {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_summary(all_results: List[Dict]):
    """Print comprehensive summary table."""
    print("\n" + "="*80)
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print("="*80)
    
    # Compression table
    print("\n📊 COMPRESSION RATIOS")
    print("-"*80)
    print(f"{'Model':<20} {'4-bit':<12} {'2-bit':<12} {'1-bit':<12}")
    print("-"*80)
    
    for r in all_results:
        if r and "compression" in r:
            model = r['model']
            c4 = r['compression']['4bit']['compression_ratio']
            c2 = r['compression']['2bit']['compression_ratio']
            c1 = r['compression']['1bit']['compression_ratio']
            print(f"{model:<20} {c4:>10.1f}× {c2:>10.1f}× {c1:>10.1f}×")
    
    # Speed overhead table
    print("\n⚡ SPEED OVERHEAD (%)")
    print("-"*80)
    print(f"{'Model':<20} {'4-bit':<12} {'2-bit':<12} {'1-bit':<12}")
    print("-"*80)
    
    for r in all_results:
        if r and "compression" in r:
            model = r['model']
            o4 = r['compression']['4bit']['overhead_pct']
            o2 = r['compression']['2bit']['overhead_pct']
            o1 = r['compression']['1bit']['overhead_pct']
            print(f"{model:<20} {o4:>+10.1f}% {o2:>+10.1f}% {o1:>+10.1f}%")
    
    # Quality table
    print("\n🎯 QUALITY PRESERVATION (Perplexity Change)")
    print("-"*80)
    print(f"{'Model':<20} {'4-bit':<15} {'2-bit':<15}")
    print("-"*80)
    
    for r in all_results:
        if r and "quality" in r:
            model = r['model']
            q4 = r['quality']['4bit']['quality_change_pct']
            q2 = r['quality']['2bit']['quality_change_pct']
            print(f"{model:<20} {q4:>+13.2f}% {q2:>+13.2f}%")
    
    # Needle-in-haystack table
    print("\n🪡 NEEDLE-IN-HAYSTACK (Retrieval Accuracy)")
    print("-"*80)
    print(f"{'Model':<20} {'Baseline':<15} {'4-bit Turbo':<15} {'Delta':<10}")
    print("-"*80)
    
    for r in all_results:
        if r and "needle_in_haystack" in r:
            model = r['model']
            base = r['needle_in_haystack']['baseline']['score_pct']
            tq = r['needle_in_haystack']['turboquant']['score_pct']
            delta = tq - base
            print(f"{model:<20} {base:>13.1f}% {tq:>13.1f}% {delta:>+8.1f}%")
    
    print("\n" + "="*80)


def save_results(all_results: List[Dict]):
    """Save results to JSON."""
    output = {
        "benchmark": "TurboQuantCPU Comprehensive",
        "timestamp": datetime.now().isoformat(),
        "models_tested": list(BENCHMARK_MODELS.keys()),
        "results": [r for r in all_results if r],
    }
    
    os.makedirs("results", exist_ok=True)
    filename = f"results/comprehensive_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")


def main():
    print("="*80)
    print("TurboQuantCPU - Comprehensive Benchmark (3 Models)")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {', '.join(BENCHMARK_MODELS.keys())}")
    print()
    
    all_results = []
    
    for model_key, config in BENCHMARK_MODELS.items():
        result = benchmark_model(model_key, config)
        if result:
            all_results.append(result)
    
    # Print summary
    if all_results:
        print_summary(all_results)
        save_results(all_results)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print(f"Models tested: {len(all_results)}/{len(BENCHMARK_MODELS)}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()
