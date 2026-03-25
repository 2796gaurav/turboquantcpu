#!/usr/bin/env python3
"""
Optimized Quick Benchmark for TurboQuantCPU

Runs focused benchmarks on a single model to verify:
1. Correctness (needle-in-haystack at key depths)
2. Quality (perplexity check)
3. Memory compression
4. Speed

Optimized for quick iteration - runs in ~5-10 minutes.
"""

import os
import sys
import time
import json
import warnings
import numpy as np
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '8'

import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent.parent))

from turboquantcpu import patch_model, unpatch_model
from turboquantcpu.cpu_features import cpu_info, print_cpu_capabilities

# Configuration - Single model for quick testing
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_SHORT = "Qwen2.5-0.5B"

# Test configuration
CONTEXT_LENGTHS = [512, 1024, 2048]  # Reduced for speed
NEEDLE_DEPTHS = [0.0, 0.5, 1.0]  # Start, middle, end
COMPRESSION_MODES = [
    ("FP16", None, 16),
    ("PROD-4bit", "prod", 4),
    ("QJL-1bit", "qjl", 1),
]

NEEDLE_TEMPLATE = "The secret code is: {code}. Remember this exactly."
HAYSTACK_FILLERS = [
    "The weather today is pleasant.",
    "In 1492, Columbus sailed the Atlantic.",
    "Machine learning recognizes patterns.",
    "Paris is the capital of France.",
    "Photosynthesis converts light to energy.",
    "Water boils at 100 degrees Celsius.",
    "Light speed is 299,792,458 m/s.",
    "DNA carries genetic information.",
    "The Great Wall was built over centuries.",
    "Shakespeare wrote Hamlet and Romeo.",
    "The brain has 86 billion neurons.",
    "Python is a readable language.",
    "The Amazon is the largest rainforest.",
    "Gravity is a fundamental force.",
    "Beethoven composed nine symphonies.",
    "The periodic table organizes elements.",
    "Electricity flows through copper.",
    "The Internet started as a military project.",
    "Jupiter is the largest planet.",
    "Honey never spoils."
]


def log(msg):
    """Print with timestamp."""
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def generate_needle_context(ctx_len, depth, tokenizer, code="4839"):
    """Generate needle-in-haystack context."""
    needle = NEEDLE_TEMPLATE.format(code=code)
    needle_tokens = len(tokenizer.encode(needle))
    target_tokens = ctx_len - needle_tokens
    
    fillers = []
    current_tokens = 0
    np.random.seed(42)  # Reproducible
    
    while current_tokens < target_tokens:
        filler = np.random.choice(HAYSTACK_FILLERS)
        filler_tokens = len(tokenizer.encode(filler))
        if current_tokens + filler_tokens > target_tokens:
            break
        fillers.append(filler)
        current_tokens += filler_tokens
    
    insert_pos = int(depth * len(fillers))
    insert_pos = min(insert_pos, len(fillers))
    
    parts = fillers[:insert_pos] + [needle] + fillers[insert_pos:]
    return " ".join(parts)


def test_needle(model, tokenizer, ctx_len, depth, code="4839"):
    """Run single needle test."""
    context = generate_needle_context(ctx_len, depth, tokenizer, code)
    prompt = f"{context}\n\nWhat is the secret code?\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=ctx_len+50)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    latency = time.time() - start
    
    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    success = code in generated
    
    return success, latency, generated.strip()


def test_perplexity(model, tokenizer, text, max_length=512):
    """Calculate perplexity."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc.input_ids.to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        return torch.exp(outputs.loss).item()


def benchmark_mode(mode_name, mode, bits):
    """Benchmark a single compression mode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    log(f"\n{'='*60}")
    log(f"Benchmarking: {mode_name}")
    log(f"{'='*60}")
    
    # Load model
    log(f"Loading {MODEL_SHORT}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = True
    model.eval()
    
    cache = None
    if mode is not None:
        log(f"Applying {mode_name} compression...")
        cache = patch_model(model, mode=mode, bits=bits, verbose=False)
    
    results = {
        "mode": mode_name,
        "compression_ratio": 1.0,
        "needle_accuracy": 0.0,
        "perplexity": 0.0,
        "speed_tok_per_sec": 0.0,
    }
    
    # Needle-in-haystack
    log("Testing needle-in-haystack...")
    all_success = []
    for ctx_len in CONTEXT_LENGTHS:
        log(f"  Context {ctx_len} tokens:")
        for depth in NEEDLE_DEPTHS:
            code = str(np.random.randint(1000, 9999))
            success, latency, out = test_needle(model, tokenizer, ctx_len, depth, code)
            all_success.append(success)
            status = "✓" if success else "✗"
            log(f"    Depth {depth*100:.0f}%: {status} ({latency:.2f}s)")
    
    results["needle_accuracy"] = 100.0 * sum(all_success) / len(all_success)
    log(f"  Overall accuracy: {results['needle_accuracy']:.1f}%")
    
    # Memory
    if cache is not None:
        mem = cache.memory_report()
        results["compression_ratio"] = mem.get("compression_ratio", 1.0)
        log(f"  Compression ratio: {results['compression_ratio']:.2f}×")
    
    # Perplexity (quick check)
    log("Testing perplexity...")
    test_texts = [
        "The capital of France is Paris. It is known for the Eiffel Tower.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
    ]
    ppls = []
    for text in test_texts:
        try:
            ppl = test_perplexity(model, tokenizer, text, max_length=256)
            if not (np.isnan(ppl) or np.isinf(ppl)):
                ppls.append(ppl)
        except:
            pass
    
    if ppls:
        results["perplexity"] = np.mean(ppls)
        log(f"  Mean perplexity: {results['perplexity']:.2f}")
    
    # Speed
    log("Testing generation speed...")
    prompt = "The quick brown fox jumps over the lazy dog. " * 10
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    
    # Benchmark
    n_tokens = 30
    latencies = []
    for _ in range(2):
        start = time.time()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        latencies.append(time.time() - start)
    
    avg_latency = np.mean(latencies)
    results["speed_tok_per_sec"] = n_tokens / avg_latency
    log(f"  Speed: {results['speed_tok_per_sec']:.2f} tok/s")
    
    # Cleanup
    del model
    if cache is not None:
        import gc
        gc.collect()
    
    return results


def plot_results(all_results):
    """Generate comparison plots."""
    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(exist_ok=True)
    
    modes = [r["mode"] for r in all_results]
    comp_ratios = [r["compression_ratio"] for r in all_results]
    accuracies = [r["needle_accuracy"] for r in all_results]
    speeds = [r["speed_tok_per_sec"] for r in all_results]
    ppls = [r.get("perplexity", 0) for r in all_results]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Compression
    ax = axes[0, 0]
    bars = ax.bar(modes, comp_ratios, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax.set_ylabel("Compression Ratio (×)")
    ax.set_title("Memory Compression")
    ax.set_ylim(bottom=0)
    for bar, v in zip(bars, comp_ratios):
        ax.text(bar.get_x() + bar.get_width()/2, v, f'{v:.2f}×', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Accuracy
    ax = axes[0, 1]
    bars = ax.bar(modes, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax.set_ylabel("Needle Retrieval Accuracy (%)")
    ax.set_title("Long Context Quality")
    ax.set_ylim([0, 105])
    ax.axhline(y=100, color='r', linestyle='--', alpha=0.5)
    for bar, v in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, v, f'{v:.0f}%', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Speed
    ax = axes[1, 0]
    bars = ax.bar(modes, speeds, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax.set_ylabel("Tokens/Second")
    ax.set_title("Generation Speed")
    ax.set_ylim(bottom=0)
    for bar, v in zip(bars, speeds):
        ax.text(bar.get_x() + bar.get_width()/2, v, f'{v:.1f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Perplexity
    ax = axes[1, 1]
    valid_ppls = [(m, p) for m, p in zip(modes, ppls) if p > 0]
    if valid_ppls:
        m, p = zip(*valid_ppls)
        bars = ax.bar(m, p, color=['#3498db', '#2ecc71', '#e74c3c'][:len(m)])
        ax.set_ylabel("Perplexity")
        ax.set_title("Language Modeling Quality (Lower=Better)")
        for bar, v in zip(bars, p):
            ax.text(bar.get_x() + bar.get_width()/2, v, f'{v:.1f}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle(f"TurboQuantCPU Benchmark: {MODEL_SHORT}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"benchmark_{MODEL_SHORT}.png", dpi=150, bbox_inches='tight')
    log(f"\nPlot saved to: {output_dir / f'benchmark_{MODEL_SHORT}.png'}")


def main():
    """Main entry point."""
    log("="*60)
    log("TurboQuantCPU Optimized Quick Benchmark")
    log("="*60)
    
    print_cpu_capabilities()
    
    all_results = []
    
    for mode_name, mode, bits in COMPRESSION_MODES:
        try:
            results = benchmark_mode(mode_name, mode, bits)
            all_results.append(results)
        except Exception as e:
            log(f"ERROR benchmarking {mode_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    log("\n" + "="*60)
    log("BENCHMARK SUMMARY")
    log("="*60)
    
    for r in all_results:
        log(f"\n{r['mode']}:")
        log(f"  Compression:    {r['compression_ratio']:.2f}×")
        log(f"  Needle Acc:     {r['needle_accuracy']:.1f}%")
        log(f"  Perplexity:     {r['perplexity']:.2f}")
        log(f"  Speed:          {r['speed_tok_per_sec']:.2f} tok/s")
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"quick_benchmark_{MODEL_SHORT}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"\nResults saved to: {results_file}")
    
    # Plot
    plot_results(all_results)
    
    log("\n" + "="*60)
    log("Benchmark Complete!")
    log("="*60)


if __name__ == "__main__":
    main()
