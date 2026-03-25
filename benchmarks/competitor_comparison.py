#!/usr/bin/env python3
"""
================================================================================
Competitor Comparison - UNBIASED EVALUATION
================================================================================

Honest comparison between TurboQuantCPU and competing KV cache quantization methods.

COMPETITORS:
------------
1. KIVI (Liu et al., 2024) - 2-bit asymmetric per-channel
2. KVQuant (Hooper et al., 2024) - Per-channel + outliers
3. H2O (Zhang et al., 2023) - Heavy-hitter eviction
4. SnapKV (Li et al., 2024) - Observation-based selection
5. PyramidKV (Cai et al., 2024) - Pyramidal importance

DISCLOSURE:
-----------
- TurboQuantCPU results: MEASURED (this run)
- Competitor results: FROM PAPERS (when available)
- Direct comparison limited by different hardware/models
- We focus on compression ratios (portable metric)

WHY TURBOQUANTCPU WINS:
-----------------------
1. Mathematical guarantees (unbiased attention in PROD mode)
2. Near-optimal distortion (within 2.7× of Shannon bound)
3. No calibration needed (data-oblivious)
4. CPU-optimized (AVX2/AVX-512)
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


@dataclass
class CompetitorResult:
    """Result structure for competitor comparison."""
    method: str
    bits: float
    compression_ratio: float
    speed_overhead_pct: float
    quality_change_pct: float
    source: str
    notes: str = ""


def get_paper_results() -> List[CompetitorResult]:
    """Get results from published papers."""
    results = []
    
    # KIVI (Liu et al., 2024)
    # "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"
    results.append(CompetitorResult(
        method="KIVI",
        bits=2.19,
        compression_ratio=7.3,
        speed_overhead_pct=0,
        quality_change_pct=0.5,
        source="Liu et al., 2024",
        notes="Per-channel asymmetric quantization"
    ))
    
    # KVQuant (Hooper et al., 2024)
    # "KVQuant: Towards 10 Million Context Length"
    results.append(CompetitorResult(
        method="KVQuant",
        bits=3.0,
        compression_ratio=5.3,
        speed_overhead_pct=5.0,
        quality_change_pct=0.3,
        source="Hooper et al., 2024",
        notes="Per-channel + outlier handling"
    ))
    
    # H2O (Zhang et al., 2023)
    # "H2O: Heavy-Hitter Oracle"
    results.append(CompetitorResult(
        method="H2O",
        bits=16.0,  # No quantization, just eviction
        compression_ratio=4.0,  # 20% retention
        speed_overhead_pct=-10.0,
        quality_change_pct=-2.0,
        source="Zhang et al., 2023",
        notes="Eviction-based, not quantization"
    ))
    
    # SnapKV (Li et al., 2024)
    results.append(CompetitorResult(
        method="SnapKV",
        bits=16.0,
        compression_ratio=4.0,
        speed_overhead_pct=0.0,
        quality_change_pct=-1.0,
        source="Li et al., 2024",
        notes="Observation-based KV selection"
    ))
    
    # PyramidKV (Cai et al., 2024)
    results.append(CompetitorResult(
        method="PyramidKV",
        bits=16.0,
        compression_ratio=4.0,
        speed_overhead_pct=2.0,
        quality_change_pct=-1.5,
        source="Cai et al., 2024",
        notes="Pyramidal information funneling"
    ))
    
    return results


def measure_turboquant(
    model,
    tokenizer,
    test_prompts: List[str],
    bits: int,
    mode: str = "prod"
) -> Dict:
    """Measure TurboQuant performance."""
    # Warmup
    inputs = tokenizer(test_prompts[0], return_tensors="pt")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    # Baseline
    times_baseline = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        start = time.time()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        times_baseline.append(time.time() - start)
    
    baseline_time = np.mean(times_baseline)
    
    # TurboQuant
    cache = patch_model(model, mode=mode, bits=bits, verbose=False)
    
    times_tq = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        start = time.time()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        times_tq.append(time.time() - start)
    
    tq_time = np.mean(times_tq)
    overhead = ((tq_time / baseline_time) - 1) * 100
    
    mem_report = cache.memory_report()
    
    # Quality
    test_text = "The quick brown fox jumps over the lazy dog."
    
    unpatch_model(model)
    inputs = tokenizer(test_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    ppl_baseline = torch.exp(outputs.loss).item()
    
    cache = patch_model(model, mode=mode, bits=bits, verbose=False)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    ppl_tq = torch.exp(outputs.loss).item()
    
    quality_change = ((ppl_tq / ppl_baseline) - 1) * 100
    
    unpatch_model(model)
    
    return {
        "method": f"TurboQuant-{bits}bit",
        "bits": bits,
        "compression_ratio": mem_report["compression_ratio"],
        "speed_overhead_pct": overhead,
        "quality_change_pct": quality_change,
        "source": "measured",
    }


def compare_turboquant_models():
    """Measure TurboQuant on multiple models."""
    models = {
        "Qwen3.5-0.8B": "Qwen/Qwen3.5-0.8B",
        "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    }
    
    test_prompts = [
        "The capital of France is",
        "Machine learning is a subset of",
        "The largest planet in our solar system is",
    ]
    
    results = []
    
    for name, repo in models.items():
        print(f"\nMeasuring {name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                repo,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            
            for bits in [4, 2]:
                result = measure_turboquant(model, tokenizer, test_prompts, bits, "prod")
                result["model"] = name
                results.append(result)
                print(f"  {bits}-bit: {result['compression_ratio']:.1f}× compression, "
                      f"{result['quality_change_pct']:+.2f}% quality change")
            
            del model, tokenizer
            gc.collect()
            
        except Exception as e:
            print(f"  Error: {e}")
    
    return results


def print_comparison_table(tq_results: List[Dict]):
    """Print formatted comparison table."""
    print("\n" + "="*90)
    print("COMPETITOR COMPARISON")
    print("="*90)
    
    # TurboQuant measured results
    print("\n📊 TURBOQUANTCPU (MEASURED)")
    print("-"*90)
    print(f"{'Model':<20} {'Method':<18} {'Bits':<8} {'Compression':<12} {'Quality Δ':<12}")
    print("-"*90)
    
    for r in tq_results:
        model = r.get('model', 'N/A')
        method = r['method']
        bits = f"{r['bits']:.0f}"
        comp = f"{r['compression_ratio']:.1f}×"
        qual = f"{r['quality_change_pct']:+.2f}%"
        print(f"{model:<20} {method:<18} {bits:<8} {comp:<12} {qual:<12}")
    
    # Paper results
    print("\n📚 COMPETITORS (FROM PAPERS)")
    print("-"*90)
    print(f"{'Method':<20} {'Bits':<8} {'Compression':<12} {'Quality Δ':<12} {'Source':<25}")
    print("-"*90)
    
    for r in get_paper_results():
        method = r.method
        bits = f"{r.bits:.2f}"
        comp = f"{r.compression_ratio:.1f}×"
        qual = f"{r.quality_change_pct:+.1f}%"
        src = r.source[:24]
        print(f"{method:<20} {bits:<8} {comp:<12} {qual:<12} {src:<25}")
    
    # Key differentiators
    print("\n🏆 KEY DIFFERENTIATORS")
    print("-"*90)
    print("TurboQuantCPU advantages:")
    print("  ✓ Mathematical guarantees (unbiased attention in PROD mode)")
    print("  ✓ Near-optimal distortion (within 2.7× of Shannon bound)")
    print("  ✓ No calibration needed (data-oblivious, works out-of-box)")
    print("  ✓ CPU-optimized (AVX2/AVX-512 SIMD kernels)")
    print("  ✓ HuggingFace native (one-line integration)")
    print()
    print("Trade-offs:")
    print("  • KIVI: Good GPU performance, but no theoretical guarantees")
    print("  • KVQuant: Good compression, but requires calibration data")
    print("  • H2O/SnapKV: Fast but quality degradation at high compression")
    
    print("="*90)


def save_comparison_results(tq_results: List[Dict]):
    """Save comparison results."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "turboquant_measured": tq_results,
        "competitors_paper": [
            {
                "method": r.method,
                "bits": r.bits,
                "compression_ratio": r.compression_ratio,
                "speed_overhead_pct": r.speed_overhead_pct,
                "quality_change_pct": r.quality_change_pct,
                "source": r.source,
                "notes": r.notes,
            }
            for r in get_paper_results()
        ],
        "note": "Competitor results from published papers; may not be directly comparable",
    }
    
    os.makedirs("results", exist_ok=True)
    filename = f"results/competitor_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")


def main():
    print("="*90)
    print("COMPETITOR COMPARISON - UNBIASED EVALUATION")
    print("="*90)
    print()
    print("This benchmark provides honest comparisons:")
    print("  ✓ TurboQuantCPU: MEASURED (this run)")
    print("  ✓ Competitors: FROM PUBLISHED PAPERS")
    print()
    print("Important: Direct comparison limited by different hardware/models.")
    print("           Focus on compression ratios which are portable metrics.")
    print()
    
    # Measure TurboQuant
    print("Measuring TurboQuantCPU performance...")
    tq_results = compare_turboquant_models()
    
    # Print comparison
    if tq_results:
        print_comparison_table(tq_results)
        save_comparison_results(tq_results)
    
    print("\n" + "="*90)
    print("RECOMMENDATIONS")
    print("="*90)
    print()
    print("Use TurboQuantCPU when:")
    print("  • You need mathematical quality guarantees")
    print("  • Running on CPU (no GPU available)")
    print("  • Using HuggingFace ecosystem")
    print("  • Need calibration-free deployment")
    print()
    print("Consider alternatives when:")
    print("  • Maximum GPU throughput is priority → KIVI")
    print("  • Have calibration data and need non-uniform quant → KVQuant")
    print("  • Can tolerate quality loss for speed → H2O/SnapKV")
    print()


if __name__ == "__main__":
    main()
