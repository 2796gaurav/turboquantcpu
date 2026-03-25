#!/usr/bin/env python3
"""
Comprehensive Unbiased Benchmark for TurboQuantCPU

This benchmark implements:
1. Needle-in-Haystack: Long context retrieval at multiple depths
2. Perplexity Evaluation: WikiText-2 validation set
3. Memory vs Quality Trade-off: Multiple compression levels
4. Speed Benchmark: End-to-end generation speed

Models tested:
- Qwen/Qwen2.5-0.5B-Instruct (proxy for Qwen3.5-0.8B if unavailable)

Compression modes:
- FP16 baseline (no compression)
- 4-bit PROD (TurboQuant recommended)
- 1-bit QJL (maximum compression)
- 2-bit MSE (optional)

Author: TurboQuantCPU Research
"""

import os
import sys
import time
import json
import math
import random
import warnings
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from turboquantcpu import patch_model, unpatch_model, PatchConfig
from turboquantcpu.cpu_features import cpu_info, print_cpu_capabilities

# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark suite."""
    # Models to test
    models: List[str] = None
    
    # Compression modes: (name, mode, bits)
    compression_modes: List[Tuple[str, str, int]] = None
    
    # Needle-in-haystack settings
    context_lengths: List[int] = None
    needle_depths: List[float] = None  # Fractions of context length
    
    # Perplexity settings
    perplexity_max_tokens: int = 1024
    
    # Generation settings
    max_new_tokens: int = 50
    temperature: float = 0.0  # Greedy for reproducibility
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Device
    device: str = "cpu"
    
    def __post_init__(self):
        if self.models is None:
            self.models = [
                "Qwen/Qwen2.5-0.5B-Instruct",  # Using 0.5B as proxy for 0.8B
            ]
        if self.compression_modes is None:
            self.compression_modes = [
                ("FP16", None, 16),  # Baseline
                ("PROD-4bit", "prod", 4),
                ("QJL-1bit", "qjl", 1),
                ("MSE-2bit", "mse", 2),
            ]
        if self.context_lengths is None:
            self.context_lengths = [512, 1024, 2048, 4096]
        if self.needle_depths is None:
            self.needle_depths = [0.0, 0.25, 0.5, 0.75, 1.0]


# ==============================================================================
# Needle-in-Haystack Dataset
# ==============================================================================

NEEDLE_TEMPLATE = "The secret code is: {code}. Remember this code exactly."

HAYSTACK_FILLERS = [
    "The weather today is pleasant with a slight breeze.",
    "In 1492, Christopher Columbus sailed across the Atlantic Ocean.",
    "Machine learning algorithms can recognize patterns in data.",
    "The capital of France is Paris, known for the Eiffel Tower.",
    "Photosynthesis is the process by which plants convert light into energy.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The speed of light in vacuum is approximately 299,792,458 meters per second.",
    "DNA stands for deoxyribonucleic acid and carries genetic information.",
    "The Great Wall of China was built over many centuries by different dynasties.",
    "Shakespeare wrote many famous plays including Hamlet and Romeo and Juliet.",
    "The human brain contains approximately 86 billion neurons.",
    "Python is a popular programming language known for its readability.",
    "The Amazon rainforest is the largest tropical rainforest in the world.",
    "Gravity is one of the four fundamental forces of nature.",
    "Beethoven composed nine symphonies despite losing his hearing.",
    "The periodic table organizes chemical elements by atomic number.",
    "Electricity flows through conductors like copper and aluminum.",
    "The Internet was originally developed as a military research project.",
    "Jupiter is the largest planet in our solar system.",
    "Honey never spoils and can remain edible for thousands of years.",
]


def generate_needle_context(context_length: int, depth: float, tokenizer, needle_code: str = "4839") -> str:
    """
    Generate a needle-in-haystack context with the needle at specified depth.
    
    Args:
        context_length: Target context length in tokens
        depth: Position of needle (0.0 = start, 1.0 = end)
        tokenizer: HuggingFace tokenizer
        needle_code: The secret code to embed
    
    Returns:
        Context string with needle embedded
    """
    needle = NEEDLE_TEMPLATE.format(code=needle_code)
    needle_tokens = len(tokenizer.encode(needle))
    
    # Calculate how many tokens we need for haystack
    target_tokens = context_length - needle_tokens
    
    # Build haystack
    fillers = []
    current_tokens = 0
    
    while current_tokens < target_tokens:
        filler = random.choice(HAYSTACK_FILLERS)
        filler_tokens = len(tokenizer.encode(filler))
        
        if current_tokens + filler_tokens > target_tokens:
            # Try to fill remaining with a partial sentence
            break
        
        fillers.append(filler)
        current_tokens += filler_tokens
    
    # Insert needle at specified depth
    total_fillers = len(fillers)
    insert_position = int(depth * total_fillers)
    insert_position = min(insert_position, total_fillers)
    
    context_parts = fillers[:insert_position] + [needle] + fillers[insert_position:]
    context = " ".join(context_parts)
    
    return context


def create_needle_prompt(context: str, question: str = "What is the secret code?") -> str:
    """Create the prompt for needle-in-haystack test."""
    return f"{context}\n\nQuestion: {question}\nAnswer:"


# ==============================================================================
# Perplexity Calculation
# ==============================================================================

def calculate_perplexity(model, tokenizer, text: str, max_length: int = 2048) -> float:
    """
    Calculate perplexity on a text sample.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        text: Text to evaluate
        max_length: Maximum sequence length
    
    Returns:
        Perplexity value
    """
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings.input_ids.to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    
    return perplexity


def load_wikitext_samples(n_samples: int = 10) -> List[str]:
    """Load samples from WikiText-2 dataset."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        samples = []
        for i, item in enumerate(dataset):
            if len(item['text']) > 200:  # Only longer texts
                samples.append(item['text'])
            if len(samples) >= n_samples:
                break
        return samples
    except Exception as e:
        print(f"Could not load WikiText: {e}")
        # Return fallback samples
        return [
            "The quick brown fox jumps over the lazy dog. " * 50,
            "Machine learning is a subset of artificial intelligence. " * 40,
            "The capital of France is Paris. It is known for its cuisine. " * 45,
        ][:n_samples]


# ==============================================================================
# Benchmark Functions
# ==============================================================================

class BenchmarkRunner:
    """Main benchmark runner class."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}
        self.setup_directories()
        
    def setup_directories(self):
        """Create output directories."""
        self.output_dir = Path(__file__).parent / "results"
        self.output_dir.mkdir(exist_ok=True)
        self.plots_dir = Path(__file__).parent / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
    def log(self, message: str):
        """Print and log message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def load_model(self, model_name: str, compression_mode: Optional[Tuple] = None):
        """
        Load a model with optional TurboQuant compression.
        
        Args:
            model_name: HuggingFace model name
            compression_mode: (name, mode, bits) tuple or None for baseline
        
        Returns:
            model, tokenizer, cache (if compression enabled)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.log(f"Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use FP32 for CPU
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.config.use_cache = True
        model.eval()
        
        cache = None
        if compression_mode is not None and compression_mode[1] is not None:
            name, mode, bits = compression_mode
            self.log(f"  Applying {name} compression (mode={mode}, bits={bits})...")
            cache = patch_model(model, mode=mode, bits=bits, verbose=False)
        
        return model, tokenizer, cache
    
    def run_needle_in_haystack(
        self, 
        model, 
        tokenizer, 
        context_length: int, 
        depth: float,
        needle_code: str = "4839"
    ) -> Dict[str, Any]:
        """
        Run a single needle-in-haystack test.
        
        Returns:
            Dict with success, latency, output, expected_code
        """
        # Generate context
        context = generate_needle_context(context_length, depth, tokenizer, needle_code)
        prompt = create_needle_prompt(context)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=context_length + 100)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        actual_context_length = inputs['input_ids'].shape[1]
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,  # Greedy for reproducibility
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        latency = time.time() - start_time
        
        # Decode
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Check if code is present
        success = needle_code in generated_text
        
        return {
            "success": success,
            "latency": latency,
            "output": generated_text.strip(),
            "expected_code": needle_code,
            "actual_context_length": actual_context_length,
        }
    
    def run_needle_benchmark(self, model_name: str, compression_mode: Optional[Tuple]) -> Dict:
        """Run full needle-in-haystack benchmark for a model/mode."""
        model, tokenizer, cache = self.load_model(model_name, compression_mode)
        
        mode_name = compression_mode[0] if compression_mode else "FP16"
        self.log(f"Running needle-in-haystack for {model_name} with {mode_name}...")
        
        results = {
            "model": model_name,
            "mode": mode_name,
            "context_results": {},
        }
        
        # Generate unique needle codes for each test
        random.seed(self.config.seed)
        needle_codes = [f"{random.randint(1000, 9999)}" for _ in range(100)]
        code_idx = 0
        
        for ctx_len in self.config.context_lengths:
            self.log(f"  Testing context length: {ctx_len}")
            depth_results = {}
            
            for depth in self.config.needle_depths:
                needle_code = needle_codes[code_idx % len(needle_codes)]
                code_idx += 1
                
                try:
                    result = self.run_needle_in_haystack(
                        model, tokenizer, ctx_len, depth, needle_code
                    )
                    depth_results[depth] = result
                    status = "✓" if result["success"] else "✗"
                    self.log(f"    Depth {depth:.1%}: {status} ({result['latency']:.2f}s)")
                except Exception as e:
                    self.log(f"    Depth {depth:.1%}: ERROR - {e}")
                    depth_results[depth] = {"success": False, "error": str(e)}
            
            results["context_results"][ctx_len] = depth_results
        
        # Calculate overall accuracy
        all_success = [
            r["success"] 
            for ctx_res in results["context_results"].values() 
            for r in ctx_res.values()
            if "success" in r
        ]
        results["overall_accuracy"] = sum(all_success) / len(all_success) if all_success else 0.0
        
        # Memory report
        if cache is not None:
            mem_report = cache.memory_report()
            results["memory"] = mem_report
            self.log(f"  Compression ratio: {mem_report.get('compression_ratio', 0):.2f}x")
        
        # Cleanup
        del model
        if cache is not None:
            unpatch_model(model if 'model' in locals() else None)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return results
    
    def run_perplexity_benchmark(self, model_name: str, compression_mode: Optional[Tuple]) -> Dict:
        """Run perplexity benchmark."""
        model, tokenizer, cache = self.load_model(model_name, compression_mode)
        
        mode_name = compression_mode[0] if compression_mode else "FP16"
        self.log(f"Running perplexity for {model_name} with {mode_name}...")
        
        # Load samples
        samples = load_wikitext_samples(self.config.perplexity_max_tokens // 100)
        
        perplexities = []
        for i, text in enumerate(samples[:5]):  # Limit to 5 samples for speed
            try:
                ppl = calculate_perplexity(model, tokenizer, text, max_length=512)
                perplexities.append(ppl)
                self.log(f"  Sample {i+1}: PPL = {ppl:.2f}")
            except Exception as e:
                self.log(f"  Sample {i+1}: ERROR - {e}")
        
        results = {
            "model": model_name,
            "mode": mode_name,
            "perplexities": perplexities,
            "mean_perplexity": np.mean(perplexities) if perplexities else float('inf'),
            "std_perplexity": np.std(perplexities) if perplexities else 0.0,
        }
        
        self.log(f"  Mean PPL: {results['mean_perplexity']:.2f} ± {results['std_perplexity']:.2f}")
        
        # Cleanup
        del model
        if cache is not None:
            unpatch_model(model if 'model' in locals() else None)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return results
    
    def run_speed_benchmark(self, model_name: str, compression_mode: Optional[Tuple]) -> Dict:
        """Run generation speed benchmark."""
        model, tokenizer, cache = self.load_model(model_name, compression_mode)
        
        mode_name = compression_mode[0] if compression_mode else "FP16"
        self.log(f"Running speed benchmark for {model_name} with {mode_name}...")
        
        # Test prompt
        prompt = "The quick brown fox jumps over the lazy dog. " * 20
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Warmup
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        # Benchmark
        latencies = []
        tokens_generated = 50
        
        for _ in range(3):  # 3 runs
            start = time.time()
            with torch.no_grad():
                _ = model.generate(
                    **inputs, 
                    max_new_tokens=tokens_generated, 
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            latencies.append(time.time() - start)
        
        mean_latency = np.mean(latencies)
        tokens_per_sec = tokens_generated / mean_latency
        
        results = {
            "model": model_name,
            "mode": mode_name,
            "mean_latency": mean_latency,
            "tokens_per_sec": tokens_per_sec,
            "latencies": latencies,
        }
        
        self.log(f"  Speed: {tokens_per_sec:.2f} tok/s ({mean_latency:.2f}s for {tokens_generated} tokens)")
        
        # Cleanup
        del model
        if cache is not None:
            unpatch_model(model if 'model' in locals() else None)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return results
    
    def run_full_benchmark(self):
        """Run complete benchmark suite."""
        self.log("=" * 70)
        self.log("TurboQuantCPU Comprehensive Unbiased Benchmark")
        self.log("=" * 70)
        
        # Print CPU info
        print_cpu_capabilities()
        
        # Set seed
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        
        all_results = {
            "config": asdict(self.config),
            "timestamp": datetime.now().isoformat(),
            "cpu_info": {
                "brand": cpu_info().brand,
                "simd": cpu_info().best_simd,
            },
            "needle": {},
            "perplexity": {},
            "speed": {},
        }
        
        for model_name in self.config.models:
            model_short = model_name.split("/")[-1]
            self.log(f"\n{'='*70}")
            self.log(f"Benchmarking: {model_short}")
            self.log(f"{'='*70}")
            
            all_results["needle"][model_short] = {}
            all_results["perplexity"][model_short] = {}
            all_results["speed"][model_short] = {}
            
            for comp_mode in self.config.compression_modes:
                mode_name = comp_mode[0]
                
                try:
                    # Needle-in-haystack
                    needle_res = self.run_needle_benchmark(model_name, comp_mode)
                    all_results["needle"][model_short][mode_name] = needle_res
                    
                    # Perplexity
                    ppl_res = self.run_perplexity_benchmark(model_name, comp_mode)
                    all_results["perplexity"][model_short][mode_name] = ppl_res
                    
                    # Speed
                    speed_res = self.run_speed_benchmark(model_name, comp_mode)
                    all_results["speed"][model_short][mode_name] = speed_res
                    
                except Exception as e:
                    self.log(f"ERROR benchmarking {model_short} with {mode_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Save results
        results_file = self.output_dir / f"benchmark_results_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        self.log(f"\nResults saved to: {results_file}")
        
        self.results = all_results
        return all_results
    
    def generate_plots(self):
        """Generate visualization plots from results."""
        if not self.results:
            self.log("No results to plot. Run benchmark first.")
            return
        
        self.log("\nGenerating plots...")
        
        # 1. Needle-in-Haystack Heatmap
        self._plot_needle_heatmap()
        
        # 2. Perplexity Comparison
        self._plot_perplexity()
        
        # 3. Speed Comparison
        self._plot_speed()
        
        # 4. Memory vs Quality Trade-off
        self._plot_memory_quality_tradeoff()
        
        # 5. Summary Dashboard
        self._plot_summary_dashboard()
        
        self.log(f"Plots saved to: {self.plots_dir}")
    
    def _plot_needle_heatmap(self):
        """Plot needle-in-haystack heatmaps for each model/mode."""
        for model_name, model_results in self.results["needle"].items():
            fig, axes = plt.subplots(1, len(self.config.compression_modes), 
                                     figsize=(5*len(self.config.compression_modes), 5))
            if len(self.config.compression_modes) == 1:
                axes = [axes]
            
            for idx, (mode_name, results) in enumerate(model_results.items()):
                if mode_name not in results.get("context_results", {}):
                    continue
                
                # Build accuracy matrix
                ctx_lengths = sorted(self.config.context_lengths)
                depths = sorted(self.config.needle_depths)
                
                accuracy_matrix = np.zeros((len(ctx_lengths), len(depths)))
                
                for i, ctx_len in enumerate(ctx_lengths):
                    if ctx_len in results.get("context_results", {}):
                        for j, depth in enumerate(depths):
                            if depth in results["context_results"][ctx_len]:
                                success = results["context_results"][ctx_len][depth].get("success", False)
                                accuracy_matrix[i, j] = 1.0 if success else 0.0
                
                ax = axes[idx]
                im = ax.imshow(accuracy_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
                ax.set_xticks(range(len(depths)))
                ax.set_xticklabels([f"{d:.0%}" for d in depths])
                ax.set_yticks(range(len(ctx_lengths)))
                ax.set_yticklabels([f"{c//1024}K" for c in ctx_lengths])
                ax.set_xlabel("Needle Depth")
                ax.set_ylabel("Context Length")
                ax.set_title(f"{mode_name}\nAcc: {results.get('overall_accuracy', 0):.0%}")
                plt.colorbar(im, ax=ax, label="Accuracy")
            
            plt.suptitle(f"Needle-in-Haystack: {model_name}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"needle_heatmap_{model_name}.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    def _plot_perplexity(self):
        """Plot perplexity comparison."""
        fig, axes = plt.subplots(1, len(self.results["perplexity"]), 
                                 figsize=(8*len(self.results["perplexity"]), 6))
        if len(self.results["perplexity"]) == 1:
            axes = [axes]
        
        for idx, (model_name, model_results) in enumerate(self.results["perplexity"].items()):
            modes = []
            ppls = []
            errors = []
            
            for mode_name, results in model_results.items():
                modes.append(mode_name)
                ppls.append(results.get("mean_perplexity", 0))
                errors.append(results.get("std_perplexity", 0))
            
            ax = axes[idx]
            bars = ax.bar(modes, ppls, yerr=errors, capsize=5, 
                         color=['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'][:len(modes)])
            ax.set_ylabel("Perplexity")
            ax.set_title(f"{model_name}")
            ax.set_ylim(bottom=0)
            
            # Add value labels
            for bar, ppl in zip(bars, ppls):
                height = bar.get_height()
                if not np.isnan(ppl) and not np.isinf(ppl):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{ppl:.1f}',
                           ha='center', va='bottom', fontsize=9)
            
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle("Perplexity Comparison (Lower is Better)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.plots_dir / "perplexity_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_speed(self):
        """Plot generation speed comparison."""
        fig, axes = plt.subplots(1, len(self.results["speed"]), 
                                 figsize=(8*len(self.results["speed"]), 6))
        if len(self.results["speed"]) == 1:
            axes = [axes]
        
        for idx, (model_name, model_results) in enumerate(self.results["speed"].items()):
            modes = []
            speeds = []
            
            for mode_name, results in model_results.items():
                modes.append(mode_name)
                speeds.append(results.get("tokens_per_sec", 0))
            
            ax = axes[idx]
            bars = ax.bar(modes, speeds, 
                         color=['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'][:len(modes)])
            ax.set_ylabel("Tokens/Second")
            ax.set_title(f"{model_name}")
            ax.set_ylim(bottom=0)
            
            # Add value labels
            for bar, speed in zip(bars, speeds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{speed:.1f}',
                       ha='center', va='bottom', fontsize=9)
            
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle("Generation Speed Comparison (Higher is Better)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.plots_dir / "speed_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_quality_tradeoff(self):
        """Plot memory vs quality trade-off."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Collect data
        for model_name in self.results["needle"].keys():
            compressions = []
            accuracies = []
            ppls = []
            mode_names = []
            
            for mode_name in self.config.compression_modes:
                name = mode_name[0]
                if name in self.results["needle"].get(model_name, {}):
                    needle_res = self.results["needle"][model_name][name]
                    ppl_res = self.results["perplexity"].get(model_name, {}).get(name, {})
                    
                    comp_ratio = needle_res.get("memory", {}).get("compression_ratio", 1.0)
                    acc = needle_res.get("overall_accuracy", 0)
                    ppl = ppl_res.get("mean_perplexity", float('inf'))
                    
                    if not np.isnan(ppl) and not np.isinf(ppl):
                        compressions.append(comp_ratio)
                        accuracies.append(acc * 100)  # Convert to percentage
                        ppls.append(ppl)
                        mode_names.append(name)
            
            # Plot Accuracy vs Compression
            axes[0].scatter(compressions, accuracies, s=100, label=model_name)
            for i, name in enumerate(mode_names):
                axes[0].annotate(name, (compressions[i], accuracies[i]), 
                               textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
            
            # Plot Perplexity vs Compression
            valid_data = [(c, p) for c, p in zip(compressions, ppls) if p < 1000]
            if valid_data:
                c, p = zip(*valid_data)
                axes[1].scatter(c, p, s=100, label=model_name)
        
        axes[0].set_xlabel("Compression Ratio (×)")
        axes[0].set_ylabel("Needle Retrieval Accuracy (%)")
        axes[0].set_title("Long Context Retrieval Quality")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 105])
        
        axes[1].set_xlabel("Compression Ratio (×)")
        axes[1].set_ylabel("Perplexity")
        axes[1].set_title("Language Modeling Quality (Lower is Better)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle("Memory-Quality Trade-off", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.plots_dir / "memory_quality_tradeoff.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_summary_dashboard(self):
        """Create a comprehensive summary dashboard."""
        n_models = len(self.results["needle"])
        fig = plt.figure(figsize=(16, 4 * n_models))
        
        for idx, model_name in enumerate(self.results["needle"].keys()):
            # Get data
            modes = []
            comp_ratios = []
            accuracies = []
            speeds = []
            
            for mode_name in self.config.compression_modes:
                name = mode_name[0]
                if name in self.results["needle"].get(model_name, {}):
                    needle_res = self.results["needle"][model_name][name]
                    speed_res = self.results["speed"].get(model_name, {}).get(name, {})
                    
                    modes.append(name)
                    comp_ratios.append(needle_res.get("memory", {}).get("compression_ratio", 1.0))
                    accuracies.append(needle_res.get("overall_accuracy", 0) * 100)
                    speeds.append(speed_res.get("tokens_per_sec", 0))
            
            # Plot 1: Compression Ratio
            ax1 = plt.subplot(n_models, 3, idx*3 + 1)
            bars = ax1.bar(range(len(modes)), comp_ratios, color='#3498db')
            ax1.set_xticks(range(len(modes)))
            ax1.set_xticklabels(modes, rotation=45, ha='right')
            ax1.set_ylabel("Compression Ratio (×)")
            ax1.set_title(f"{model_name}\nMemory Compression")
            ax1.grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(comp_ratios):
                ax1.text(i, v, f'{v:.1f}×', ha='center', va='bottom', fontsize=9)
            
            # Plot 2: Accuracy
            ax2 = plt.subplot(n_models, 3, idx*3 + 2)
            bars = ax2.bar(range(len(modes)), accuracies, color='#2ecc71')
            ax2.set_xticks(range(len(modes)))
            ax2.set_xticklabels(modes, rotation=45, ha='right')
            ax2.set_ylabel("Needle Retrieval Accuracy (%)")
            ax2.set_title("Long Context Retrieval")
            ax2.set_ylim([0, 105])
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Perfect')
            
            # Plot 3: Speed
            ax3 = plt.subplot(n_models, 3, idx*3 + 3)
            bars = ax3.bar(range(len(modes)), speeds, color='#e67e22')
            ax3.set_xticks(range(len(modes)))
            ax3.set_xticklabels(modes, rotation=45, ha='right')
            ax3.set_ylabel("Tokens/Second")
            ax3.set_title("Generation Speed")
            ax3.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle("TurboQuantCPU Comprehensive Benchmark Summary", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.plots_dir / "summary_dashboard.png", dpi=150, bbox_inches='tight')
        plt.close()


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TurboQuantCPU Comprehensive Benchmark")
    parser.add_argument("--models", nargs="+", default=None,
                       help="Models to benchmark (default: Qwen2.5-0.5B, Llama-3.2-1B)")
    parser.add_argument("--context-lengths", nargs="+", type=int, default=None,
                       help="Context lengths to test")
    parser.add_argument("--skip-needle", action="store_true",
                       help="Skip needle-in-haystack benchmark")
    parser.add_argument("--skip-perplexity", action="store_true",
                       help="Skip perplexity benchmark")
    parser.add_argument("--skip-speed", action="store_true",
                       help="Skip speed benchmark")
    parser.add_argument("--skip-plots", action="store_true",
                       help="Skip plot generation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Create config
    config = BenchmarkConfig(
        seed=args.seed,
    )
    
    if args.models:
        config.models = args.models
    if args.context_lengths:
        config.context_lengths = args.context_lengths
    
    # Run benchmark
    runner = BenchmarkRunner(config)
    
    try:
        results = runner.run_full_benchmark()
        
        if not args.skip_plots:
            runner.generate_plots()
        
        # Print summary
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        
        for model_name, model_results in results["needle"].items():
            print(f"\n{model_name}:")
            for mode_name, mode_results in model_results.items():
                acc = mode_results.get("overall_accuracy", 0) * 100
                comp = mode_results.get("memory", {}).get("compression_ratio", 1.0)
                print(f"  {mode_name:12s}: Acc={acc:5.1f}%  Comp={comp:.1f}×")
        
        print("\n" + "="*70)
        
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nBenchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
