#!/usr/bin/env python3
"""
Comprehensive benchmark suite for TurboQuantCPU.

This runs unbiased benchmarks with:
- Real LLM model configurations (Llama, Mistral, Qwen, etc.)
- Multiple sequence lengths
- Statistical significance (multiple runs)
- Memory profiling
- Correctness verification

Output: JSON results + visual reports
"""

import json
import time
import os
import sys
import gc
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Add parent to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquantcpu import (
    TurboQuantizer, TurboMode, QJLQuantizer, PolarQuantizer,
    ValueQuantizer, ValueMode,
    cpu_info, fwht_numpy, get_signs, randomized_fwht,
)


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    name: str
    metric: str
    value: float
    unit: str
    std: float = 0.0
    samples: int = 1
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class TurboQuantBenchmarks:
    """Comprehensive benchmark suite."""
    
    def __init__(self, output_dir: str = "benchmarks/results"):
        self.output_dir = output_dir
        self.results: List[BenchmarkResult] = []
        self.cpu_info = cpu_info()
        
        os.makedirs(output_dir, exist_ok=True)
        
    def save_results(self, filename: str = None):
        """Save results to JSON."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "cpu": self.cpu_info.brand,
                "simd": self.cpu_info.best_simd,
                "cores": self.cpu_info.num_cores,
                "threads": self.cpu_info.num_threads,
                "c_extension": self.cpu_info.c_kernel_simd,
            },
            "results": [asdict(r) for r in self.results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Results saved to: {filepath}")
        return filepath
    
    def _timeit(self, func, repeats: int = 10, warmup: int = 3) -> Tuple[float, float]:
        """Time a function with warmup and multiple repeats."""
        # Warmup
        for _ in range(warmup):
            func()
        
        # Actual timing
        times = []
        for _ in range(repeats):
            gc.collect()
            t0 = time.perf_counter()
            func()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms
        
        return np.mean(times), np.std(times)
    
    def benchmark_fwht_correctness(self) -> Dict[str, Any]:
        """Verify FWHT mathematical properties."""
        print("\n" + "="*70)
        print("BENCHMARK: FWHT Correctness")
        print("="*70)
        
        results = {}
        
        # Test 1: Self-inverse property
        for d in [64, 128, 256]:
            x = np.random.randn(100, d).astype(np.float32)
            y = fwht_numpy(x.copy())
            z = fwht_numpy(y.copy())
            max_err = np.abs(z - x).max()
            
            result = BenchmarkResult(
                name=f"fwht_self_inverse_d{d}",
                metric="max_error",
                value=float(max_err),
                unit="absolute",
                samples=100
            )
            self.results.append(result)
            results[f"self_inverse_d{d}"] = max_err
            print(f"  d={d:3d}: max_error={max_err:.3e} {'✓' if max_err < 1e-4 else '✗'}")
        
        # Test 2: Norm preservation
        for d in [64, 128, 256]:
            x = np.random.randn(100, d).astype(np.float32)
            norms_before = np.linalg.norm(x, axis=1)
            y = fwht_numpy(x.copy())
            norms_after = np.linalg.norm(y, axis=1)
            rel_err = np.abs(norms_before - norms_after).mean() / norms_before.mean()
            
            result = BenchmarkResult(
                name=f"fwht_norm_preservation_d{d}",
                metric="relative_error",
                value=float(rel_err),
                unit="relative",
                samples=100
            )
            self.results.append(result)
            results[f"norm_preservation_d{d}"] = rel_err
            print(f"  d={d:3d}: norm_error={rel_err:.3e} {'✓' if rel_err < 1e-4 else '✗'}")
        
        return results
    
    def benchmark_fwht_speed(self) -> Dict[str, Any]:
        """Benchmark FWHT performance."""
        print("\n" + "="*70)
        print("BENCHMARK: FWHT Speed")
        print("="*70)
        
        results = {}
        
        for d in [64, 128, 256, 512]:
            for seq_len in [512, 1024, 4096]:
                x = np.random.randn(seq_len, d).astype(np.float32)
                
                mean_time, std_time = self._timeit(
                    lambda: fwht_numpy(x.copy()),
                    repeats=10, warmup=3
                )
                
                # Calculate throughput
                elements_per_sec = (seq_len * d) / (mean_time / 1000)
                
                result = BenchmarkResult(
                    name=f"fwht_speed_d{d}_seq{seq_len}",
                    metric="time_ms",
                    value=float(mean_time),
                    unit="ms",
                    std=float(std_time),
                    samples=10
                )
                self.results.append(result)
                
                results[f"d{d}_seq{seq_len}"] = {
                    "time_ms": mean_time,
                    "throughput_Melem/s": elements_per_sec / 1e6
                }
                print(f"  d={d:3d}, seq={seq_len:5d}: {mean_time:7.3f} ± {std_time:5.3f} ms  "
                      f"({elements_per_sec/1e6:.1f} Melem/s)")
        
        return results
    
    def benchmark_qjl_unbiasedness(self) -> Dict[str, Any]:
        """Verify QJL unbiased estimator property."""
        print("\n" + "="*70)
        print("BENCHMARK: QJL Unbiasedness")
        print("="*70)
        
        results = {}
        
        configs = [
            (64, 4, 256),
            (128, 8, 512),
            (128, 8, 4096),
        ]
        
        for head_dim, num_heads, seq_len in configs:
            keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
            qjl = QJLQuantizer(head_dim, num_heads, layer_idx=0)
            state = qjl.compress(keys)
            
            # Test with multiple queries
            biases = []
            maes = []
            
            for _ in range(50):
                query = np.random.randn(1, num_heads, head_dim).astype(np.float32)
                true_scores = np.einsum("bhd,shd->bhs", query, keys)
                est_scores = qjl.scores(query, state)
                
                biases.append(float((est_scores - true_scores).mean()))
                maes.append(float(np.abs(est_scores - true_scores).mean()))
            
            mean_bias = np.mean(biases)
            std_bias = np.std(biases)
            mean_mae = np.mean(maes)
            
            result_bias = BenchmarkResult(
                name=f"qjl_bias_d{head_dim}_h{num_heads}_seq{seq_len}",
                metric="mean_bias",
                value=float(mean_bias),
                unit="absolute",
                std=float(std_bias),
                samples=50
            )
            self.results.append(result_bias)
            
            result_mae = BenchmarkResult(
                name=f"qjl_mae_d{head_dim}_h{num_heads}_seq{seq_len}",
                metric="mean_absolute_error",
                value=float(mean_mae),
                unit="absolute",
                samples=50
            )
            self.results.append(result_mae)
            
            results[f"d{head_dim}_h{num_heads}_seq{seq_len}"] = {
                "bias": mean_bias,
                "bias_std": std_bias,
                "mae": mean_mae,
                "is_unbiased": abs(mean_bias) < 0.1
            }
            
            status = "✓" if abs(mean_bias) < 0.1 else "✗"
            print(f"  d={head_dim}, h={num_heads}, seq={seq_len:4d}: "
                  f"bias={mean_bias:+.4f}±{std_bias:.4f}, mae={mean_mae:.4f} {status}")
        
        return results
    
    def benchmark_compression_ratios(self) -> Dict[str, Any]:
        """Benchmark compression ratios."""
        print("\n" + "="*70)
        print("BENCHMARK: Compression Ratios")
        print("="*70)
        
        results = {}
        
        configs = [
            ("QJL-1bit", lambda d, h: QJLQuantizer(d, h, layer_idx=0)),
            ("Turbo-MSE-2bit", lambda d, h: TurboQuantizer(d, h, mode="mse", bits=2, layer_idx=0)),
            ("Turbo-MSE-4bit", lambda d, h: TurboQuantizer(d, h, mode="mse", bits=4, layer_idx=0)),
            ("Turbo-PROD-4bit", lambda d, h: TurboQuantizer(d, h, mode="prod", bits=4, layer_idx=0)),
            ("Polar-4bit", lambda d, h: PolarQuantizer(d, h, n_r_bits=2, n_theta_bits=2, layer_idx=0)),
        ]
        
        for name, quantizer_fn in configs:
            for head_dim in [64, 128]:
                for seq_len in [1024, 4096]:
                    num_heads = 8
                    
                    keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
                    quantizer = quantizer_fn(head_dim, num_heads)
                    state = quantizer.compress(keys)
                    
                    orig_bytes = seq_len * num_heads * head_dim * 2  # FP16
                    compressed_bytes = state.memory_bytes()
                    ratio = orig_bytes / compressed_bytes
                    
                    result = BenchmarkResult(
                        name=f"compression_{name}_d{head_dim}_seq{seq_len}",
                        metric="compression_ratio",
                        value=float(ratio),
                        unit="x",
                        samples=1
                    )
                    self.results.append(result)
                    
                    key = f"{name}_d{head_dim}_seq{seq_len}"
                    results[key] = {
                        "ratio": ratio,
                        "orig_MB": orig_bytes / 1e6,
                        "compressed_MB": compressed_bytes / 1e6
                    }
                    
                    print(f"  {name:20s} d={head_dim} seq={seq_len:5d}: "
                          f"{ratio:5.2f}× ({orig_bytes/1e6:.2f}MB → {compressed_bytes/1e6:.2f}MB)")
        
        return results
    
    def benchmark_inference_speed(self) -> Dict[str, Any]:
        """Benchmark inference (score computation) speed."""
        print("\n" + "="*70)
        print("BENCHMARK: Inference Speed")
        print("="*70)
        
        results = {}
        
        configs = [
            ("FP32", None),
            ("QJL-1bit", lambda d, h: QJLQuantizer(d, h, layer_idx=0)),
            ("Turbo-PROD-4bit", lambda d, h: TurboQuantizer(d, h, mode="prod", bits=4, layer_idx=0)),
        ]
        
        head_dim, num_heads = 128, 8
        
        for seq_len in [512, 1024, 2048, 4096]:
            print(f"\n  Sequence length: {seq_len}")
            
            keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
            query = np.random.randn(1, num_heads, head_dim).astype(np.float32)
            
            for name, quantizer_fn in configs:
                if quantizer_fn is None:
                    # Baseline FP32
                    mean_time, std_time = self._timeit(
                        lambda: np.einsum("bhd,shd->bhs", query, keys),
                        repeats=10, warmup=3
                    )
                else:
                    quantizer = quantizer_fn(head_dim, num_heads)
                    state = quantizer.compress(keys)
                    
                    mean_time, std_time = self._timeit(
                        lambda: quantizer.scores(query, state),
                        repeats=10, warmup=3
                    )
                
                result = BenchmarkResult(
                    name=f"inference_{name}_seq{seq_len}",
                    metric="time_ms",
                    value=float(mean_time),
                    unit="ms",
                    std=float(std_time),
                    samples=10
                )
                self.results.append(result)
                
                results[f"{name}_seq{seq_len}"] = {
                    "time_ms": mean_time,
                    "std_ms": std_time
                }
                
                print(f"    {name:20s}: {mean_time:7.3f} ± {std_time:5.3f} ms")
        
        return results
    
    def benchmark_real_model_configs(self) -> Dict[str, Any]:
        """Benchmark with real LLM configurations."""
        print("\n" + "="*70)
        print("BENCHMARK: Real Model Configurations")
        print("="*70)
        
        results = {}
        
        models = [
            ("Llama-3.1-8B", 32, 8, 32, 128),
            ("Mistral-7B", 32, 8, 32, 128),
            ("Qwen2.5-7B", 28, 4, 28, 128),
            ("Phi-3-mini", 32, 32, 32, 96),
        ]
        
        seq_len = 4096
        
        for name, layers, kv_heads, q_heads, head_dim in models:
            quantizer = TurboQuantizer(
                head_dim=head_dim,
                num_kv_heads=kv_heads,
                num_q_heads=q_heads,
                mode="prod",
                bits=4,
                layer_idx=0
            )
            
            # Memory for one layer
            report = quantizer.memory_report(seq_len)
            
            # Scale to full model
            orig_total = report["original_fp16_MB"] * layers
            comp_total = report["compressed_MB"] * layers
            
            result = BenchmarkResult(
                name=f"model_{name}_memory",
                metric="memory_savings",
                value=float(comp_total),
                unit="MB",
                samples=1
            )
            self.results.append(result)
            
            results[name] = {
                "layers": layers,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "orig_MB": orig_total,
                "compressed_MB": comp_total,
                "ratio": orig_total / comp_total
            }
            
            print(f"  {name:20s}: {orig_total:6.1f}MB → {comp_total:6.1f}MB  "
                  f"({orig_total/comp_total:.1f}× reduction)")
        
        return results
    
    def run_all(self):
        """Run all benchmarks."""
        print("\n" + "╔" + "="*68 + "╗")
        print("║" + " "*15 + "TurboQuantCPU Benchmark Suite" + " "*24 + "║")
        print("╚" + "="*68 + "╝")
        
        print(f"\nSystem: {self.cpu_info.brand}")
        print(f"SIMD: {self.cpu_info.best_simd}")
        print(f"Cores: {self.cpu_info.num_cores} physical / {self.cpu_info.num_threads} logical")
        print(f"C Extension: {self.cpu_info.c_kernel_simd}")
        
        # Run all benchmarks
        self.benchmark_fwht_correctness()
        self.benchmark_fwht_speed()
        self.benchmark_qjl_unbiasedness()
        self.benchmark_compression_ratios()
        self.benchmark_inference_speed()
        self.benchmark_real_model_configs()
        
        # Save results
        filepath = self.save_results()
        
        print("\n" + "="*70)
        print("All benchmarks completed!")
        print(f"Results saved to: {filepath}")
        print("="*70)
        
        return self.results


def main():
    benchmarks = TurboQuantBenchmarks()
    benchmarks.run_all()


if __name__ == "__main__":
    main()
