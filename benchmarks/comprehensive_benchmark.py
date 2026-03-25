#!/usr/bin/env python3
"""
Comprehensive TurboQuantCPU Benchmark Suite with Visualizations

This benchmark suite tests:
1. Correctness (mathematical guarantees)
2. Memory compression ratios
3. Speed (various sequence lengths)
4. Quality (accuracy vs baseline)
5. Long context behavior
6. Multi-model compatibility

Generates plots for README and documentation.
"""

import json
import time
import gc
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquantcpu import (
    TurboQuantizer, TurboMode, QJLQuantizer, PolarQuantizer,
    ValueQuantizer, ValueMode, CompressedKVCache, CacheConfig,
    cpu_info, fwht_numpy, get_signs, randomized_fwht,
    run_correctness_check
)

# Constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    name: str
    metric: str
    value: float
    unit: str
    std: float = 0.0
    samples: int = 1
    metadata: Dict = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}


class ComprehensiveBenchmark:
    """Comprehensive benchmark suite with visualizations."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.cpu_info = cpu_info()
        self.system_info = {
            "cpu": self.cpu_info.brand,
            "simd": self.cpu_info.best_simd,
            "cores": self.cpu_info.num_cores,
            "threads": self.cpu_info.num_threads,
            "c_extension": self.cpu_info.c_kernel_simd,
            "timestamp": datetime.now().isoformat(),
        }
        print("="*70)
        print("TurboQuantCPU Comprehensive Benchmark Suite")
        print("="*70)
        print(f"System: {self.system_info['cpu']}")
        print(f"SIMD: {self.system_info['simd']}")
        print(f"Cores: {self.system_info['cores']} physical / {self.system_info['threads']} logical")
        print(f"C Extension: {self.system_info['c_extension']}")
        print("="*70)
    
    def save_results(self, filename: str = None):
        """Save results to JSON."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_comprehensive_{timestamp}.json"
        
        filepath = os.path.join(RESULTS_DIR, filename)
        
        data = {
            "system": self.system_info,
            "results": [asdict(r) for r in self.results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Results saved to: {filepath}")
        return filepath
    
    def _timeit(self, func, repeats: int = 10, warmup: int = 3) -> Tuple[float, float]:
        """Time a function with warmup."""
        for _ in range(warmup):
            func()
        
        times = []
        for _ in range(repeats):
            gc.collect()
            t0 = time.perf_counter()
            func()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        
        return np.mean(times), np.std(times)
    
    def benchmark_correctness(self) -> Dict[str, Any]:
        """Verify all mathematical guarantees."""
        print("\n" + "="*70)
        print("BENCHMARK 1/6: Correctness Verification")
        print("="*70)
        
        results = {}
        
        # Run built-in correctness check
        correctness = run_correctness_check(verbose=False)
        
        # FWHT tests
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
                samples=100,
                metadata={"dim": d}
            )
            self.results.append(result)
            results[f"fwht_d{d}"] = max_err
            status = "✓" if max_err < 1e-4 else "✗"
            print(f"  FWHT self-inverse d={d:3d}: {max_err:.3e} {status}")
        
        # QJL unbiasedness
        for seq_len in [512, 4096]:
            keys = np.random.randn(seq_len, 8, 128).astype(np.float32)
            qjl = QJLQuantizer(128, 8, layer_idx=0)
            state = qjl.compress(keys)
            
            biases = []
            for _ in range(50):
                query = np.random.randn(1, 8, 128).astype(np.float32)
                true = np.einsum("bhd,shd->bhs", query, keys)
                est = qjl.scores(query, state)
                biases.append(float((est - true).mean()))
            
            mean_bias = np.mean(biases)
            std_bias = np.std(biases)
            
            result = BenchmarkResult(
                name=f"qjl_unbiasedness_seq{seq_len}",
                metric="mean_bias",
                value=float(mean_bias),
                unit="absolute",
                std=float(std_bias),
                samples=50,
                metadata={"seq_len": seq_len}
            )
            self.results.append(result)
            status = "✓" if abs(mean_bias) < 0.1 else "✗"
            print(f"  QJL unbiasedness seq={seq_len:5d}: bias={mean_bias:+.4f}±{std_bias:.4f} {status}")
        
        return results
    
    def benchmark_compression(self) -> Dict[str, Any]:
        """Benchmark compression ratios."""
        print("\n" + "="*70)
        print("BENCHMARK 2/6: Compression Ratios")
        print("="*70)
        
        results = {}
        
        configs = [
            ("QJL-1bit", lambda d, h: QJLQuantizer(d, h, layer_idx=0), 1),
            ("Turbo-MSE-2bit", lambda d, h: TurboQuantizer(d, h, mode="mse", bits=2, layer_idx=0), 2),
            ("Turbo-MSE-4bit", lambda d, h: TurboQuantizer(d, h, mode="mse", bits=4, layer_idx=0), 4),
            ("Turbo-PROD-4bit", lambda d, h: TurboQuantizer(d, h, mode="prod", bits=4, layer_idx=0), 4),
            ("Polar-4bit", lambda d, h: PolarQuantizer(d, h, n_r_bits=2, n_theta_bits=2, layer_idx=0), 4),
        ]
        
        compression_data = []
        
        for name, quantizer_fn, bits in configs:
            for head_dim in [64, 128]:
                for seq_len in [1024, 4096]:
                    num_heads = 8
                    
                    keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
                    quantizer = quantizer_fn(head_dim, num_heads)
                    state = quantizer.compress(keys)
                    
                    orig_bytes = seq_len * num_heads * head_dim * 2
                    compressed_bytes = state.memory_bytes()
                    ratio = orig_bytes / compressed_bytes
                    
                    result = BenchmarkResult(
                        name=f"compression_{name}_d{head_dim}_seq{seq_len}",
                        metric="compression_ratio",
                        value=float(ratio),
                        unit="x",
                        samples=1,
                        metadata={
                            "method": name,
                            "head_dim": head_dim,
                            "seq_len": seq_len,
                            "bits": bits,
                            "orig_mb": orig_bytes / 1e6,
                            "compressed_mb": compressed_bytes / 1e6
                        }
                    )
                    self.results.append(result)
                    
                    compression_data.append({
                        'method': name,
                        'head_dim': head_dim,
                        'seq_len': seq_len,
                        'ratio': ratio
                    })
                    
                    print(f"  {name:20s} d={head_dim:3d} seq={seq_len:5d}: {ratio:5.2f}×")
        
        # Create compression plot
        self._plot_compression(compression_data)
        
        return results
    
    def _plot_compression(self, data):
        """Plot compression ratios."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        methods = sorted(set(d['method'] for d in data))
        head_dims = sorted(set(d['head_dim'] for d in data))
        
        x = np.arange(len(methods))
        width = 0.35
        
        for i, head_dim in enumerate(head_dims):
            ratios = [d['ratio'] for d in data if d['head_dim'] == head_dim and d['seq_len'] == 4096]
            ax.bar(x + i*width, ratios, width, label=f'd={head_dim}')
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Compression Ratio (×)')
        ax.set_title('Compression Ratios by Method and Dimension')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'compression_ratios.png'), dpi=150)
        plt.close()
        print(f"  ✓ Saved compression plot")
    
    def benchmark_speed(self) -> Dict[str, Any]:
        """Benchmark inference speed."""
        print("\n" + "="*70)
        print("BENCHMARK 3/6: Inference Speed")
        print("="*70)
        
        results = {}
        
        head_dim, num_heads = 128, 8
        
        configs = [
            ("FP32", None),
            ("QJL-1bit", lambda d, h: QJLQuantizer(d, h, layer_idx=0)),
            ("Turbo-PROD-4bit", lambda d, h: TurboQuantizer(d, h, mode="prod", bits=4, layer_idx=0)),
        ]
        
        speed_data = {name: [] for name, _ in configs}
        
        for seq_len in [512, 1024, 2048, 4096, 8192]:
            print(f"\n  Sequence length: {seq_len:,}")
            
            keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
            query = np.random.randn(1, num_heads, head_dim).astype(np.float32)
            
            for name, quantizer_fn in configs:
                if quantizer_fn is None:
                    # FP32 baseline
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
                    name=f"speed_{name}_seq{seq_len}",
                    metric="time_ms",
                    value=float(mean_time),
                    unit="ms",
                    std=float(std_time),
                    samples=10,
                    metadata={"method": name, "seq_len": seq_len}
                )
                self.results.append(result)
                
                speed_data[name].append((seq_len, mean_time))
                print(f"    {name:20s}: {mean_time:7.3f} ± {std_time:5.3f} ms")
        
        # Create speed plot
        self._plot_speed(speed_data)
        
        return results
    
    def _plot_speed(self, data):
        """Plot speed comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for method, points in data.items():
            if points:
                seq_lens, times = zip(*points)
                ax.plot(seq_lens, times, marker='o', label=method, linewidth=2)
        
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Inference Speed vs Sequence Length')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'speed_comparison.png'), dpi=150)
        plt.close()
        print(f"  ✓ Saved speed plot")
    
    def benchmark_quality(self) -> Dict[str, Any]:
        """Benchmark quality (accuracy vs baseline)."""
        print("\n" + "="*70)
        print("BENCHMARK 4/6: Quality (Accuracy vs Baseline)")
        print("="*70)
        
        results = {}
        head_dim, num_heads = 128, 8
        seq_len = 4096
        
        # Generate test data
        keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
        
        # Test queries
        n_queries = 100
        queries = [np.random.randn(1, num_heads, head_dim).astype(np.float32) for _ in range(n_queries)]
        
        # Compute true attention
        true_scores = [np.einsum("bhd,shd->bhs", q, keys) for q in queries]
        
        configs = [
            ("QJL", QJLQuantizer(head_dim, num_heads, layer_idx=0)),
            ("Turbo-MSE-4bit", TurboQuantizer(head_dim, num_heads, mode="mse", bits=4, layer_idx=1)),
            ("Turbo-PROD-4bit", TurboQuantizer(head_dim, num_heads, mode="prod", bits=4, layer_idx=2)),
            ("Polar-4bit", PolarQuantizer(head_dim, num_heads, n_r_bits=2, n_theta_bits=2, layer_idx=3)),
        ]
        
        quality_data = []
        
        for name, quantizer in configs:
            state = quantizer.compress(keys)
            
            maes = []
            biases = []
            
            for q, true in zip(queries, true_scores):
                est = quantizer.scores(q, state)
                mae = np.abs(est - true).mean()
                bias = (est - true).mean()
                maes.append(mae)
                biases.append(bias)
            
            mean_mae = np.mean(maes)
            std_mae = np.std(maes)
            mean_bias = np.mean(biases)
            std_bias = np.std(biases)
            
            result = BenchmarkResult(
                name=f"quality_{name}",
                metric="mae",
                value=float(mean_mae),
                unit="absolute",
                std=float(std_mae),
                samples=n_queries,
                metadata={
                    "method": name,
                    "mean_bias": float(mean_bias),
                    "std_bias": float(std_bias)
                }
            )
            self.results.append(result)
            
            quality_data.append({
                'method': name,
                'mae': mean_mae,
                'bias': abs(mean_bias)
            })
            
            print(f"  {name:20s}: MAE={mean_mae:.4f}±{std_mae:.4f}, Bias={mean_bias:+.4f}±{std_bias:.4f}")
        
        # Create quality plot
        self._plot_quality(quality_data)
        
        return results
    
    def _plot_quality(self, data):
        """Plot quality comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        methods = [d['method'] for d in data]
        maes = [d['mae'] for d in data]
        biases = [d['bias'] for d in data]
        
        # MAE plot
        ax1.bar(methods, maes, color='steelblue')
        ax1.set_ylabel('Mean Absolute Error')
        ax1.set_title('Reconstruction Error by Method')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Bias plot
        ax2.bar(methods, biases, color='coral')
        ax2.set_ylabel('|Bias|')
        ax2.set_title('Attention Score Bias by Method')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'quality_comparison.png'), dpi=150)
        plt.close()
        print(f"  ✓ Saved quality plot")
    
    def benchmark_long_context(self) -> Dict[str, Any]:
        """Benchmark long context behavior."""
        print("\n" + "="*70)
        print("BENCHMARK 5/6: Long Context Scaling")
        print("="*70)
        
        results = {}
        head_dim, num_heads = 128, 8
        
        quantizer = TurboQuantizer(head_dim, num_heads, mode="prod", bits=4, layer_idx=0)
        
        context_lengths = [1024, 2048, 4096, 8192, 16384, 32768]
        memory_data = []
        
        for seq_len in context_lengths:
            keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
            state = quantizer.compress(keys)
            
            orig_mb = seq_len * num_heads * head_dim * 2 / 1e6
            comp_mb = state.memory_bytes() / 1e6
            ratio = orig_mb / comp_mb
            
            result = BenchmarkResult(
                name=f"long_context_seq{seq_len}",
                metric="memory_mb",
                value=float(comp_mb),
                unit="MB",
                samples=1,
                metadata={
                    "seq_len": seq_len,
                    "orig_mb": orig_mb,
                    "ratio": ratio
                }
            )
            self.results.append(result)
            
            memory_data.append({
                'seq_len': seq_len,
                'orig': orig_mb,
                'compressed': comp_mb
            })
            
            print(f"  Context {seq_len:6d}: {orig_mb:8.1f} MB → {comp_mb:8.1f} MB ({ratio:.1f}×)")
        
        # Create long context plot
        self._plot_long_context(memory_data)
        
        return results
    
    def _plot_long_context(self, data):
        """Plot long context scaling."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        seq_lens = [d['seq_len'] for d in data]
        orig = [d['orig'] for d in data]
        compressed = [d['compressed'] for d in data]
        
        ax.plot(seq_lens, orig, marker='o', label='FP16 Baseline', linewidth=2)
        ax.plot(seq_lens, compressed, marker='s', label='Turbo-4bit', linewidth=2)
        
        ax.set_xlabel('Context Length')
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Memory Scaling with Context Length')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'long_context_scaling.png'), dpi=150)
        plt.close()
        print(f"  ✓ Saved long context plot")
    
    def benchmark_real_models(self) -> Dict[str, Any]:
        """Benchmark with real model configurations."""
        print("\n" + "="*70)
        print("BENCHMARK 6/6: Real Model Configurations")
        print("="*70)
        
        results = {}
        
        models = [
            ("Qwen2.5-0.5B", 24, 2, 14, 64),
            ("Qwen2.5-1.5B", 28, 2, 12, 128),
            ("Qwen2.5-7B", 28, 4, 28, 128),
            ("Llama-3.1-8B", 32, 8, 32, 128),
            ("Mistral-7B", 32, 8, 32, 128),
            ("Llama-3.1-70B", 80, 8, 64, 128),
        ]
        
        seq_len = 4096
        model_data = []
        
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
            
            # Scale to full model (keys + values, both K and V)
            orig_total = report["original_fp16_MB"] * layers * 2  # K + V
            comp_total = report["compressed_MB"] * layers * 2
            
            result = BenchmarkResult(
                name=f"model_{name}",
                metric="memory_mb",
                value=float(comp_total),
                unit="MB",
                samples=1,
                metadata={
                    "model": name,
                    "layers": layers,
                    "kv_heads": kv_heads,
                    "head_dim": head_dim,
                    "orig_mb": orig_total,
                    "ratio": orig_total / comp_total
                }
            )
            self.results.append(result)
            
            model_data.append({
                'model': name,
                'orig': orig_total,
                'compressed': comp_total
            })
            
            print(f"  {name:20s}: {orig_total:7.1f} MB → {comp_total:7.1f} MB ({orig_total/comp_total:.1f}×)")
        
        # Create model comparison plot
        self._plot_models(model_data)
        
        return results
    
    def _plot_models(self, data):
        """Plot model comparison."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = [d['model'] for d in data]
        orig = [d['orig'] for d in data]
        compressed = [d['compressed'] for d in data]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax.bar(x - width/2, orig, width, label='FP16 Baseline', color='lightcoral')
        ax.bar(x + width/2, compressed, width, label='Turbo-4bit', color='steelblue')
        
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Memory Usage: Real Model Configurations (Context=4096)')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'), dpi=150)
        plt.close()
        print(f"  ✓ Saved model comparison plot")
    
    def generate_summary(self):
        """Generate summary report."""
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        
        # Get key metrics
        qjl_compression = [r.value for r in self.results 
                          if r.name.startswith("compression_QJL") and r.metadata.get("head_dim") == 128]
        prod_compression = [r.value for r in self.results 
                           if r.name.startswith("compression_Turbo-PROD") and r.metadata.get("head_dim") == 128]
        
        if qjl_compression:
            print(f"\nQJL Compression (d=128): {np.mean(qjl_compression):.1f}×")
        if prod_compression:
            print(f"Turbo-PROD Compression (d=128): {np.mean(prod_compression):.1f}×")
        
        print(f"\nTotal benchmarks run: {len(self.results)}")
        print(f"Plots generated in: {PLOTS_DIR}")
        print(f"Results saved in: {RESULTS_DIR}")
        print("="*70)
    
    def run_all(self):
        """Run all benchmarks."""
        self.benchmark_correctness()
        self.benchmark_compression()
        self.benchmark_speed()
        self.benchmark_quality()
        self.benchmark_long_context()
        self.benchmark_real_models()
        
        self.generate_summary()
        
        # Save results
        filepath = self.save_results()
        
        return self.results


def main():
    """Run comprehensive benchmarks."""
    benchmark = ComprehensiveBenchmark()
    results = benchmark.run_all()
    
    print("\n" + "="*70)
    print("All benchmarks completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
