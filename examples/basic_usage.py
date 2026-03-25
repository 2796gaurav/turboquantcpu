"""
Basic usage examples for TurboQuantCPU.
"""

import numpy as np
from turboquantcpu import (
    TurboQuantizer, TurboMode,
    QJLQuantizer,
    PolarQuantizer,
    print_cpu_capabilities,
)


def example_basic_compression():
    """Basic compression and decompression example."""
    print("=" * 60)
    print("Example: Basic Compression")
    print("=" * 60)
    
    head_dim = 128
    num_heads = 8
    seq_len = 1024
    
    # Create quantizer
    quantizer = TurboQuantizer(
        head_dim=head_dim,
        num_kv_heads=num_heads,
        mode=TurboMode.PROD,
        bits=4,
        layer_idx=0,
    )
    
    # Generate random keys
    keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
    
    # Compress
    state = quantizer.compress(keys)
    
    # Compute attention scores
    query = np.random.randn(1, num_heads, head_dim).astype(np.float32)
    scores = quantizer.scores(query, state)
    
    # Check compression
    orig_bytes = seq_len * num_heads * head_dim * 2  # FP16
    compressed_bytes = state.memory_bytes()
    ratio = orig_bytes / compressed_bytes
    
    print(f"Original size: {orig_bytes / 1024 / 1024:.2f} MB")
    print(f"Compressed size: {compressed_bytes / 1024 / 1024:.2f} MB")
    print(f"Compression ratio: {ratio:.1f}×")
    print(f"Score shape: {scores.shape}")
    print()


def example_qjl_quantization():
    """QJL 1-bit quantization example."""
    print("=" * 60)
    print("Example: QJL 1-bit Quantization")
    print("=" * 60)
    
    head_dim = 128
    num_heads = 8
    seq_len = 1024
    
    qjl = QJLQuantizer(head_dim, num_heads, layer_idx=0)
    
    keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
    state = qjl.compress(keys)
    
    # Test unbiasedness
    query = np.random.randn(1, num_heads, head_dim).astype(np.float32)
    true_scores = np.einsum("bhd,shd->bhs", query, keys)
    est_scores = qjl.scores(query, state)
    
    bias = np.abs((est_scores - true_scores).mean())
    
    print(f"Sequence length: {seq_len}")
    print(f"Compression ratio: {seq_len * num_heads * head_dim * 2 / state.memory_bytes():.1f}×")
    print(f"Estimator bias: {bias:.5f} (should be ~0)")
    print()


def example_compare_modes():
    """Compare different quantization modes."""
    print("=" * 60)
    print("Example: Compare Quantization Modes")
    print("=" * 60)
    
    head_dim = 128
    num_heads = 8
    seq_len = 1024
    
    keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
    query = np.random.randn(1, num_heads, head_dim).astype(np.float32)
    true_scores = np.einsum("bhd,shd->bhs", query, keys)
    
    modes = [
        ("QJL", QJLQuantizer(head_dim, num_heads, layer_idx=0)),
        ("Turbo-MSE-4bit", TurboQuantizer(head_dim, num_heads, mode="mse", bits=4, layer_idx=1)),
        ("Turbo-PROD-4bit", TurboQuantizer(head_dim, num_heads, mode="prod", bits=4, layer_idx=2)),
        ("Polar-4bit", PolarQuantizer(head_dim, num_heads, n_r_bits=2, n_theta_bits=2, layer_idx=3)),
    ]
    
    for name, quantizer in modes:
        state = quantizer.compress(keys)
        est_scores = quantizer.scores(query, state)
        
        mae = np.abs(est_scores - true_scores).mean()
        bias = (est_scores - true_scores).mean()
        ratio = seq_len * num_heads * head_dim * 2 / state.memory_bytes()
        
        print(f"{name:20s} | MAE: {mae:.4f} | Bias: {bias:+.4f} | Ratio: {ratio:.1f}×")
    
    print()


def example_memory_report():
    """Show detailed memory report."""
    print("=" * 60)
    print("Example: Memory Report")
    print("=" * 60)
    
    from turboquantcpu import TurboQuantizer, TurboMode
    
    configs = [
        ("Llama-3.1-8B", 32, 8, 32, 128),
        ("Mistral-7B", 32, 8, 32, 128),
        ("Qwen2.5-7B", 28, 4, 28, 128),
    ]
    
    for name, layers, kv_heads, q_heads, head_dim in configs:
        quantizer = TurboQuantizer(
            head_dim=head_dim,
            num_kv_heads=kv_heads,
            num_q_heads=q_heads,
            mode=TurboMode.PROD,
            bits=4,
            layer_idx=0,
        )
        
        report = quantizer.memory_report(seq_len=4096)
        orig_mb = report["original_fp16_MB"] * layers
        comp_mb = report["compressed_MB"] * layers
        
        print(f"{name:20s} | FP16: {orig_mb:6.1f} MB | Compressed: {comp_mb:6.1f} MB | Ratio: {orig_mb/comp_mb:.1f}×")
    
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 12 + "TurboQuantCPU Examples" + " " * 24 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")
    
    # Show CPU capabilities
    print_cpu_capabilities()
    print()
    
    # Run examples
    example_basic_compression()
    example_qjl_quantization()
    example_compare_modes()
    example_memory_report()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
