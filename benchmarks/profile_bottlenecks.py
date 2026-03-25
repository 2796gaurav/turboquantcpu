#!/usr/bin/env python3
"""
Profile bottlenecks in TurboQuantCPU.
"""

import sys
import os
import time
import numpy as np
import cProfile
import pstats
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquantcpu import QJLQuantizer, TurboQuantizer
from turboquantcpu.fwht import _FOUND_C, _ACTIVE_BACKEND, FWHTBackend


def profile_qjl_compression():
    """Profile QJL compression."""
    print("\n" + "="*70)
    print("PROFILING: QJL Compression")
    print("="*70)
    
    head_dim, num_heads = 128, 8
    seq_len = 1024
    
    keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
    qjl = QJLQuantizer(head_dim, num_heads, layer_idx=0)
    
    pr = cProfile.Profile()
    pr.enable()
    
    for _ in range(10):
        state = qjl.compress(keys)
    
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())


def profile_qjl_scores():
    """Profile QJL score computation."""
    print("\n" + "="*70)
    print("PROFILING: QJL Score Computation")
    print("="*70)
    
    head_dim, num_heads = 128, 8
    seq_len = 1024
    
    keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
    query = np.random.randn(1, num_heads, head_dim).astype(np.float32)
    
    qjl = QJLQuantizer(head_dim, num_heads, layer_idx=0)
    state = qjl.compress(keys)
    
    pr = cProfile.Profile()
    pr.enable()
    
    for _ in range(10):
        scores = qjl.scores(query, state)
    
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())


def check_backend_usage():
    """Check which backend is being used."""
    print("\n" + "="*70)
    print("BACKEND CHECK")
    print("="*70)
    print(f"C Extension Found: {_FOUND_C}")
    print(f"Active Backend: {_ACTIVE_BACKEND}")


def benchmark_step_by_step():
    """Benchmark each step of compression."""
    print("\n" + "="*70)
    print("STEP-BY-STEP BENCHMARK")
    print("="*70)
    
    from turboquantcpu.fwht import randomized_fwht_batch, get_signs, pack_bits
    
    head_dim, num_heads = 128, 8
    seq_len = 1024
    d_p2 = 128
    
    keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
    signs = get_signs(d_p2, seed=42, as_numpy=True)
    
    # Step 1: Norm computation
    t0 = time.perf_counter()
    for _ in range(100):
        norms = np.linalg.norm(keys, axis=-1)
    t1 = time.perf_counter()
    print(f"  Norm computation:     {(t1-t0)/100*1000:.3f} ms")
    
    # Step 2: Normalization
    t0 = time.perf_counter()
    for _ in range(100):
        keys_n = keys / norms[..., None].clip(1e-8)
    t1 = time.perf_counter()
    print(f"  Normalization:        {(t1-t0)/100*1000:.3f} ms")
    
    # Step 3: FWHT rotation
    t0 = time.perf_counter()
    for _ in range(100):
        rotated = randomized_fwht_batch(keys_n[:, 0, :], signs)
    t1 = time.perf_counter()
    print(f"  FWHT (per head):      {(t1-t0)/100*1000:.3f} ms")
    print(f"  FWHT (all heads):     {(t1-t0)/100*1000*num_heads:.3f} ms")
    
    # Step 4: Sign extraction
    t0 = time.perf_counter()
    for _ in range(100):
        signs_out = np.sign(rotated).astype(np.int8)
    t1 = time.perf_counter()
    print(f"  Sign extraction:      {(t1-t0)/100*1000:.3f} ms")
    
    # Step 5: Bit packing
    t0 = time.perf_counter()
    for _ in range(100):
        packed = pack_bits(signs_out)
    t1 = time.perf_counter()
    print(f"  Bit packing:          {(t1-t0)/100*1000:.3f} ms")
    
    # Total theoretical
    print(f"\n  Theoretical total:    ~{((t1-t0)/100*1000*num_heads + 0.5 + 2.9):.1f} ms")


def main():
    check_backend_usage()
    benchmark_step_by_step()
    profile_qjl_compression()
    profile_qjl_scores()


if __name__ == "__main__":
    main()
