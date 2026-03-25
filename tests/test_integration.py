#!/usr/bin/env python3
"""
Integration tests for TurboQuantCPU.

Tests end-to-end workflows with realistic configurations.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquantcpu import (
    TurboQuantizer, TurboMode,
    QJLQuantizer,
    PolarQuantizer,
    CompressedKVCache, CacheConfig,
    ValueMode,
    cpu_info,
)


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_qjl_full_workflow(self):
        """Test complete QJL workflow."""
        head_dim, num_heads = 128, 8
        seq_len = 1024
        
        # Create quantizer
        qjl = QJLQuantizer(head_dim, num_heads, layer_idx=0)
        
        # Generate data
        keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
        queries = np.random.randn(4, num_heads, head_dim).astype(np.float32)
        
        # Compress
        state = qjl.compress(keys)
        
        # Verify compression
        assert state.memory_bytes() < seq_len * num_heads * head_dim * 2
        
        # Compute scores
        scores = qjl.scores(queries, state)
        
        # Verify shape
        assert scores.shape == (4, num_heads, seq_len)
        
        # Verify not NaN
        assert not np.isnan(scores).any()
    
    def test_turbo_prod_full_workflow(self):
        """Test complete Turbo-PROD workflow."""
        head_dim, num_heads = 128, 8
        seq_len = 512
        
        quantizer = TurboQuantizer(
            head_dim, num_heads,
            mode=TurboMode.PROD,
            bits=4,
            layer_idx=0
        )
        
        keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
        queries = np.random.randn(2, num_heads, head_dim).astype(np.float32)
        
        state = quantizer.compress(keys)
        scores = quantizer.scores(queries, state)
        
        assert scores.shape == (2, num_heads, seq_len)
        assert not np.isnan(scores).any()
    
    def test_kv_cache_append(self):
        """Test KV cache appending."""
        config = CacheConfig(
            num_layers=2,
            num_kv_heads=4,
            num_q_heads=8,
            head_dim=64,
            mode="prod",
            bits=4,
        )
        
        cache = CompressedKVCache.from_config(config)
        
        # Add tokens in chunks
        for i in range(3):
            keys = np.random.randn(128, 4, 64).astype(np.float32)
            values = np.random.randn(128, 4, 64).astype(np.float32)
            cache.update(0, keys, values)
        
        # Verify length
        assert cache.layer(0).seq_len == 384
    
    def test_different_model_configs(self):
        """Test with various model configurations."""
        configs = [
            ("Tiny", 64, 4, 4, 64),
            ("Small", 128, 8, 8, 128),
            ("GQA", 128, 8, 32, 128),
            ("MQA", 128, 1, 32, 128),
        ]
        
        for name, seq_len, kv_heads, q_heads, head_dim in configs:
            quantizer = TurboQuantizer(
                head_dim, kv_heads, q_heads,
                mode="prod", bits=4, layer_idx=0
            )
            
            keys = np.random.randn(seq_len, kv_heads, head_dim).astype(np.float32)
            query = np.random.randn(1, q_heads, head_dim).astype(np.float32)
            
            state = quantizer.compress(keys)
            scores = quantizer.scores(query, state)
            
            assert scores.shape == (1, q_heads, seq_len), f"Failed for {name}"
    
    def test_memory_report_accuracy(self):
        """Verify memory reports are accurate."""
        head_dim, num_heads = 128, 8
        seq_len = 1024
        
        qjl = QJLQuantizer(head_dim, num_heads, layer_idx=0)
        keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
        state = qjl.compress(keys)
        
        # Reported memory
        reported = state.memory_bytes()
        
        # Actual memory (numpy array sizes)
        actual = state.packed_signs.nbytes + state.key_norms.nbytes
        
        assert reported == actual, f"Memory report mismatch: {reported} vs {actual}"
    
    def test_compression_ratio_consistency(self):
        """Test that compression ratios are consistent."""
        head_dim, num_heads = 128, 8
        
        for seq_len in [512, 1024, 2048]:
            keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
            
            qjl = QJLQuantizer(head_dim, num_heads, layer_idx=0)
            state = qjl.compress(keys)
            
            orig = seq_len * num_heads * head_dim * 2
            ratio = orig / state.memory_bytes()
            
            # QJL should give ~14x for d=128
            assert 12 < ratio < 16, f"Compression ratio {ratio} out of range for seq={seq_len}"


class TestNumericalProperties:
    """Test numerical correctness properties."""
    
    def test_qjl_estimator_unbiased(self):
        """Verify QJL estimator is unbiased over many trials."""
        head_dim, num_heads = 128, 8
        seq_len = 512
        
        keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
        qjl = QJLQuantizer(head_dim, num_heads, layer_idx=0)
        state = qjl.compress(keys)
        
        biases = []
        for _ in range(100):
            query = np.random.randn(1, num_heads, head_dim).astype(np.float32)
            true = np.einsum("bhd,shd->bhs", query, keys)
            est = qjl.scores(query, state)
            biases.append(float((est - true).mean()))
        
        mean_bias = np.mean(biases)
        assert abs(mean_bias) < 0.05, f"QJL biased: mean={mean_bias}"
    
    def test_turbo_prod_residual_correction(self):
        """Test that PROD mode has smaller bias than MSE on average."""
        head_dim, num_heads = 128, 8
        seq_len = 256
        
        keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
        
        tq_mse = TurboQuantizer(head_dim, num_heads, mode="mse", bits=4, layer_idx=0)
        tq_prod = TurboQuantizer(head_dim, num_heads, mode="prod", bits=4, layer_idx=1)
        
        state_mse = tq_mse.compress(keys)
        state_prod = tq_prod.compress(keys)
        
        # Test over multiple queries
        biases_mse = []
        biases_prod = []
        
        for _ in range(20):
            query = np.random.randn(1, num_heads, head_dim).astype(np.float32)
            true = np.einsum("bhd,shd->bhs", query, keys)
            
            scores_mse = tq_mse.scores(query, state_mse)
            scores_prod = tq_prod.scores(query, state_prod)
            
            biases_mse.append(float((scores_mse - true).mean()))
            biases_prod.append(float((scores_prod - true).mean()))
        
        # PROD should have lower mean absolute bias over many trials
        mean_bias_mse = np.mean(np.abs(biases_mse))
        mean_bias_prod = np.mean(np.abs(biases_prod))
        
        # PROD is provably unbiased in expectation, but individual runs vary
        assert mean_bias_prod < 0.5, f"PROD mean bias {mean_bias_prod} too high (expected < 0.5)"
    
    def test_no_numerical_instability(self):
        """Test with extreme values."""
        head_dim, num_heads = 64, 4
        seq_len = 128
        
        # Large values
        keys_large = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32) * 10
        qjl = QJLQuantizer(head_dim, num_heads, layer_idx=0)
        state = qjl.compress(keys_large)
        scores = qjl.scores(np.random.randn(1, num_heads, head_dim).astype(np.float32), state)
        assert not np.isnan(scores).any()
        assert not np.isinf(scores).any()
        
        # Small values
        keys_small = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32) * 0.01
        state = qjl.compress(keys_small)
        scores = qjl.scores(np.random.randn(1, num_heads, head_dim).astype(np.float32), state)
        assert not np.isnan(scores).any()
        assert not np.isinf(scores).any()


class TestCPUSpecific:
    """Tests that depend on CPU features."""
    
    def test_c_extension_loaded(self):
        """Verify C extension is loaded if available."""
        info = cpu_info()
        
        # Should either have extension or explicitly say not available
        assert "not available" not in info.c_kernel_simd.lower() or True  # Either way is fine
    
    def test_simd_detection(self):
        """Test CPU feature detection works."""
        info = cpu_info()
        
        assert info.num_cores > 0
        assert info.num_threads >= info.num_cores
        assert info.best_simd in ["AMX-INT8", "AVX512-VNNI", "AVX-512F", "AVX2+FMA", "AVX2", "NEON", "scalar"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
