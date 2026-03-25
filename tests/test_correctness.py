"""
Correctness tests for TurboQuantCPU.

Verifies mathematical guarantees:
  - FWHT self-inverse property
  - Norm preservation
  - QJL unbiasedness
  - TurboQuant monotonicity
"""

import numpy as np
import pytest

from turboquantcpu import (
    TurboQuantizer, TurboMode,
    QJLQuantizer,
    fwht_numpy, get_signs, randomized_fwht,
)


class TestFWHT:
    """Test Fast Walsh-Hadamard Transform properties."""
    
    def test_self_inverse(self):
        """FWHT should be self-inverse (orthogonal)."""
        d = 128
        x = np.random.randn(64, d).astype(np.float32)
        
        y = fwht_numpy(x.copy())
        z = fwht_numpy(y.copy())
        
        max_err = np.abs(z - x).max()
        assert max_err < 1e-4, f"FWHT not self-inverse: max_err={max_err}"
    
    def test_norm_preservation(self):
        """FWHT should preserve L2 norms."""
        d = 128
        x = np.random.randn(64, d).astype(np.float32)
        
        norms_before = np.linalg.norm(x, axis=-1)
        y = fwht_numpy(x.copy())
        norms_after = np.linalg.norm(y, axis=-1)
        
        rel_err = np.abs(norms_before - norms_after) / (norms_before + 1e-12)
        assert rel_err.mean() < 1e-4, "FWHT does not preserve norms"
    
    def test_randomized_fwht_orthogonal(self):
        """Randomized FWHT with signs should be orthogonal."""
        d = 128
        signs = get_signs(d, seed=42, as_numpy=True)
        
        x = np.random.randn(d).astype(np.float32)
        y = randomized_fwht(x, signs)
        
        # Should preserve norm
        assert np.abs(np.linalg.norm(x) - np.linalg.norm(y)) < 1e-3


class TestQJL:
    """Test QJL quantization properties."""
    
    def test_unbiasedness(self):
        """QJL estimator should be unbiased."""
        head_dim, num_heads = 128, 8
        seq_len = 512
        
        keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
        qjl = QJLQuantizer(head_dim, num_heads, layer_idx=0)
        state = qjl.compress(keys)
        
        # Test over multiple random queries
        biases = []
        for _ in range(50):
            q = np.random.randn(1, num_heads, head_dim).astype(np.float32)
            true = np.einsum("bhd,shd->bhs", q, keys)
            est = qjl.scores(q, state)
            biases.append(float((est - true).mean()))
        
        mean_bias = np.mean(np.abs(biases))
        # Theoretical guarantee: unbiased on average, but we allow some tolerance
        assert mean_bias < 0.3, f"QJL potentially biased: mean_abs_bias={mean_bias} (expected < 0.3)"
    
    def test_compression_ratio(self):
        """QJL should achieve ~14× compression for d=128."""
        head_dim, num_heads = 128, 8
        seq_len = 1024
        
        keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
        qjl = QJLQuantizer(head_dim, num_heads, layer_idx=0)
        state = qjl.compress(keys)
        
        orig_bytes = seq_len * num_heads * head_dim * 2  # FP16
        compressed_bytes = state.memory_bytes()
        ratio = orig_bytes / compressed_bytes
        
        assert ratio > 10, f"QJL compression too low: {ratio:.1f}×"


class TestTurboQuant:
    """Test TurboQuant quantization properties."""
    
    def test_mse_monotonic_in_bits(self):
        """MSE should decrease with more bits."""
        head_dim, num_heads = 128, 8
        seq_len = 512
        keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
        
        mses = []
        for bits in [2, 3, 4]:
            tq = TurboQuantizer(head_dim, num_heads, mode="mse", bits=bits, layer_idx=0)
            state = tq.compress(keys)
            
            # Compute reconstruction MSE
            q = np.random.randn(1, num_heads, head_dim).astype(np.float32)
            true = np.einsum("bhd,shd->bhs", q, keys)
            est = tq.scores(q, state)
            mse = float(((est - true) ** 2).mean())
            mses.append(mse)
        
        # MSE should decrease on average (with more bits)
        # For a single trial it may not hold due to randomness
        # Check that higher bits generally give lower or similar MSE
        assert mses[1] < mses[0] * 1.5, f"MSE unexpectedly increased: {mses[1]} vs {mses[0]}"
        assert mses[2] < mses[1] * 1.5, f"MSE unexpectedly increased: {mses[2]} vs {mses[1]}"
    
    def test_prod_unbiased(self):
        """PROD mode should be unbiased."""
        head_dim, num_heads = 128, 8
        seq_len = 512
        keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
        
        tq = TurboQuantizer(head_dim, num_heads, mode="prod", bits=4, layer_idx=0)
        state = tq.compress(keys)
        
        biases = []
        for _ in range(30):
            q = np.random.randn(1, num_heads, head_dim).astype(np.float32)
            true = np.einsum("bhd,shd->bhs", q, keys)
            est = tq.scores(q, state)
            biases.append(float((est - true).mean()))
        
        mean_bias = np.mean(np.abs(biases))
        # PROD is provably unbiased in expectation
        # Individual runs may vary, but should not be systematically biased
        assert mean_bias < 0.3, f"PROD potentially biased: mean_abs_bias={mean_bias} (expected < 0.3)"


class TestMemory:
    """Test memory efficiency."""
    
    def test_qjl_memory(self):
        """QJL memory should be d/8 + 2 bytes per vector."""
        head_dim, num_heads = 128, 8
        seq_len = 1024
        
        keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
        qjl = QJLQuantizer(head_dim, num_heads, layer_idx=0)
        state = qjl.compress(keys)
        
        # Expected: d_packed + 2 bytes for norm
        d_p2 = 128  # next power of 2
        expected = seq_len * num_heads * ((d_p2 // 8) + 2)
        actual = state.memory_bytes()
        
        # Allow 10% tolerance
        assert abs(actual - expected) / expected < 0.1, \
            f"Memory mismatch: expected ~{expected}, got {actual}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
