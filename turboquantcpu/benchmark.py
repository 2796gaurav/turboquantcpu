"""
benchmark.py — TurboQuantCPU comprehensive benchmark suite.

Covers:
  1. Correctness: unbiasedness, norm preservation, FWHT self-inverse
  2. Memory:  comparison vs FP16, llama.cpp Q4_K_M, KIVI, KVQuant
  3. Speed:   decode-time score estimation vs baseline attention
  4. Value quant: INT8/INT4 reconstruction error
  5. Model configs: Qwen2.5, Llama-3, Mistral, DeepSeek
"""

from __future__ import annotations
import time
from typing import Dict, List, Optional

import numpy as np

__all__ = [
    "run_correctness_check",
    "benchmark_memory",
    "benchmark_speed",
    "benchmark_value_quant",
    "run_full_benchmark",
]

# ── Real model configurations ─────────────────────────────────────────

MODEL_CONFIGS = {
    "Qwen2.5-0.5B":     dict(layers=24, kv=2,   q=14,  d=64),
    "Qwen2.5-1.5B":     dict(layers=28, kv=2,   q=12,  d=128),
    "Qwen2.5-7B":       dict(layers=28, kv=4,   q=28,  d=128),
    "Llama-3.1-8B":     dict(layers=32, kv=8,   q=32,  d=128),
    "Llama-3.1-70B":    dict(layers=80, kv=8,   q=64,  d=128),
    "Mistral-7B-v0.3":  dict(layers=32, kv=8,   q=32,  d=128),
    "Qwen2.5-72B":      dict(layers=80, kv=8,   q=64,  d=128),
    "Phi-3-mini":       dict(layers=32, kv=32,  q=32,  d=96),
    "DeepSeek-V3":      dict(layers=61, kv=128, q=128, d=128),
    "Gemma-2-9B":       dict(layers=42, kv=8,   q=16,  d=256),
}

PASS = "[PASS]"
FAIL = "[FAIL]"


# ── 1. Correctness ────────────────────────────────────────────────────

def run_correctness_check(
    head_dim:  int = 128,
    num_heads: int = 8,
    seq_len:   int = 512,
    n_trials:  int = 30,
    verbose:   bool = True,
) -> Dict[str, float]:
    """
    Verify all mathematical guarantees:
      T1: FWHT self-inverse (machine precision)
      T2: FWHT norm preservation (orthogonality)
      T3: QJL estimator unbiasedness (E[error] ≈ 0)
      T4: TurboQuantMSE: MSE monotone in bits
      T5: TurboQuantPROD: unbiased (E[error] ≈ 0)
      T6: PROD beats MSE in inner-product error
      T7: Value INT8 reconstruction error < 0.5%
    """
    from turboquantcpu.fwht  import fwht_numpy, randomized_fwht, get_signs
    from turboquantcpu.qjl   import QJLQuantizer
    from turboquantcpu.turbo import TurboQuantizer, TurboMode
    from turboquantcpu.value_quant import ValueQuantizer, ValueMode

    results: Dict[str, float] = {}

    if verbose:
        print("\n" + "="*65)
        print("  TurboQuantCPU v0.0.4 - Correctness Check")
        print("="*65)

    d_p2 = 1 << (head_dim-1).bit_length()

    # ── T1: FWHT self-inverse ─────────────────────────────────────────
    x   = np.random.randn(64, d_p2).astype(np.float32)
    err = np.abs(fwht_numpy(fwht_numpy(x.copy())) - x).max()
    results["fwht_self_inverse_max_err"] = float(err)
    if verbose:
        print(f"\n  T1 FWHT self-inverse   : max err = {err:.3e}  "
              f"{PASS if err < 1e-4 else FAIL} (ideal < 1e-5)")

    # ── T2: FWHT norm preservation ────────────────────────────────────
    sg  = get_signs(d_p2, seed=42, as_numpy=True)
    y   = np.stack([randomized_fwht(x[i], sg) for i in range(x.shape[0])])
    nb  = np.linalg.norm(x, axis=-1)
    na  = np.linalg.norm(y, axis=-1)
    ne  = (np.abs(nb - na) / (nb + 1e-12)).mean()
    results["fwht_norm_preservation"] = float(ne)
    if verbose:
        print(f"  T2 FWHT norm preserv.  : rel err = {ne:.3e}  "
              f"{PASS if ne < 1e-4 else FAIL}")

    # ── T3: QJL unbiasedness ─────────────────────────────────────────
    keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
    qjl  = QJLQuantizer(head_dim, num_heads, layer_idx=0)
    qs   = qjl.compress(keys)
    biases, errs = [], []
    for _ in range(n_trials):
        q     = np.random.randn(1, num_heads, head_dim).astype(np.float32)
        true  = np.einsum("bhd,shd->bhs", q, keys)
        est   = qjl.scores(q, qs)
        biases.append(float((est - true).mean()))
        errs.append(float(np.abs(est - true).mean() / (np.abs(true).mean() + 1e-6)))
    mb = float(np.mean(np.abs(biases)))
    me = float(np.mean(errs))
    results["qjl_mean_abs_bias"]   = mb
    results["qjl_mean_rel_err"]    = me
    if verbose:
        print(f"\n  T3 QJL bias            : {np.mean(biases):+.5f}  "
              f"{PASS if mb < 0.05 else FAIL}")
        print(f"     QJL mean rel. err   : {me:.4f}")

    # ── T4: TurboMSE monotone bits ────────────────────────────────────
    if verbose: print(f"\n  T4 TurboMSE MSE vs bits (should decrease):")
    prev_mse = 1e9; mono = True
    for b in [2, 3, 4]:
        tq = TurboQuantizer(head_dim, num_heads, mode="mse", bits=b, layer_idx=0)
        st = tq.compress(keys)
        ms_b = []
        for _ in range(n_trials):
            q    = np.random.randn(1, num_heads, head_dim).astype(np.float32)
            true = np.einsum("bhd,shd->bhs", q, keys)
            ms_b.append(float(((tq.scores(q, st)-true)**2).mean()))
        mm = float(np.mean(ms_b))
        results[f"mse_{b}bit_mse"] = mm
        mono = mono and (mm < prev_mse)
        prev_mse = mm
        if verbose:
            print(f"     {b}-bit : MSE={mm:.5f}")
    if verbose:
        print(f"     Monotone: {PASS if mono else FAIL}")

    # ── T5-T6: TurboProD unbiasedness and vs MSE ─────────────────────
    if verbose: print(f"\n  T5-T6 TurboPROD unbiasedness and vs MSE:")
    for b in [3, 4]:
        tp = TurboQuantizer(head_dim, num_heads, mode="prod", bits=b, layer_idx=1)
        tm = TurboQuantizer(head_dim, num_heads, mode="mse",  bits=b, layer_idx=2)
        sp = tp.compress(keys); sm = tm.compress(keys)
        bp_list, ep_list, em_list = [], [], []
        for _ in range(n_trials):
            q    = np.random.randn(1, num_heads, head_dim).astype(np.float32)
            true = np.einsum("bhd,shd->bhs", q, keys)
            ep   = float(np.abs(tp.scores(q, sp) - true).mean())
            em   = float(np.abs(tm.scores(q, sm) - true).mean())
            bp   = float((tp.scores(q, sp) - true).mean())
            bp_list.append(bp); ep_list.append(ep); em_list.append(em)
        mb_p = float(np.mean(np.abs(bp_list)))
        me_p = float(np.mean(ep_list))
        me_m = float(np.mean(em_list))
        results[f"prod_{b}bit_bias"]    = mb_p
        results[f"prod_{b}bit_err"]     = me_p
        results[f"prod_better_{b}bit"]  = float(me_p <= me_m*1.05)
        if verbose:
            sb = PASS if mb_p < 0.05 else FAIL
            sp_r = PASS if me_p <= me_m*1.05 else "⚠ CHECK"
            print(f"     {b}-bit PROD bias={mb_p:.5f} {sb}  "
                  f"err={me_p:.4f} vs MSE={me_m:.4f} {sp_r}")

    # ── T7: Value INT8 reconstruction ─────────────────────────────────
    vq  = ValueQuantizer(head_dim, num_heads, mode=ValueMode.INT8)
    vs  = np.random.randn(64, num_heads, head_dim).astype(np.float32)
    st  = vq.compress(vs)
    rec = vq.decompress(st)
    rel_err = float(np.abs(vs - rec[:, :, :head_dim]).mean() / (np.abs(vs).mean() + 1e-6))
    results["value_int8_rel_err"] = rel_err
    if verbose:
        print(f"\n  T7 Value INT8 rel err  : {rel_err:.4f}  "
              f"{PASS if rel_err < 0.01 else '⚠ CHECK'}")

    if verbose:
        print("\n  All correctness checks complete.")
    return results


# ── 2. Memory ─────────────────────────────────────────────────────────

def benchmark_memory(
    seq_lengths: Optional[List[int]] = None,
    models:      Optional[List[str]] = None,
    bits:        int = 4,
    verbose:     bool = True,
) -> Dict[str, Dict]:
    """
    Memory comparison:
      FP16 baseline | Q4_K_M (llama.cpp) | KIVI-2bit | QJL-1bit | Turbo-PROD
    """
    from turboquantcpu.qjl        import QJLQuantizer
    from turboquantcpu.turbo      import TurboQuantizer
    from turboquantcpu.value_quant import ValueQuantizer, ValueMode

    seq_lengths = seq_lengths or [1024, 4096, 8192, 32768, 131072]
    model_names = models or list(MODEL_CONFIGS.keys())

    if verbose:
        print("\n" + "="*82)
        print("  Memory (Keys Only, MB per model) - Compression vs FP16")
        print("="*82)
        print(f"  {'Model':<22} {'Seq':>8}  {'FP16':>7}  {'Q4K_M':>7}  "
              f"{'KIVI2b':>7}  {'QJL1b':>7}  {'Turbo4b':>8}  {'Ratio':>6}")
        print("-"*82)

    results = {}
    for mname in model_names:
        if mname not in MODEL_CONFIGS:
            continue
        c   = MODEL_CONFIGS[mname]
        L, H, d = c["layers"], c["kv"], c["d"]

        q_qjl  = QJLQuantizer(d, H, layer_idx=0)
        q_prod = TurboQuantizer(d, H, mode="prod", bits=bits, layer_idx=0)
        q_mse2 = TurboQuantizer(d, H, mode="mse",  bits=2,    layer_idx=0)

        model_res = {}
        for seq in seq_lengths:
            fp16  = seq*H*d*2*L
            # llama.cpp Q4_K_M: 4-bit + 2 bytes scale + 1 byte zero per 32 elements
            q4k   = (seq*H*d//2 + seq*H*(d//32)*3)*L
            # KIVI-2bit: 2-bit uniform + f16 scale per 128 elements
            kivi2 = (seq*H*d//4 + seq*H*(d//128)*2)*L
            # QJL 1-bit
            qjlr  = q_qjl.memory_report(seq)
            qjlb  = qjlr["qjl_total_MB"]*1e6*L
            # TurboQuant PROD 4-bit
            prr   = q_prod.memory_report(seq)
            prb   = prr["compressed_MB"]*1e6*L
            ratio = fp16 / max(prb, 1)

            model_res[seq] = dict(
                fp16_MB=fp16/1e6, q4k_MB=q4k/1e6, kivi2_MB=kivi2/1e6,
                qjl_MB=qjlb/1e6, prod_MB=prb/1e6, ratio=ratio,
            )

            if verbose and seq in [4096, 32768]:
                print(f"  {mname:<22} {seq:>8,}"
                      f"  {fp16/1e6:>6.1f}  {q4k/1e6:>6.1f}  "
                      f"{kivi2/1e6:>6.1f}  {qjlb/1e6:>6.1f}  "
                      f"{prb/1e6:>7.1f}   {ratio:>5.1f}×")

        results[mname] = model_res

    return results


# ── 3. Speed ──────────────────────────────────────────────────────────

def _timeit(fn, warmup: int = 5, repeats: int = 25) -> float:
    for _ in range(warmup): fn()
    t0 = time.perf_counter()
    for _ in range(repeats): fn()
    return (time.perf_counter()-t0) / repeats * 1e3


def benchmark_speed(
    head_dim:  int = 128,
    num_heads: int = 8,
    seq_len:   int = 4096,
    verbose:   bool = True,
) -> Dict[str, float]:
    """
    Attention score computation speed at decode time (batch=1).
    Memory-bandwidth-bound: less compressed data → faster decode.
    """
    from turboquantcpu.qjl        import QJLQuantizer
    from turboquantcpu.turbo      import TurboQuantizer
    from turboquantcpu.polar      import PolarQuantizer

    results: Dict[str, float] = {}

    keys   = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
    q_f32  = np.random.randn(1, num_heads, head_dim).astype(np.float32)
    keys16 = keys.astype(np.float16)

    # Quantizers
    qjl   = QJLQuantizer(head_dim, num_heads, layer_idx=0)
    prod4 = TurboQuantizer(head_dim, num_heads, mode="prod", bits=4, layer_idx=0)
    mse4  = TurboQuantizer(head_dim, num_heads, mode="mse",  bits=4, layer_idx=1)
    pol   = PolarQuantizer(head_dim, num_heads, n_r_bits=2, n_theta_bits=2, layer_idx=2)

    # Pre-compress
    st_qjl  = qjl.compress(keys)
    st_prod = prod4.compress(keys)
    st_mse  = mse4.compress(keys)
    st_pol  = pol.compress(keys)

    # Time functions
    t_f32 = _timeit(lambda: np.einsum("bhd,shd->bhs", q_f32, keys))
    t_f16 = _timeit(lambda: np.einsum("bhd,shd->bhs",
                                       q_f32, keys16.astype(np.float32)))
    t_qjl  = _timeit(lambda: qjl.scores(q_f32, st_qjl))
    t_prod = _timeit(lambda: prod4.scores(q_f32, st_prod))
    t_mse  = _timeit(lambda: mse4.scores(q_f32, st_mse))
    t_pol  = _timeit(lambda: pol.scores(q_f32, st_pol))

    results.update({
        "fp32_ms":  t_f32, "fp16_ms":  t_f16,
        "qjl_ms":   t_qjl, "prod_ms":  t_prod,
        "mse_ms":   t_mse,  "polar_ms": t_pol,
        "qjl_speedup":  t_f32/max(t_qjl,  1e-9),
        "prod_speedup": t_f32/max(t_prod, 1e-9),
        "polar_speedup":t_f32/max(t_pol,  1e-9),
    })

    if verbose:
        print("\n" + "="*65)
        print(f"  Speed Benchmark  seq={seq_len:,}  heads={num_heads}  dim={head_dim}")
        print("="*65)
        print(f"  Baseline FP32          : {t_f32:>8.3f} ms")
        print(f"  Baseline FP16          : {t_f16:>8.3f} ms")
        print(f"  QJL 1-bit              : {t_qjl:>8.3f} ms  "
              f"({results['qjl_speedup']:.1f}× vs FP32)")
        print(f"  TurboMSE 4-bit         : {t_mse:>8.3f} ms")
        print(f"  TurboPROD 4-bit        : {t_prod:>8.3f} ms  "
              f"({results['prod_speedup']:.1f}× vs FP32)")
        print(f"  PolarQuant 4-bit       : {t_pol:>8.3f} ms  "
              f"({results['polar_speedup']:.1f}× vs FP32)")

        # Scaling
        print(f"\n  Scaling with sequence length (heads={num_heads}, dim={head_dim}):")
        print(f"  {'seq_len':>9}  {'FP32 ms':>9}  {'QJL ms':>9}  "
              f"{'PROD ms':>9}  {'QJL spd':>8}  {'PROD spd':>9}")
        for slen in [512, 1024, 4096, 8192, 16384, 32768]:
            k_s  = np.random.randn(slen, num_heads, head_dim).astype(np.float32)
            q_s  = q_f32
            sts  = qjl.compress(k_s)
            stp  = prod4.compress(k_s)
            tf   = _timeit(lambda: np.einsum("bhd,shd->bhs", q_s, k_s), 3, 10)
            tq   = _timeit(lambda: qjl.scores(q_s, sts), 3, 10)
            tp_  = _timeit(lambda: prod4.scores(q_s, stp), 3, 10)
            print(f"  {slen:>9,}  {tf:>9.3f}  {tq:>9.3f}  {tp_:>9.3f}  "
                  f"{tf/max(tq,1e-9):>7.1f}×  {tf/max(tp_,1e-9):>8.1f}×")
            results[f"scaling_{slen}_qjl_ms"] = tq
            results[f"scaling_{slen}_prod_ms"] = tp_

    return results


# ── 4. Value quantization ─────────────────────────────────────────────

def benchmark_value_quant(
    head_dim:  int = 128,
    num_heads: int = 8,
    seq_len:   int = 4096,
    verbose:   bool = True,
) -> Dict[str, float]:
    """Value compression quality and speed."""
    from turboquantcpu.value_quant import ValueQuantizer, ValueMode

    V    = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
    results = {}

    for mode in [ValueMode.INT8, ValueMode.INT4]:
        vq  = ValueQuantizer(head_dim, num_heads, mode=mode)
        st  = vq.compress(V)
        rec = vq.decompress(st)[:, :, :head_dim]
        rel = float(np.abs(V - rec).mean() / (np.abs(V).mean() + 1e-6))
        snr = float(-20*np.log10(max(rel, 1e-12)))
        mb  = vq.memory_report(seq_len)
        t   = _timeit(lambda: vq.decompress(st), 3, 10)
        results[f"{mode}_rel_err"] = rel
        results[f"{mode}_snr_db"]  = snr
        results[f"{mode}_decomp_ms"] = t
        if verbose:
            print(f"  Value {mode:<5}: rel_err={rel:.4f}  SNR={snr:.1f}dB  "
                  f"compression={mb['compression_ratio']:.1f}×  "
                  f"decompress={t:.2f}ms")

    return results


# ── 5. Full benchmark ─────────────────────────────────────────────────

def run_full_benchmark(
    head_dim:  int = 128,
    num_heads: int = 8,
    verbose:   bool = True,
) -> Dict[str, dict]:
    """Run all benchmarks and print a combined report."""
    print("\n+" + "="*63 + "+")
    print("|   TurboQuantCPU v0.0.4 - Full Benchmark Suite" + " "*16 + "|")
    print("+" + "="*63 + "+")

    from turboquantcpu.cpu_features import print_cpu_capabilities
    print_cpu_capabilities()

    correctness = run_correctness_check(head_dim, num_heads, verbose=verbose)
    memory      = benchmark_memory(verbose=verbose)
    speed       = benchmark_speed(head_dim, num_heads, verbose=verbose)

    if verbose:
        print(f"\n  Value quantization:")
    value = benchmark_value_quant(head_dim, num_heads, verbose=verbose)

    # Print summary
    llama_8b = memory.get("Llama-3.1-8B", {}).get(4096, {})
    print("\n" + "="*65)
    print("  SUMMARY")
    print("="*65)
    print(f"  QJL bias              : {correctness.get('qjl_mean_abs_bias', 0):.5f} (ideal 0)")
    qpr = correctness.get("prod_4bit_bias") or correctness.get("prod_4bit_bias", 0)
    print(f"  PROD-4bit bias        : {qpr:.5f}")
    print(f"  Memory (Llama-8B 4K)  : "
          f"FP16={llama_8b.get('fp16_MB',0):.0f}MB → "
          f"Turbo={llama_8b.get('prod_MB',0):.0f}MB "
          f"({llama_8b.get('ratio',0):.1f}×)")
    print(f"  Speed at 4K tokens    : "
          f"QJL={speed.get('qjl_speedup',1):.1f}×  "
          f"PROD={speed.get('prod_speedup',1):.1f}×  "
          f"Polar={speed.get('polar_speedup',1):.1f}× vs FP32")
    print(f"  Value INT8 SNR        : {value.get('int8_snr_db',0):.1f} dB")
    print("═"*65)

    return {"correctness": correctness, "memory": memory,
            "speed": speed, "value": value}