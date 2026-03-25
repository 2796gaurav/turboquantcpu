"""
cpu_features.py — Runtime CPU capability detection for TurboQuantCPU.

Detects: SSE4.2, AVX, AVX2, FMA, AVX-512F/BW/VNNI, AMX-INT8/BF16,
         ARM NEON, SVE/SVE2, Apple Silicon, OpenMP thread count.

Also probes the compiled C extension to confirm actual runtime SIMD level.
"""

from __future__ import annotations
import ctypes
import os, sys, platform, functools, subprocess
from dataclasses import dataclass, field
from typing import Optional

__all__ = ["CPUInfo", "cpu_info", "print_cpu_capabilities"]


@dataclass
class CPUInfo:
    """Complete snapshot of CPU capabilities relevant to TurboQuantCPU."""
    # Identification
    brand:          str  = "unknown"
    arch:           str  = "unknown"    # x86_64 | aarch64 | arm
    vendor:         str  = "unknown"    # intel | amd | apple | arm
    num_cores:      int  = 1
    num_threads:    int  = 1
    cache_l3_kb:    int  = 0

    # x86 SIMD
    has_sse42:       bool = False
    has_avx:         bool = False
    has_avx2:        bool = False
    has_fma:         bool = False
    has_avx512f:     bool = False
    has_avx512bw:    bool = False
    has_avx512vnni:  bool = False
    has_avx512vbmi:  bool = False
    has_amx_int8:    bool = False
    has_amx_bf16:    bool = False
    has_amx_fp16:    bool = False

    # ARM SIMD
    has_neon:        bool = False
    has_sve:         bool = False
    has_sve2:        bool = False
    has_dotprod:     bool = False  # ARMv8.2 i8 dot product

    # Runtime info from C extension
    c_kernel_simd:   str  = "unknown"
    openmp_threads:  int  = 1

    # Memory bandwidth estimate (GB/s) — useful for throughput estimates
    mem_bandwidth_gb: float = 0.0

    flags: set = field(default_factory=set)

    @property
    def best_simd(self) -> str:
        if self.has_amx_int8:     return "AMX-INT8"
        if self.has_avx512vnni:   return "AVX512-VNNI"
        if self.has_avx512f:      return "AVX-512F"
        if self.has_avx2:         return "AVX2+FMA" if self.has_fma else "AVX2"
        if self.has_avx:          return "AVX"
        if self.has_sve2:         return "SVE2"
        if self.has_sve:          return "SVE"
        if self.has_neon:         return "NEON"
        if self.has_dotprod:      return "NEON+DotProd"
        return "scalar"

    @property
    def simd_width_f32(self) -> int:
        """Number of f32 elements processed per SIMD instruction."""
        if self.has_avx512f:  return 16
        if self.has_avx2:     return 8
        if self.has_avx:      return 8
        if self.has_sse42:    return 4
        if self.has_neon:     return 4
        return 1

    @property
    def fwht_speedup_estimate(self) -> float:
        """Rough FWHT speedup vs scalar baseline."""
        return {16: 12.0, 8: 6.0, 4: 3.0, 1: 1.0}[self.simd_width_f32]

    @property
    def torch_compile_ready(self) -> bool:
        return self.has_avx2 or self.has_neon


def _parse_proc_cpuinfo() -> CPUInfo:
    info = CPUInfo()
    info.arch = platform.machine().lower()

    try:
        text = open("/proc/cpuinfo").read()
    except OSError:
        return info

    flags_set: set = set()
    for line in text.splitlines():
        lw = line.lower()
        if lw.startswith("model name") and info.brand == "unknown":
            info.brand = line.split(":",1)[1].strip()
        if lw.startswith("vendor_id") and info.vendor == "unknown":
            v = line.split(":",1)[1].strip().lower()
            if "genuineintel" in v:   info.vendor = "intel"
            elif "authenticamd" in v: info.vendor = "amd"
        if lw.startswith("flags") or lw.startswith("features"):
            flags_set.update(lw.split(":")[1].split())
        if lw.startswith("cpu cores"):
            try: info.num_cores   = int(line.split(":")[1])
            except ValueError: pass
        if lw.startswith("siblings"):
            try: info.num_threads = int(line.split(":")[1])
            except ValueError: pass
        if "cache size" in lw:
            try:
                kb = int(line.split(":")[1].split()[0])
                info.cache_l3_kb = max(info.cache_l3_kb, kb)
            except (ValueError, IndexError): pass

    info.flags = flags_set
    info.has_sse42       = "sse4_2"      in flags_set
    info.has_avx         = "avx"         in flags_set
    info.has_avx2        = "avx2"        in flags_set
    info.has_fma         = "fma"         in flags_set
    info.has_avx512f     = "avx512f"     in flags_set
    info.has_avx512bw    = "avx512bw"    in flags_set
    info.has_avx512vnni  = "avx512_vnni" in flags_set
    info.has_avx512vbmi  = "avx512vbmi"  in flags_set
    info.has_amx_int8    = "amx_int8"    in flags_set
    info.has_amx_bf16    = "amx_bf16"    in flags_set
    info.has_amx_fp16    = "amx_fp16"    in flags_set
    info.has_neon        = "neon" in flags_set or "asimd" in flags_set
    info.has_sve         = "sve"         in flags_set
    info.has_sve2        = "sve2"        in flags_set
    info.has_dotprod     = "asimddp"     in flags_set

    return info


def _parse_sysctl_macos() -> CPUInfo:
    info = CPUInfo()
    info.arch = platform.machine().lower()

    def sysctl(k):
        try:
            return subprocess.check_output(
                ["sysctl","-n",k], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return ""

    brand = sysctl("machdep.cpu.brand_string") or sysctl("hw.model")
    info.brand = brand

    if "apple" in brand.lower() or info.arch == "arm64":
        info.vendor    = "apple"
        info.has_neon  = True
        info.has_dotprod = True
        # Apple Silicon has AMX coprocessor (used by Accelerate.framework)
        info.has_amx_int8 = True  # M1+
    else:
        feat = (sysctl("machdep.cpu.features") + " " +
                sysctl("machdep.cpu.leaf7_features")).lower()
        info.vendor        = "intel"
        info.has_sse42     = "sse4.2"   in feat
        info.has_avx       = "avx1.0"  in feat or "avx"   in feat
        info.has_avx2      = "avx2"     in feat
        info.has_fma       = "fma"      in feat
        info.has_avx512f   = "avx512f"  in feat
        info.has_avx512bw  = "avx512bw" in feat
        info.has_avx512vnni= "avx512vnni" in feat

    try:
        info.num_threads = int(sysctl("hw.logicalcpu"))
        info.num_cores   = int(sysctl("hw.physicalcpu"))
    except (ValueError, TypeError):
        pass

    return info


def _probe_c_extension(info: CPUInfo) -> CPUInfo:
    """Load C extension (if compiled) and query its runtime SIMD level."""
    try:
        from ._kernels import get_lib
        lib = get_lib()
        if lib is not None and hasattr(lib, 'tqcpu_version'):
            lib.tqcpu_version.restype = ctypes.c_char_p
            ver = lib.tqcpu_version().decode('utf-8')
            info.c_kernel_simd = ver
            # Parse OpenMP thread count from version string
            if "OpenMP=" in ver:
                try:
                    info.openmp_threads = int(ver.split("OpenMP=")[1].split()[0])
                except Exception:
                    pass
            return info
    except Exception:
        pass
    
    # Fallback to old method
    try:
        from ._tqcpu_kernels import tqcpu_version  # type: ignore
        ver = tqcpu_version().decode() if isinstance(tqcpu_version(), bytes) else str(tqcpu_version())
        info.c_kernel_simd = ver
        # Parse OpenMP thread count from version string
        if "OpenMP=" in ver:
            try:
                info.openmp_threads = int(ver.split("OpenMP=")[1].split()[0])
            except Exception:
                pass
    except ImportError:
        info.c_kernel_simd = "C extension not compiled"

    # Also check OpenMP via environment
    if info.openmp_threads == 1:
        try:
            import multiprocessing
            info.openmp_threads = multiprocessing.cpu_count()
        except Exception:
            pass

    return info


@functools.lru_cache(maxsize=1)
def cpu_info() -> CPUInfo:
    """Return a cached CPUInfo for the current host."""
    sys_ = platform.system()
    if sys_ == "Linux":
        info = _parse_proc_cpuinfo()
    elif sys_ == "Darwin":
        info = _parse_sysctl_macos()
    else:
        info = CPUInfo()
        info.arch = platform.machine().lower()

    info = _probe_c_extension(info)
    return info


_BAR = "=" * 52

def print_cpu_capabilities() -> None:
    ci = cpu_info()
    print(_BAR)
    print("  TurboQuantCPU v0.0.6 - Host CPU Capabilities")
    print(_BAR)
    print(f"  CPU     : {ci.brand}")
    print(f"  Arch    : {ci.arch}  ({ci.vendor})")
    print(f"  Cores   : {ci.num_cores} phys / {ci.num_threads} logical")
    print(f"  SIMD    : {ci.best_simd}  ({ci.simd_width_f32}×f32/instr)")
    print(f"  OpenMP  : {ci.openmp_threads} threads")
    print(f"  Kernel  : {ci.c_kernel_simd}")
    print()
    feats = [
        ("AVX2",           ci.has_avx2),
        ("FMA",            ci.has_fma),
        ("AVX-512F",       ci.has_avx512f),
        ("AVX-512BW",      ci.has_avx512bw),
        ("AVX-512VNNI",    ci.has_avx512vnni),
        ("AVX-512VBMI",    ci.has_avx512vbmi),
        ("AMX-INT8",       ci.has_amx_int8),
        ("AMX-BF16",       ci.has_amx_bf16),
        ("ARM NEON",       ci.has_neon),
        ("ARM DotProd",    ci.has_dotprod),
        ("ARM SVE",        ci.has_sve),
        ("ARM SVE2",       ci.has_sve2),
    ]
    for name, has in feats:
        print(f"  {'[Y]' if has else '[N]'}  {name}")
    print()
    print(f"  FWHT speedup estimate: ~{ci.fwht_speedup_estimate:.0f}× vs scalar")
    print(_BAR)