"""
Setup script for TurboQuantCPU.

This script handles:
- Package metadata
- C extension compilation with SIMD detection
- Platform-specific optimizations
"""

import os
import sys
import platform
import subprocess
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

# Package metadata
AUTHOR = "Gaurav Chauhan"
AUTHOR_EMAIL = "2796gaurav@gmail.com"
VERSION = "0.3.0"
DESCRIPTION = (
    "TurboQuantCPU — Near-optimal KV cache quantization for CPU LLM inference "
    "with mathematical guarantees. Based on TurboQuant (ICLR 2026)."
)


# ── CPU feature detection ────────────────────────────────────────────

def _has_flag(compiler, flag):
    """Return True if the compiler supports `flag`."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as f:
        f.write("int main(){return 0;}")
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flag])
        return True
    except Exception:
        return False
    finally:
        try:
            os.unlink(fname)
            os.unlink(fname.replace(".c", ".o"))
        except Exception:
            pass


class SmartBuild(build_ext):
    """Build C extension with best available SIMD flags."""

    def build_extension(self, ext):
        cc = self.compiler

        # Base flags
        base = ["-O3", "-ffast-math", "-fPIC", "-std=c11"]
        extra = []

        arch = platform.machine().lower()

        if arch in ("x86_64", "amd64", "i386", "i686"):
            # Test in priority order
            if _has_flag(cc, "-mavx512vnni"):
                extra += ["-mavx512f", "-mavx512bw", "-mavx512vnni",
                          "-mavx512vl", "-mfma"]
            elif _has_flag(cc, "-mavx512f"):
                extra += ["-mavx512f", "-mavx512bw", "-mfma"]
            elif _has_flag(cc, "-mavx2"):
                extra += ["-mavx2", "-mfma"]
            elif _has_flag(cc, "-msse4.1"):
                extra += ["-msse4.1"]
            if _has_flag(cc, "-mpopcnt"):
                extra.append("-mpopcnt")

        elif arch in ("aarch64", "arm64"):
            extra += ["-march=armv8-a+simd"]

        # OpenMP
        omp_flag = "-fopenmp"
        if platform.system() == "Darwin":
            # Homebrew llvm on macOS
            try:
                llvm = subprocess.check_output(
                    ["brew", "--prefix", "libomp"], stderr=subprocess.DEVNULL
                ).decode().strip()
                omp_flag = f"-Xpreprocessor -fopenmp -I{llvm}/include"
                ext.extra_link_args = [f"-L{llvm}/lib", "-lomp"]
            except Exception:
                omp_flag = None  # OpenMP optional on macOS
        if omp_flag and _has_flag(cc, "-fopenmp"):
            extra.append("-fopenmp")
            if not hasattr(ext, "extra_link_args") or ext.extra_link_args is None:
                ext.extra_link_args = ["-fopenmp"]

        ext.extra_compile_args = base + extra
        super().build_extension(ext)


# ── Extension definition ─────────────────────────────────────────────

kernels_ext = Extension(
    "turboquantcpu._tqcpu_kernels",
    sources=["turboquantcpu/_kernels/kernels.c"],
    extra_compile_args=[],   # overridden by SmartBuild
    extra_link_args=["-lm"],
    language="c",
)

# ── Read long description ────────────────────────────────────────────

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# ── Setup ────────────────────────────────────────────────────────────

setup(
    name="turboquantcpu",
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=AUTHOR,
    maintainer_email=AUTHOR_EMAIL,
    url="https://github.com/2796gaurav/turboquantcpu",
    download_url="https://pypi.org/project/turboquantcpu/",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "examples*", "benchmarks*", "docs*"]),
    package_data={
        "turboquantcpu": ["_kernels/*.c"],
    },
    ext_modules=[kernels_ext],
    cmdclass={"build_ext": SmartBuild},
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "torch>=2.0",
    ],
    extras_require={
        "hf": ["transformers>=4.38", "accelerate>=0.26", "safetensors>=0.4"],
        "serve": ["fastapi>=0.110", "uvicorn>=0.27", "httpx>=0.27"],
        "gguf": ["gguf>=0.6.0"],
        "dev": [
            "pytest>=7",
            "hypothesis",
            "matplotlib",
            "tqdm",
            "psutil",
            "mkdocs>=1.5",
            "mkdocs-material>=9.0",
            "mkdocstrings[python]>=0.20",
        ],
        "all": [
            "transformers>=4.38",
            "accelerate>=0.26",
            "safetensors>=0.4",
            "fastapi>=0.110",
            "uvicorn>=0.27",
            "httpx>=0.27",
            "gguf>=0.6.0",
            "pytest>=7",
            "hypothesis",
            "matplotlib",
            "tqdm",
            "psutil",
        ],
    },
    entry_points={
        "console_scripts": [
            "turboquantcpu=turboquantcpu.cli.main:main",
            "tqcpu=turboquantcpu.cli.main:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: C",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
        "Environment :: CPU",
    ],
    keywords=[
        "llm", "quantization", "kv-cache", "inference", "cpu", "simd",
        "avx512", "turboquant", "qjl", "transformers", "memory-optimization",
        "long-context", "edge-computing"
    ],
    project_urls={
        "Homepage": "https://github.com/2796gaurav/turboquantcpu",
        "Documentation": "https://2796gaurav.github.io/turboquantcpu",
        "Bug Tracker": "https://github.com/2796gaurav/turboquantcpu/issues",
        "Source Code": "https://github.com/2796gaurav/turboquantcpu",
        "Paper": "https://arxiv.org/abs/2504.19874",
        "Changelog": "https://github.com/2796gaurav/turboquantcpu/blob/main/CHANGELOG.md",
    },
    zip_safe=False,
)
