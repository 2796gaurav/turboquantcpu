"""
Setup script for TurboQuantCPU.

This script handles:
- Package metadata
- C extension compilation with SIMD detection (optional)
- Platform-specific optimizations
- Graceful fallback to pure Python if C extensions fail
"""

import os
import sys
import platform
import subprocess
import warnings
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

# Package metadata
AUTHOR = "Gaurav Chauhan"
AUTHOR_EMAIL = "2796gaurav@gmail.com"
VERSION = "0.0.11"
DESCRIPTION = "Near-optimal KV cache quantization for CPU LLM inference"


# ── Optional C Extension Build ───────────────────────────────────────

class OptionalBuildExt(build_ext):
    """Build C extension with best available SIMD flags, but allow failure."""

    def run(self):
        """Attempt to build extension, but don't fail if it doesn't work."""
        try:
            super().run()
        except Exception as e:
            warnings.warn(
                f"\n{'='*70}\n"
                f"C extension build failed: {e}\n"
                f"Falling back to pure Python implementation.\n"
                f"Performance will be reduced but functionality is preserved.\n"
                f"To build C extensions manually, see documentation.\n"
                f"{'='*70}\n"
            )
            # Don't fail the build
            pass

    def build_extension(self, ext):
        """Build with platform-specific flags."""
        cc = self.compiler
        arch = platform.machine().lower()
        system = platform.system()
        
        # Base flags (universal)
        base = ["-O2"]
        extra = []
        
        try:
            if system == "Linux" and arch in ("x86_64", "amd64"):
                # Linux x86_64
                if self._has_flag(cc, "-mavx2"):
                    extra = ["-mavx2", "-mfma"]
                elif self._has_flag(cc, "-msse4.1"):
                    extra = ["-msse4.1"]
                    
                # OpenMP on Linux
                if self._has_flag(cc, "-fopenmp"):
                    extra.append("-fopenmp")
                    ext.extra_link_args = ["-fopenmp"]
                    
            elif system == "Darwin":
                # macOS - skip OpenMP, use native arch
                if arch == "arm64":
                    # Apple Silicon - no special flags needed, compiler handles it
                    extra = ["-arch", "arm64"]
                else:
                    # Intel Mac
                    if self._has_flag(cc, "-mavx2"):
                        extra = ["-mavx2", "-mfma"]
                        
            elif system == "Windows":
                # Windows MSVC
                extra = ["/O2"]
                # Don't use GCC-specific flags on Windows
                
        except Exception:
            # Any error in flag detection, use minimal flags
            pass
        
        ext.extra_compile_args = base + extra
        
        try:
            super().build_extension(ext)
        except Exception as e:
            raise RuntimeError(f"C extension build failed: {e}")
    
    def _has_flag(self, compiler, flag):
        """Check if compiler supports a flag."""
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
                for ext in ['.o', '.obj']:
                    if os.path.exists(fname.replace(".c", ext)):
                        os.unlink(fname.replace(".c", ext))
            except Exception:
                pass


# ── Extension definition (optional) ─────────────────────────────────

kernels_ext = Extension(
    "turboquantcpu._tqcpu_kernels",
    sources=["turboquantcpu/_kernels/kernels.c"],
    extra_compile_args=[],  # Set by OptionalBuildExt
    extra_link_args=[],
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
    # C extension is optional - build will succeed even if it fails
    ext_modules=[kernels_ext],
    cmdclass={"build_ext": OptionalBuildExt},
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
