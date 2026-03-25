"""
C kernel extension loader for TurboQuantCPU.

This module handles loading the compiled C extension with SIMD optimizations.
The extension provides:
  - AVX-512/AVX2 optimized FWHT
  - Fast bit packing/unpacking
  - Optimized dot products (f32 · i8)
  - AMX tile operations (when available)
"""

import ctypes
import os
import warnings

_lib = None
_FOUND_C = False


def _load_c_extension():
    """Attempt to load the compiled C extension."""
    global _lib, _FOUND_C
    
    if _FOUND_C:
        return _lib
    
    # Search paths
    module_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = [
        os.path.join(module_dir, "_tqcpu_kernels.so"),
        os.path.join(module_dir, "_tqcpu_kernels.dylib"),
        os.path.join(module_dir, "_tqcpu_kernels.dll"),
        os.path.join(os.path.dirname(module_dir), "_tqcpu_kernels.so"),
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            try:
                lib = ctypes.CDLL(path)
                
                # Verify key functions exist
                lib.tqcpu_fwht_f32
                lib.tqcpu_rfwht_batch_f32
                lib.tqcpu_dot_f32_i8
                lib.tqcpu_pack_signs
                
                _lib = lib
                _FOUND_C = True
                return lib
            except (OSError, AttributeError):
                continue
    
    # Try importing as Python extension module
    try:
        from . import _tqcpu_kernels
        _FOUND_C = True
        return _tqcpu_kernels
    except ImportError:
        pass
    
    return None


def get_lib():
    """Get the C library handle, or None if not available."""
    global _lib
    if _lib is None:
        _lib = _load_c_extension()
    return _lib


def is_available():
    """Check if C extension is available."""
    return get_lib() is not None


def get_version():
    """Get C extension version string."""
    lib = get_lib()
    if lib is None:
        return "not available"
    
    try:
        # Try calling version function
        if hasattr(lib, 'tqcpu_version'):
            lib.tqcpu_version.restype = ctypes.c_char_p
            return lib.tqcpu_version().decode('utf-8')
    except Exception:
        pass
    
    return "available (version unknown)"


# Load on import
_load_c_extension()
