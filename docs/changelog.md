# Changelog

All notable changes to TurboQuantCPU will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.4] - 2026-03-25

### Fixed
- Removed minify plugin from MkDocs
- Fixed package metadata for PyPI
- All Unicode replaced with ASCII for Windows

## [0.0.2] - 2026-03-25

### Fixed
- Made C extensions optional for cross-platform builds
- Fixed Windows/macOS build issues
- Improved build error handling

## [0.0.1] - 2026-03-25

### Added
- Initial release of TurboQuantCPU
- QJL 1-bit quantization algorithm
- TurboQuant-MSE optimal quantization
- TurboQuant-PROD unbiased estimator
- PolarQuant outlier-resistant quantization
- HuggingFace Transformers integration
- AVX2/AVX-512/NEON SIMD kernels
- H2O sparse attention support
- Comprehensive test suite (19 tests)
- Full documentation with MkDocs
- GitHub Actions CI/CD

### Features
- 4-14× memory compression
- Mathematical guarantees on quality
- CPU-optimized SIMD kernels
- One-line HuggingFace integration
- Real model benchmarks

## [Unreleased]

### Planned
- Intel AMX tile accelerator support
- GGUF format support
- Inference server mode
- GPU kernel implementation
- Additional sparse attention methods
- Multi-GPU support

---

## Version History

### Versioning Policy

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

### Support

- Current stable: 0.3.x
- Python: 3.9+
- PyTorch: 2.0+

---

For detailed commit history, see [GitHub commits](https://github.com/2796gaurav/turboquantcpu/commits/main).
