# Contributing

Thank you for your interest in contributing to TurboQuantCPU! We welcome contributions from the community.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or fix

```bash
git clone https://github.com/2796gaurav/turboquantcpu.git
cd turboquantcpu
git checkout -b feature/your-feature-name
```

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Build C extensions
python setup.py build_ext --inplace
```

## Code Style

- Follow PEP 8 for Python code
- Use type hints where applicable
- Add docstrings in Google style
- Keep functions focused and small

## Testing

All contributions must include tests:

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_correctness.py::TestQJL::test_unbiasedness -v

# Run with coverage
pytest tests/ --cov=turboquantcpu --cov-report=html
```

## Adding New Features

### Adding a New Quantizer

1. Create a new file in `turboquantcpu/`
2. Implement the quantizer class
3. Add tests in `tests/`
4. Update documentation

Example structure:

```python
# turboquantcpu/my_quantizer.py

from dataclasses import dataclass
import numpy as np

@dataclass
class MyQuantizerState:
    """State for my quantizer."""
    data: np.ndarray
    
    def memory_bytes(self) -> int:
        return self.data.nbytes

class MyQuantizer:
    """My awesome quantizer."""
    
    def __init__(self, head_dim: int, num_heads: int):
        self.head_dim = head_dim
        self.num_heads = num_heads
    
    def compress(self, keys: np.ndarray) -> MyQuantizerState:
        """Compress keys."""
        # Implementation
        pass
    
    def scores(self, query: np.ndarray, state: MyQuantizerState) -> np.ndarray:
        """Compute scores."""
        # Implementation
        pass
```

### Adding C Kernels

1. Add function to `turboquantcpu/_kernels/kernels.c`
2. Add Python binding in `turboquantcpu/fwht.py`
3. Update CPU detection if needed

## Documentation

- Update README.md for user-facing changes
- Add docstrings to all public APIs
- Update docs/ for significant features
- Run `mkdocs serve` to preview docs locally

## Pull Request Process

1. Update README.md with details of changes if applicable
2. Add tests for new functionality
3. Ensure all tests pass
4. Update documentation
5. Submit PR with clear description

## Commit Messages

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test changes
- `refactor:` Code refactoring
- `perf:` Performance improvements

Example:
```
feat: add PolarQuant algorithm

Implement PolarQuant for outlier-resistant quantization.
- Add polar transformation after FWHT
- Rayleigh quantization for radius
- Uniform quantization for angle
```

## Questions?

- Open an issue for bugs
- Start a discussion for features
- Email: 2796gaurav@gmail.com

## Code of Conduct

Be respectful and constructive in all interactions.
