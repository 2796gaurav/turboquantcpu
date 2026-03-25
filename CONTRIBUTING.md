# Contributing to TurboQuantCPU

Thank you for your interest in contributing to TurboQuantCPU! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, constructive, and inclusive in all interactions.

## How to Contribute

### Reporting Issues

1. Check if the issue already exists in the [issue tracker](https://github.com/2796gaurav/turboquantcpu/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - System information (OS, Python version, CPU)
   - Minimal code example if applicable

### Suggesting Features

1. Open an issue with the "feature request" label
2. Describe the use case and proposed solution
3. Discuss alternatives considered

### Contributing Code

#### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/2796gaurav/turboquantcpu.git
cd turboquantcpu

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Build C extensions (optional but recommended)
python setup.py build_ext --inplace
```

#### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=turboquantcpu --cov-report=html

# Run specific test file
pytest tests/test_correctness.py

# Run with verbose output
pytest tests/ -v
```

#### Code Style

We follow PEP 8 with some modifications:

```bash
# Format code
black turboquantcpu/ tests/ examples/

# Check style
flake8 turboquantcpu/ tests/

# Type checking
mypy turboquantcpu/
```

#### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

### Pull Request Process

1. **Fork and Branch**: Create a feature branch from `main`
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make Changes**: Write code following our style guidelines

3. **Add Tests**: Ensure new code has test coverage

4. **Update Documentation**: Update docstrings, README, and docs/

5. **Run Tests**: All tests must pass
   ```bash
   pytest tests/
   ```

6. **Commit**: Use clear, descriptive commit messages
   ```bash
   git commit -m "feat: add support for AVX-512 VL"
   ```

7. **Push and PR**: Push to your fork and create a pull request

### Commit Message Format

We use conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Test changes
- `chore:` Build/process changes

Example:
```
feat: add H2O eviction policy support

Implements Heavy-Hitter Oracle eviction for long contexts.
Adds configuration options for heavy_budget and recent_budget.

Closes #123
```

### Code Review

All submissions require review before merging. Maintainers will:
- Review code quality and correctness
- Ensure tests pass
- Verify documentation is updated
- Check for breaking changes

## Development Guidelines

### Project Structure

```
turboquantcpu/
├── turboquantcpu/      # Main package
│   ├── __init__.py     # Public API exports
│   ├── turbo.py        # TurboQuant implementation
│   ├── qjl.py          # QJL 1-bit quantization
│   ├── polar.py        # PolarQuant implementation
│   ├── kv_cache.py     # KV cache management
│   ├── patch.py        # HuggingFace integration
│   └── ...
├── tests/              # Test suite
├── examples/           # Usage examples
├── benchmarks/         # Benchmarking scripts
├── docs/               # Documentation
└── setup.py            # Package configuration
```

### Adding New Quantization Modes

To add a new quantization mode:

1. Create a new file in `turboquantcpu/` (e.g., `new_mode.py`)
2. Implement the quantizer class with required interface:
   - `compress(keys)` → state
   - `decompress(state)` → keys
   - `scores(query, state)` → attention scores
3. Add tests in `tests/test_correctness.py`
4. Update `__init__.py` to export the new class
5. Add example in `examples/02_quantization_modes.py`
6. Update documentation

### C Extension Development

For performance-critical code:

1. Add C/C++ code in `turboquantcpu/_kernels/`
2. Update `setup.py` with new source files
3. Provide NumPy fallback for compatibility
4. Add feature detection in `cpu_features.py`

### Documentation

- Use Google-style docstrings
- Include type hints
- Provide code examples
- Update mkdocs.yml if adding new pages

## Release Process

Maintainers follow this process for releases:

1. Update version in `__init__.py`
2. Update CHANGELOG.md
3. Create git tag: `git tag v0.x.x`
4. Push tag: `git push origin v0.x.x`
5. Build and upload to PyPI

## Questions?

- Join discussions in GitHub Issues
- Check existing documentation
- Reach out to maintainers

Thank you for contributing!
