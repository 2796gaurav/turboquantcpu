"""
Model loading and conversion utilities.

Supports:
  - HuggingFace Transformers models
  - GGUF format (llama.cpp compatible)
  - Safetensors format
  - PyTorch checkpoint format
"""

__all__ = [
    "load_model",
    "convert_gguf",
    "convert_safetensors",
]


def load_model(model_path: str, **kwargs):
    """Load a model from various formats."""
    raise NotImplementedError("Model loading coming soon!")


def convert_gguf(input_path: str, output_path: str, **kwargs):
    """Convert model to GGUF format."""
    raise NotImplementedError("GGUF conversion coming soon!")


def convert_safetensors(input_path: str, output_path: str, **kwargs):
    """Convert model to Safetensors format."""
    raise NotImplementedError("Safetensors conversion coming soon!")
