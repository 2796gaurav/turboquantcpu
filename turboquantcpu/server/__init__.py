"""
OpenAI-compatible inference server.

Provides HTTP API for TurboQuantCPU-optimized LLM inference.
"""

__all__ = ["create_app", "run_server"]


def create_app():
    """Create FastAPI application."""
    raise NotImplementedError("Server coming soon!")


def run_server(host: str = "0.0.0.0", port: int = 8000, **kwargs):
    """Run the inference server."""
    raise NotImplementedError("Server coming soon!")
