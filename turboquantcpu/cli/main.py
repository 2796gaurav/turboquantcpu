"""
Command-line interface for TurboQuantCPU.

Usage:
    turboquantcpu benchmark [correctness|memory|speed|all]
    turboquantcpu info
    turboquantcpu serve [options]
    turboquantcpu convert [model_path] [options]
"""

import argparse
import sys
from typing import List, Optional


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run benchmarks."""
    from turboquantcpu import run_full_benchmark, run_correctness_check
    from turboquantcpu import benchmark_memory, benchmark_speed
    
    benchmark_type = args.type or "all"
    
    if benchmark_type == "correctness":
        print("Running correctness checks...")
        results = run_correctness_check(verbose=True)
        return 0 if all(v < 0.1 for k, v in results.items() if "bias" in k) else 1
    
    elif benchmark_type == "memory":
        print("Running memory benchmarks...")
        benchmark_memory(verbose=True)
        return 0
    
    elif benchmark_type == "speed":
        print("Running speed benchmarks...")
        benchmark_speed(verbose=True)
        return 0
    
    elif benchmark_type == "all":
        print("Running full benchmark suite...")
        run_full_benchmark(verbose=True)
        return 0
    
    else:
        print(f"Unknown benchmark type: {benchmark_type}")
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Show system information."""
    from turboquantcpu import print_cpu_capabilities
    from turboquantcpu._kernels import get_version
    
    print_cpu_capabilities()
    print(f"\n  C Extension: {get_version()}")
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert model formats."""
    print("Model conversion - coming soon!")
    print("Planned formats: Safetensors, PyTorch")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Start inference server."""
    print("Server mode - coming soon!")
    print("Will provide OpenAI-compatible API")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="turboquantcpu",
        description="TurboQuantCPU — Ultra-fast CPU KV cache quantization",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Benchmark command
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run benchmarks",
        aliases=["bench"],
    )
    bench_parser.add_argument(
        "type",
        nargs="?",
        choices=["correctness", "memory", "speed", "all"],
        help="Type of benchmark to run",
    )
    bench_parser.add_argument(
        "--head-dim",
        type=int,
        default=128,
        help="Head dimension for benchmarks (default: 128)",
    )
    bench_parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of heads for benchmarks (default: 8)",
    )
    bench_parser.set_defaults(func=cmd_benchmark)
    
    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show system information",
    )
    info_parser.set_defaults(func=cmd_info)
    
    # Convert command
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert model formats",
    )
    convert_parser.add_argument("model_path", help="Path to input model")
    convert_parser.add_argument("-o", "--output", help="Output path")
    convert_parser.add_argument(
        "-f", "--format",
        choices=["safetensors", "pytorch"],
        default="safetensors",
        help="Output format",
    )
    convert_parser.set_defaults(func=cmd_convert)
    
    # Serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start inference server",
    )
    serve_parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to model",
    )
    serve_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    serve_parser.add_argument(
        "-p", "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    serve_parser.add_argument(
        "--mode",
        choices=["prod", "mse", "qjl", "polar"],
        default="prod",
        help="Quantization mode (default: prod)",
    )
    serve_parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="Bits per coordinate (default: 4)",
    )
    serve_parser.set_defaults(func=cmd_serve)
    
    # Parse args
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
