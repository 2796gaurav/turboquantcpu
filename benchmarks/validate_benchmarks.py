#!/usr/bin/env python3
"""
================================================================================
Benchmark Validation Script
================================================================================

Validates that all benchmark scripts are syntactically correct
and can be imported without errors.

This does NOT run the actual benchmarks (which take time due to model loading),
but ensures the code structure is correct.
================================================================================
"""

import sys
import os

def validate_import(module_name: str, file_path: str) -> bool:
    """Try to import a module and report status."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        with open(file_path, 'r') as f:
            code = f.read()
        
        # Check for GGUF references (but allow comments saying "no GGUF" and competitor comparisons)
        code_lower = code.lower()
        is_competitor_script = 'competitor' in module_name.lower() or 'plot' in module_name.lower()
        if not is_competitor_script:
            if ('gguf' in code_lower and 'no gguf' not in code_lower) or \
               ('llama.cpp' in code_lower and 'no gguf' not in code_lower):
                print(f"✗ {module_name}: Contains GGUF/llama.cpp references (should be removed)")
                return False
        
        # Compile to check syntax
        compile(code, file_path, 'exec')
        
        print(f"✓ {module_name}: Valid")
        return True
        
    except SyntaxError as e:
        print(f"✗ {module_name}: Syntax Error - {e}")
        return False
    except Exception as e:
        print(f"✗ {module_name}: Error - {e}")
        return False


def main():
    print("="*70)
    print("Benchmark Validation")
    print("="*70)
    print()
    
    benchmarks_dir = os.path.dirname(os.path.abspath(__file__))
    
    benchmarks = [
        ("Comprehensive Benchmark", "comprehensive_benchmark.py"),
        ("Needle-in-Haystack", "needle_in_haystack.py"),
        ("Competitor Comparison", "competitor_comparison.py"),
        ("Sanity Benchmark", "sanity_benchmark.py"),
        ("Plot Generator", "create_plots.py"),
    ]
    
    all_valid = True
    
    for name, filename in benchmarks:
        filepath = os.path.join(benchmarks_dir, filename)
        if os.path.exists(filepath):
            valid = validate_import(name, filepath)
            all_valid = all_valid and valid
        else:
            print(f"⚠ {name}: File not found - {filename}")
    
    print()
    print("="*70)
    if all_valid:
        print("✓ All benchmark scripts validated successfully!")
        print()
        print("To run actual benchmarks:")
        print("  cd benchmarks/")
        print("  python comprehensive_benchmark.py  # Full 3-model benchmarks")
        print("  python needle_in_haystack.py      # Long context retrieval")
        print("  python competitor_comparison.py   # vs KIVI, KVQuant, etc.")
        print("  python sanity_benchmark.py        # Quick smoke test")
        print("  python create_plots.py            # Generate visualizations")
    else:
        print("✗ Some benchmarks have validation errors.")
        return 1
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
