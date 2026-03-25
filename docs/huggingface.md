# HuggingFace Integration

TurboQuantCPU provides seamless integration with HuggingFace Transformers. Patch any model in one line.

## Basic Integration

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquantcpu import patch_model
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float32,
    device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Patch with TurboQuant (one line!)
cache = patch_model(model, mode="prod", bits=4)

# Generate as usual
inputs = tokenizer("Explain quantum computing:", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0]))

# View memory savings
print(cache.memory_report())
```

## Patch Configuration

```python
from turboquantcpu import PatchConfig

cfg = PatchConfig(
    mode="prod",           # "qjl", "mse", "prod", "polar"
    bits=4,                # 1-8 bits
    max_seq_len=32768,     # Sliding window
    value_mode="int8",     # "fp16", "int8", "int4"
    use_h2o=False,         # Enable H2O sparse attention
    h2o_heavy_budget=128,
    h2o_recent_budget=64,
)

cache = patch_model(model, cfg=cfg)
```

## Supported Models

TurboQuantCPU works with any model that uses standard attention:

| Model Family | Status | Notes |
|--------------|--------|-------|
| **Llama** | ✅ Fully supported | Llama 2, 3, 3.1 |
| **Mistral** | ✅ Fully supported | All versions |
| **Qwen** | ✅ Fully supported | Qwen 2, 2.5 |
| **Phi** | ✅ Fully supported | Phi-2, Phi-3 |
| **Gemma** | ✅ Fully supported | Gemma 2 |
| **Mixtral** | ✅ Fully supported | MoE models |
| **Falcon** | ✅ Fully supported | All versions |
| **Custom** | ✅ Should work | If using standard attention |

## Memory Report

```python
# After generation
report = cache.memory_report()

print(f"Original FP16: {report['original_fp16_MB']:.1f} MB")
print(f"Compressed: {report['compressed_MB']:.1f} MB")
print(f"Compression ratio: {report['compression_ratio']:.1f}×")
```

## Unpatching

To remove TurboQuantCPU and restore original behavior:

```python
from turboquantcpu import unpatch_model

n_hooks = unpatch_model(model)
print(f"Removed {n_hooks} patches")
```

## Complete Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquantcpu import patch_model, unpatch_model

def compare_memory(model_name):
    """Compare memory usage with and without TurboQuantCPU."""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Baseline
    print("=== Baseline (No Compression) ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    
    inputs = tokenizer("Test prompt:", return_tensors="pt")
    
    import psutil, os
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024**2
    
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=50, use_cache=True)
    
    mem_after = process.memory_info().rss / 1024**2
    baseline_mem = mem_after - mem_before
    print(f"Memory increase: {baseline_mem:.1f} MB")
    
    del model
    import gc
    gc.collect()
    
    # With TurboQuantCPU
    print("\n=== With TurboQuantCPU ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    
    cache = patch_model(model, mode="prod", bits=4, verbose=False)
    
    mem_before = process.memory_info().rss / 1024**2
    
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=50, use_cache=True)
    
    mem_after = process.memory_info().rss / 1024**2
    turbo_mem = mem_after - mem_before
    
    print(f"Memory increase: {turbo_mem:.1f} MB")
    print(f"Savings: {baseline_mem / turbo_mem:.1f}×")
    
    # Report from cache
    report = cache.memory_report()
    print(f"\nCache report: {report['compression_ratio']:.1f}× compression")

# Run comparison
compare_memory("Qwen/Qwen2.5-0.5B-Instruct")
```

## Tips

1. **Start with PROD mode**: Best trade-off between compression and quality
2. **Use 4 bits**: Good default for most models
3. **Enable H2O for very long contexts**: Combines compression with token eviction
4. **Monitor memory**: Use `cache.memory_report()` to verify savings
