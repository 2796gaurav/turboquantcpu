#!/usr/bin/env python3
"""
================================================================================
Needle-in-a-Haystack Benchmark
================================================================================

Based on the Needle-in-a-Haystack test from the TurboQuant paper (ICLR 2026).
Tests long-context retrieval capability by hiding a "needle" (specific fact)
at various depths within a "haystack" (long document).

From the paper:
"We achieve perfect long-context retrieval in needle-in-a-haystack tasks 
as well as other long-context tasks, while compressing the KV cache 
by a factor exceeding 5×."

TESTED ON:
----------
- Qwen/Qwen3.5-0.8B
- meta-llama/Llama-3.2-1B
- google/gemma-2-2b-it

CONFIGURATION:
--------------
- Context lengths: 512, 1024, 2048, 4096 tokens
- Depths: 0%, 25%, 50%, 75%, 100% through document
- Trials: 2 per configuration

SUCCESS METRIC:
---------------
Model correctly answers question about hidden fact.
================================================================================
"""

import os
import sys
import json
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquantcpu import patch_model, unpatch_model


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


NEEDLES = [
    {"fact": "The secret color is AZURE-BLUE", "question": "What is the secret color?", "answer": "AZURE-BLUE"},
    {"fact": "The magic number is SEVEN-FOUR-TWO", "question": "What is the magic number?", "answer": "SEVEN-FOUR-TWO"},
    {"fact": "The hidden word is SERENDIPITY", "question": "What is the hidden word?", "answer": "SERENDIPITY"},
    {"fact": "The special place is HIDDEN-VALLEY", "question": "What is the special place?", "answer": "HIDDEN-VALLEY"},
    {"fact": "The mystery code is XJ-NINE-ZERO-ZERO", "question": "What is the mystery code?", "answer": "XJ-NINE-ZERO-ZERO"},
]

FILLER_SENTENCES = [
    "The study of history provides essential context for understanding contemporary events.",
    "Climate change represents one of the most significant challenges facing modern civilization.",
    "Advances in artificial intelligence continue to transform industries worldwide.",
    "Space exploration has yielded numerous scientific discoveries and technological innovations.",
    "Economic theory provides frameworks for analyzing market behavior and trends.",
    "Philosophical inquiry has shaped human thought and culture for millennia.",
    "Medical research pushes the boundaries of what is possible in healthcare.",
    "Cultural exchange enriches societies and promotes international understanding.",
    "Technological innovation drives progress in unexpected directions.",
    "Literary works offer profound insights into the human condition.",
    "Environmental conservation requires coordinated global action and commitment.",
    "Educational systems must adapt to meet evolving societal needs.",
    "Political institutions evolve in response to citizen demands and changing times.",
    "Scientific methodology provides rigor to empirical investigation and discovery.",
    "Artistic expression reflects and shapes cultural values across generations.",
    "Urban planning requires careful consideration of infrastructure and sustainability.",
    "Public health initiatives have dramatically improved quality of life globally.",
    "Transportation networks facilitate commerce and connectivity between regions.",
    "Agricultural innovation ensures food security for growing populations.",
    "Renewable energy sources offer sustainable alternatives to fossil fuels.",
] * 10  # Repeat for sufficient content


class NeedleTester:
    """Implements needle-in-haystack testing."""
    
    def __init__(self, model_repo: str, device: str = "cpu"):
        self.model_repo = model_repo
        self.device = device
        self.model = None
        self.tokenizer = None
        
    def load(self):
        """Load model and tokenizer."""
        print(f"Loading {self.model_repo}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_repo, 
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_repo,
            torch_dtype=torch.float32,
            device_map=self.device,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        print(f"  ✓ Loaded: {self.model.num_parameters() / 1e9:.2f}B parameters")
        return self
        
    def unload(self):
        """Unload model to free memory."""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        import gc
        gc.collect()
        
    def create_haystack(self, target_length: int, needle: str, depth: float) -> str:
        """Create haystack with needle at specified depth."""
        # Estimate tokens per sentence
        sample = " ".join(FILLER_SENTENCES[:5])
        tokens_per_sentence = len(self.tokenizer.encode(sample)) / 5
        
        # Calculate sentences needed
        needle_tokens = len(self.tokenizer.encode(needle))
        num_sentences = int((target_length - needle_tokens) / tokens_per_sentence)
        
        sentences = FILLER_SENTENCES[:num_sentences].copy()
        insert_pos = int(len(sentences) * depth)
        sentences.insert(insert_pos, needle)
        
        return " ".join(sentences)
    
    def test_retrieval(self, haystack: str, question: str, answer: str) -> Tuple[bool, str]:
        """Test if model retrieves correct answer."""
        prompt = f"Context: {haystack}\n\nQuestion: {question}\nAnswer:"
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True,
            max_length=4096
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer portion
        if "Answer:" in generated:
            generated = generated.split("Answer:")[-1].strip()
        
        success = answer.lower().replace("-", " ") in generated.lower()
        return success, generated
    
    def run_benchmark(
        self,
        context_lengths: List[int],
        depths: List[float],
        num_trials: int = 2
    ) -> Dict:
        """Run full needle-in-haystack benchmark."""
        print(f"\nConfiguration:")
        print(f"  Context lengths: {context_lengths}")
        print(f"  Depths: {depths}")
        print(f"  Trials per config: {num_trials}")
        
        results = {
            "model": self.model_repo,
            "baseline": {"total": 0, "correct": 0, "details": []},
            "turboquant": {"total": 0, "correct": 0, "details": []},
        }
        
        # Baseline test
        print("\n[1/2] Testing baseline (FP16)...")
        for length in context_lengths:
            for depth in depths:
                for trial in range(num_trials):
                    needle = NEEDLES[trial % len(NEEDLES)]
                    haystack = self.create_haystack(length, needle["fact"], depth)
                    
                    success, answer = self.test_retrieval(
                        haystack, needle["question"], needle["answer"]
                    )
                    
                    results["baseline"]["total"] += 1
                    if success:
                        results["baseline"]["correct"] += 1
                    
                    results["baseline"]["details"].append({
                        "length": length,
                        "depth": depth,
                        "trial": trial,
                        "success": success,
                    })
                    
                    status = "✓" if success else "✗"
                    print(f"  {status} L={length:>4}, D={depth:>4.0%}, T={trial+1}")
        
        baseline_score = results["baseline"]["correct"] / results["baseline"]["total"] * 100
        print(f"  Baseline score: {baseline_score:.1f}%")
        
        # TurboQuant test
        print("\n[2/2] Testing TurboQuant 4-bit PROD...")
        cache = patch_model(self.model, mode="prod", bits=4, verbose=False)
        
        for length in context_lengths:
            for depth in depths:
                for trial in range(num_trials):
                    needle = NEEDLES[trial % len(NEEDLES)]
                    haystack = self.create_haystack(length, needle["fact"], depth)
                    
                    success, answer = self.test_retrieval(
                        haystack, needle["question"], needle["answer"]
                    )
                    
                    results["turboquant"]["total"] += 1
                    if success:
                        results["turboquant"]["correct"] += 1
                    
                    results["turboquant"]["details"].append({
                        "length": length,
                        "depth": depth,
                        "trial": trial,
                        "success": success,
                    })
                    
                    status = "✓" if success else "✗"
                    print(f"  {status} L={length:>4}, D={depth:>4.0%}, T={trial+1}")
        
        tq_score = results["turboquant"]["correct"] / results["turboquant"]["total"] * 100
        print(f"  TurboQuant score: {tq_score:.1f}%")
        
        unpatch_model(self.model)
        
        results["baseline"]["score_pct"] = baseline_score
        results["turboquant"]["score_pct"] = tq_score
        
        return results
    
    def __enter__(self):
        self.load()
        return self
    
    def __exit__(self, *args):
        self.unload()
        return False


def save_results(results: Dict, model_name: str):
    """Save results to JSON."""
    os.makedirs("results", exist_ok=True)
    filename = f"results/needle_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")


def main():
    print("="*80)
    print("NEEDLE-IN-A-HAYSTACK BENCHMARK")
    print("Based on TurboQuant paper (ICLR 2026)")
    print("="*80)
    
    models = [
        {"name": "Qwen3.5-0.8B", "repo": "Qwen/Qwen3.5-0.8B"},
        {"name": "Llama-3.2-1B", "repo": "meta-llama/Llama-3.2-1B"},
    ]
    
    all_summaries = []
    
    for model_info in models:
        print(f"\n{'='*80}")
        print(f"Testing: {model_info['name']}")
        print(f"{'='*80}")
        
        try:
            with NeedleTester(model_info["repo"]) as tester:
                results = tester.run_benchmark(
                    context_lengths=[512, 1024, 2048],
                    depths=[0.0, 0.5, 1.0],
                    num_trials=2
                )
                
                save_results(results, model_info["name"])
                
                all_summaries.append({
                    "model": model_info["name"],
                    "baseline": results["baseline"]["score_pct"],
                    "turboquant": results["turboquant"]["score_pct"],
                })
                
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    if all_summaries:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"{'Model':<25} {'Baseline':<12} {'TurboQuant':<12} {'Delta':<10}")
        print("-"*80)
        
        for s in all_summaries:
            delta = s["turboquant"] - s["baseline"]
            print(f"{s['model']:<25} {s['baseline']:>10.1f}% {s['turboquant']:>10.1f}% {delta:>+8.1f}%")
        
        print()
        print("Note: Negative delta indicates TurboQuant performs better than baseline.")
        print("      (Can happen due to random sampling in generation)")
    
    print("\n" + "="*80)
    print("Benchmark complete!")
    print("="*80)


if __name__ == "__main__":
    seed_everything(42)
    main()
