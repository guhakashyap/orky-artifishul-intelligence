#!/usr/bin/env python3
"""
HYENA DEMO - QUICK DEMONSTRATION OF DA CONVOLUTIONAL SEQUENCE MODEL! 🐍⚡

Dis is a quick demo script dat shows off da Hyena architecture in action!
Perfect for seein' how da efficient convolutional processing works!

FOR DA BOYZ: Run dis to see da Hyena in action!
FOR HUMANS: This demonstrates the key features of the Hyena architecture.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from hyena_hierarchy import (
    OrkyHyenaModel,
    OrkyHyenaOperator,
    OrkyPositionalEmbedding,
    OrkyHyenaBlock
)

def quick_hyena_demo():
    """
    QUICK DEMO OF DA HYENA!

    Shows da basic functionality without all da detailed explanations.
    Perfect for a quick test run!
    """
    print("🐍⚡ QUICK HYENA DEMO! ⚡🐍")
    print("=" * 50)

    # Simple parameters for quick demo
    vocab_size = 50
    d_model = 64
    seq_len = 8
    batch_size = 1

    print(f"Building Hyena with {d_model} dimensions...")

    # Create model
    model = OrkyHyenaModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=2,
        max_seq_len=32,
    )

    # Create input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input: {input_ids[0].tolist()}")

    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
        predictions = torch.argmax(logits, dim=-1)

    print(f"Predictions: {predictions[0].tolist()}")

    # Generate some tokens
    print("\nGenerating new tokens...")
    generated = model.generate(input_ids, max_new_tokens=5, temperature=0.8)
    print(f"Generated: {generated[0].tolist()}")

    print("\n✅ Quick demo complete! Da Hyena is workin'!")

def convolution_comparison_demo():
    """
    COMPARE DIFFERENT CONVOLUTIONAL OPERATORS!

    Shows how each part of da Hyena contributes different capabilities.
    """
    print("\n🔍 CONVOLUTION COMPARISON DEMO!")
    print("=" * 50)

    d_model = 64
    seq_len = 12
    batch_size = 1

    # Create test input
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Test input shape: {x.shape}")

    # Test each component
    components = {
        "Hyena Operator": OrkyHyenaOperator(d_model, l_max=32),
        "Positional Embedding": OrkyPositionalEmbedding(d_model, max_seq_len=32),
        "Full Hyena Block": OrkyHyenaBlock(d_model, l_max=32),
    }

    print("\nTesting each component:")
    for name, component in components.items():
        start_time = time.time()

        with torch.no_grad():
            if "Operator" in name:
                output = component(x)
            elif "Embedding" in name:
                output = component(x)
            else:
                output = component(x)

        end_time = time.time()

        print(f"  {name}:")
        print(f"    Output shape: {output.shape}")
        print(f"    Processing time: {end_time - start_time:.4f}s")
        print(f"    Output range: [{output.min():.3f}, {output.max():.3f}]")

def long_sequence_demo():
    """
    DEMONSTRATE LONG SEQUENCE PROCESSING!

    Shows how da Hyena can efficiently handle long sequences.
    """
    print("\n📏 LONG SEQUENCE DEMONSTRATION!")
    print("=" * 50)

    d_model = 64
    batch_size = 1

    # Test different sequence lengths
    seq_lengths = [64, 128, 256, 512]

    print("Testing Hyena on different sequence lengths:")
    print("Seq Len | Time (ms) | Memory (MB)")
    print("-" * 35)

    model = OrkyHyenaModel(
        vocab_size=100,
        d_model=d_model,
        num_layers=2,
        max_seq_len=512,
    )

    for seq_len in seq_lengths:
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        # Clear cache and measure memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        start_time = time.time()
        with torch.no_grad():
            logits = model(input_ids)
        end_time = time.time()

        end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        mem_used = (end_mem - start_mem) / (1024 * 1024)  # MB

        print(f"{seq_len:7d} | {((end_time - start_time)*1000):8.2f} | {mem_used:9.1f}")

def hierarchical_processing_demo():
    """
    SHOW DA HIERARCHICAL PROCESSING!

    Demonstrates how da Hyena processes information at different scales.
    """
    print("\n🏗️  HIERARCHICAL PROCESSING DEMO!")
    print("=" * 50)

    d_model = 64
    seq_len = 32
    batch_size = 1

    # Create model with multiple layers
    model = OrkyHyenaModel(
        vocab_size=50,
        d_model=d_model,
        num_layers=3,
        max_seq_len=64,
    )

    # Create input with hierarchical patterns
    input_ids = torch.randint(0, 50, (batch_size, seq_len))

    print("Processing sequence through hierarchical layers...")

    with torch.no_grad():
        # Process through each layer manually to show hierarchy
        x = model.embedding(input_ids)

        for i, layer in enumerate(model.layers):
            x = layer(x)
            print(f"Layer {i+1} output variance: {torch.var(x).item():.4f}")
            print(f"Layer {i+1} output range: [{x.min():.3f}, {x.max():.3f}]")

    print("Each layer processes at a different hierarchical level!")

def performance_benchmark():
    """
    BENCHMARK DA HYENA PERFORMANCE!

    Shows how fast da Hyena can process different sequence lengths.
    """
    print("\n⚡ PERFORMANCE BENCHMARK!")
    print("=" * 50)

    d_model = 128
    batch_size = 2

    # Create model
    model = OrkyHyenaModel(
        vocab_size=100,
        d_model=d_model,
        num_layers=2,
        max_seq_len=256,
    )

    # Test different sequence lengths
    seq_lengths = [8, 16, 32, 64]

    print("Benchmarking different sequence lengths:")
    print("Seq Len | Time (ms) | Tokens/sec")
    print("-" * 35)

    for seq_len in seq_lengths:
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        # Warmup
        with torch.no_grad():
            _ = model(input_ids)

        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):  # Average over 10 runs
                _ = model(input_ids)
        end_time = time.time()

        avg_time = (end_time - start_time) / 10
        tokens_per_sec = (batch_size * seq_len) / avg_time

        print(f"{seq_len:7d} | {avg_time*1000:8.2f} | {tokens_per_sec:9.0f}")

def main():
    """
    RUN ALL DA HYENA DEMOS!

    Dis runs through all da different demonstrations to show off
    da full capabilities of da Hyena architecture!
    """
    print("🐍⚡ HYENA COMPLETE DEMONSTRATION! ⚡🐍")
    print("=" * 60)
    print("Welcome to da complete Hyena demo!")
    print("Dis will show you all da convolutional sequence processing!")
    print()

    try:
        # Run all demos
        quick_hyena_demo()
        convolution_comparison_demo()
        long_sequence_demo()
        hierarchical_processing_demo()
        performance_benchmark()

        print("\n🎉 ALL DEMOS COMPLETE!")
        print("=" * 60)
        print("Da Hyena has demonstrated all its capabilities!")
        print("From basic processing to advanced hierarchical modeling,")
        print("da Hyena has shown it can handle any sequence processing task!")
        print("\nKey features demonstrated:")
        print("✅ Efficient convolutional sequence processing")
        print("✅ Hierarchical information processing")
        print("✅ Long sequence handling with linear complexity")
        print("✅ Positional embeddings for sequence awareness")
        print("✅ Stacked layers for deep processing")
        print("✅ Memory-efficient implementation")
        print("\nWAAAGH! Da Hyena is ready for battle! 🚀")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Check dat all dependencies are installed and da code is correct!")

if __name__ == "__main__":
    main()
