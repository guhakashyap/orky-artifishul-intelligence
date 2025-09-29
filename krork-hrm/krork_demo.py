#!/usr/bin/env python3
"""
KRORK DEMO - DA HIERARCHICAL REASONING DEMONSTRATION! 🧠⚡

Dis demo shows off da Krork-HRM architecture with hierarchical reasoning!
Watch how da Warboss does strategic plannin' while da boyz handle tactical execution!

FOR DA BOYZ: Dis shows how da ancient Krork intelligence works with hierarchical
thinkin' where da Warboss plans da big strategy while da boyz handle da details!
FOR HUMANS: This demonstrates the Hierarchical Reasoning Model (HRM) architecture
with brain-inspired hierarchical processing for efficient reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from krork_hrm import OrkyKrorkHRM, create_orky_krork_hrm

def quick_krork_demo():
    """
    QUICK KRORK DEMO - SHOW OFF DA HIERARCHICAL REASONING!

    Dis shows how da Krork-HRM model works with hierarchical Warboss + Boyz coordination!
    """
    print("🧠⚡ QUICK KRORK DEMO - DA HIERARCHICAL REASONING! ⚡🧠")
    print("=" * 60)

    # Create a small Krork-HRM model for demo
    model = create_orky_krork_hrm(
        da_vocab_size=50,
        da_orky_model_size=32,
        da_warboss_hidden_size=64,
        da_boyz_hidden_size=32,
        da_num_layers=2,
        da_max_seq_len=16
    )

    # Test with different input patterns
    test_inputs = [
        torch.tensor([[1, 2, 3, 4, 5]]),  # Simple sequence
        torch.tensor([[10, 20, 30, 40, 49]]),  # Different range
        torch.tensor([[1, 1, 1, 1, 1]]),  # Repeated pattern
    ]

    for i, input_tokens in enumerate(test_inputs):
        print(f"\n🔵 Test {i+1}: Input tokens {input_tokens[0].tolist()}")
        
        # Forward pass
        with torch.no_grad():
            logits = model.unleash_da_krork_reasoning(input_tokens)
            predictions = F.softmax(logits, dim=-1)
            
        print(f"   Output shape: {logits.shape}")
        print(f"   Max prediction: {predictions.max().item():.4f}")
        print(f"   Prediction entropy: {-torch.sum(predictions * torch.log(predictions + 1e-8), dim=-1).mean().item():.4f}")

    print("\n✅ Quick Krork demo complete! WAAAGH!")

def hierarchical_reasoning_demo():
    """
    HIERARCHICAL REASONING DEMO - SHOW DA WARBOSS + BOYZ COORDINATION!

    Dis shows how da Warboss does strategic plannin' while da boyz handle tactical execution!
    """
    print("\n🧠⚡ HIERARCHICAL REASONING DEMO - DA WARBOSS + BOYZ COORDINATION! ⚡🧠")
    print("=" * 60)

    # Create a model with more layers for better hierarchical reasoning
    model = create_orky_krork_hrm(
        da_vocab_size=100,
        da_orky_model_size=64,
        da_warboss_hidden_size=128,
        da_boyz_hidden_size=64,
        da_num_layers=3,  # More layers for better hierarchical reasoning
        da_max_seq_len=32
    )

    # Test with different reasoning patterns
    test_patterns = [
        "Simple sequential reasoning",
        "Complex pattern recognition",
        "Hierarchical problem solving",
        "Strategic planning simulation",
        "Tactical execution simulation"
    ]

    test_inputs = [
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]]),
        torch.tensor([[15, 23, 7, 41, 2, 38, 19, 33]]),
        torch.tensor([[1, 2, 1, 2, 1, 2, 1, 2]]),
        torch.tensor([[95, 96, 97, 98, 99, 95, 96, 97]])
    ]

    for pattern_name, input_tokens in zip(test_patterns, test_inputs):
        print(f"\n🔵 Pattern: {pattern_name}")
        print(f"   Input: {input_tokens[0].tolist()}")
        
        # Forward pass to get hierarchical reasoning
        with torch.no_grad():
            logits = model.unleash_da_krork_reasoning(input_tokens)
            predictions = F.softmax(logits, dim=-1)
            
            print(f"   Output shape: {logits.shape}")
            print(f"   Max prediction: {predictions.max().item():.4f}")
            print(f"   Prediction entropy: {-torch.sum(predictions * torch.log(predictions + 1e-8), dim=-1).mean().item():.4f}")

    print("\n✅ Hierarchical reasoning demo complete! WAAAGH!")

def krork_vs_transformer_comparison():
    """
    KRORK VS TRANSFORMER COMPARISON - SHOW DA EFFICIENCY OF HIERARCHICAL REASONING!

    Dis compares Krork-HRM with transformer models to show da efficiency gains!
    """
    print("\n🧠⚡ KRORK VS TRANSFORMER COMPARISON - DA EFFICIENCY BATTLE! ⚡🧠")
    print("=" * 60)

    # Create Krork-HRM model
    krork_model = create_orky_krork_hrm(
        da_vocab_size=100,
        da_orky_model_size=64,
        da_warboss_hidden_size=128,
        da_boyz_hidden_size=64,
        da_num_layers=2,
        da_max_seq_len=32
    )

    # Create equivalent transformer model for comparison
    class TransformerModel(nn.Module):
        def __init__(self, vocab_size, model_size, num_layers):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, model_size)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(model_size, nhead=8, dim_feedforward=128, batch_first=True)
                for _ in range(num_layers)
            ])
            self.output_proj = nn.Linear(model_size, vocab_size)
            
        def forward(self, x):
            x = self.embedding(x)
            for layer in self.layers:
                x = layer(x)
            return self.output_proj(x)

    transformer_model = TransformerModel(100, 64, 2)

    # Compare parameter counts
    krork_params = sum(p.numel() for p in krork_model.parameters())
    transformer_params = sum(p.numel() for p in transformer_model.parameters())

    print(f"🔵 Krork-HRM Parameters: {krork_params:,}")
    print(f"🔵 Transformer Parameters: {transformer_params:,}")
    print(f"🔵 Parameter Ratio: {krork_params / transformer_params:.2f}x")

    # Test inference speed
    test_input = torch.randint(0, 100, (1, 16))
    
    # Krork-HRM inference
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = krork_model.unleash_da_krork_reasoning(test_input)
    krork_time = time.time() - start_time

    # Transformer inference
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = transformer_model(test_input)
    transformer_time = time.time() - start_time

    print(f"\n🔵 Krork-HRM Inference Time: {krork_time:.4f}s")
    print(f"🔵 Transformer Inference Time: {transformer_time:.4f}s")
    print(f"🔵 Speed Ratio: {transformer_time / krork_time:.2f}x")

    print("\n✅ Krork vs Transformer comparison complete! WAAAGH!")

def krork_scaling_demo():
    """
    KRORK SCALING DEMO - SHOW HOW KRORK-HRM SCALES WITH MORE LAYERS!

    Dis shows how da Krork-HRM model scales with more hierarchical layers!
    """
    print("\n🧠⚡ KRORK SCALING DEMO - DA HIERARCHICAL SCALING! ⚡🧠")
    print("=" * 60)

    layer_counts = [1, 2, 3, 4]
    
    for num_layers in layer_counts:
        print(f"\n🔵 Testing with {num_layers} layers:")
        
        # Create model with different layer counts
        model = create_orky_krork_hrm(
            da_vocab_size=100,
            da_orky_model_size=64,
            da_warboss_hidden_size=128,
            da_boyz_hidden_size=64,
            da_num_layers=num_layers,
            da_max_seq_len=32
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total Parameters: {total_params:,}")
        
        # Test inference
        test_input = torch.randint(0, 100, (1, 8))
        start_time = time.time()
        with torch.no_grad():
            for _ in range(50):
                _ = model.unleash_da_krork_reasoning(test_input)
        inference_time = time.time() - start_time
        print(f"   Inference Time: {inference_time:.4f}s")

    print("\n✅ Krork scaling demo complete! WAAAGH!")

def reasoning_task_demo():
    """
    REASONING TASK DEMO - SHOW KRORK-HRM ON COMPLEX REASONING TASKS!

    Dis shows how da Krork-HRM model handles complex reasoning tasks!
    """
    print("\n🧠⚡ REASONING TASK DEMO - DA COMPLEX REASONING! ⚡🧠")
    print("=" * 60)

    # Create a model for complex reasoning
    model = create_orky_krork_hrm(
        da_vocab_size=100,
        da_orky_model_size=64,
        da_warboss_hidden_size=128,
        da_boyz_hidden_size=64,
        da_num_layers=3,
        da_max_seq_len=32
    )

    # Test with different reasoning tasks
    reasoning_tasks = [
        "Sequential pattern completion",
        "Hierarchical problem decomposition",
        "Strategic planning simulation",
        "Tactical execution coordination",
        "Complex reasoning chain"
    ]

    test_inputs = [
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),  # Sequential
        torch.tensor([[1, 1, 2, 2, 3, 3, 4, 4]]),  # Hierarchical
        torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]]),  # Strategic
        torch.tensor([[1, 2, 1, 2, 1, 2, 1, 2]]),  # Tactical
        torch.tensor([[1, 2, 3, 1, 2, 3, 1, 2]])   # Complex
    ]

    for task_name, input_tokens in zip(reasoning_tasks, test_inputs):
        print(f"\n🔵 Task: {task_name}")
        print(f"   Input: {input_tokens[0].tolist()}")
        
        # Forward pass
        with torch.no_grad():
            logits = model.unleash_da_krork_reasoning(input_tokens)
            predictions = F.softmax(logits, dim=-1)
            
            print(f"   Output shape: {logits.shape}")
            print(f"   Max prediction: {predictions.max().item():.4f}")
            print(f"   Prediction entropy: {-torch.sum(predictions * torch.log(predictions + 1e-8), dim=-1).mean().item():.4f}")

    print("\n✅ Reasoning task demo complete! WAAAGH!")

def main():
    """
    MAIN DEMO FUNCTION - RUN ALL DA KRORK DEMONSTRATIONS!

    Dis runs all da Krork demos to show off da hierarchical reasoning system!
    """
    print("🧠⚡ KRORK DEMO - DA HIERARCHICAL REASONING DEMONSTRATION! ⚡🧠")
    print("=" * 80)
    print("Dis demo shows off da Krork-HRM architecture with hierarchical reasoning!")
    print("Watch how da Warboss does strategic plannin' while da boyz handle tactical execution!")
    print("=" * 80)

    # Run all demos
    quick_krork_demo()
    hierarchical_reasoning_demo()
    krork_vs_transformer_comparison()
    krork_scaling_demo()
    reasoning_task_demo()

    print("\n🎉 ALL KRORK DEMOS COMPLETE! WAAAGH! 🎉")
    print("Da Krork-HRM architecture shows how hierarchical reasoning can solve complex problems!")
    print("Da Warboss handles strategic plannin' while da boyz handle tactical execution!")

if __name__ == "__main__":
    main()
