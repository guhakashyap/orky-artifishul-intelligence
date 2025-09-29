#!/usr/bin/env python3
"""
MOE DEMO - DA MIXTURE OF EXPERTS DEMONSTRATION! 🐍⚡

Dis demo shows off da MoE architecture with different Ork clan specializations!
Watch how different experts get activated for different types of tasks!

FOR DA BOYZ: Dis shows how different Ork clans specialize in different jobs!
FOR HUMANS: This demonstrates the Mixture of Experts architecture with task specialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from moe_waaagh import OrkyMoEModel, create_orky_moe_model

def quick_moe_demo():
    """
    QUICK MOE DEMO - SHOW OFF DA CLAN SPECIALIZATION!

    Dis shows how da MoE model works with different Ork clan specializations!
    """
    print("🐍⚡ QUICK MOE DEMO - DA CLAN SPECIALIZATION! ⚡🐍")
    print("=" * 60)

    # Create a small MoE model for demo
    model = create_orky_moe_model(
        da_vocab_size=50,
        da_orky_model_size=32,
        da_num_layers=2,
        da_num_experts=4,
        da_expert_hidden_size=64,
        da_max_seq_len=16
    )

    # Test with different input patterns (keeping within vocab size of 50)
    test_inputs = [
        torch.tensor([[1, 2, 3, 4, 5]]),  # Simple sequence
        torch.tensor([[10, 20, 30, 40, 49]]),  # Different range (max 49 for vocab 50)
        torch.tensor([[1, 1, 1, 1, 1]]),  # Repeated pattern
    ]

    for i, input_tokens in enumerate(test_inputs):
        print(f"\n🔵 Test {i+1}: Input tokens {input_tokens[0].tolist()}")
        
        # Forward pass
        with torch.no_grad():
            logits = model.unleash_da_moe_waaagh(input_tokens)
            predictions = F.softmax(logits, dim=-1)
            
        print(f"   Output shape: {logits.shape}")
        print(f"   Max prediction: {predictions.max().item():.4f}")
        print(f"   Prediction entropy: {-torch.sum(predictions * torch.log(predictions + 1e-8), dim=-1).mean().item():.4f}")

    print("\n✅ Quick MoE demo complete! WAAAGH!")

def expert_activation_demo():
    """
    EXPERT ACTIVATION DEMO - SHOW HOW DIFFERENT EXPERTS GET ACTIVATED!

    Dis shows how different Ork clans get activated for different tasks!
    """
    print("\n🐍⚡ EXPERT ACTIVATION DEMO - DA CLAN SELECTION! ⚡🐍")
    print("=" * 60)

    # Create a model with more experts for better visualization
    model = create_orky_moe_model(
        da_vocab_size=100,
        da_orky_model_size=64,
        da_num_layers=1,  # Single layer for clarity
        da_num_experts=8,  # More experts to see specialization
        da_expert_hidden_size=128,
        da_max_seq_len=32
    )

    # Test with different input patterns to see expert activation
    test_patterns = [
        "Simple ascending sequence",
        "Repeated pattern",
        "Random tokens",
        "High frequency tokens",
        "Low frequency tokens"
    ]

    test_inputs = [
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]]),
        torch.tensor([[15, 23, 7, 41, 2, 38, 19, 33]]),
        torch.tensor([[1, 2, 1, 2, 1, 2, 1, 2]]),
        torch.tensor([[95, 96, 97, 98, 99, 95, 96, 97]])  # Keep within vocab size 100
    ]

    for pattern_name, input_tokens in zip(test_patterns, test_inputs):
        print(f"\n🔵 Pattern: {pattern_name}")
        print(f"   Input: {input_tokens[0].tolist()}")
        
        # Forward pass to get routing information
        with torch.no_grad():
            # Get routing weights from the first MoE layer
            moe_layer = model.da_moe_layers[0]
            routing_weights, expert_indices = moe_layer.da_ork_router.route_da_ork_clans(
                model.da_token_embeddin(input_tokens) + model.da_positional_embeddin(torch.arange(input_tokens.size(1)).unsqueeze(0))
            )
            
            print(f"   Expert indices: {expert_indices[0].tolist()}")
            print(f"   Routing weights: {routing_weights[0].mean(dim=0).tolist()}")
            
            # Show which experts are most active
            active_experts = expert_indices[0].flatten().unique()
            print(f"   Active experts: {active_experts.tolist()}")

    print("\n✅ Expert activation demo complete! WAAAGH!")

def moe_vs_dense_comparison():
    """
    MOE VS DENSE COMPARISON - SHOW DA EFFICIENCY OF CLAN SPECIALIZATION!

    Dis compares MoE with dense models to show da efficiency gains!
    """
    print("\n🐍⚡ MOE VS DENSE COMPARISON - DA EFFICIENCY BATTLE! ⚡🐍")
    print("=" * 60)

    # Create MoE model
    moe_model = create_orky_moe_model(
        da_vocab_size=100,
        da_orky_model_size=64,
        da_num_layers=2,
        da_num_experts=4,
        da_expert_hidden_size=128,
        da_max_seq_len=32
    )

    # Create equivalent dense model for comparison
    class DenseModel(nn.Module):
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

    dense_model = DenseModel(100, 64, 2)

    # Compare parameter counts
    moe_params = sum(p.numel() for p in moe_model.parameters())
    dense_params = sum(p.numel() for p in dense_model.parameters())

    print(f"🔵 MoE Model Parameters: {moe_params:,}")
    print(f"🔵 Dense Model Parameters: {dense_params:,}")
    print(f"🔵 Parameter Ratio: {moe_params / dense_params:.2f}x")

    # Test inference speed
    test_input = torch.randint(0, 100, (1, 16))
    
    # MoE inference
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = moe_model.unleash_da_moe_waaagh(test_input)
    moe_time = time.time() - start_time

    # Dense inference
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = dense_model(test_input)
    dense_time = time.time() - start_time

    print(f"\n🔵 MoE Inference Time: {moe_time:.4f}s")
    print(f"🔵 Dense Inference Time: {dense_time:.4f}s")
    print(f"🔵 Speed Ratio: {dense_time / moe_time:.2f}x")

    print("\n✅ MoE vs Dense comparison complete! WAAAGH!")

def moe_scaling_demo():
    """
    MOE SCALING DEMO - SHOW HOW MOE SCALES WITH MORE EXPERTS!

    Dis shows how da MoE model scales with more Ork clans!
    """
    print("\n🐍⚡ MOE SCALING DEMO - DA CLAN SCALING! ⚡🐍")
    print("=" * 60)

    expert_counts = [2, 4, 8, 16]
    
    for num_experts in expert_counts:
        print(f"\n🔵 Testing with {num_experts} experts:")
        
        # Create model with different expert counts
        model = create_orky_moe_model(
            da_vocab_size=100,
            da_orky_model_size=64,
            da_num_layers=1,
            da_num_experts=num_experts,
            da_expert_hidden_size=128,
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
                _ = model.unleash_da_moe_waaagh(test_input)
        inference_time = time.time() - start_time
        print(f"   Inference Time: {inference_time:.4f}s")

    print("\n✅ MoE scaling demo complete! WAAAGH!")

def main():
    """
    MAIN DEMO FUNCTION - RUN ALL DA MOE DEMONSTRATIONS!

    Dis runs all da MoE demos to show off da clan specialization system!
    """
    print("🐍⚡ MOE DEMO - DA MIXTURE OF EXPERTS DEMONSTRATION! ⚡🐍")
    print("=" * 80)
    print("Dis demo shows off da MoE architecture with different Ork clan specializations!")
    print("Watch how different experts get activated for different types of tasks!")
    print("=" * 80)

    # Run all demos
    quick_moe_demo()
    expert_activation_demo()
    moe_vs_dense_comparison()
    moe_scaling_demo()

    print("\n🎉 ALL MOE DEMOS COMPLETE! WAAAGH! 🎉")
    print("Da MoE architecture shows how specialized Ork clans can work together!")
    print("Each expert focuses on their specialty, makin' da whole system more efficient!")

if __name__ == "__main__":
    main()
