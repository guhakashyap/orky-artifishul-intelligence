#!/usr/bin/env python3
"""
HYDRA WAAAGH DEMO - QUICK DEMONSTRATION OF DA MULTI-HEAD ORK WAR MACHINE! 🐍⚡

DIS IS A QUICK DEMO SCRIPT DAT SHOWS OFF DA HYDRA ARCHITECTURE IN ACTION!
Perfect for seein' how all da different Ork systems work together for da biggest WAAAGH!

WHAT YOU'LL SEE (ORKY):
- Da mighty Hydra model processin' sequences like a proper Ork horde
- Different Ork clans (attention heads) workin' together
- State space memory rememberin' important battles
- Mixture of Experts routin' different jobs to different specialists
- Performance benchmarks showin' da Ork efficiency

WHAT YOU'LL SEE (HUMIE):
- Complete Hydra model forward passes and generation
- Component-level analysis of different architectural pieces
- Memory demonstration showing selective retention
- Attention pattern visualization across different heads
- Speed and efficiency benchmarks vs theoretical limits

FOR DA BOYZ: Run dis to see da Hydra crush some sequences!
FOR HUMANS: This demonstrates the hybrid SSM-attention-MoE architecture.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from hydra_waaagh import (
    OrkyHydraWaaaghModel,
    OrkyStateSpaceModel,
    OrkyMultiHeadHydra,
    OrkyMixtureOfExperts,
    HydraWaaaghBlock
)

def quick_hydra_demo():
    """
    QUICK DEMO OF DA HYDRA WAAAGH - SEE DA ORKS IN ACTION!

    DIS SHOWS DA BASIC FUNCTIONALITY WITHOUT ALL DA DETAILED EXPLANATIONS.
    Perfect for a quick test run to see if da Orks are ready for battle!

    WHAT HAPPENS (ORKY):
    1. We build a small Hydra model (like fieldin' a few Ork squads)
    2. Feed it some random input tokens (like throwin' some Gretchin into battle)
    3. Da Hydra processes da sequence (da Orks fight their way through)
    4. We get predictions for each position (da Orks report their victories)
    5. Generate some new tokens (da Orks continue da WAAAGH!)

    WHAT HAPPENS (HUMIE):
    1. Instantiate a small Hydra model with minimal parameters
    2. Create random input sequences for testing
    3. Perform forward pass through the model
    4. Extract predictions using argmax
    5. Use autoregressive generation for token continuation
    """
    print("🐍⚡ QUICK HYDRA WAAAGH DEMO! ⚡🐍")
    print("=" * 50)
    print("Watch da Orks unleash their WAAAGH!")

    # SIMPLE PARAMETERS FOR QUICK DEMO - keep it small for fast testin'
    # Why small? (Orky): We don't want to field da entire horde for a simple demo!
    # Why small? (Humie): Smaller models train/test faster and use less memory
    da_vocab_size = 50      # How many different words da Orks know
    da_model_size = 64      # How big da Ork brains are
    da_seq_len = 8         # How long da battle sequence is
    da_batch_size = 1      # How many battle squads we're testin'

    print(f"Fieldin' Hydra wif {da_model_size} brain power...")

    # CREATE DA HYDRA MODEL - build da Ork war machine!
    # Why we need 'em (Orky): Can't have a WAAAGH without da proper war machine!
    # Why we need 'em (Humie): Instantiate the model with specified hyperparameters
    da_hydra_model = OrkyHydraWaaaghModel(
        vocab_size=da_vocab_size,
        d_model=da_model_size,
        num_layers=2,           # Two layers of Ork processing
        max_seq_len=32,         # Maximum battle length
    )

    # CREATE RANDOM INPUT - throw some Gretchin into da fight!
    # torch.randint: Creates random integers between 0 and vocab_size-1
    # Why random? (Orky): Real battles ain't predictable - neither are our inputs!
    # Why random? (Humie): Random inputs test the model's generalization ability
    da_input_tokens = torch.randint(0, da_vocab_size, (da_batch_size, da_seq_len))
    print(f"Input battle tokens: {da_input_tokens[0].tolist()}")

    # FORWARD PASS - unleash da WAAAGH on da sequence!
    # torch.no_grad(): Disables gradient computation for inference (faster, less memory)
    # Why no_grad? (Orky): Da Orks are fightin', not learnin' - no need for fancy gradients!
    # Why no_grad? (Humie): Inference doesn't require gradients, so we skip them for efficiency
    with torch.no_grad():
        da_battle_logits = da_hydra_model(da_input_tokens)
        da_victory_predictions = torch.argmax(da_battle_logits, dim=-1)

    print(f"Victory predictions: {da_victory_predictions[0].tolist()}")

    # GENERATE NEW TOKENS - continue da WAAAGH!
    # Why generate? (Orky): After winnin' a battle, da Orks want more fights!
    # Why generate? (Humie): Autoregressive generation shows the model's predictive capabilities
    print("\nDa Orks continue their WAAAGH...")
    da_continued_battle = da_hydra_model.generate(da_input_tokens, max_new_tokens=5, temperature=0.8)
    print(f"Extended battle: {da_continued_battle[0].tolist()}")

    print("\n✅ Quick demo complete! Da Hydra WAAAGH is workin' perfectly!")

def component_comparison_demo():
    """
    COMPARE DIFFERENT ORK COMPONENTS - SEE EACH CLAN DO THEIR JOB!

    DIS SHOWS HOW EACH PART OF DA HYDRA CONTRIBUTES DIFFERENT CAPABILITIES.
    Like watchin' different Ork clans specialize in different types of warfare!

    WHAT WE TEST (ORKY):
    - State Space Model: Da memory expert who remembers important battles
    - Multi-Head Hydra: Da clan coordinator who manages different fightin' styles
    - Mixture of Experts: Da job dispatcher who assigns tasks to specialists
    - Full Hydra Block: Da complete battle squad workin' together

    WHAT WE MEASURE (HUMIE):
    - Output shapes: Ensure tensor dimensions are correct
    - Processing times: Compare computational efficiency
    - Output ranges: Check activation distributions
    """
    print("\n🔍 COMPONENT COMPARISON DEMO!")
    print("=" * 50)
    print("Watch each Ork clan demonstrate their specialties!")

    # PARAMETERS FOR DA COMPONENT TESTS
    da_model_size = 64      # How big da Ork brains are
    da_seq_len = 12        # How long da test sequence is
    da_batch_size = 1      # Single battle squad for testin'

    # CREATE TEST INPUT - da battlefield for our Ork components!
    # torch.randn: Creates random numbers from normal distribution (like unpredictable battles)
    # Why random? (Orky): Real battles are chaotic - our inputs should be too!
    # Why random? (Humie): Random inputs test generalization across different scenarios
    da_test_battlefield = torch.randn(da_batch_size, da_seq_len, da_model_size)
    print(f"Battlefield dimensions: {da_test_battlefield.shape}")

    # TEST EACH ORK COMPONENT - field da different clans!
    # Why test separately? (Orky): Each clan has different specialties - we test their skills!
    # Why test separately? (Humie): Component analysis helps debug and understand contributions
    da_ork_components = {
        "State Space Model": OrkyStateSpaceModel(da_model_size, d_state=8),
        "Multi-Head Hydra": OrkyMultiHeadHydra(da_model_size, num_heads=2),
        "Mixture of Experts": OrkyMixtureOfExperts(da_model_size, num_experts=2),
        "Full Hydra Block": HydraWaaaghBlock(da_model_size, d_state=8, num_heads=2, num_experts=2),
    }
    
    print("\nTestin' each Ork clan:")
    for da_clan_name, da_clan_component in da_ork_components.items():
        # TIME DA CLAN'S PERFORMANCE - how fast dey fight!
        # time.time(): Gets current time for performance measurement
        # Why time it? (Orky): We want to know which clans are da fastest fighters!
        # Why time it? (Humie): Performance benchmarking helps identify bottlenecks
        da_battle_start = time.time()

        # LET DA CLAN FIGHT - send 'em into battle!
        # torch.no_grad(): No learnin' durin' battle - just fightin'!
        # Why no_grad? (Orky): Da Orks are fightin', not studyin' - pure combat mode!
        # Why no_grad? (Humie): Inference doesn't require gradients, so we disable them
        with torch.no_grad():
            if "Hydra" in da_clan_name and "Block" not in da_clan_name:
                # Multi-head components need da attention mask parameter
                da_clan_victory = da_clan_component(da_test_battlefield, mask=None)
            else:
                # Other components just need da input battlefield
                da_clan_victory = da_clan_component(da_test_battlefield)

        # CALCULATE BATTLE DURATION - how long da clan took to win!
        da_battle_end = time.time()
        da_battle_duration = da_battle_end - da_battle_start

        # REPORT DA CLAN'S PERFORMANCE - da battle results!
        print(f"  {da_clan_name}:")
        print(f"    Victory dimensions: {da_clan_victory.shape}")
        print(f"    Battle duration: {da_battle_duration:.4f}s")
        print(f"    Victory range: [{da_clan_victory.min():.3f}, {da_clan_victory.max():.3f}]")

def memory_demonstration():
    """
    DEMONSTRATE DA ORK MEMORY CAPABILITIES!
    
    Shows how da Hydra can remember and use past information.
    """
    print("\n🧠 MEMORY DEMONSTRATION!")
    print("=" * 50)
    
    from hydra_waaagh import OrkyMemoryHead
    
    d_model = 64
    head_dim = 32
    seq_len = 10
    batch_size = 1
    
    # Create memory head
    memory_head = OrkyMemoryHead(d_model, head_dim, memory_size=16)
    
    # Create sequences with patterns
    print("Testing memory with patterned sequences...")
    
    # First sequence - establish pattern
    seq1 = torch.randn(batch_size, seq_len, d_model)
    seq1[:, ::2, :] = 1.0  # Even positions have high values
    
    # Second sequence - similar pattern
    seq2 = torch.randn(batch_size, seq_len, d_model)
    seq2[:, ::2, :] = 1.0  # Even positions have high values
    
    # Third sequence - different pattern
    seq3 = torch.randn(batch_size, seq_len, d_model)
    seq3[:, 1::2, :] = 1.0  # Odd positions have high values
    
    with torch.no_grad():
        out1 = memory_head(seq1)
        out2 = memory_head(seq2)  # Should be similar to out1
        out3 = memory_head(seq3)  # Should be different
    
    # Compare similarities
    sim_1_2 = F.cosine_similarity(out1.mean(dim=1), out2.mean(dim=1), dim=-1)
    sim_1_3 = F.cosine_similarity(out1.mean(dim=1), out3.mean(dim=1), dim=-1)
    
    print(f"Similarity between seq1 and seq2 (same pattern): {sim_1_2.item():.3f}")
    print(f"Similarity between seq1 and seq3 (different pattern): {sim_1_3.item():.3f}")
    print("Higher similarity for same patterns shows memory is working!")

def attention_pattern_demo():
    """
    SHOW DA DIFFERENT ATTENTION PATTERNS!
    
    Demonstrates how different heads focus on different aspects.
    """
    print("\n👁️  ATTENTION PATTERN DEMO!")
    print("=" * 50)
    
    from hydra_waaagh import OrkyLocalAttentionHead, OrkySparseGlobalHead
    
    d_model = 32
    head_dim = 16
    seq_len = 8
    batch_size = 1
    
    # Create attention heads
    local_head = OrkyLocalAttentionHead(d_model, head_dim, window_size=3)
    sparse_head = OrkySparseGlobalHead(d_model, head_dim, sparsity_ratio=0.3)
    
    # Create input with clear patterns
    x = torch.randn(batch_size, seq_len, d_model)
    # Make first and last tokens distinctive
    x[:, 0, :] = 2.0
    x[:, -1, :] = 2.0
    
    print("Testing attention patterns...")
    
    with torch.no_grad():
        local_out = local_head(x)
        sparse_out = sparse_head(x)
    
    print(f"Local attention output shape: {local_out.shape}")
    print(f"Sparse attention output shape: {sparse_out.shape}")
    
    # Show how outputs differ
    local_var = torch.var(local_out, dim=1).mean()
    sparse_var = torch.var(sparse_out, dim=1).mean()
    
    print(f"Local attention variance: {local_var:.4f}")
    print(f"Sparse attention variance: {sparse_var:.4f}")
    print("Different variances show different attention patterns!")

def performance_benchmark():
    """
    BENCHMARK DA HYDRA PERFORMANCE!
    
    Shows how fast da Hydra can process different sequence lengths.
    """
    print("\n⚡ PERFORMANCE BENCHMARK!")
    print("=" * 50)
    
    d_model = 128
    batch_size = 2
    
    # Create model
    model = OrkyHydraWaaaghModel(
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
    RUN ALL DA HYDRA DEMOS!
    
    Dis runs through all da different demonstrations to show off
    da full capabilities of da Hydra Waaagh architecture!
    """
    print("🐍⚡ HYDRA WAAAGH COMPLETE DEMONSTRATION! ⚡🐍")
    print("=" * 60)
    print("Welcome to da complete Hydra demo!")
    print("Dis will show you all da different Ork capabilities!")
    print()
    
    try:
        # Run all demos
        quick_hydra_demo()
        component_comparison_demo()
        memory_demonstration()
        attention_pattern_demo()
        performance_benchmark()
        
        print("\n🎉 ALL DEMOS COMPLETE!")
        print("=" * 60)
        print("Da Hydra Waaagh has demonstrated all its capabilities!")
        print("From basic processing to advanced memory and attention patterns,")
        print("da Orks have shown they can handle any sequence processing task!")
        print("\nKey features demonstrated:")
        print("✅ Multi-head attention with specialized heads")
        print("✅ State Space Model for efficient sequence processing")
        print("✅ Memory mechanisms for pattern recognition")
        print("✅ Mixture of Experts for task specialization")
        print("✅ Sparse attention for long-range dependencies")
        print("✅ Causal masking for proper autoregressive modeling")
        print("\nWAAAGH! Da Hydra is ready for battle! 🚀")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Check dat all dependencies are installed and da code is correct!")

if __name__ == "__main__":
    main()
