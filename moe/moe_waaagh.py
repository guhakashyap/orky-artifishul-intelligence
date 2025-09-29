#!/usr/bin/env python3
"""
MOE WAAAGH! - DA MIXTURE OF EXPERTS ORK WAR MACHINE! 🐍⚡

Dis is da MoE architecture - like havin' different Ork clans specializin' in different jobs!
Instead of one big Ork tryin' to do everyfin', we got specialists who only work when needed!

WHAT IS DA MOE?
- Multiple expert "clans" (like different Ork specialists)
- Router mechanism (like a Warboss assignin' tasks)
- Sparse activation (only da right experts work)
- Efficient processing (don't waste energy on wrong specialists)
- Task specialization (each expert is really good at their job)

FOR DA BOYZ: Dis shows how different Ork clans can specialize and work together!
FOR HUMANS: This implements the Mixture of Experts architecture for efficient task routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List
import time

class OrkyExpert(nn.Module):
    """
    DA ORK EXPERT - A SPECIALIZED ORK CLAN!

    Each expert is like a different Ork clan that's really good at one specific job.
    Da Blood Axes are great at close combat, da Deathskulls are artillery experts,
    da Bad Moons handle logistics, and da Goffs plan da big strategies!

    WHY WE NEED DIS (ORKY):
    Instead of one Ork tryin' to do everyfin' (and doin' it badly), we got
    specialists who are really good at their specific jobs! Each expert
    focuses on what dey do best!

    WHY WE NEED DIS (HUMIE):
    Expert modules provide specialized processing for different types of tasks.
    Each expert learns to handle specific patterns or domains effectively.
    """

    def __init__(self, da_orky_model_size: int, da_expert_hidden_size: int):
        super().__init__()
        # DA MODEL DIMENSION - how big da Ork thoughts are
        # Why we need 'em (Orky): Each expert needs enough brain power for their specialty!
        # Why we need 'em (Humie): Model dimension determines input/output capacity
        self.da_orky_model_size = da_orky_model_size

        # DA EXPERT HIDDEN SIZE - how much brain power dis expert gets
        # Why we need 'em (Orky): Different experts need different amounts of brain power!
        # Why we need 'em (Humie): Hidden size determines the expert's processing capacity
        self.da_expert_hidden_size = da_expert_hidden_size

        # DA EXPERT INPUT PROJECTION - turn general Ork thoughts into expert thoughts
        # Why we need 'em (Orky): Each expert needs to understand da general WAAAGH energy!
        # Why we need 'em (Humie): Input projection adapts general features for expert processing
        self.da_expert_input_proj = nn.Linear(da_orky_model_size, da_expert_hidden_size)

        # DA EXPERT PROCESSING - da main expert brain work
        # Why we need 'em (Orky): Dis is where da expert does their specialized finkin'!
        # Why we need 'em (Humie): Hidden layer provides the expert's core processing
        self.da_expert_processor = nn.Linear(da_expert_hidden_size, da_expert_hidden_size)

        # DA EXPERT OUTPUT PROJECTION - turn expert thoughts back into general Ork thoughts
        # Why we need 'em (Orky): Da expert needs to share their wisdom wif da rest of da horde!
        # Why we need 'em (Humie): Output projection converts expert features back to model dimension
        self.da_expert_output_proj = nn.Linear(da_expert_hidden_size, da_orky_model_size)

        # DA EXPERT ACTIVATION - da expert's special energy surge
        # Why we need 'em (Orky): Each expert has their own way of channelin' WAAAGH energy!
        # Why we need 'em (Humie): Activation function provides non-linearity for expert processing
        self.da_expert_activation = nn.SiLU()

    def do_da_expert_specialization(self, da_general_thoughts: torch.Tensor) -> torch.Tensor:
        """
        DO DA EXPERT SPECIALIZATION - WHERE DA EXPERT SHINES!

        DIS IS WHERE DA EXPERT DOES THEIR SPECIALIZED WORK!

        da_general_thoughts: (batch_size, seq_len, da_orky_model_size) - general Ork thoughts
        Returns: (batch_size, seq_len, da_orky_model_size) - expert-processed thoughts

        DIS IS LIKE WATCHIN' A SPECIALIST ORK CLAN DO THEIR JOB:
        1. Take da general WAAAGH energy and adapt it for their specialty
        2. Process it through their specialized brain
        3. Convert it back into general wisdom dat others can understand
        """
        # STEP 1: ADAPT DA GENERAL THOUGHTS - make 'em suitable for dis expert
        # Why we need 'em (Orky): Each expert needs to understand da general WAAAGH in their own way!
        # Why we need 'em (Humie): Input projection adapts general features for expert processing
        da_expert_input = self.da_expert_input_proj(da_general_thoughts)

        # STEP 2: DA EXPERT PROCESSING - where da specialization happens
        # Why we need 'em (Orky): Dis is where da expert does their specialized finkin'!
        # Why we need 'em (Humie): Hidden layer provides the expert's core processing
        da_expert_processing = self.da_expert_processor(da_expert_input)

        # STEP 3: DA EXPERT ACTIVATION - da expert's special energy surge
        # Why we need 'em (Orky): Each expert has their own way of channelin' WAAAGH energy!
        # Why we need 'em (Humie): Activation function provides non-linearity for expert processing
        da_expert_energy = self.da_expert_activation(da_expert_processing)

        # STEP 4: CONVERT BACK TO GENERAL WISDOM - share da expert knowledge
        # Why we need 'em (Orky): Da expert needs to share their wisdom wif da rest of da horde!
        # Why we need 'em (Humie): Output projection converts expert features back to model dimension
        da_expert_wisdom = self.da_expert_output_proj(da_expert_energy)

        return da_expert_wisdom

class OrkyRouter(nn.Module):
    """
    DA ORK ROUTER - DA WARBOSS WHO ASSIGNS TASKS!

    Dis is like a smart Warboss who looks at da situation and decides which
    Ork clans should handle it! Da Blood Axes for close combat, da Deathskulls
    for artillery, da Bad Moons for logistics, etc.

    WHY WE NEED DIS (ORKY):
    A good Warboss knows which Ork clans are best for which jobs! Dis router
    is like dat smart Warboss who assigns tasks to da right specialists!

    WHY WE NEED DIS (HUMIE):
    Router determines which experts should be activated for each input.
    Uses learned routing to efficiently distribute tasks to appropriate specialists.
    """

    def __init__(self, da_orky_model_size: int, da_num_experts: int, da_top_k: int = 2):
        super().__init__()
        # DA MODEL DIMENSION - how big da Ork thoughts are
        # Why we need 'em (Orky): Da router needs to understand da general WAAAGH energy!
        # Why we need 'em (Humie): Model dimension determines input capacity
        self.da_orky_model_size = da_orky_model_size

        # DA NUMBER OF EXPERTS - how many Ork clans we got
        # Why we need 'em (Orky): More clans = more specialized options!
        # Why we need 'em (Humie): Number of experts determines specialization capacity
        self.da_num_experts = da_num_experts

        # DA TOP-K SELECTION - how many experts to activate
        # Why we need 'em (Orky): Sometimes we need multiple clans workin' together!
        # Why we need 'em (Humie): Top-k routing balances specialization with diversity
        self.da_top_k = da_top_k

        # DA ROUTING NETWORK - da Warboss's decision-making brain
        # Why we need 'em (Orky): Da router needs to be smart about which clans to activate!
        # Why we need 'em (Humie): Routing network learns to assign tasks to appropriate experts
        self.da_routing_network = nn.Linear(da_orky_model_size, da_num_experts)

    def route_da_ork_clans(self, da_input_thoughts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ROUTE DA ORK CLANS - DA WARBOSS ASSIGNS TASKS!

        DIS IS WHERE DA SMART WARBOSS DECIDES WHICH CLANS TO ACTIVATE!

        da_input_thoughts: (batch_size, seq_len, da_orky_model_size) - da input thoughts
        Returns: (routing_weights, expert_indices) - which experts to use and how much

        DIS IS LIKE WATCHIN' A WARBOSS ANALYZE DA BATTLEFIELD:
        1. Look at da situation and understand wot needs to be done
        2. Decide which Ork clans are best for dis job
        3. Assign tasks to da selected clans
        """
        # GET DA ROUTING LOGITS - da Warboss's decision scores
        # Why we need 'em (Orky): Da router needs to score each clan's suitability!
        # Why we need 'em (Humie): Routing logits determine expert selection probabilities
        da_routing_logits = self.da_routing_network(da_input_thoughts)

        # APPLY SOFTMAX - convert scores to probabilities
        # Why we need 'em (Orky): Da router needs to decide how much to trust each clan!
        # Why we need 'em (Humie): Softmax converts logits to probability distribution
        da_routing_probs = F.softmax(da_routing_logits, dim=-1)

        # SELECT TOP-K EXPERTS - choose da best clans for dis job
        # Why we need 'em (Orky): Sometimes we need multiple clans workin' together!
        # Why we need 'em (Humie): Top-k selection balances specialization with diversity
        da_top_k_probs, da_expert_indices = torch.topk(da_routing_probs, self.da_top_k, dim=-1)

        # NORMALIZE DA WEIGHTS - balance da expert contributions
        # Why we need 'em (Orky): Da selected clans need to work together properly!
        # Why we need 'em (Humie): Normalization ensures proper expert weighting
        da_normalized_weights = da_top_k_probs / da_top_k_probs.sum(dim=-1, keepdim=True)

        return da_normalized_weights, da_expert_indices

class OrkyMoELayer(nn.Module):
    """
    DA MOE LAYER - DA COMPLETE ORK CLAN COORDINATION SYSTEM!

    Dis combines da router and experts into one complete system. Like havin'
    a Warboss who can assign tasks to different Ork clans and coordinate
    their work into one mighty WAAAGH!

    WHY WE NEED DIS (ORKY):
    A complete Ork war machine needs both da smart Warboss (router) and
    da specialized clans (experts) workin' together! Dis layer coordinates
    everything into one efficient system!

    WHY WE NEED DIS (HUMIE):
    MoE layer combines routing and expert processing for efficient task specialization.
    Provides sparse activation while maintaining high performance through expert routing.
    """

    def __init__(
        self,
        da_orky_model_size: int,
        da_num_experts: int,
        da_expert_hidden_size: int,
        da_top_k: int = 2
    ):
        super().__init__()
        # DA MODEL DIMENSION - how big da Ork thoughts are
        # Why we need 'em (Orky): Da MoE layer needs to handle da general WAAAGH energy!
        # Why we need 'em (Humie): Model dimension determines input/output capacity
        self.da_orky_model_size = da_orky_model_size

        # DA NUMBER OF EXPERTS - how many Ork clans we got
        # Why we need 'em (Orky): More clans = more specialized options!
        # Why we need 'em (Humie): Number of experts determines specialization capacity
        self.da_num_experts = da_num_experts

        # DA EXPERT HIDDEN SIZE - how much brain power each expert gets
        # Why we need 'em (Orky): Each expert needs enough brain power for their specialty!
        # Why we need 'em (Humie): Hidden size determines expert processing capacity
        self.da_expert_hidden_size = da_expert_hidden_size

        # DA TOP-K SELECTION - how many experts to activate
        # Why we need 'em (Orky): Sometimes we need multiple clans workin' together!
        # Why we need 'em (Humie): Top-k routing balances specialization with diversity
        self.da_top_k = da_top_k

        # DA ROUTER - da smart Warboss who assigns tasks
        # Why we need 'em (Orky): We need a smart Warboss to coordinate da clans!
        # Why we need 'em (Humie): Router determines which experts to activate
        self.da_ork_router = OrkyRouter(da_orky_model_size, da_num_experts, da_top_k)

        # DA EXPERTS - da specialized Ork clans
        # Why we need 'em (Orky): We need different clans for different jobs!
        # Why we need 'em (Humie): Experts provide specialized processing capabilities
        self.da_ork_experts = nn.ModuleList([
            OrkyExpert(da_orky_model_size, da_expert_hidden_size)
            for _ in range(da_num_experts)
        ])

    def coordinate_da_ork_clans(self, da_input_thoughts: torch.Tensor) -> torch.Tensor:
        """
        COORDINATE DA ORK CLANS - DA COMPLETE MOE PROCESSING!

        DIS IS WHERE DA WARBOSS AND DA CLANS WORK TOGETHER!

        da_input_thoughts: (batch_size, seq_len, da_orky_model_size) - da input thoughts
        Returns: (batch_size, seq_len, da_orky_model_size) - da processed thoughts

        DIS IS LIKE WATCHIN' A COMPLETE ORK WAR MACHINE IN ACTION:
        1. Da Warboss (router) analyzes da situation
        2. Da Warboss assigns tasks to da right clans (experts)
        3. Da selected clans do their specialized work
        4. Da results are combined into one mighty WAAAGH!
        """
        # STEP 1: ROUTE DA CLANS - da Warboss assigns tasks
        # Why we need 'em (Orky): Da router needs to decide which clans to activate!
        # Why we need 'em (Humie): Routing determines expert selection and weighting
        da_routing_weights, da_expert_indices = self.da_ork_router.route_da_ork_clans(da_input_thoughts)

        # STEP 2: PROCESS THROUGH DA SELECTED EXPERTS - da clans do their work
        # Why we need 'em (Orky): Da selected clans need to do their specialized jobs!
        # Why we need 'em (Humie): Expert processing provides specialized feature transformation
        da_expert_outputs = []
        for da_batch_idx in range(da_input_thoughts.size(0)):
            da_batch_outputs = []
            for da_seq_idx in range(da_input_thoughts.size(1)):
                # Get da input for dis position
                da_position_input = da_input_thoughts[da_batch_idx, da_seq_idx:da_seq_idx+1, :]
                
                # Process through da selected experts for dis position
                da_position_output = torch.zeros_like(da_position_input)
                for da_expert_idx in range(self.da_top_k):
                    da_expert_id = da_expert_indices[da_batch_idx, da_seq_idx, da_expert_idx]
                    da_expert_weight = da_routing_weights[da_batch_idx, da_seq_idx, da_expert_idx]
                    
                    # Get da expert output
                    da_expert_output = self.da_ork_experts[da_expert_id].do_da_expert_specialization(da_position_input)
                    
                    # Weight and accumulate
                    da_position_output += da_expert_weight * da_expert_output
                
                da_batch_outputs.append(da_position_output)
            
            da_expert_outputs.append(torch.cat(da_batch_outputs, dim=0))
        
        # STEP 3: COMBINE DA RESULTS - merge all da expert outputs
        # Why we need 'em (Orky): Da different clans need to coordinate their results!
        # Why we need 'em (Humie): Output combination integrates expert contributions
        da_final_output = torch.stack(da_expert_outputs, dim=0)

        return da_final_output

class OrkyMoEModel(nn.Module):
    """
    DA COMPLETE MOE MODEL - DA ULTIMATE ORK CLAN COORDINATION MACHINE!

    Dis is da full MoE architecture with embedding, multiple MoE layers,
    and output projection. Ready to process any sequence like a proper
    coordinated Ork horde with specialized clans!

    WHY WE NEED DIS (ORKY):
    Dis is da complete Ork war machine! It takes humie words, turns 'em into
    Ork battle cries, processes 'em through multiple layers of specialized
    Ork clans, and outputs predictions for da next mighty victory! Like
    havin' an entire coordinated Ork army workin' together!

    WHY WE NEED DIS (HUMIE):
    Complete MoE model with token embeddings, positional encoding, stacked
    MoE layers, and output projection. Achieves efficient processing through
    sparse expert activation while maintaining high performance.
    """

    def __init__(
        self,
        da_vocab_size: int,
        da_orky_model_size: int,
        da_num_layers: int,
        da_num_experts: int,
        da_expert_hidden_size: int,
        da_max_seq_len: int = 1024,
        da_dropout_rate: float = 0.1
    ):
        super().__init__()
        # DA VOCABULARY SIZE - how many different words da Orks know
        # Why we need 'em (Orky): Da bigger da vocabulary, da more battle cries da Orks know!
        # Why we need 'em (Humie): Vocabulary size determines the number of possible tokens
        self.da_vocab_size = da_vocab_size

        # DA MODEL DIMENSION - how big da Ork brains are
        # Why we need 'em (Orky): Bigger brains = more powerful WAAAGH processing!
        # Why we need 'em (Humie): Model dimension determines representational capacity
        self.da_orky_model_size = da_orky_model_size

        # DA NUMBER OF LAYERS - how many MoE layers we stack
        # Why we need 'em (Orky): More layers = deeper clan coordination!
        # Why we need 'em (Humie): More layers enable deeper feature learning
        self.da_num_layers = da_num_layers

        # DA NUMBER OF EXPERTS - how many Ork clans we got
        # Why we need 'em (Orky): More clans = more specialized options!
        # Why we need 'em (Humie): Number of experts determines specialization capacity
        self.da_num_experts = da_num_experts

        # DA EXPERT HIDDEN SIZE - how much brain power each expert gets
        # Why we need 'em (Orky): Each expert needs enough brain power for their specialty!
        # Why we need 'em (Humie): Hidden size determines expert processing capacity
        self.da_expert_hidden_size = da_expert_hidden_size

        # DA MAXIMUM SEQUENCE LENGTH - how long da battle can be
        # Why we need 'em (Orky): Even da biggest WAAAGH has limits!
        # Why we need 'em (Humie): Maximum context length for positional embeddings
        self.da_max_seq_len = da_max_seq_len

        # DA TOKEN EMBEDDING - turn humie words into mighty Ork battle cries!
        # Why we need 'em (Orky): Orks don't fink in puny humie words - dey fink in WAAAGHs!
        # Why we need 'em (Humie): Token embeddings convert discrete tokens to continuous vectors
        self.da_token_embeddin = nn.Embedding(da_vocab_size, da_orky_model_size)

        # DA POSITIONAL EMBEDDING - da WAAAGH positional awareness
        # Why we need 'em (Orky): Every Ork needs to know their place in da horde!
        # Why we need 'em (Humie): Positional embeddings provide sequence order information
        self.da_positional_embeddin = nn.Embedding(da_max_seq_len, da_orky_model_size)

        # DA DROPOUT - some Orks get distracted during battle
        # Why we need 'em (Orky): Even Orks sometimes get distracted - makes 'em more robust!
        # Why we need 'em (Humie): Dropout prevents overfitting and improves generalization
        self.da_ork_distraction = nn.Dropout(da_dropout_rate)

        # DA MOE LAYERS - stacked Ork clan coordination systems
        # Why we need 'em (Orky): Multiple layers of clan coordination for maximum WAAAGH!
        # Why we need 'em (Humie): Stacked MoE layers enable deep feature learning with specialization
        self.da_moe_layers = nn.ModuleList([
            OrkyMoELayer(da_orky_model_size, da_num_experts, da_expert_hidden_size)
            for _ in range(da_num_layers)
        ])

        # DA OUTPUT NORMALIZATION - final Ork thought cleanup
        # Why we need 'em (Orky): Even da best Ork thoughts need a final polish!
        # Why we need 'em (Humie): Final normalization stabilizes output and improves training
        self.da_final_ork_norm = nn.LayerNorm(da_orky_model_size)

        # DA OUTPUT PROJECTION - turn Ork thoughts back into humie words
        # Why we need 'em (Orky): Orks need to communicate their wisdom to da humies!
        # Why we need 'em (Humie): Output projection predicts next token probabilities
        self.da_output_proj = nn.Linear(da_orky_model_size, da_vocab_size, bias=False)

        # TIE EMBEDDINGS - save memory like a thrifty Ork
        # Why we need 'em (Orky): Smart Orks reuse their equipment to save resources!
        # Why we need 'em (Humie): Weight tying reduces parameters and improves efficiency
        self.da_output_proj.weight = self.da_token_embeddin.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """INITIALIZE DA WEIGHTS LIKE A PROPER ORK WARBOSS!"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def unleash_da_moe_waaagh(self, da_input_tokens: torch.Tensor) -> torch.Tensor:
        """
        UNLEASH DA MOE WAAAGH - DA ULTIMATE CLAN COORDINATION BATTLE!

        DIS IS WHERE DA ENTIRE ORK ARMY WITH SPECIALIZED CLANS GOES TO WAR:
        1. Convert humie words into mighty Ork battle cries
        2. Add positional awareness so every Ork knows their place
        3. Process through multiple layers of specialized Ork clan coordination
        4. Emerge with predictions for da next mighty victory!

        da_input_tokens: (batch_size, seq_len) - da input token indices
        Returns: (batch_size, seq_len, da_vocab_size) - predictions for each position
        """
        # GET DA SEQUENCE LENGTH - how long is dis battle?
        da_current_seq_len = da_input_tokens.size(1)

        # STEP 1: DA TOKEN EMBEDDING - convert humie words to Ork battle cries!
        da_ork_battle_cries = self.da_token_embeddin(da_input_tokens)

        # STEP 2: ADD POSITIONAL AWARENESS - every Ork knows their place!
        da_position_indices = torch.arange(da_current_seq_len, device=da_input_tokens.device).unsqueeze(0)
        da_positional_formation = self.da_positional_embeddin(da_position_indices)
        da_position_aware_thoughts = da_ork_battle_cries + da_positional_formation

        # STEP 3: DROPOUT FOR ROBUSTNESS - some Orks get distracted
        da_robust_thoughts = self.da_ork_distraction(da_position_aware_thoughts)

        # STEP 4: PROCESS THROUGH DA MOE LAYERS - multiple clan coordination systems!
        da_processed_thoughts = da_robust_thoughts
        for da_moe_layer in self.da_moe_layers:
            da_processed_thoughts = da_moe_layer.coordinate_da_ork_clans(da_processed_thoughts)

        # STEP 5: FINAL NORMALIZATION - polish da Ork thoughts
        da_polished_thoughts = self.da_final_ork_norm(da_processed_thoughts)

        # STEP 6: GET DA LOGITS - predict da next mighty victory!
        da_victory_predictions = self.da_output_proj(da_polished_thoughts)

        return da_victory_predictions

def create_orky_moe_model(
    da_vocab_size: int = 50000,
    da_orky_model_size: int = 512,
    da_num_layers: int = 8,
    da_num_experts: int = 8,
    da_expert_hidden_size: int = 1024,
    da_max_seq_len: int = 2048
) -> OrkyMoEModel:
    """
    CREATE A FULL-SIZED MOE MODEL FOR REAL BATTLES!

    Dis creates a big MoE model ready for serious sequence processing.
    Like fielding a full Ork army with multiple specialized clans!
    """
    print(f"🐍⚡ Creating Orky MoE Model!")
    print(f"   Vocab Size: {da_vocab_size}")
    print(f"   Model Dim: {da_orky_model_size}")
    print(f"   Layers: {da_num_layers}")
    print(f"   Experts: {da_num_experts}")
    print(f"   Expert Hidden: {da_expert_hidden_size}")
    print(f"   Max Seq Len: {da_max_seq_len}")

    model = OrkyMoEModel(
        da_vocab_size=da_vocab_size,
        da_orky_model_size=da_orky_model_size,
        da_num_layers=da_num_layers,
        da_num_experts=da_num_experts,
        da_expert_hidden_size=da_expert_hidden_size,
        da_max_seq_len=da_max_seq_len,
    )

    # Count da parameters like countin' da horde
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Model Size: {total_params * 4 / (1024**2):.1f} MB")

    return model

if __name__ == "__main__":
    # QUICK TEST OF DA MOE MODEL
    print("🐍⚡ Testing da MoE Model! ⚡🐍")

    # Create a small model for testing
    model = create_orky_moe_model(
        da_vocab_size=100,
        da_orky_model_size=64,
        da_num_layers=2,
        da_num_experts=4,
        da_expert_hidden_size=128,
        da_max_seq_len=32
    )

    # Test forward pass
    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, 100, (batch_size, seq_len))

    print(f"\nInput shape: {input_ids.shape}")
    print(f"Input tokens: {input_ids[0].tolist()}")

    logits = model.unleash_da_moe_waaagh(input_ids)
    print(f"Output shape: {logits.shape}")

    print("\n✅ MoE model test complete! WAAAGH!")
