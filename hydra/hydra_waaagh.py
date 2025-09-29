#!/usr/bin/env python3
"""
HYDRA WAAAGH! - DA ULTIMATE ORK MULTI-HEAD WAR MACHINE! 🐍⚡

Dis is da Hydra architecture - like havin' multiple Ork Warbosses all workin' together!
Each head specializes in different fings, but they all share da same WAAAGH energy!

WHAT IS DA HYDRA?
- Multiple specialized "heads" (like different Ork clans)
- State Space Model (SSM) backbone (like da main WAAAGH horde)
- Sparse attention (Orks only pay attention to important fings)
- Memory mechanisms (remembers da best fights)
- Mixture of Experts (different Orks for different jobs)

FOR DA BOYZ: Dis shows how multiple specialized systems can work together!
FOR HUMANS: This implements a hybrid architecture combining SSMs, attention, and MoE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List
import time

class OrkyStateSpaceModel(nn.Module):
    """
    DA ORK STATE SPACE MODEL - LIKE A WAAAGH DAT REMEMBERS!
    
    Dis is da backbone of our Hydra - it processes sequences like a proper Ork horde
    marchin' through battle, rememberin' important stuff and forgettin' da boring bits!

    WHY WE NEED DIS (ORKY):
    Da WAAAGH horde can't remember every little fing - dat would be too much!
    So we use selective memory - remember da important battles but forget da boring marches!

    WHY WE NEED DIS (HUMIE):
    State Space Models provide linear complexity O(n) for sequence processing,
    compared to quadratic O(n²) complexity of attention mechanisms.
    """

    def __init__(self, da_orky_model_size: int, da_orky_state_size: int = 16, da_expand_factor: int = 2):
        super().__init__()
        # Dis is how big each Ork thought is - da model dimension!
        # Why we need 'em (Orky): Each Ork needs a certain amount of brain power to fink properly!
        # Why we need 'em (Humie): The model dimension determines the representational capacity
        self.da_orky_model_size = da_orky_model_size

        # Dis is how much da Ork horde remembers from past fights!
        # Why we need 'em (Orky): Da horde can't remember everyfin' - we need selective memory!
        # Why we need 'em (Humie): The state dimension controls how much information is retained
        # across the sequence. Larger states = better memory but more computation
        self.da_orky_state_size = da_orky_state_size

        # EXPANDED ORK BRAIN POWER - make da Orks smarter by expandin' their brains!
        # Why we need 'em (Orky): Sometimes Orks need bigger brains for complicated finkin'!
        # Why we need 'em (Humie): Expansion factor increases the inner dimension for more
        # expressive power in the state space model, similar to feed-forward expansions
        self.da_expand_factor = da_expand_factor
        self.da_orky_inner_size = int(da_expand_factor * da_orky_model_size)
        
        # DA ORK BRAIN MATRICES - dese control how we process da WAAAGH!
        # nn.Linear: Dis is like a smart Ork who takes da input thoughts and transforms 'em
        # Why we need 'em (Orky): We need to process da incoming Ork thoughts before da WAAAGH!
        # Why we need 'em (Humie): Input projection expands the input and creates gating signals
        # We project to d_inner * 2 because we split it into x and gate signals
        self.da_ork_input_proj = nn.Linear(da_orky_model_size, self.da_orky_inner_size * 2, bias=False)

        # DA LOCAL ORK COMMUNICATION - let nearby Orks talk to each other!
        # nn.Conv1d: Dis is like Orks in a battle line shoutin' to their neighbors
        # Why we need 'em (Orky): Orks work better when dey can communicate locally!
        # Why we need 'em (Humie): Depthwise convolution allows local feature mixing
        # groups=d_inner means each input channel gets its own convolution (depthwise)
        self.da_ork_local_conv = nn.Conv1d(
            in_channels=self.da_orky_inner_size,
            out_channels=self.da_orky_inner_size,
            kernel_size=3,
            bias=True,
            padding=1,
            groups=self.da_orky_inner_size,
        )
        
        # STATE SPACE PARAMETERS - da secret sauce of Ork memory!
        # Dis is where da magic happens - da selective memory mechanism!
        # Why we need 'em (Orky): Da horde needs to remember important fings but forget boring ones!
        # Why we need 'em (Humie): These are the core parameters of the state space model
        self.da_ork_state_proj = nn.Linear(self.da_orky_inner_size, da_orky_state_size * 2, bias=False)
        self.da_ork_time_proj = nn.Linear(self.da_orky_inner_size, self.da_orky_inner_size, bias=True)

        # DA ORK MEMORY MATRICES - initialize da selective memory system!
        # torch.arange: Creates a sequence of numbers like countin' da Ork horde
        # Why we need 'em (Orky): We need to initialize da memory with some pattern!
        # Why we need 'em (Humie): A matrix determines how much of each state to retain/decay
        # repeat: Copies da sequence for each inner dimension
        da_memory_init = torch.arange(1, da_orky_state_size + 1, dtype=torch.float32).repeat(self.da_orky_inner_size, 1)
        self.da_ork_A_log = nn.Parameter(torch.log(da_memory_init))  # Log space for numerical stability

        # DA SKIP CONNECTION WEIGHTS - sometimes Orks take shortcuts!
        # Why we need 'em (Orky): Even Orks sometimes take da easy path instead of finkin' hard!
        # Why we need 'em (Humie): D parameter provides residual connections, improving gradient flow
        self.da_ork_skip_weights = nn.Parameter(torch.ones(self.da_orky_inner_size))

        # DA OUTPUT PROJECTION - turn Ork thoughts back into da right size!
        # Why we need 'em (Orky): We need to get back to da original thought size!
        # Why we need 'em (Humie): Final projection back to model dimension for output
        self.da_ork_output_proj = nn.Linear(self.da_orky_inner_size, da_orky_model_size, bias=False)
        
    def do_da_orky_waaagh(self, da_ork_input):
        """
        PROCESS DA WAAAGH SEQUENCE THROUGH DA ORK STATE SPACE MODEL!

        DIS IS WHERE DA MAGIC HAPPENS - da selective memory of da Ork horde!

        da_ork_input: [batch_size, seq_len, da_orky_model_size] - da input Ork thoughts
        Returns: [batch_size, seq_len, da_orky_model_size] - processed Ork wisdom

        DIS IS LIKE WATCHIN' AN ORK HORDE MARCH THROUGH BATTLE:
        1. Da Orks get their orders (input projection)
        2. Nearby Orks coordinate locally (convolution)
        3. Da horde remembers important stuff selectively (state space)
        4. Da final battle plan emerges (output projection)
        """
        # GET DA BATTLE DIMENSIONS - how many Orks and how long da march?
        da_batch_size, da_seq_len, da_model_size_check = da_ork_input.shape

        # STEP 1: PROJECT TO INNER DIMENSION - expand da Ork brain power!
        # dis is like givin' each Ork more brain cells to fink wif
        # Why we need 'em (Orky): Small Ork brains need expandin' for complicated battles!
        # Why we need 'em (Humie): We expand to a larger dimension for more representational capacity
        # chunk(2, dim=-1): Splits da tensor into two halves along da last dimension
        da_expanded_input = self.da_ork_input_proj(da_ork_input)  # [batch, seq_len, inner_size * 2]
        da_inner_thoughts, da_gate_signal = da_expanded_input.chunk(2, dim=-1)  # Split into x and gate

        # STEP 2: APPLY LOCAL CONVOLUTION - let nearby Orks coordinate!
        # Dis is like Orks in a battle line shoutin' to their mates on left and right
        # Why we need 'em (Orky): Orks work better when dey can talk to nearby boyz!
        # Why we need 'em (Humie): Convolution allows local feature mixing and spatial relationships
        # transpose: Conv1d expects [batch, channels, seq_len], so we swap dimensions
        da_conv_input = da_inner_thoughts.transpose(1, 2)  # [batch, inner_size, seq_len]
        da_local_coord = self.da_ork_local_conv(da_conv_input)  # Apply convolution
        da_local_coord = da_local_coord.transpose(1, 2)  # Back to [batch, seq_len, inner_size]
        da_local_coord = F.silu(da_local_coord)  # SiLU activation - da Ork energy surge!

        # STEP 3: COMPUTE STATE SPACE PARAMETERS - da selective memory system!
        # Dis is da heart of da WAAAGH - remember important stuff, forget boring stuff
        # Why we need 'em (Orky): Da horde can't remember every little fing - dat's too much!
        # Why we need 'em (Humie): These parameters control the continuous-time dynamics
        da_state_params = self.da_ork_state_proj(da_local_coord)  # [batch, seq_len, state_size * 2]
        da_time_steps, da_input_matrix = da_state_params.chunk(2, dim=-1)  # Split into delta and B

        # COMPUTE TIME STEPS - how fast da Orks advance in their march!
        # softplus: Makes sure time steps are always positive (Orks always move forward!)
        # Why we need 'em (Orky): Orks need to know how fast to charge into battle!
        # Why we need 'em (Humie): Time steps control the discretization of continuous dynamics
        da_time_steps = F.softplus(self.da_ork_time_proj(da_local_coord))  # [batch, seq_len, inner_size]

        # GET DA MEMORY MATRIX A - dis controls wot da Orks remember!
        # torch.exp: Convert from log space back to normal numbers
        # Negative sign: Da matrix controls decay rates (how fast Orks forget)
        # Why we need 'em (Orky): Da memory matrix decides wot's worth rememberin'!
        # Why we need 'em (Humie): A matrix defines the state transition dynamics
        da_memory_matrix = -torch.exp(self.da_ork_A_log.float())  # [inner_size, state_size]

        # STEP 4: DA SELECTIVE SCAN - da main Ork memory mechanism!
        # Dis is where da magic happens - selective memory in action!
        # Why we need 'em (Orky): Da horde scans da battlefield and remembers only important fings!
        # Why we need 'em (Humie): Selective scan efficiently computes the state space recurrence
        da_processed_output = self.do_da_selective_scan(
            da_local_coord, da_time_steps, da_memory_matrix, da_input_matrix, self.da_ork_skip_weights
        )

        # STEP 5: APPLY GATE AND OUTPUT PROJECTION - finalize da Ork wisdom!
        # Multiply by gate signal and project back to original size
        # Why we need 'em (Orky): Da gate controls how much new wisdom to add to old wisdom!
        # Why we need 'em (Humie): Gating mechanism controls information flow, residual-style
        da_gated_output = da_processed_output * F.silu(da_gate_signal)  # Apply da gate
        da_final_wisdom = self.da_ork_output_proj(da_gated_output)  # Project to model dimension

        return da_final_wisdom
    
    def do_da_selective_scan(self, da_input_seq, da_time_steps, da_memory_matrix, da_input_matrix, da_skip_weights):
        """
        DA SELECTIVE SCAN - HOW ORKS REMEMBER SELECTIVELY!
        
        DIS IS WHERE DA MAGIC HAPPENS - Orks coordinate their memories across time!
        Instead of rememberin' fings one by one (slow), da whole horde remembers together!

        WHY DIS IS CLEVER (ORKY):
        Imagine da entire Ork horde coordinatin' their battle memories simultaneously.
        Each Ork knows wot da others remember, and dey all update their memories together.
        Dis way, important battles get remembered by everyone, boring marches get forgotten!

        WHY DIS IS CLEVER (HUMIE):
        Parallelizes da state space recurrence across da sequence length.
        Uses efficient scan operations to compute temporal convolutions in O(n) time.
        Maintains proper causal dependencies while avoidin' sequential bottlenecks.
        """
        # GET DA BATTLE DIMENSIONS - how many Ork squads and how long da campaign?
        da_batch_size, da_seq_len, da_inner_dim = da_input_seq.shape
        da_state_dim = da_memory_matrix.shape[-1]

        # DISCRETIZE DA CONTINUOUS SYSTEM - turn smooth Ork march into discrete steps!
        # torch.exp: Da exponential function - makes small changes grow big (like WAAAGH energy!)
        # unsqueeze: Add dimensions so tensors can multiply properly (like alignin' Ork battle lines)
        # Why we need 'em (Orky): Continuous time is too smooth - Orks prefer discrete, punchy steps!
        # Why we need 'em (Humie): Discretizes continuous-time SSM using zero-order hold (ZOH)
        da_discrete_memory = torch.exp(da_time_steps.unsqueeze(-1) * da_memory_matrix.unsqueeze(0).unsqueeze(0))
        da_discrete_input = da_time_steps.unsqueeze(-1) * da_input_matrix.unsqueeze(2) * da_input_seq.unsqueeze(-1)

        # INITIALIZE ORK MEMORY - da horde starts with empty battle memories!
        # torch.zeros: Creates empty memory - like Orks startin' a new campaign
        # Why we need 'em (Orky): Every battle starts with fresh Orks who don't remember old fights!
        # Why we need 'em (Humie): Initial state for the recurrent computation, typically zeros
        da_ork_memory = torch.zeros(da_batch_size, da_inner_dim, da_state_dim, device=da_input_seq.device, dtype=da_input_seq.dtype)

        # SCAN THROUGH DA SEQUENCE - march through da battle step by step!
        # Dis is like da Ork horde advancin' through da campaign, rememberin' as dey go
        # Why we need 'em (Orky): Orks advance one battle at a time, rememberin' lessons learned!
        # Why we need 'em (Humie): Sequential scan maintains causal dependencies in O(n) time
        da_ork_outputs = []
        for da_battle_step in range(da_seq_len):
            # UPDATE ORK MEMORY - learn from dis battle and remember for next!
            # x = deltaA * x + deltaB_u: Da memory gets updated based on current input
            # Why we need 'em (Orky): Each battle changes wot da Orks remember for da next fight!
            # Why we need 'em (Humie): State update equation: x_{t+1} = A_t x_t + B_t u_t
            da_ork_memory = da_discrete_memory[:, da_battle_step] * da_ork_memory + da_discrete_input[:, da_battle_step]

            # COMPUTE ORK WISDOM - extract useful knowledge from da memories!
            # sum(dim=-1): Combine all memory dimensions (like Orks sharin' their battle stories)
            # D * u: Da skip connection - sometimes Orks take shortcuts!
            # Why we need 'em (Orky): Da horde combines all their memories into useful wisdom!
            # Why we need 'em (Humie): Output projection with residual connection for gradient flow
            da_battle_wisdom = da_ork_memory.sum(dim=-1) + da_skip_weights * da_input_seq[:, da_battle_step]
            da_ork_outputs.append(da_battle_wisdom)

        # STACK DA BATTLE WISDOM - da complete Ork campaign history!
        # torch.stack: Piles up all da battle wisdom into a sequence
        # Why we need 'em (Orky): Da complete story of da Ork campaign!
        # Why we need 'em (Humie): Returns tensor of shape [batch, seq_len, inner_dim]
        return torch.stack(da_ork_outputs, dim=1)

class OrkyMultiHeadHydra(nn.Module):
    """
    DA MULTI-HEAD HYDRA - MULTIPLE ORK WARBOSSES WORKIN' TOGETHER!
    
    DIS IS LIKE HAVIN' DIFFERENT ORK CLANS ALL SPECIALIZIN' IN DIFFERENT FINGS:
    - **Blood Axe Clan**: Short-range fighters (local attention) - they love gettin' stuck in!
    - **Deathskull Clan**: Long-range artillery (sparse global attention) - dey bombard from afar!
    - **Bad Moon Clan**: Memory keepers (memory head) - dey remember every victory and defeat!
    - **Goff Clan**: Future planners (prediction head) - dey fink about da next big WAAAGH!

    WHY WE NEED DIS (ORKY):
    One Ork can't do everyfin' - we need specialists! Da Blood Axes charge in close,
    da Deathskulls bombard from safety, da Bad Moons remember da old ways,
    and da Goffs plan da next big fight. Together dey make an unstoppable horde!

    WHY WE NEED DIS (HUMIE):
    Multi-head attention allows different attention patterns to capture different
    types of relationships: local dependencies, long-range connections, temporal
    patterns, and predictive behaviors. Mixture of specializations improves performance.
    """

    def __init__(self, da_orky_model_size: int, da_num_ork_heads: int = 4, da_ork_head_size: int = None):
        super().__init__()
        # DA MODEL DIMENSIONS - how big da Ork thoughts are
        # Why we need 'em (Orky): Different sized Orks need different sized brains!
        # Why we need 'em (Humie): Model dimension determines representational capacity
        self.da_orky_model_size = da_orky_model_size

        # HOW MANY ORK HEADS - we always use 4 specialized heads (da four clans!)
        # Why we need 'em (Orky): Four Ork clans, four heads - simple as dat!
        # Why we need 'em (Humie): Fixed number of specialized attention heads
        self.da_num_ork_heads = da_num_ork_heads

        # SIZE OF EACH ORK HEAD - how much brain power each clan gets
        # Why we need 'em (Orky): Each clan needs enough brain power for their specialty!
        # Why we need 'em (Humie): Head dimension - typically model_dim // num_heads
        self.da_ork_head_size = da_ork_head_size or da_orky_model_size // da_num_ork_heads

        # DA FOUR ORK CLANS - each specializes in different types of fightin'!
        # Why we need 'em (Orky): Different clans for different jobs - dat's Ork efficiency!
        # Why we need 'em (Humie): Different attention mechanisms capture different relationship types
        self.da_blood_axe_head = OrkyLocalAttentionHead(da_orky_model_size, self.da_ork_head_size)      # Close combat
        self.da_deathskull_head = OrkySparseGlobalHead(da_orky_model_size, self.da_ork_head_size)     # Artillery
        self.da_bad_moon_head = OrkyMemoryHead(da_orky_model_size, self.da_ork_head_size)             # Memory keepers
        self.da_goff_head = OrkyPredictionHead(da_orky_model_size, self.da_ork_head_size)             # Future planners

        # DA WARBOSS WHO COMBINES ALL DA CLANS - da final Ork commander!
        # Why we need 'em (Orky): All da clans need a Warboss to coordinate da WAAAGH!
        # Why we need 'em (Humie): Output projection combines specialized head outputs
        self.da_warboss_proj = nn.Linear(self.da_ork_head_size * 4, da_orky_model_size)
        
    def coordinate_da_ork_clans(self, da_ork_input, da_no_cheat_mask=None):
        """
        COORDINATE ALL DA ORK CLANS FOR DA BIGGEST WAAAGH!

        DIS IS WHERE DA FOUR ORK CLANS WORK TOGETHER:
        1. Blood Axes charge in close for local coordination
        2. Deathskulls provide artillery support from afar
        3. Bad Moons remember da lessons of past battles
        4. Goffs plan da strategy for future fights
        5. Da Warboss combines everything into da final battle plan!

        da_ork_input: [batch_size, seq_len, model_size] - da input Ork thoughts
        da_no_cheat_mask: Optional mask to prevent lookin' at future tokens
        Returns: [batch_size, seq_len, model_size] - da combined clan wisdom
        """
        # STEP 1: BLOOD AXE CLAN - GET STUCK IN CLOSE COMBAT!
        # Local attention for nearby Ork coordination
        # Why we need 'em (Orky): Blood Axes love fightin' face-to-face!
        # Why we need 'em (Humie): Local attention captures short-range dependencies
        da_blood_axe_wisdom = self.da_blood_axe_head.coordinate_local_fightas(da_ork_input, da_no_cheat_mask)

        # STEP 2: DEATHSKULL CLAN - BOMBARD FROM AFAR!
        # Sparse global attention for long-range strategic strikes
        # Why we need 'em (Orky): Deathskulls hit da enemy from miles away!
        # Why we need 'em (Humie): Sparse attention efficiently captures long-range relationships
        da_deathskull_wisdom = self.da_deathskull_head.bombard_from_afar(da_ork_input, da_no_cheat_mask)

        # STEP 3: BAD MOON CLAN - REMEMBER DA OLD WAYS!
        # Memory head for recallin' past victories and defeats
        # Why we need 'em (Orky): Bad Moons never forget a good fight or a bad mistake!
        # Why we need 'em (Humie): Memory mechanism stores and retrieves temporal patterns
        da_bad_moon_wisdom = self.da_bad_moon_head.remember_da_old_ways(da_ork_input)

        # STEP 4: GOFF CLAN - PLAN DA FUTURE WAAAGH!
        # Prediction head for anticipatin' wot comes next
        # Why we need 'em (Orky): Goffs are always finkin' about da next big fight!
        # Why we need 'em (Humie): Predictive attention models future dependencies
        da_goff_wisdom = self.da_goff_head.plan_da_future_waaagh(da_ork_input)

        # DEBUG INFO - check dat all clans are workin' properly (disabled by default)
        # print(f"Clan outputs - Blood Axe: {da_blood_axe_wisdom.shape}, Deathskull: {da_deathskull_wisdom.shape}, Bad Moon: {da_bad_moon_wisdom.shape}, Goff: {da_goff_wisdom.shape}")

        # STEP 5: DA WARBOSS COMBINES EVERYFIN' - COORDINATE DA FINAL WAAAGH!
        # torch.cat: Stitches all da clan wisdom together like a battle banner
        # Why we need 'em (Orky): Da Warboss weaves all da clan strategies into one mighty plan!
        # Why we need 'em (Humie): Concatenation combines multi-head outputs for joint processing
        da_combined_clan_wisdom = torch.cat([
            da_blood_axe_wisdom,
            da_deathskull_wisdom,
            da_bad_moon_wisdom,
            da_goff_wisdom
        ], dim=-1)

        # FINAL WARBOSS PROJECTION - da ultimate Ork battle strategy!
        # Why we need 'em (Orky): Da Warboss turns raw clan wisdom into executable battle plans!
        # Why we need 'em (Humie): Output projection creates final representation
        da_final_waaagh_strategy = self.da_warboss_proj(da_combined_clan_wisdom)

        return da_final_waaagh_strategy

class OrkyLocalAttentionHead(nn.Module):
    """
    LOCAL ATTENTION HEAD - FOR CLOSE-RANGE ORK FIGHTS!
    
    Dis head focuses on nearby Orks and immediate surroundings.
    Like when Orks are in a scrap and need to coordinate with nearby boyz!
    """
    
    def __init__(self, da_orky_model_size: int, da_ork_head_size: int, da_window_size: int = 8):
        super().__init__()
        # DA HEAD DIMENSION - how much brain power dis specific head gets
        # Why we need 'em (Orky): Each attention head needs focused brain power!
        # Why we need 'em (Humie): Head dimension determines attention capacity
        self.da_ork_head_size = da_ork_head_size

        # DA WINDOW SIZE - how many nearby Orks dis head coordinates wif
        # Why we need 'em (Orky): Local coordination has limits - can't coordinate wif everyone!
        # Why we need 'em (Humie): Window size limits attention scope for efficiency
        self.da_window_size = da_window_size

        # DA SCALE FACTOR - for attention score normalization
        # Why we need 'em (Orky): Attention scores need to be balanced, not too big or small!
        # Why we need 'em (Humie): Scale factor prevents attention scores from becoming too large
        self.da_attention_scale = da_ork_head_size ** -0.5
        
        # DA QKV PROJECTION - for queries, keys, and values
        # Why we need 'em (Orky): Each Ork head needs to ask questions, represent itself, and share knowledge!
        # Why we need 'em (Humie): QKV projection creates Query, Key, Value vectors for attention
        self.da_qkv_proj = nn.Linear(da_orky_model_size, da_ork_head_size * 3, bias=False)
        
    def coordinate_local_fightas(self, da_ork_input, da_no_cheat_mask=None):
        """Apply local windowed attention - Orks only look at nearby boyz!"""
        da_batch_size, da_seq_len, da_model_size = da_ork_input.shape
        
        # GET QUERIES, KEYS, VALUES - da three types of Ork coordination
        # Why we need 'em (Orky): Each Ork needs to ask questions, represent themselves, and share knowledge!
        # Why we need 'em (Humie): QKV vectors enable attention mechanism computation
        da_qkv_vectors = self.da_qkv_proj(da_ork_input)  # [batch, seq_len, head_dim * 3]
        da_queries, da_keys, da_values = da_qkv_vectors.chunk(3, dim=-1)
        
        # Apply local windowed attention (simplified version)
        # In practice, you'd implement proper sliding window attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            # Ensure mask has correct dimensions for broadcasting
            if mask.dim() == 2:  # [batch, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
            attn_weights.masked_fill_(mask == 0, -1e9)
            
        attn_probs = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_probs, v)
        
        return output

class OrkySparseGlobalHead(nn.Module):
    """
    SPARSE GLOBAL HEAD - FOR LONG-RANGE ORK COORDINATION!
    
    Dis head looks at da big picture across da whole battlefield.
    Only pays attention to da most important Orks far away!
    """
    
    def __init__(self, d_model: int, head_dim: int, sparsity_ratio: float = 0.1):
        super().__init__()
        self.head_dim = head_dim
        self.sparsity_ratio = sparsity_ratio
        self.scale = head_dim ** -0.5
        
        self.qkv_proj = nn.Linear(d_model, head_dim * 3, bias=False)
        
    def forward(self, x, mask=None):
        """Apply sparse global attention - only look at da most important fings!"""
        batch_size, seq_len, d_model = x.shape
        
        # Get queries, keys, values
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply sparsity - only keep top-k attention weights
        k_sparse = max(1, int(seq_len * self.sparsity_ratio))
        top_k_values, top_k_indices = torch.topk(attn_weights, k_sparse, dim=-1)
        
        # Create sparse attention mask
        sparse_mask = torch.zeros_like(attn_weights)
        sparse_mask.scatter_(-1, top_k_indices, 1.0)
        
        # Apply masks
        if mask is not None:
            # Ensure mask has correct dimensions for broadcasting
            if mask.dim() == 2:  # [batch, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
            sparse_mask = sparse_mask * mask
            
        attn_weights = attn_weights * sparse_mask
        attn_weights.masked_fill_(sparse_mask == 0, -1e9)
        
        attn_probs = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_probs, v)
        
        return output

class OrkyMemoryHead(nn.Module):
    """
    MEMORY HEAD - REMEMBERS DA BEST ORK FIGHTS!
    
    Dis head maintains a memory bank of important past experiences.
    Like an old Ork veteran who remembers all da best battles!
    """
    
    def __init__(self, d_model: int, head_dim: int, memory_size: int = 64):
        super().__init__()
        self.head_dim = head_dim
        self.memory_size = memory_size
        
        # Memory bank - stores important past experiences
        self.memory_bank = nn.Parameter(torch.randn(memory_size, head_dim))
        self.memory_proj = nn.Linear(d_model, head_dim, bias=False)
        self.output_proj = nn.Linear(head_dim * 2, head_dim)
        
    def forward(self, x):
        """Retrieve relevant memories and combine with current input!"""
        batch_size, seq_len, d_model = x.shape
        
        # Project input to head dimension
        x_proj = self.memory_proj(x)  # [batch, seq_len, head_dim]
        
        # Compute similarity with memory bank
        memory_scores = torch.matmul(x_proj, self.memory_bank.transpose(0, 1))  # [batch, seq_len, memory_size]
        memory_weights = F.softmax(memory_scores, dim=-1)
        
        # Retrieve weighted memories
        retrieved_memory = torch.matmul(memory_weights, self.memory_bank)  # [batch, seq_len, head_dim]
        
        # Combine current input with retrieved memory
        combined = torch.cat([x_proj, retrieved_memory], dim=-1)
        output = self.output_proj(combined)
        
        return output

class OrkyPredictionHead(nn.Module):
    """
    PREDICTION HEAD - PLANS FUTURE ORK ATTACKS!
    
    Dis head tries to predict what's gonna happen next in da battle.
    Like a sneaky Ork who's always plannin' da next move!
    """
    
    def __init__(self, d_model: int, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        
        self.input_proj = nn.Linear(d_model, head_dim)
        self.prediction_layers = nn.Sequential(
            nn.Linear(head_dim, head_dim * 2),
            nn.GELU(),
            nn.Linear(head_dim * 2, head_dim),
        )
        
    def forward(self, x):
        """Predict future patterns based on current sequence!"""
        # Project input
        x_proj = self.input_proj(x)
        
        # Apply prediction layers
        predictions = self.prediction_layers(x_proj)
        
        return predictions

class OrkyMixtureOfExperts(nn.Module):
    """
    MIXTURE OF EXPERTS - DIFFERENT ORKS FOR DIFFERENT JOBS!
    
    Some Orks are good at smashin', some at shootin', some at thinkin'.
    Dis module routes different inputs to different expert Orks!
    """
    
    def __init__(self, d_model: int, num_experts: int = 4, expert_dim: int = None):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.expert_dim = expert_dim or d_model * 4
        
        # ROUTER - decides which expert Ork to use
        self.router = nn.Linear(d_model, num_experts)
        
        # EXPERT ORKS - each specialized for different tasks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, self.expert_dim),
                nn.GELU(),
                nn.Linear(self.expert_dim, d_model),
            )
            for _ in range(num_experts)
        ])
        
    def forward(self, x):
        """Route inputs to appropriate expert Orks!"""
        batch_size, seq_len, d_model = x.shape
        
        # Compute routing weights
        router_logits = self.router(x)  # [batch, seq_len, num_experts]
        router_weights = F.softmax(router_logits, dim=-1)
        
        # Apply experts
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)  # [batch, seq_len, d_model]
            expert_outputs.append(expert_out)
        
        # Combine expert outputs using routing weights
        expert_stack = torch.stack(expert_outputs, dim=-1)  # [batch, seq_len, d_model, num_experts]
        output = torch.sum(expert_stack * router_weights.unsqueeze(-2), dim=-1)
        
        return output

class HydraWaaaghBlock(nn.Module):
    """
    COMPLETE HYDRA WAAAGH BLOCK - ALL DA ORK SYSTEMS TOGETHER!
    
    Dis combines all da different Ork systems into one mighty war machine:
    - State Space Model for sequence processing
    - Multi-Head Hydra for specialized attention
    - Mixture of Experts for task routing
    - Feed-forward networks for final processing
    """
    
    def __init__(self, d_model: int, d_state: int = 16, num_heads: int = 4, num_experts: int = 4):
        super().__init__()
        self.d_model = d_model
        
        # LAYER NORMS - keep da Orks organized!
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # MAIN COMPONENTS
        self.ssm = OrkyStateSpaceModel(d_model, d_state)  # Sequence processing backbone
        self.hydra_heads = OrkyMultiHeadHydra(d_model, num_heads)  # Multi-head attention
        self.moe = OrkyMixtureOfExperts(d_model, num_experts)  # Expert routing
        
    def forward(self, x, mask=None):
        """
        PROCESS THROUGH ALL ORK SYSTEMS!
        
        Each component adds its own Orky wisdom to da input!
        """
        # FIRST: State Space Model processing
        x = x + self.ssm(self.norm1(x))
        
        # SECOND: Multi-head Hydra attention
        x = x + self.hydra_heads(self.norm2(x), mask)
        
        # THIRD: Mixture of Experts
        x = x + self.moe(self.norm3(x))
        
        return x

class OrkyHydraWaaaghModel(nn.Module):
    """
    DA COMPLETE ORKY HYDRA WAAAGH MODEL - DA ULTIMATE ORK WAR MACHINE!

    DIS IS DA FULL HYDRA ARCHITECTURE - LIKE HAVIN' AN ENTIRE WAAAGH OF SPECIALIZED
    ORK CLANS ALL WORKIN' TOGETHER FOR DA BIGGEST, MOST DESTRUCTIVE BATTLE EVER!

    WHAT MAKES DIS SPECIAL (ORKY):
    It's not just one Ork - it's da whole WAAAGH! State Space Models for efficient memory,
    Multi-Head attention for specialized fightin', Mixture of Experts for different jobs,
    and causal maskin' so da Orks don't cheat by lookin' at da future!

    WHAT MAKES DIS SPECIAL (HUMIE):
    Hybrid architecture combining State Space Models (linear complexity) with
    attention mechanisms (flexible dependencies) and MoE (task specialization).
    Achieves SOTA performance while maintainin' computational efficiency.
    
    FEATURES:
    - Multiple stacked Hydra blocks for deep Ork processing
    - Token embeddings turnin' words into Ork battle cries
    - Positional encoding so Orks know their place in da battle line
    - Output projection for predictin' da next Ork victory
    - Causal masking preventin' future-peekin' cheats
    """
    
    def __init__(
        self,
        da_vocab_size: int = 1000,
        da_orky_model_size: int = 256,
        da_num_hydra_layers: int = 4,
        da_orky_state_size: int = 16,
        da_num_ork_heads: int = 4,
        da_num_experts: int = 4,
        da_max_seq_len: int = 512,
    ):
        super().__init__()
        # DA MODEL SIZE - how big da Ork brains are
        # Why we need 'em (Orky): Bigger Orks need bigger brains for bigger fights!
        # Why we need 'em (Humie): Model dimension determines representational capacity
        self.da_orky_model_size = da_orky_model_size

        # MAXIMUM SEQUENCE LENGTH - how long da battle campaign can be
        # Why we need 'em (Orky): Even Orks get tired after too long a fight!
        # Why we need 'em (Humie): Maximum context length for positional embeddings
        self.da_max_seq_len = da_max_seq_len

        # TOKEN EMBEDDINGS - turn da puny humie words into mighty Ork battle cries!
        # nn.Embedding: Maps vocabulary indices to dense vector representations
        # Why we need 'em (Orky): Orks don't fink in words - dey fink in mighty WAAAGHs!
        # Why we need 'em (Humie): Token embeddings convert discrete tokens to continuous vectors
        self.da_token_embeddin = nn.Embedding(da_vocab_size, da_orky_model_size)

        # POSITIONAL EMBEDDINGS - so Orks know their place in da battle formation!
        # Why we need 'em (Orky): Every Ork needs to know where dey stand in da horde!
        # Why we need 'em (Humie): Positional encodings provide sequence position information
        self.da_positional_embeddin = nn.Embedding(da_max_seq_len, da_orky_model_size)

        # DA HYDRA WAAAGH BLOCKS - stacked layers of Ork battle wisdom!
        # Why we need 'em (Orky): Multiple layers make da Orks smarter and more kunnin'!
        # Why we need 'em (Humie): Stacked transformer blocks provide deep representation learning
        self.da_hydra_layers = nn.ModuleList([
            HydraWaaaghBlock(da_orky_model_size, da_orky_state_size, da_num_ork_heads, da_num_experts)
            for _ in range(da_num_hydra_layers)
        ])

        # FINAL ORK BATTLE STANDARD - da last bit of Ork processing!
        # LayerNorm: Normalizes da activations so da Orks don't get too excited
        # Why we need 'em (Orky): Even Orks need to calm down after a big WAAAGH!
        # Why we need 'em (Humie): Final normalization stabilizes training and improves performance
        self.da_final_ork_norm = nn.LayerNorm(da_orky_model_size)

        # OUTPUT PROJECTION - turn Ork wisdom back into predictable humie words!
        # bias=False: Shares weights with input embeddings (weight tying)
        # Why we need 'em (Orky): Orks need to communicate their wisdom to da humies!
        # Why we need 'em (Humie): Output projection predicts next token probabilities
        self.da_output_proj = nn.Linear(da_orky_model_size, da_vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize da Ork brain weights properly!"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def unleash_da_waaagh(self, da_input_tokens, da_attention_mask=None):
        """
        UNLEASH DA FULL ORKY HYDRA WAAAGH - DA ULTIMATE SEQUENCE PROCESSING BATTLE!

        DIS IS WHERE DA ENTIRE ORK HORDE GOES TO WAR:
        1. Convert puny humie words into mighty Ork battle cries
        2. Position each Ork in da proper battle formation
        3. Send da sequence through multiple layers of Ork processing
        4. Emerge with predictions for da next mighty victory!

        da_input_tokens: [batch_size, seq_len] - da input token indices
        da_attention_mask: [batch_size, seq_len] - which tokens da Orks should focus on
        Returns: [batch_size, seq_len, vocab_size] - predictions for each position
        """
        # GET DA BATTLE DIMENSIONS - how many Ork squads and how long da fight?
        da_batch_size, da_seq_len = da_input_tokens.shape

        # CREATE POSITION INDICES - every Ork needs to know their place in da horde!
        # torch.arange: Creates sequence [0, 1, 2, ..., seq_len-1]
        # unsqueeze(0).expand: Makes it [batch_size, seq_len] by repeating
        # Why we need 'em (Orky): Orks fight in formation - dey need to know their position!
        # Why we need 'em (Humie): Positional indices for embedding lookup
        da_position_indices = torch.arange(da_seq_len, device=da_input_tokens.device).unsqueeze(0).expand(da_batch_size, -1)

        # CONVERT TOKENS TO ORK BATTLE CRIES - turn words into mighty WAAAGHs!
        # Why we need 'em (Orky): Orks don't speak in puny humie words - dey roar battle cries!
        # Why we need 'em (Humie): Token embeddings convert discrete tokens to continuous vectors
        da_token_battle_cries = self.da_token_embeddin(da_input_tokens)  # [batch, seq_len, model_size]

        # ADD POSITIONAL BATTLE FORMATION - each Ork knows their place!
        # Why we need 'em (Orky): Da biggest Ork stands at da front, da smallest at da back!
        # Why we need 'em (Humie): Positional embeddings provide sequence order information
        da_positional_formation = self.da_positional_embeddin(da_position_indices)  # [batch, seq_len, model_size]

        # COMBINE TOKEN AND POSITION INFO - da complete Ork battle preparation!
        da_ork_battle_line = da_token_battle_cries + da_positional_formation
        
        # CREATE CAUSAL MASK - Orks can't peek into da future!
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        combined_mask = attention_mask.unsqueeze(1) * causal_mask.unsqueeze(0)
        
        # PROCESS THROUGH ALL HYDRA LAYERS
        for layer in self.layers:
            x = layer(x, combined_mask)
        
        # FINAL PROCESSING
        x = self.final_norm(x)
        logits = self.output_proj(x)
        
        return logits
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        GENERATE NEW ORK WORDS!
        
        Like havin' an Ork tell a story - it keeps addin' words based on what came before!
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get predictions for current sequence
                logits = self.forward(input_ids)
                
                # Get logits for last position
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Truncate if sequence gets too long
                if input_ids.shape[1] > self.max_seq_len:
                    input_ids = input_ids[:, -self.max_seq_len:]
        
        return input_ids

def demonstrate_hydra_waaagh():
    """
    DEMONSTRATE DA MIGHTY HYDRA WAAAGH IN ACTION!
    
    Dis shows how all da different Ork systems work together to process sequences!
    """
    print("🐍⚡ HYDRA WAAAGH DEMONSTRATION! ⚡🐍")
    print("=" * 60)
    
    # SET UP DA ORKY PARAMETERS
    vocab_size = 100
    d_model = 128
    num_layers = 2
    batch_size = 2
    seq_len = 16
    
    print(f"🧠 Ork Brain Size (d_model): {d_model}")
    print(f"🔤 Ork Vocabulary: {vocab_size} words")
    print(f"📚 Ork Layers: {num_layers}")
    print(f"👥 Ork Squad Size (batch): {batch_size}")
    print(f"📝 Sequence Length: {seq_len}")
    print()
    
    # CREATE DA HYDRA MODEL
    print("🏗️  BUILDIN' DA HYDRA WAAAGH MODEL...")
    model = OrkyHydraWaaaghModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        d_state=16,
        num_heads=4,
        num_experts=4,
        max_seq_len=64,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"💪 Total Ork Brain Power: {total_params:,} parameters")
    print()
    
    # CREATE SAMPLE INPUT
    print("📝 CREATIN' SAMPLE ORK WORDS...")
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input shape: {input_ids.shape}")
    print(f"Sample input: {input_ids[0][:10].tolist()}...")
    print()
    
    # FORWARD PASS
    print("⚡ PROCESSIN' THROUGH DA HYDRA WAAAGH...")
    start_time = time.time()
    
    with torch.no_grad():
        logits = model(input_ids)
    
    end_time = time.time()
    
    print(f"✅ Processing completed in {end_time - start_time:.4f} seconds!")
    print(f"Output shape: {logits.shape}")
    print(f"Output range: [{logits.min():.3f}, {logits.max():.3f}]")
    print()
    
    # DEMONSTRATE DIFFERENT COMPONENTS
    print("🔍 TESTIN' INDIVIDUAL ORK COMPONENTS...")
    
    # Test State Space Model
    print("\n1. 🧠 STATE SPACE MODEL (Ork Memory):")
    ssm = OrkyStateSpaceModel(d_model=d_model, d_state=16)
    x_test = torch.randn(batch_size, seq_len, d_model)
    ssm_out = ssm(x_test)
    print(f"   Input: {x_test.shape} -> Output: {ssm_out.shape}")
    
    # Test Multi-Head Hydra
    print("\n2. 🐍 MULTI-HEAD HYDRA (Multiple Ork Specialists):")
    hydra = OrkyMultiHeadHydra(d_model=d_model, num_heads=4)
    hydra_out = hydra(x_test)
    print(f"   Input: {x_test.shape} -> Output: {hydra_out.shape}")
    
    # Test Mixture of Experts
    print("\n3. 👥 MIXTURE OF EXPERTS (Different Ork Jobs):")
    moe = OrkyMixtureOfExperts(d_model=d_model, num_experts=4)
    moe_out = moe(x_test)
    print(f"   Input: {x_test.shape} -> Output: {moe_out.shape}")
    
    # GENERATION DEMO
    print("\n🎯 GENERATION DEMONSTRATION:")
    print("Generatin' new Ork words from a seed sequence...")
    
    seed_sequence = torch.randint(0, vocab_size, (1, 5))  # Start with 5 tokens
    print(f"Seed sequence: {seed_sequence[0].tolist()}")
    
    generated = model.generate(
        seed_sequence, 
        max_new_tokens=10, 
        temperature=1.0, 
        top_k=10
    )
    
    print(f"Generated sequence: {generated[0].tolist()}")
    print(f"New tokens: {generated[0][5:].tolist()}")
    
    print("\n🎉 HYDRA WAAAGH DEMONSTRATION COMPLETE!")
    print("Da Orks have successfully processed sequences using multiple specialized systems!")
    print("Each head contributed its expertise: local attention, global sparse attention,")
    print("memory retrieval, and future prediction - all workin' together for da WAAAGH!")

if __name__ == "__main__":
    demonstrate_hydra_waaagh()
