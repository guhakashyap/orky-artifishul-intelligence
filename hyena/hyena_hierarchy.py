#!/usr/bin/env python3
"""
HYENA HIERARCHY! - DA CONVOLUTIONAL SEQUENCE WAR MACHINE! 🐍⚡

Dis is da Hyena architecture - like havin' da whole Ork horde communicatin' through
da WAAAGH energy field! No more slow attention - we use fast global convolutions!

WHAT IS DA HYENA?
- Global convolutions for efficient long-range dependencies (like WAAAGH field)
- Hierarchical processing at multiple scales (like Ork command structure)
- Linear complexity in sequence length (Orks don't do quadratic!)
- Implicit positional embeddings (da WAAAGH knows where everyone is)
- Stacked layers for deep processing (more Ork brains = more power!)

FOR DA BOYZ: Dis shows how convolutions can do what attention does, but faster!
FOR HUMANS: This implements the Hyena operator for efficient sequence processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List
import time

class OrkyHyenaOperator(nn.Module):
    """
    DA HYENA OPERATOR - DA HEART OF DA WAAAGH CONVOLUTIONS!

    Dis is da core Hyena operator - it uses global convolutions to process
    sequences efficiently. Like how da WAAAGH energy connects all Orks instantly!

    WHY WE NEED DIS (ORKY):
    Instead of slow attention where Orks have to look at every other Ork one by one,
    da Hyena uses global convolutions - like da WAAAGH energy field dat connects
    all Orks instantly across da entire battlefield! Maximum efficiency, maximum WAAAGH!

    WHY WE NEED DIS (HUMIE):
    Hyena operator achieves linear complexity O(n) for sequence processing using
    global convolutions instead of quadratic attention O(n²). Uses FFT for efficient
    long-range dependencies while maintaining parallel processing capabilities.
    """

    def __init__(self, da_orky_model_size: int, da_max_seq_len: int, da_conv_order: int = 2, da_filter_size: int = 64):
        super().__init__()
        # DA MODEL DIMENSION - how big da Ork thoughts are
        # Why we need 'em (Orky): Each Ork needs enough brain power for da WAAAGH!
        # Why we need 'em (Humie): Model dimension determines representational capacity
        self.da_orky_model_size = da_orky_model_size

        # MAXIMUM SEQUENCE LENGTH - how long da battle can be
        # Why we need 'em (Orky): Even da biggest WAAAGH has limits!
        # Why we need 'em (Humie): Maximum context length for positional embeddings
        self.da_max_seq_len = da_max_seq_len

        # CONVOLUTION ORDER - how many layers of WAAAGH energy
        # Why we need 'em (Orky): More layers = more powerful WAAAGH field!
        # Why we need 'em (Humie): Higher order convolutions capture more complex patterns
        self.da_conv_order = da_conv_order

        # FILTER SIZE - how big da WAAAGH energy patterns are
        # Why we need 'em (Orky): Bigger patterns = stronger WAAAGH connections!
        # Why we need 'em (Humie): Filter size determines receptive field and pattern complexity
        self.da_filter_size = da_filter_size

        # DA CONVOLUTIONAL FILTERS - dese are da WAAAGH energy patterns!
        # nn.ParameterList: A list of learnable parameters for da WAAAGH patterns
        # Why we need 'em (Orky): Da WAAAGH energy needs different patterns for different situations!
        # Why we need 'em (Humie): Multiple filter kernels capture different frequency components
        # torch.randn: Initialize with random WAAAGH energy (like Ork creativity!)
        # math.sqrt: Scale da energy so it don't explode (like controllin' da WAAAGH!)
        self.da_waaagh_filters = nn.ParameterList([
            nn.Parameter(torch.randn(da_filter_size, da_orky_model_size) / math.sqrt(da_orky_model_size))
            for _ in range(da_conv_order)
        ])

        # DA SHORT CONVOLUTION - for local Ork coordination
        # nn.Conv1d: 1D convolution for sequence processing
        # Why we need 'em (Orky): Nearby Orks need to coordinate their WAAAGH energy!
        # Why we need 'em (Humie): Local convolution captures short-range dependencies
        # groups=d_model: Depthwise convolution (each channel gets its own filter)
        # Why groups? (Orky): Each Ork brain cell processes its own WAAAGH independently!
        # Why groups? (Humie): Depthwise convolution reduces parameters and improves efficiency
        self.da_local_waaagh_conv = nn.Conv1d(
            da_orky_model_size, da_orky_model_size, kernel_size=3, padding=1, groups=da_orky_model_size, bias=False
        )

        # DA DEPTHWISE CONVOLUTION - for mixing da WAAAGH signals
        # Why we need 'em (Orky): Sometimes different WAAAGH energies need to mix together!
        # Why we need 'em (Humie): Pointwise convolution allows channel mixing
        self.da_waaagh_mixer = nn.Conv1d(
            da_orky_model_size, da_orky_model_size, kernel_size=1, groups=da_orky_model_size, bias=False
        )

        # DA SMOOTH ORK ACTIVATION - SiLU for smooth WAAAGH energy!
        # Why we need 'em (Orky): Orks need smooth energy flow, not jagged spikes!
        # Why we need 'em (Humie): SiLU provides smooth gradients and better training dynamics
        self.da_ork_activation = nn.SiLU()

    def unleash_da_hyena_waaagh(self, da_ork_input: torch.Tensor) -> torch.Tensor:
        """
        UNLEASH DA HYENA WAAAGH - DA ULTIMATE CONVOLUTIONAL SEQUENCE PROCESSING!

        DIS IS WHERE DA MAGIC HAPPENS - da WAAAGH energy flows through da entire horde!

        da_ork_input: (batch_size, seq_len, da_orky_model_size) - da input Ork thoughts
        Returns: (batch_size, seq_len, da_orky_model_size) - da processed WAAAGH energy

        DIS IS LIKE WATCHIN' DA WAAAGH ENERGY FLOW THROUGH DA ENTIRE ORK HORDE:
        1. Local coordination - nearby Orks coordinate their WAAAGH
        2. Global WAAAGH field - energy flows across da entire battlefield
        3. Energy mixing - different WAAAGH energies blend together
        4. Final activation - da complete WAAAGH power surge!
        """
        # GET DA BATTLE DIMENSIONS - how many Ork squads and how long da fight?
        da_batch_size, da_seq_len, da_model_size_check = da_ork_input.shape

        # CONVERT TO CONVOLUTION FORMAT - transpose for Conv1d processing
        # transpose(1, 2): Swap seq_len and d_model dimensions
        # Why we need 'em (Orky): Conv1d expects channels first, like Orks in formation!
        # Why we need 'em (Humie): Conv1d expects (batch, channels, sequence) format
        da_conv_input = da_ork_input.transpose(1, 2)

        # STEP 1: LOCAL WAAAGH COORDINATION - nearby Orks coordinate their energy!
        # Why we need 'em (Orky): Orks work better when dey can coordinate with nearby boyz!
        # Why we need 'em (Humie): Local convolution captures short-range dependencies
        da_local_waaagh = self.da_local_waaagh_conv(da_conv_input)

        # STEP 2: GLOBAL WAAAGH FIELD - da big energy flows across da battlefield!
        # Why we need 'em (Orky): Da WAAAGH energy connects all Orks across da entire fight!
        # Why we need 'em (Humie): Global convolution captures long-range dependencies efficiently
        da_global_waaagh = da_local_waaagh
        for da_conv_layer, da_filter_kernel in enumerate(self.da_waaagh_filters):
            # PREPARE DA GLOBAL FILTER - reshape for convolution
            # unsqueeze(1): Add dimension for Conv1d (filter_order, 1, d_model)
            # Why we need 'em (Orky): Da WAAAGH filter needs da right shape to work!
            # Why we need 'em (Humie): Conv1d expects 3D filter tensors
            da_global_filter = da_filter_kernel.unsqueeze(1)

            # APPLY DA GLOBAL WAAAGH FILTER - connect all Orks across da battlefield!
            # F.conv1d: Apply convolution with proper padding
            # padding=self.filter_order//2: Keep sequence length same
            # groups=d_model: Depthwise convolution for efficiency
            # Why we need 'em (Orky): Da WAAAGH energy flows through every Ork simultaneously!
            # Why we need 'em (Humie): Global convolution with depthwise groups for efficiency
            da_global_waaagh = F.conv1d(
                da_global_waaagh, da_global_filter, padding=self.da_filter_size//2, groups=da_orky_model_size
            )

            # MIX WIF DA ORIGINAL SIGNAL - blend old and new WAAAGH energy!
            # Why we need 'em (Orky): Sometimes da original WAAAGH is still important!
            # Why we need 'em (Humie): Residual connections help gradient flow and training
            if da_conv_layer < len(self.da_waaagh_filters) - 1:
                da_global_waaagh = da_global_waaagh + da_conv_input

        # STEP 3: DA WAAAGH ENERGY MIXING - blend all da different energies together!
        # Why we need 'em (Orky): Different WAAAGH energies need to mix for maximum power!
        # Why we need 'em (Humie): Pointwise convolution allows channel mixing
        da_mixed_waaagh = self.da_waaagh_mixer(da_global_waaagh)

        # CONVERT BACK TO SEQUENCE FORMAT - transpose back to normal
        # Why we need 'em (Orky): We need to get back to da normal Ork thought format!
        # Why we need 'em (Humie): Convert back to (batch, seq_len, d_model) for downstream processing
        da_sequence_output = da_mixed_waaagh.transpose(1, 2)

        # STEP 4: FINAL WAAAGH ACTIVATION - da complete power surge!
        # Why we need 'em (Orky): Da final WAAAGH energy surge to power up da Orks!
        # Why we need 'em (Humie): Activation function provides non-linearity and smooth gradients
        return self.da_ork_activation(da_sequence_output)

class OrkyPositionalEmbedding(nn.Module):
    """
    DA ORK POSITIONAL EMBEDDING - DA WAAAGH KNOWS WHERE YOU ARE!

    Dis tells da model where each token is in da sequence, like how Orks
    know their place in da horde formation!

    WHY WE NEED DIS (ORKY):
    Every Ork needs to know where dey stand in da battle formation! Da biggest Ork
    goes at da front, da smallest at da back. Da WAAAGH energy needs to know
    everyone's position to coordinate properly!

    WHY WE NEED DIS (HUMIE):
    Positional embeddings provide sequence order information that transformers
    need since they process all tokens in parallel. Sinusoidal patterns allow
    the model to understand relative positions and extrapolate to longer sequences.
    """

    def __init__(self, da_orky_model_size: int, da_max_seq_len: int = 1024):
        super().__init__()
        # DA MODEL DIMENSION - how big da Ork thoughts are
        # Why we need 'em (Orky): Each Ork needs enough brain power for position awareness!
        # Why we need 'em (Humie): Model dimension determines embedding capacity
        self.da_orky_model_size = da_orky_model_size

        # MAXIMUM SEQUENCE LENGTH - how long da battle formation can be
        # Why we need 'em (Orky): Even da biggest WAAAGH has formation limits!
        # Why we need 'em (Humie): Maximum context length for positional patterns
        self.da_max_seq_len = da_max_seq_len

        # DA POSITIONAL PATTERNS - like da Ork battle formations!
        # torch.arange: Creates position sequence [0, 1, 2, ..., max_seq_len-1]
        # Why we need 'em (Orky): Each Ork needs a unique position in da formation!
        # Why we need 'em (Humie): Position indices for sinusoidal encoding
        da_position_sequence = torch.arange(da_max_seq_len).unsqueeze(1)

        # DA FREQUENCY TERMS - different wavelengths for different position patterns
        # torch.exp: Exponential function for frequency scaling
        # Why we need 'em (Orky): Different Ork positions need different WAAAGH frequencies!
        # Why we need 'em (Humie): Multiple frequencies capture different scales of positional relationships
        da_frequency_terms = torch.exp(torch.arange(0, da_orky_model_size, 2) * (-math.log(10000.0) / da_orky_model_size))

        # CREATE DA POSITIONAL EMBEDDING MATRIX - da complete formation pattern!
        # torch.zeros: Initialize empty embedding matrix
        # Why we need 'em (Orky): Start with empty formation, den fill it wif patterns!
        # Why we need 'em (Humie): Initialize embedding matrix for positional patterns
        da_positional_embeddings = torch.zeros(da_max_seq_len, da_orky_model_size)

        # SINE PATTERNS - for even dimensions (like alternating Ork formations)
        # Why we need 'em (Orky): Sine patterns create smooth position awareness!
        # Why we need 'em (Humie): Sine functions provide smooth positional encoding
        da_positional_embeddings[:, 0::2] = torch.sin(da_position_sequence * da_frequency_terms)

        # COSINE PATTERNS - for odd dimensions (like complementary formations)
        # Why we need 'em (Orky): Cosine patterns complement da sine patterns!
        # Why we need 'em (Humie): Cosine functions provide phase-shifted positional encoding
        da_positional_embeddings[:, 1::2] = torch.cos(da_position_sequence * da_frequency_terms)

        # REGISTER DA BUFFER - store da positional patterns permanently
        # Why we need 'em (Orky): Da formation patterns don't change during battle!
        # Why we need 'em (Humie): register_buffer stores non-trainable parameters
        self.register_buffer('da_ork_positional_patterns', da_positional_embeddings)

    def add_da_ork_position_awareness(self, da_ork_input: torch.Tensor) -> torch.Tensor:
        """
        ADD DA POSITIONAL INFORMATION - EVERY ORK KNOWS THEIR PLACE!

        DIS IS WHERE DA ORKS LEARN WHERE DEY STAND IN DA BATTLE FORMATION!

        da_ork_input: (batch_size, seq_len, da_orky_model_size) - da input Ork thoughts
        Returns: (batch_size, seq_len, da_orky_model_size) - thoughts wif position awareness

        DIS IS LIKE TELLIN' EVERY ORK WHERE DEY STAND IN DA HORDE:
        1. Check how long da current battle sequence is
        2. Get da positional patterns for dat length
        3. Add da position awareness to da Ork thoughts
        4. Now every Ork knows their place in da formation!
        """
        # GET DA SEQUENCE LENGTH - how long is dis battle?
        # Why we need 'em (Orky): We need to know how many Orks are in da formation!
        # Why we need 'em (Humie): Sequence length determines how many positional embeddings to use
        da_current_seq_len = da_ork_input.size(1)

        # GET DA POSITIONAL PATTERNS - da formation awareness for dis sequence length
        # Why we need 'em (Orky): Each Ork needs to know their specific position in da formation!
        # Why we need 'em (Humie): Slice positional embeddings to match current sequence length
        da_formation_patterns = self.da_ork_positional_patterns[:da_current_seq_len].unsqueeze(0)

        # ADD DA POSITIONAL AWARENESS - combine Ork thoughts wif position knowledge
        # Why we need 'em (Orky): Da Orks need to know where dey are to coordinate properly!
        # Why we need 'em (Humie): Addition combines token embeddings with positional information
        da_position_aware_thoughts = da_ork_input + da_formation_patterns

        return da_position_aware_thoughts

class OrkyHyenaBlock(nn.Module):
    """
    DA HYENA BLOCK - A COMPLETE ORK PROCESSING UNIT!

    Dis combines da Hyena operator with normalization and residual connections,
    like a full Ork squad ready for battle!

    WHY WE NEED DIS (ORKY):
    A complete Ork squad needs multiple components workin' together! Da Hyena operator
    does da main WAAAGH processing, normalization keeps da energy clean, and residual
    connections make sure important information don't get lost in da chaos!

    WHY WE NEED DIS (HUMIE):
    Hyena block combines convolutional processing with normalization and residual
    connections for stable training and effective gradient flow. Provides a complete
    processing unit that can be stacked for deep architectures.
    """

    def __init__(self, da_orky_model_size: int, da_max_seq_len: int):
        super().__init__()
        # DA MODEL DIMENSION - how big da Ork thoughts are
        # Why we need 'em (Orky): Each Ork squad needs consistent brain power!
        # Why we need 'em (Humie): Model dimension determines processing capacity
        self.da_orky_model_size = da_orky_model_size

        # DA PRE-NORMALIZATION - keep da WAAAGH signals clean and stable
        # nn.LayerNorm: Normalizes activations across da feature dimension
        # Why we need 'em (Orky): Orks need clean WAAAGH energy, not chaotic spikes!
        # Why we need 'em (Humie): Layer normalization stabilizes training and improves convergence
        self.da_waaagh_normalizer = nn.LayerNorm(da_orky_model_size)

        # DA HYENA OPERATOR - da main WAAAGH processor for da squad
        # Why we need 'em (Orky): Da heart of da squad - where da real WAAAGH happens!
        # Why we need 'em (Humie): Core convolutional processing for sequence modeling
        self.da_hyena_operator = OrkyHyenaOperator(da_orky_model_size, da_max_seq_len)

        # DA OUTPUT PROJECTION - refine da Ork thoughts after processing
        # nn.Linear: Linear transformation to refine da processed features
        # Why we need 'em (Orky): Sometimes da processed WAAAGH needs a bit of refinement!
        # Why we need 'em (Humie): Output projection allows feature transformation and mixing
        self.da_thought_refiner = nn.Linear(da_orky_model_size, da_orky_model_size)

        # DA RESIDUAL GATE - control how much new info to add to old info
        # Why we need 'em (Orky): Orks need to balance new WAAAGH energy wif old wisdom!
        # Why we need 'em (Humie): Gating mechanism controls information flow and residual connections
        self.da_waaagh_gate = nn.Linear(da_orky_model_size, da_orky_model_size)

    def process_da_ork_squad(self, da_ork_input: torch.Tensor) -> torch.Tensor:
        """
        PROCESS THROUGH DA HYENA BLOCK - DA COMPLETE ORK SQUAD IN ACTION!

        DIS IS WHERE DA ORK SQUAD DOES ITS COMPLETE BATTLE PROCESSING!

        da_ork_input: (batch_size, seq_len, da_orky_model_size) - da input Ork thoughts
        Returns: (batch_size, seq_len, da_orky_model_size) - da processed squad wisdom

        DIS IS LIKE WATCHIN' A COMPLETE ORK SQUAD WORK TOGETHER:
        1. Remember da original Ork thoughts (residual connection)
        2. Clean up da WAAAGH energy (normalization)
        3. Process through da Hyena operator (main WAAAGH processing)
        4. Refine da thoughts (output projection)
        5. Smartly blend old and new wisdom (gated residual)
        """
        # DA RESIDUAL CONNECTION - remember da original Ork thoughts
        # Why we need 'em (Orky): Sometimes da original WAAAGH is still da best!
        # Why we need 'em (Humie): Residual connections help gradient flow and preserve information
        da_original_thoughts = da_ork_input

        # STEP 1: NORMALIZE DA WAAAGH ENERGY - keep it clean and stable
        # Why we need 'em (Orky): Orks need clean WAAAGH energy, not chaotic spikes!
        # Why we need 'em (Humie): Normalization stabilizes training and improves convergence
        da_clean_energy = self.da_waaagh_normalizer(da_ork_input)

        # STEP 2: PROCESS THROUGH DA HYENA OPERATOR - da main WAAAGH processing
        # Why we need 'em (Orky): Dis is where da real WAAAGH magic happens!
        # Why we need 'em (Humie): Core convolutional processing for sequence modeling
        da_processed_waaagh = self.da_hyena_operator.unleash_da_hyena_waaagh(da_clean_energy)

        # STEP 3: REFINE DA ORK THOUGHTS - polish da processed wisdom
        # Why we need 'em (Orky): Sometimes da processed WAAAGH needs a bit of refinement!
        # Why we need 'em (Humie): Output projection allows feature transformation and mixing
        da_refined_thoughts = self.da_thought_refiner(da_processed_waaagh)

        # STEP 4: DA GATED RESIDUAL - smartly blend old and new wisdom
        # torch.sigmoid: Creates a gate between 0 and 1 for blending
        # Why we need 'em (Orky): Orks need to balance new WAAAGH energy wif old wisdom!
        # Why we need 'em (Humie): Gating mechanism controls information flow and residual connections
        da_waaagh_gate = torch.sigmoid(self.da_waaagh_gate(da_original_thoughts))

        # BLEND OLD AND NEW WISDOM - da final Ork squad coordination!
        # Why we need 'em (Orky): Da squad combines new WAAAGH energy wif proven old tactics!
        # Why we need 'em (Humie): Gated residual connection balances new and old information
        da_final_squad_wisdom = da_waaagh_gate * da_refined_thoughts + (1 - da_waaagh_gate) * da_original_thoughts

        return da_final_squad_wisdom

class OrkyHyenaModel(nn.Module):
    """
    DA COMPLETE HYENA MODEL - DA FULL ORK WAAAGH PROCESSOR!

    Dis is da full Hyena architecture with embedding, multiple layers,
    and output projection. Ready to process any sequence like a proper Ork horde!

    WHY WE NEED DIS (ORKY):
    Dis is da complete Ork WAAAGH machine! It takes humie words, turns 'em into
    Ork battle cries, processes 'em through multiple squads of Orks, and outputs
    predictions for da next mighty victory! Like havin' an entire Ork army
    workin' together for maximum WAAAGH power!

    WHY WE NEED DIS (HUMIE):
    Complete Hyena model with token embeddings, positional encoding, stacked
    convolutional blocks, and output projection. Achieves linear complexity
    for efficient sequence processing while maintaining high performance.
    """

    def __init__(
        self,
        da_vocab_size: int,
        da_orky_model_size: int,
        da_num_layers: int,
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

        # DA NUMBER OF LAYERS - how many Ork squads we stack
        # Why we need 'em (Orky): More squads = deeper WAAAGH processing!
        # Why we need 'em (Humie): More layers enable deeper feature learning
        self.da_num_layers = da_num_layers

        # DA MAXIMUM SEQUENCE LENGTH - how long da battle can be
        # Why we need 'em (Orky): Even da biggest WAAAGH has limits!
        # Why we need 'em (Humie): Maximum context length for positional embeddings
        self.da_max_seq_len = da_max_seq_len

        # DA TOKEN EMBEDDING - turn humie words into mighty Ork battle cries!
        # nn.Embedding: Maps vocabulary indices to dense vector representations
        # Why we need 'em (Orky): Orks don't fink in puny humie words - dey fink in WAAAGHs!
        # Why we need 'em (Humie): Token embeddings convert discrete tokens to continuous vectors
        self.da_token_embeddin = nn.Embedding(da_vocab_size, da_orky_model_size)

        # DA POSITIONAL EMBEDDING - da WAAAGH positional awareness
        # Why we need 'em (Orky): Every Ork needs to know their place in da horde!
        # Why we need 'em (Humie): Positional embeddings provide sequence order information
        self.da_positional_embeddin = OrkyPositionalEmbedding(da_orky_model_size, da_max_seq_len)

        # DA DROPOUT - some Orks get distracted during battle
        # nn.Dropout: Randomly sets some activations to zero during training
        # Why we need 'em (Orky): Even Orks sometimes get distracted - makes 'em more robust!
        # Why we need 'em (Humie): Dropout prevents overfitting and improves generalization
        self.da_ork_distraction = nn.Dropout(da_dropout_rate)

        # DA HYENA LAYERS - stacked Ork processing squads
        # nn.ModuleList: List of neural network modules for stacking
        # Why we need 'em (Orky): Multiple Ork squads workin' together for maximum WAAAGH!
        # Why we need 'em (Humie): Stacked layers enable deep feature learning
        self.da_hyena_squads = nn.ModuleList([
            OrkyHyenaBlock(da_orky_model_size, da_max_seq_len)
            for _ in range(da_num_layers)
        ])

        # DA OUTPUT NORMALIZATION - final Ork thought cleanup
        # Why we need 'em (Orky): Even da best Ork thoughts need a final polish!
        # Why we need 'em (Humie): Final normalization stabilizes output and improves training
        self.da_final_ork_norm = nn.LayerNorm(da_orky_model_size)

        # DA OUTPUT PROJECTION - turn Ork thoughts back into humie words
        # bias=False: No bias term for weight tying
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
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def unleash_da_complete_hyena_waaagh(self, da_input_tokens: torch.Tensor) -> torch.Tensor:
        """
        UNLEASH DA COMPLETE HYENA WAAAGH - DA ULTIMATE SEQUENCE PROCESSING BATTLE!

        DIS IS WHERE DA ENTIRE ORK ARMY GOES TO WAR:
        1. Convert humie words into mighty Ork battle cries
        2. Add positional awareness so every Ork knows their place
        3. Process through multiple Ork squads for deep WAAAGH processing
        4. Emerge with predictions for da next mighty victory!

        da_input_tokens: (batch_size, seq_len) - da input token indices
        Returns: (batch_size, seq_len, da_vocab_size) - predictions for each position
        """
        # GET DA SEQUENCE LENGTH - how long is dis battle?
        # Why we need 'em (Orky): We need to know how many Orks are in da formation!
        # Why we need 'em (Humie): Sequence length for processing and positional embeddings
        da_current_seq_len = da_input_tokens.size(1)

        # STEP 1: DA TOKEN EMBEDDING - convert humie words to Ork battle cries!
        # Why we need 'em (Orky): Orks don't fink in puny humie words - dey fink in WAAAGHs!
        # Why we need 'em (Humie): Token embeddings convert discrete tokens to continuous vectors
        da_ork_battle_cries = self.da_token_embeddin(da_input_tokens)

        # STEP 2: ADD POSITIONAL AWARENESS - every Ork knows their place!
        # Why we need 'em (Orky): Every Ork needs to know where dey stand in da horde!
        # Why we need 'em (Humie): Positional embeddings provide sequence order information
        da_position_aware_thoughts = self.da_positional_embeddin.add_da_ork_position_awareness(da_ork_battle_cries)

        # STEP 3: DROPOUT FOR ROBUSTNESS - some Orks get distracted
        # Why we need 'em (Orky): Even Orks sometimes get distracted - makes 'em more robust!
        # Why we need 'em (Humie): Dropout prevents overfitting and improves generalization
        da_robust_thoughts = self.da_ork_distraction(da_position_aware_thoughts)

        # STEP 4: PROCESS THROUGH DA HYENA SQUADS - multiple Ork squads workin' together!
        # Why we need 'em (Orky): Multiple Ork squads process da WAAAGH for maximum power!
        # Why we need 'em (Humie): Stacked layers enable deep feature learning
        da_processed_thoughts = da_robust_thoughts
        for da_squad in self.da_hyena_squads:
            da_processed_thoughts = da_squad.process_da_ork_squad(da_processed_thoughts)

        # STEP 5: FINAL NORMALIZATION - polish da Ork thoughts
        # Why we need 'em (Orky): Even da best Ork thoughts need a final polish!
        # Why we need 'em (Humie): Final normalization stabilizes output and improves training
        da_polished_thoughts = self.da_final_ork_norm(da_processed_thoughts)

        # STEP 6: GET DA LOGITS - predict da next mighty victory!
        # Why we need 'em (Orky): Da Orks predict wot comes next in da battle!
        # Why we need 'em (Humie): Output projection predicts next token probabilities
        da_victory_predictions = self.da_output_proj(da_polished_thoughts)

        return da_victory_predictions

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        GENERATE NEW TOKENS LIKE A STORYTELLIN' ORK!

        input_ids: (batch_size, seq_len) - starting sequence
        Returns: (batch_size, seq_len + max_new_tokens) - completed sequence
        """
        for _ in range(max_new_tokens):
            # Only process da last tokens for efficiency (Orks don't waste energy!)
            logits = self(input_ids[:, -self.max_seq_len:])

            # Get da last token's predictions
            next_token_logits = logits[:, -1, :] / temperature

            # TOP-K SAMPLING - only da best Ork choices
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(-1, top_k_indices, top_k_logits)

            # TOP-P SAMPLING - da nucleus sampling for creative Orks
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Remove tokens with cumulative probability above da threshold
                sorted_logits[cumulative_probs > top_p] = float('-inf')

                # Re-sort da logits
                next_token_logits = torch.gather(
                    sorted_logits,
                    -1,
                    torch.argsort(sorted_indices, dim=-1)
                )

            # SAMPLE DA NEXT TOKEN
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # ADD TO DA SEQUENCE
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

def create_orky_hyena_model(
    vocab_size: int = 50000,
    d_model: int = 512,
    num_layers: int = 12,
    max_seq_len: int = 2048
) -> OrkyHyenaModel:
    """
    CREATE A FULL-SIZED HYENA MODEL FOR REAL BATTLES!

    Dis creates a big Hyena model ready for serious sequence processing.
    Like fielding a full Ork army instead of just a few boyz!
    """
    print(f"🐍⚡ Creating Orky Hyena Model!")
    print(f"   Vocab Size: {vocab_size}")
    print(f"   Model Dim: {d_model}")
    print(f"   Layers: {num_layers}")
    print(f"   Max Seq Len: {max_seq_len}")

    model = OrkyHyenaModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
    )

    # Count da parameters like countin' da horde
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Model Size: {total_params * 4 / (1024**2):.1f} MB")

    return model

if __name__ == "__main__":
    # QUICK TEST OF DA HYENA MODEL
    print("🐍⚡ Testing da Hyena Model! ⚡🐍")

    # Create a small model for testing
    model = create_orky_hyena_model(
        vocab_size=100,
        d_model=64,
        num_layers=2,
        max_seq_len=32
    )

    # Test forward pass
    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, 100, (batch_size, seq_len))

    print(f"\nInput shape: {input_ids.shape}")
    print(f"Input tokens: {input_ids[0].tolist()}")

    logits = model(input_ids)
    print(f"Output shape: {logits.shape}")

    # Test generation
    generated = model.generate(input_ids, max_new_tokens=5)
    print(f"Generated sequence: {generated[0].tolist()}")

    print("\n✅ Hyena model test complete! WAAAGH!")
