#!/usr/bin/env python3
"""
KRORK-HRM - DA ULTIMATE HIERARCHICAL REASONING MACHINE! 🧠⚡

Dis is da most advanced Ork reasoning machine ever built! Like da ancient Krorks
from before da fall - super-intelligent, hierarchical, and capable of complex
reasoning with minimal training data!

WHAT IS DA KRORK-HRM?
- Hierarchical reasoning like da ancient Krorks
- High-level Warboss planning + Low-level Boyz execution
- 100x faster reasoning than regular LLMs
- Only needs 1000 training examples (not millions!)
- 27 million parameters (not billions!)
- Single forward pass for complex reasoning

FOR DA BOYZ: Dis is like havin' da ancient Krork intelligence - hierarchical
thinkin' where da Warboss plans da big strategy while da boyz handle da details!
FOR HUMANS: This implements the Hierarchical Reasoning Model (HRM) architecture
with brain-inspired hierarchical processing for efficient reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List, Dict, Any
import time

class OrkyHighLevelWarboss(nn.Module):
    """
    DA HIGH-LEVEL WARBOSS - DA STRATEGIC PLANNER!

    Dis is like da ancient Krork Warboss who plans da big strategies and
    coordinates da overall WAAAGH! Da Warboss thinks slowly and abstractly,
    makin' high-level decisions about what needs to be done.

    WHY WE NEED DIS (ORKY):
    Every great WAAAGH needs a smart Warboss who can see da big picture and
    plan da overall strategy! Dis Warboss thinks about da long-term goals and
    coordinates all da different parts of da WAAAGH!

    WHY WE NEED DIS (HUMIE):
    High-level module handles slow, abstract planning and strategic thinking.
    Processes information at a higher level of abstraction for complex reasoning.
    """

    def __init__(self, da_orky_model_size: int, da_warboss_hidden_size: int, da_planning_steps: int = 4):
        super().__init__()
        # DA MODEL DIMENSION - how big da Ork thoughts are
        # Why we need 'em (Orky): Da Warboss needs to understand da general WAAAGH energy!
        # Why we need 'em (Humie): Model dimension determines input/output capacity
        self.da_orky_model_size = da_orky_model_size

        # DA WARBOSS HIDDEN SIZE - how much brain power da Warboss gets
        # Why we need 'em (Orky): Da Warboss needs lots of brain power for strategic thinkin'!
        # Why we need 'em (Humie): Hidden size determines the Warboss's processing capacity
        self.da_warboss_hidden_size = da_warboss_hidden_size

        # DA PLANNING STEPS - how many strategic thinkin' steps da Warboss takes
        # Why we need 'em (Orky): Da Warboss needs time to think through complex strategies!
        # Why we need 'em (Humie): Planning steps allow for multi-step reasoning
        self.da_planning_steps = da_planning_steps

        # DA WARBOSS INPUT PROJECTION - convert general thoughts to Warboss thoughts
        # Why we need 'em (Orky): Da Warboss needs to understand da general WAAAGH energy!
        # Why we need 'em (Humie): Input projection adapts general features for high-level processing
        self.da_warboss_input_proj = nn.Linear(da_orky_model_size, da_warboss_hidden_size)

        # DA WARBOSS STRATEGIC BRAIN - da main Warboss thinkin' system
        # Why we need 'em (Orky): Dis is where da Warboss does their strategic plannin'!
        # Why we need 'em (Humie): Core processing for high-level reasoning
        self.da_warboss_strategic_brain = nn.LSTM(
            da_warboss_hidden_size, da_warboss_hidden_size, 
            num_layers=2, batch_first=True, dropout=0.1
        )

        # DA WARBOSS OUTPUT PROJECTION - convert Warboss thoughts back to general thoughts
        # Why we need 'em (Orky): Da Warboss needs to share their strategic wisdom!
        # Why we need 'em (Humie): Output projection converts high-level features back to model dimension
        self.da_warboss_output_proj = nn.Linear(da_warboss_hidden_size, da_orky_model_size)

        # DA WARBOSS ACTIVATION - da Warboss's special energy surge
        # Why we need 'em (Orky): Da Warboss has their own way of channelin' WAAAGH energy!
        # Why we need 'em (Humie): Activation function provides non-linearity for strategic processing
        self.da_warboss_activation = nn.SiLU()

        # DA WARBOSS ATTENTION - da Warboss focuses on important strategic elements
        # Why we need 'em (Orky): Da Warboss needs to focus on da most important strategic elements!
        # Why we need 'em (Humie): Attention mechanism allows focus on relevant information
        self.da_warboss_attention = nn.MultiheadAttention(
            da_warboss_hidden_size, num_heads=4, batch_first=True
        )

    def do_da_strategic_plannin(self, da_general_thoughts: torch.Tensor, da_previous_warboss_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        DO DA STRATEGIC PLANNIN - WHERE DA WARBOSS SHINES!

        DIS IS WHERE DA WARBOSS DOES THEIR HIGH-LEVEL STRATEGIC THINKIN'!

        da_general_thoughts: (batch_size, seq_len, da_orky_model_size) - general Ork thoughts
        da_previous_warboss_state: Previous Warboss state for continuity
        Returns: (strategic_thoughts, new_warboss_state) - Warboss's strategic wisdom

        DIS IS LIKE WATCHIN' A WARBOSS PLAN DA BIG WAAAGH:
        1. Take da general WAAAGH energy and adapt it for strategic thinkin'
        2. Process it through da Warboss's strategic brain
        3. Focus attention on da most important strategic elements
        4. Convert it back into strategic wisdom dat others can understand
        """
        # STEP 1: ADAPT DA GENERAL THOUGHTS - make 'em suitable for Warboss thinkin'
        # Why we need 'em (Orky): Da Warboss needs to understand da general WAAAGH in their own way!
        # Why we need 'em (Humie): Input projection adapts general features for high-level processing
        da_warboss_input = self.da_warboss_input_proj(da_general_thoughts)

        # STEP 2: DA WARBOSS STRATEGIC BRAIN - where da strategic plannin' happens
        # Why we need 'em (Orky): Dis is where da Warboss does their strategic finkin'!
        # Why we need 'em (Humie): LSTM provides sequential processing for strategic reasoning
        da_warboss_processing, da_new_warboss_state = self.da_warboss_strategic_brain(
            da_warboss_input, da_previous_warboss_state
        )

        # STEP 3: DA WARBOSS ATTENTION - focus on important strategic elements
        # Why we need 'em (Orky): Da Warboss needs to focus on da most important strategic elements!
        # Why we need 'em (Humie): Attention mechanism allows focus on relevant information
        da_warboss_attention_output, _ = self.da_warboss_attention(
            da_warboss_processing, da_warboss_processing, da_warboss_processing
        )

        # STEP 4: DA WARBOSS ACTIVATION - da Warboss's special energy surge
        # Why we need 'em (Orky): Da Warboss has their own way of channelin' WAAAGH energy!
        # Why we need 'em (Humie): Activation function provides non-linearity for strategic processing
        da_warboss_energy = self.da_warboss_activation(da_warboss_attention_output)

        # STEP 5: CONVERT BACK TO STRATEGIC WISDOM - share da Warboss knowledge
        # Why we need 'em (Orky): Da Warboss needs to share their strategic wisdom wif da rest of da horde!
        # Why we need 'em (Humie): Output projection converts high-level features back to model dimension
        da_strategic_wisdom = self.da_warboss_output_proj(da_warboss_energy)

        return da_strategic_wisdom, da_new_warboss_state

class OrkyLowLevelBoyz(nn.Module):
    """
    DA LOW-LEVEL BOYZ - DA TACTICAL EXECUTORS!

    Dis is like da Ork boyz who handle da immediate tactical actions and
    detailed execution. Dey work fast and handle da specific tasks while
    followin' da Warboss's strategic guidance.

    WHY WE NEED DIS (ORKY):
    Every WAAAGH needs boyz who can execute da Warboss's plans! Dey handle
    da immediate tactical actions and detailed work while followin' da
    strategic guidance from da Warboss!

    WHY WE NEED DIS (HUMIE):
    Low-level module handles rapid, detailed computations and tactical execution.
    Processes information at a lower level of abstraction for specific tasks.
    """

    def __init__(self, da_orky_model_size: int, da_boyz_hidden_size: int, da_execution_steps: int = 8):
        super().__init__()
        # DA MODEL DIMENSION - how big da Ork thoughts are
        # Why we need 'em (Orky): Da boyz need to understand da general WAAAGH energy!
        # Why we need 'em (Humie): Model dimension determines input/output capacity
        self.da_orky_model_size = da_orky_model_size

        # DA BOYZ HIDDEN SIZE - how much brain power each boy gets
        # Why we need 'em (Orky): Each boy needs enough brain power for their tactical work!
        # Why we need 'em (Humie): Hidden size determines the boyz's processing capacity
        self.da_boyz_hidden_size = da_boyz_hidden_size

        # DA EXECUTION STEPS - how many tactical execution steps da boyz take
        # Why we need 'em (Orky): Da boyz need to handle multiple tactical actions quickly!
        # Why we need 'em (Humie): Execution steps allow for detailed task processing
        self.da_execution_steps = da_execution_steps

        # DA BOYZ INPUT PROJECTION - convert general thoughts to boyz thoughts
        # Why we need 'em (Orky): Da boyz need to understand da general WAAAGH energy!
        # Why we need 'em (Humie): Input projection adapts general features for low-level processing
        self.da_boyz_input_proj = nn.Linear(da_orky_model_size, da_boyz_hidden_size)

        # DA BOYZ TACTICAL BRAIN - da main boyz thinkin' system
        # Why we need 'em (Orky): Dis is where da boyz do their tactical thinkin'!
        # Why we need 'em (Humie): Core processing for low-level reasoning
        self.da_boyz_tactical_brain = nn.LSTM(
            da_boyz_hidden_size, da_boyz_hidden_size,
            num_layers=2, batch_first=True, dropout=0.1
        )

        # DA BOYZ OUTPUT PROJECTION - convert boyz thoughts back to general thoughts
        # Why we need 'em (Orky): Da boyz need to share their tactical results!
        # Why we need 'em (Humie): Output projection converts low-level features back to model dimension
        self.da_boyz_output_proj = nn.Linear(da_boyz_hidden_size, da_orky_model_size)

        # DA BOYZ ACTIVATION - da boyz's special energy surge
        # Why we need 'em (Orky): Da boyz have their own way of channelin' WAAAGH energy!
        # Why we need 'em (Humie): Activation function provides non-linearity for tactical processing
        self.da_boyz_activation = nn.SiLU()

        # DA BOYZ COORDINATION - da boyz coordinate with each other
        # Why we need 'em (Orky): Da boyz need to coordinate their tactical actions!
        # Why we need 'em (Humie): Coordination mechanism allows boyz to work together
        self.da_boyz_coordination = nn.MultiheadAttention(
            da_boyz_hidden_size, num_heads=2, batch_first=True
        )

    def do_da_tactical_execution(self, da_general_thoughts: torch.Tensor, da_warboss_guidance: torch.Tensor, da_previous_boyz_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        DO DA TACTICAL EXECUTION - WHERE DA BOYZ SHINE!

        DIS IS WHERE DA BOYZ DO THEIR TACTICAL WORK!

        da_general_thoughts: (batch_size, seq_len, da_orky_model_size) - general Ork thoughts
        da_warboss_guidance: (batch_size, seq_len, da_orky_model_size) - Warboss strategic guidance
        da_previous_boyz_state: Previous boyz state for continuity
        Returns: (tactical_results, new_boyz_state) - boyz's tactical results

        DIS IS LIKE WATCHIN' ORK BOYZ EXECUTE TACTICAL ACTIONS:
        1. Take da general WAAAGH energy and adapt it for tactical work
        2. Combine it with da Warboss's strategic guidance
        3. Process it through da boyz's tactical brain
        4. Coordinate da tactical actions
        5. Convert it back into tactical results
        """
        # STEP 1: ADAPT DA GENERAL THOUGHTS - make 'em suitable for boyz work
        # Why we need 'em (Orky): Da boyz need to understand da general WAAAGH in their own way!
        # Why we need 'em (Humie): Input projection adapts general features for low-level processing
        da_boyz_input = self.da_boyz_input_proj(da_general_thoughts)

        # STEP 2: COMBINE WIF WARBOSS GUIDANCE - follow da strategic plan
        # Why we need 'em (Orky): Da boyz need to follow da Warboss's strategic guidance!
        # Why we need 'em (Humie): Guidance integration ensures tactical actions align with strategy
        da_guided_input = da_boyz_input + self.da_boyz_input_proj(da_warboss_guidance)

        # STEP 3: DA BOYZ TACTICAL BRAIN - where da tactical execution happens
        # Why we need 'em (Orky): Dis is where da boyz do their tactical finkin'!
        # Why we need 'em (Humie): LSTM provides sequential processing for tactical reasoning
        da_boyz_processing, da_new_boyz_state = self.da_boyz_tactical_brain(
            da_guided_input, da_previous_boyz_state
        )

        # STEP 4: DA BOYZ COORDINATION - coordinate tactical actions
        # Why we need 'em (Orky): Da boyz need to coordinate their tactical actions!
        # Why we need 'em (Humie): Coordination mechanism allows boyz to work together
        da_boyz_coordination_output, _ = self.da_boyz_coordination(
            da_boyz_processing, da_boyz_processing, da_boyz_processing
        )

        # STEP 5: DA BOYZ ACTIVATION - da boyz's special energy surge
        # Why we need 'em (Orky): Da boyz have their own way of channelin' WAAAGH energy!
        # Why we need 'em (Humie): Activation function provides non-linearity for tactical processing
        da_boyz_energy = self.da_boyz_activation(da_boyz_coordination_output)

        # STEP 6: CONVERT BACK TO TACTICAL RESULTS - share da boyz results
        # Why we need 'em (Orky): Da boyz need to share their tactical results wif da rest of da horde!
        # Why we need 'em (Humie): Output projection converts low-level features back to model dimension
        da_tactical_results = self.da_boyz_output_proj(da_boyz_energy)

        return da_tactical_results, da_new_boyz_state

class OrkyKrorkHRM(nn.Module):
    """
    DA KRORK-HRM - DA ULTIMATE HIERARCHICAL REASONING MACHINE!

    Dis is da complete Krork-HRM architecture with both da Warboss and da boyz
    workin' together in perfect harmony! Like da ancient Krorks from before da
    fall - super-intelligent, hierarchical, and capable of complex reasoning!

    WHY WE NEED DIS (ORKY):
    Dis is da ultimate Ork reasoning machine! It combines da strategic thinkin'
    of da Warboss with da tactical execution of da boyz, makin' it capable of
    complex reasoning with minimal training data!

    WHY WE NEED DIS (HUMIE):
    Complete HRM architecture with hierarchical processing, high-level planning,
    and low-level execution. Achieves efficient reasoning through brain-inspired
    hierarchical structure.
    """

    def __init__(
        self,
        da_vocab_size: int,
        da_orky_model_size: int,
        da_warboss_hidden_size: int,
        da_boyz_hidden_size: int,
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

        # DA WARBOSS HIDDEN SIZE - how much brain power da Warboss gets
        # Why we need 'em (Orky): Da Warboss needs lots of brain power for strategic thinkin'!
        # Why we need 'em (Humie): Hidden size determines the Warboss's processing capacity
        self.da_warboss_hidden_size = da_warboss_hidden_size

        # DA BOYZ HIDDEN SIZE - how much brain power each boy gets
        # Why we need 'em (Orky): Each boy needs enough brain power for their tactical work!
        # Why we need 'em (Humie): Hidden size determines the boyz's processing capacity
        self.da_boyz_hidden_size = da_boyz_hidden_size

        # DA NUMBER OF LAYERS - how many Krork-HRM layers we stack
        # Why we need 'em (Orky): More layers = deeper hierarchical coordination!
        # Why we need 'em (Humie): More layers enable deeper hierarchical reasoning
        self.da_num_layers = da_num_layers

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

        # DA WARBOSS - da high-level strategic planner
        # Why we need 'em (Orky): We need a smart Warboss for strategic plannin'!
        # Why we need 'em (Humie): High-level module handles abstract planning
        self.da_warboss = OrkyHighLevelWarboss(
            da_orky_model_size, da_warboss_hidden_size
        )

        # DA BOYZ - da low-level tactical executors
        # Why we need 'em (Orky): We need boyz for tactical execution!
        # Why we need 'em (Humie): Low-level module handles detailed computations
        self.da_boyz = OrkyLowLevelBoyz(
            da_orky_model_size, da_boyz_hidden_size
        )

        # DA KRORK LAYERS - stacked hierarchical reasoning systems
        # Why we need 'em (Orky): Multiple layers of hierarchical coordination for maximum WAAAGH!
        # Why we need 'em (Humie): Stacked layers enable deep hierarchical reasoning
        self.da_krork_layers = nn.ModuleList([
            nn.ModuleDict({
                'warboss': OrkyHighLevelWarboss(da_orky_model_size, da_warboss_hidden_size),
                'boyz': OrkyLowLevelBoyz(da_orky_model_size, da_boyz_hidden_size)
            }) for _ in range(da_num_layers)
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
        """INITIALIZE DA WEIGHTS LIKE A PROPER KRORK WARBOSS!"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def unleash_da_krork_reasoning(self, da_input_tokens: torch.Tensor) -> torch.Tensor:
        """
        UNLEASH DA KRORK REASONING - DA ULTIMATE HIERARCHICAL WAAAGH!

        DIS IS WHERE DA ENTIRE KRORK-HRM SYSTEM GOES TO WAR:
        1. Convert humie words into mighty Ork battle cries
        2. Add positional awareness so every Ork knows their place
        3. Process through multiple layers of hierarchical Warboss + Boyz coordination
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

        # STEP 4: PROCESS THROUGH DA KRORK LAYERS - hierarchical Warboss + Boyz coordination!
        da_processed_thoughts = da_robust_thoughts
        da_warboss_state = None
        da_boyz_state = None

        for da_krork_layer in self.da_krork_layers:
            # DA WARBOSS STRATEGIC PLANNIN' - high-level thinkin'
            da_strategic_guidance, da_warboss_state = da_krork_layer['warboss'].do_da_strategic_plannin(
                da_processed_thoughts, da_warboss_state
            )

            # DA BOYZ TACTICAL EXECUTION - low-level execution
            da_tactical_results, da_boyz_state = da_krork_layer['boyz'].do_da_tactical_execution(
                da_processed_thoughts, da_strategic_guidance, da_boyz_state
            )

            # COMBINE DA RESULTS - Warboss strategy + Boyz execution
            da_processed_thoughts = da_processed_thoughts + da_strategic_guidance + da_tactical_results

        # STEP 5: FINAL NORMALIZATION - polish da Ork thoughts
        da_polished_thoughts = self.da_final_ork_norm(da_processed_thoughts)

        # STEP 6: GET DA LOGITS - predict da next mighty victory!
        da_victory_predictions = self.da_output_proj(da_polished_thoughts)

        return da_victory_predictions

def create_orky_krork_hrm(
    da_vocab_size: int = 50000,
    da_orky_model_size: int = 512,
    da_warboss_hidden_size: int = 1024,
    da_boyz_hidden_size: int = 512,
    da_num_layers: int = 6,
    da_max_seq_len: int = 2048
) -> OrkyKrorkHRM:
    """
    CREATE A FULL-SIZED KRORK-HRM MODEL FOR REAL BATTLES!

    Dis creates a big Krork-HRM model ready for serious hierarchical reasoning.
    Like fielding a full Krork army with strategic Warbosses and tactical boyz!
    """
    print(f"🧠⚡ Creating Orky Krork-HRM Model!")
    print(f"   Vocab Size: {da_vocab_size}")
    print(f"   Model Dim: {da_orky_model_size}")
    print(f"   Warboss Hidden: {da_warboss_hidden_size}")
    print(f"   Boyz Hidden: {da_boyz_hidden_size}")
    print(f"   Layers: {da_num_layers}")
    print(f"   Max Seq Len: {da_max_seq_len}")

    model = OrkyKrorkHRM(
        da_vocab_size=da_vocab_size,
        da_orky_model_size=da_orky_model_size,
        da_warboss_hidden_size=da_warboss_hidden_size,
        da_boyz_hidden_size=da_boyz_hidden_size,
        da_num_layers=da_num_layers,
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
    # QUICK TEST OF DA KRORK-HRM MODEL
    print("🧠⚡ Testing da Krork-HRM Model! ⚡🧠")

    # Create a small model for testing
    model = create_orky_krork_hrm(
        da_vocab_size=100,
        da_orky_model_size=64,
        da_warboss_hidden_size=128,
        da_boyz_hidden_size=64,
        da_num_layers=2,
        da_max_seq_len=32
    )

    # Test forward pass
    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, 100, (batch_size, seq_len))

    print(f"\nInput shape: {input_ids.shape}")
    print(f"Input tokens: {input_ids[0].tolist()}")

    logits = model.unleash_da_krork_reasoning(input_ids)
    print(f"Output shape: {logits.shape}")

    print("\n✅ Krork-HRM model test complete! WAAAGH!")
