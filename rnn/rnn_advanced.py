#!/usr/bin/env python3
"""
ADVANCED RNN WAAAGH - DA ULTIMATE MEMORY BOYZ SYSTEM! 🧠⚡

Dis is da advanced RNN system with all da fancy features! We got:
- Bidirectional memory boyz (lookin' both ways!)
- Attention mechanisms (focusin' on important stuff!)
- Stacked layers (more memory boyz!)
- Dropout (preventin' overexcitement!)
- Batch normalization (keepin' da boyz stable!)

WHAT IS DA ADVANCED RNN WAAAGH?
- Enhanced RNN architectures with modern features
- Bidirectional processing for better context
- Attention mechanisms for focus
- Advanced training techniques

FOR DA BOYZ: Dis is like havin' super-smart Ork boyz who can look
both ways, focus on important stuff, and work together in layers!

FOR HUMANS: This implements advanced RNN architectures with bidirectional
processing, attention mechanisms, and modern training techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List, Dict, Any
import time

class OrkyBidirectionalRNN(nn.Module):
    """
    DA BIDIRECTIONAL RNN - DA BOYZ WHO LOOK BOTH WAYS!
    
    Dis is like havin' Ork boyz who can look both forward and backward
    in time! They can see what happened before AND what's coming next!
    
    WHY WE NEED DIS (ORKY):
    Smart Orks look both ways before fightin'! Dis is like havin'
    Ork boyz who can see da past AND da future!
    
    WHY WE NEED DIS (HUMIE):
    Bidirectional RNN processes sequences in both directions for
    better context understanding and improved performance.
    """
    
    def __init__(self, da_input_size: int, da_hidden_size: int, da_cell_type: str = "LSTM"):
        super().__init__()
        self.da_input_size = da_input_size
        self.da_hidden_size = da_hidden_size
        self.da_cell_type = da_cell_type
        
        # DA FORWARD RNN - looks forward in time
        # Why we need 'em (Orky): Da boy needs to see what's coming!
        # Why we need 'em (Humie): Forward RNN processes sequence left-to-right
        if da_cell_type == "LSTM":
            self.da_forward_rnn = nn.LSTM(da_input_size, da_hidden_size, batch_first=True)
        elif da_cell_type == "GRU":
            self.da_forward_rnn = nn.GRU(da_input_size, da_hidden_size, batch_first=True)
        else:
            self.da_forward_rnn = nn.RNN(da_input_size, da_hidden_size, batch_first=True)
        
        # DA BACKWARD RNN - looks backward in time
        # Why we need 'em (Orky): Da boy needs to see what happened before!
        # Why we need 'em (Humie): Backward RNN processes sequence right-to-left
        if da_cell_type == "LSTM":
            self.da_backward_rnn = nn.LSTM(da_input_size, da_hidden_size, batch_first=True)
        elif da_cell_type == "GRU":
            self.da_backward_rnn = nn.GRU(da_input_size, da_hidden_size, batch_first=True)
        else:
            self.da_backward_rnn = nn.RNN(da_input_size, da_hidden_size, batch_first=True)
    
    def forward(self, da_input: torch.Tensor) -> torch.Tensor:
        """
        FORWARD PASS - WHERE DA BIDIRECTIONAL BOYZ SHINE!
        
        DIS IS WHERE DA BOYZ LOOK BOTH WAYS:
        1. Forward boy looks ahead
        2. Backward boy looks behind
        3. Combine their knowledge!
        """
        # DA FORWARD PASS - look ahead
        da_forward_output, _ = self.da_forward_rnn(da_input)
        
        # DA BACKWARD PASS - look behind (reverse the sequence)
        da_backward_input = torch.flip(da_input, dims=[1])
        da_backward_output, _ = self.da_backward_rnn(da_backward_input)
        da_backward_output = torch.flip(da_backward_output, dims=[1])
        
        # COMBINE DA KNOWLEDGE - concatenate forward and backward
        da_combined_output = torch.cat([da_forward_output, da_backward_output], dim=-1)
        
        return da_combined_output

class OrkyRNNAttention(nn.Module):
    """
    DA RNN ATTENTION - DA BOYZ WHO FOCUS ON IMPORTANT STUFF!
    
    Dis is like havin' Ork boyz who can focus on da most important
    parts of what they remember! They don't just remember everything
    equally - they focus on what matters most!
    
    WHY WE NEED DIS (ORKY):
    Smart Orks focus on da most important enemies in battle!
    Dis is like havin' Ork boyz who can focus on important memories!
    
    WHY WE NEED DIS (HUMIE):
    Attention mechanism allows the model to focus on relevant parts
    of the sequence for better performance and interpretability.
    """
    
    def __init__(self, da_hidden_size: int, da_attention_size: int = None):
        super().__init__()
        self.da_hidden_size = da_hidden_size
        self.da_attention_size = da_attention_size or da_hidden_size // 2
        
        # DA ATTENTION WEIGHTS - how much to focus on each part
        # Why we need 'em (Orky): Da boy needs to decide what's important!
        # Why we need 'em (Humie): Attention weights determine focus distribution
        self.da_attention_weights = nn.Linear(da_hidden_size, self.da_attention_size)
        
        # DA CONTEXT VECTOR - da focused information
        # Why we need 'em (Orky): Da boy needs to combine focused information!
        # Why we need 'em (Humie): Context vector combines attended information
        self.da_context_vector = nn.Linear(self.da_attention_size, 1)
        
        # DA ACTIVATION - da boy's focus energy
        # Why we need 'em (Orky): Da boy has his own way of focusin'!
        # Why we need 'em (Humie): Activation function provides non-linearity
        self.da_activation = nn.Tanh()
    
    def forward(self, da_hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        FORWARD PASS - WHERE DA ATTENTION BOY FOCUSES!
        
        DIS IS WHERE DA BOY DECIDES WHAT TO FOCUS ON:
        1. Look at all da hidden states
        2. Decide which ones are most important
        3. Focus on da important ones!
        """
        # STEP 1: COMPUTE ATTENTION SCORES - how important each state is
        # Why we need 'em (Orky): Da boy needs to score each memory!
        # Why we need 'em (Humie): Attention scores determine importance weights
        da_attention_scores = self.da_attention_weights(da_hidden_states)
        da_attention_scores = self.da_activation(da_attention_scores)
        da_attention_scores = self.da_context_vector(da_attention_scores)
        da_attention_scores = da_attention_scores.squeeze(-1)
        
        # STEP 2: COMPUTE ATTENTION WEIGHTS - how much to focus on each
        # Why we need 'em (Orky): Da boy needs to decide focus amounts!
        # Why we need 'em (Humie): Softmax normalizes attention scores to probabilities
        da_attention_weights = F.softmax(da_attention_scores, dim=-1)
        
        # STEP 3: COMPUTE CONTEXT VECTOR - da focused information
        # Why we need 'em (Orky): Da boy needs to combine focused information!
        # Why we need 'em (Humie): Weighted sum creates context vector
        da_context_vector = torch.sum(da_attention_weights.unsqueeze(-1) * da_hidden_states, dim=1)
        
        return da_context_vector, da_attention_weights

class OrkyStackedRNN(nn.Module):
    """
    DA STACKED RNN - DA LAYERED MEMORY BOYZ SYSTEM!
    
    Dis is like havin' multiple layers of Ork boyz, each one smarter
    than da last! Each layer processes da information from da layer below
    and passes it up to da next layer!
    
    WHY WE NEED DIS (ORKY):
    More layers = more brain power! Dis is like havin' a hierarchy
    of Ork boyz, each one smarter than da last!
    
    WHY WE NEED DIS (HUMIE):
    Stacked RNN layers enable deeper sequential processing with
    hierarchical feature learning and better representation capacity.
    """
    
    def __init__(
        self,
        da_input_size: int,
        da_hidden_size: int,
        da_num_layers: int,
        da_cell_type: str = "LSTM",
        da_dropout: float = 0.1,
        da_bidirectional: bool = False
    ):
        super().__init__()
        self.da_input_size = da_input_size
        self.da_hidden_size = da_hidden_size
        self.da_num_layers = da_num_layers
        self.da_cell_type = da_cell_type
        self.da_dropout = da_dropout
        self.da_bidirectional = da_bidirectional
        
        # DA RNN LAYERS - stacked memory boyz
        # Why we need 'em (Orky): Multiple layers of memory boyz for maximum WAAAGH!
        # Why we need 'em (Humie): Stacked layers enable deep sequential processing
        if da_bidirectional:
            self.da_rnn = nn.LSTM(
                da_input_size, da_hidden_size, da_num_layers,
                batch_first=True, dropout=da_dropout, bidirectional=True
            )
        else:
            if da_cell_type == "LSTM":
                self.da_rnn = nn.LSTM(
                    da_input_size, da_hidden_size, da_num_layers,
                    batch_first=True, dropout=da_dropout
                )
            elif da_cell_type == "GRU":
                self.da_rnn = nn.GRU(
                    da_input_size, da_hidden_size, da_num_layers,
                    batch_first=True, dropout=da_dropout
                )
            else:
                self.da_rnn = nn.RNN(
                    da_input_size, da_hidden_size, da_num_layers,
                    batch_first=True, dropout=da_dropout
                )
        
        # DA BATCH NORMALIZATION - keepin' da boyz stable
        # Why we need 'em (Orky): Da boyz need to stay stable during training!
        # Why we need 'em (Humie): Batch normalization stabilizes training
        self.da_batch_norm = nn.BatchNorm1d(da_hidden_size * (2 if da_bidirectional else 1))
        
        # DA DROPOUT - preventin' overexcitement
        # Why we need 'em (Orky): Da boyz can't get too excited!
        # Why we need 'em (Humie): Dropout prevents overfitting
        self.da_dropout_layer = nn.Dropout(da_dropout)
    
    def forward(self, da_input: torch.Tensor) -> torch.Tensor:
        """
        FORWARD PASS - WHERE DA STACKED BOYZ WORK TOGETHER!
        
        DIS IS WHERE DA LAYERED BOYZ PROCESS INFORMATION:
        1. Each layer processes da information
        2. Pass it to da next layer
        3. Apply normalization and dropout
        """
        # DA RNN FORWARD PASS - let da stacked boyz work!
        da_output, _ = self.da_rnn(da_input)
        
        # DA BATCH NORMALIZATION - keepin' da boyz stable
        da_batch_size, da_seq_len, da_hidden_dim = da_output.shape
        da_output_reshaped = da_output.view(-1, da_hidden_dim)
        da_output_normalized = self.da_batch_norm(da_output_reshaped)
        da_output = da_output_normalized.view(da_batch_size, da_seq_len, da_hidden_dim)
        
        # DA DROPOUT - preventin' overexcitement
        da_output = self.da_dropout_layer(da_output)
        
        return da_output

class OrkyAdvancedRNN(nn.Module):
    """
    DA ADVANCED RNN - DA ULTIMATE MEMORY BOYZ SYSTEM!
    
    Dis is da most advanced RNN system with all da fancy features!
    It combines bidirectional processing, attention, stacking, and
    all da modern techniques for maximum WAAAGH!
    
    WHY WE NEED DIS (ORKY):
    Dis is da ultimate memory system! It combines all da best
    features for maximum memory power and WAAAGH energy!
    
    WHY WE NEED DIS (HUMIE):
    Advanced RNN with bidirectional processing, attention mechanisms,
    stacked layers, and modern training techniques for optimal performance.
    """
    
    def __init__(
        self,
        da_vocab_size: int,
        da_embedding_size: int,
        da_hidden_size: int,
        da_num_layers: int = 2,
        da_cell_type: str = "LSTM",
        da_bidirectional: bool = True,
        da_attention: bool = True,
        da_dropout: float = 0.1
    ):
        super().__init__()
        self.da_vocab_size = da_vocab_size
        self.da_embedding_size = da_embedding_size
        self.da_hidden_size = da_hidden_size
        self.da_num_layers = da_num_layers
        self.da_cell_type = da_cell_type
        self.da_bidirectional = da_bidirectional
        self.da_attention = da_attention
        self.da_dropout = da_dropout
        
        # DA TOKEN EMBEDDING - turn humie words into mighty Ork battle cries!
        # Why we need 'em (Orky): Orks don't fink in puny humie words - dey fink in WAAAGHs!
        # Why we need 'em (Humie): Token embeddings convert discrete tokens to continuous vectors
        self.da_token_embedding = nn.Embedding(da_vocab_size, da_embedding_size)
        
        # DA STACKED RNN - layered memory boyz
        # Why we need 'em (Orky): Multiple layers of memory boyz for maximum WAAAGH!
        # Why we need 'em (Humie): Stacked layers enable deep sequential processing
        self.da_stacked_rnn = OrkyStackedRNN(
            da_embedding_size, da_hidden_size, da_num_layers,
            da_cell_type, da_dropout, da_bidirectional
        )
        
        # DA ATTENTION MECHANISM - focusin' on important stuff
        # Why we need 'em (Orky): Da boyz need to focus on important memories!
        # Why we need 'em (Humie): Attention mechanism improves focus and performance
        if da_attention:
            da_attention_input_size = da_hidden_size * (2 if da_bidirectional else 1)
            self.da_attention_layer = OrkyRNNAttention(da_attention_input_size)
        else:
            self.da_attention_layer = None
        
        # DA OUTPUT PROJECTION - turn Ork thoughts back into humie words
        # Why we need 'em (Orky): Orks need to communicate their wisdom to da humies!
        # Why we need 'em (Humie): Output projection predicts next token probabilities
        da_output_size = da_hidden_size * (2 if da_bidirectional else 1)
        self.da_output_projection = nn.Linear(da_output_size, da_vocab_size)
        
        # TIE EMBEDDINGS - save memory like a thrifty Ork
        # Why we need 'em (Orky): Smart Orks reuse their equipment to save resources!
        # Why we need 'em (Humie): Weight tying reduces parameters and improves efficiency
        # Note: Weight tying only works when embedding_size == output_size
        if da_embedding_size == da_output_size:
            self.da_output_projection.weight = self.da_token_embedding.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """INITIALIZE DA WEIGHTS LIKE A PROPER ADVANCED RNN WARBOSS!"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, da_input_tokens: torch.Tensor) -> torch.Tensor:
        """
        FORWARD PASS - WHERE DA ADVANCED RNN SHINES!
        
        DIS IS WHERE DA ULTIMATE MEMORY SYSTEM GOES TO WAR:
        1. Convert humie words into mighty Ork battle cries
        2. Process through stacked bidirectional layers
        3. Apply attention for focus
        4. Generate predictions with maximum WAAAGH!
        """
        # STEP 1: DA TOKEN EMBEDDING - convert humie words to Ork battle cries!
        da_embedded_tokens = self.da_token_embedding(da_input_tokens)
        
        # STEP 2: DA STACKED RNN - layered memory boyz in action!
        da_rnn_output = self.da_stacked_rnn(da_embedded_tokens)
        
        # STEP 3: DA ATTENTION - focus on important stuff!
        if self.da_attention and self.da_attention_layer is not None:
            da_context_vector, da_attention_weights = self.da_attention_layer(da_rnn_output)
            # Use context vector for all time steps (broadcast)
            da_attended_output = da_context_vector.unsqueeze(1).expand(-1, da_rnn_output.size(1), -1)
        else:
            da_attended_output = da_rnn_output
        
        # STEP 4: GET DA LOGITS - predict da next mighty victory!
        da_output_logits = self.da_output_projection(da_attended_output)
        
        return da_output_logits
    
    def get_attention_weights(self, da_input_tokens: torch.Tensor) -> Optional[torch.Tensor]:
        """
        GET ATTENTION WEIGHTS - SEE WHAT DA BOYZ ARE FOCUSIN' ON!
        
        DIS SHOWS WHICH PARTS OF DA SEQUENCE DA BOYZ FOCUS ON:
        1. Process da input
        2. Get attention weights
        3. See what's important!
        """
        if not self.da_attention or self.da_attention_layer is None:
            return None
        
        da_embedded_tokens = self.da_token_embedding(da_input_tokens)
        da_rnn_output = self.da_stacked_rnn(da_embedded_tokens)
        _, da_attention_weights = self.da_attention_layer(da_rnn_output)
        
        return da_attention_weights

def create_advanced_orky_rnn(
    da_vocab_size: int = 50000,
    da_embedding_size: int = 256,
    da_hidden_size: int = 512,
    da_num_layers: int = 2,
    da_cell_type: str = "LSTM",
    da_bidirectional: bool = True,
    da_attention: bool = True,
    da_dropout: float = 0.1
) -> OrkyAdvancedRNN:
    """
    CREATE AN ADVANCED RNN WAAAGH MODEL FOR ULTIMATE MEMORY BATTLES!
    
    Dis creates da most advanced RNN model with all da fancy features!
    Like fielding da ultimate memory system with bidirectional processing,
    attention mechanisms, and all da modern techniques!
    """
    print(f"🧠⚡ Creating Advanced Orky RNN Waaagh Model!")
    print(f"   Vocab Size: {da_vocab_size}")
    print(f"   Embedding Size: {da_embedding_size}")
    print(f"   Hidden Size: {da_hidden_size}")
    print(f"   Layers: {da_num_layers}")
    print(f"   Cell Type: {da_cell_type}")
    print(f"   Bidirectional: {da_bidirectional}")
    print(f"   Attention: {da_attention}")
    print(f"   Dropout: {da_dropout}")
    
    model = OrkyAdvancedRNN(
        da_vocab_size=da_vocab_size,
        da_embedding_size=da_embedding_size,
        da_hidden_size=da_hidden_size,
        da_num_layers=da_num_layers,
        da_cell_type=da_cell_type,
        da_bidirectional=da_bidirectional,
        da_attention=da_attention,
        da_dropout=da_dropout
    )
    
    # Count da parameters like countin' da horde
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Model Size: {total_params * 4 / (1024**2):.1f} MB")
    
    return model

if __name__ == "__main__":
    # QUICK TEST OF DA ADVANCED RNN WAAAGH MODEL
    print("🧠⚡ Testing da Advanced RNN Waaagh Model! ⚡🧠")
    
    # Create a small advanced model for testing
    model = create_advanced_orky_rnn(
        da_vocab_size=100,
        da_embedding_size=32,
        da_hidden_size=64,
        da_num_layers=2,
        da_cell_type="LSTM",
        da_bidirectional=True,
        da_attention=True,
        da_dropout=0.1
    )
    
    # Test forward pass
    batch_size, seq_len = 1, 8
    input_tokens = torch.randint(0, 100, (batch_size, seq_len))
    
    print(f"\nInput shape: {input_tokens.shape}")
    print(f"Input tokens: {input_tokens[0].tolist()}")
    
    logits = model(input_tokens)
    print(f"Output shape: {logits.shape}")
    
    # Test attention weights
    attention_weights = model.get_attention_weights(input_tokens)
    if attention_weights is not None:
        print(f"Attention weights shape: {attention_weights.shape}")
        print(f"Attention weights: {attention_weights[0].tolist()}")
    
    print("\n✅ Advanced RNN Waaagh model test complete! WAAAGH!")
