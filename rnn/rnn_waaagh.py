#!/usr/bin/env python3
"""
RNN WAAAGH - DA MEMORY BOYZ! 🧠⚡

Dis is da RNN (Recurrent Neural Network) architecture - like havin' Ork boyz
who remember things in order and pass da memory down da line!

WHAT IS DA RNN WAAAGH?
- Recurrent Neural Networks for sequential memory
- Like Ork boyz who remember things in order
- Processes sequences one step at a time
- Maintains hidden state for memory

FOR DA BOYZ: Dis is like havin' Ork boyz who remember things in order and
pass da memory down da line! Each boy remembers what happened before and
passes it to da next boy in da chain!

FOR HUMANS: This implements RNN architecture with LSTM and GRU variants
for sequential processing with memory mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List, Dict, Any
import time

class OrkyRNNCell(nn.Module):
    """
    DA RNN CELL - DA BASIC MEMORY BOY!

    Dis is like a basic Ork boy who remembers what happened before and
    combines it with new information. He's simple but effective!

    WHY WE NEED DIS (ORKY):
    Every WAAAGH needs basic boyz who can remember things in order!
    Dis is like havin' Ork boyz who pass information down da line!

    WHY WE NEED DIS (HUMIE):
    Basic RNN cell processes sequential data with hidden state for memory.
    """

    def __init__(self, da_input_size: int, da_hidden_size: int):
        super().__init__()
        # DA INPUT SIZE - how big da input is
        # Why we need 'em (Orky): Da boy needs to understand da input size!
        # Why we need 'em (Humie): Input size determines input dimension
        self.da_input_size = da_input_size

        # DA HIDDEN SIZE - how much memory da boy has
        # Why we need 'em (Orky): Da boy needs enough memory for his job!
        # Why we need 'em (Humie): Hidden size determines memory capacity
        self.da_hidden_size = da_hidden_size

        # DA INPUT WEIGHTS - how da boy processes new information
        # Why we need 'em (Orky): Da boy needs to understand new information!
        # Why we need 'em (Humie): Input weights process current input
        self.da_input_weights = nn.Linear(da_input_size, da_hidden_size)

        # DA HIDDEN WEIGHTS - how da boy processes his memory
        # Why we need 'em (Orky): Da boy needs to process his memory!
        # Why we need 'em (Humie): Hidden weights process previous hidden state
        self.da_hidden_weights = nn.Linear(da_hidden_size, da_hidden_size)

        # DA ACTIVATION - da boy's energy surge
        # Why we need 'em (Orky): Da boy has his own way of channelin' WAAAGH energy!
        # Why we need 'em (Humie): Activation function provides non-linearity
        self.da_activation = nn.Tanh()

    def do_da_rnn_step(self, da_input: torch.Tensor, da_hidden: torch.Tensor) -> torch.Tensor:
        """
        DO DA RNN STEP - WHERE DA MEMORY BOY SHINES!

        DIS IS WHERE DA MEMORY BOY PROCESSES NEW INFORMATION!

        da_input: (batch_size, da_input_size) - new information
        da_hidden: (batch_size, da_hidden_size) - previous memory
        Returns: (batch_size, da_hidden_size) - new memory

        DIS IS LIKE WATCHIN' A MEMORY BOY WORK:
        1. Process da new information
        2. Process da previous memory
        3. Combine 'em together
        4. Apply da activation for energy surge
        """
        # STEP 1: PROCESS DA NEW INFORMATION - what's happening now
        # Why we need 'em (Orky): Da boy needs to understand new information!
        # Why we need 'em (Humie): Input processing handles current input
        da_input_processed = self.da_input_weights(da_input)

        # STEP 2: PROCESS DA PREVIOUS MEMORY - what happened before
        # Why we need 'em (Orky): Da boy needs to process his memory!
        # Why we need 'em (Humie): Hidden processing handles previous state
        da_hidden_processed = self.da_hidden_weights(da_hidden)

        # STEP 3: COMBINE DA INFORMATION - new + memory
        # Why we need 'em (Orky): Da boy needs to combine new info with memory!
        # Why we need 'em (Humie): Combination integrates current input with previous state
        da_combined = da_input_processed + da_hidden_processed

        # STEP 4: APPLY DA ACTIVATION - energy surge
        # Why we need 'em (Orky): Da boy has his own way of channelin' WAAAGH energy!
        # Why we need 'em (Humie): Activation function provides non-linearity
        da_new_hidden = self.da_activation(da_combined)

        return da_new_hidden

class OrkyLSTMCell(nn.Module):
    """
    DA LSTM CELL - DA SMART MEMORY BOY!

    Dis is like a smart Ork boy who has better memory management. He can
    forget unimportant stuff and remember important stuff selectively!

    WHY WE NEED DIS (ORKY):
    Every WAAAGH needs smart boyz who can manage their memory better!
    Dis is like havin' Ork boyz who can forget boring stuff and remember
    important stuff selectively!

    WHY WE NEED DIS (HUMIE):
    LSTM cell provides better memory management with forget, input, and output gates
    for selective memory and gradient flow.
    """

    def __init__(self, da_input_size: int, da_hidden_size: int):
        super().__init__()
        # DA INPUT SIZE - how big da input is
        # Why we need 'em (Orky): Da boy needs to understand da input size!
        # Why we need 'em (Humie): Input size determines input dimension
        self.da_input_size = da_input_size

        # DA HIDDEN SIZE - how much memory da boy has
        # Why we need 'em (Orky): Da boy needs enough memory for his job!
        # Why we need 'em (Humie): Hidden size determines memory capacity
        self.da_hidden_size = da_hidden_size

        # DA FORGET GATE - da boy forgets unimportant stuff
        # Why we need 'em (Orky): Da boy needs to forget boring stuff!
        # Why we need 'em (Humie): Forget gate controls what to forget from cell state
        self.da_forget_gate = nn.Linear(da_input_size + da_hidden_size, da_hidden_size)

        # DA INPUT GATE - da boy decides what to remember
        # Why we need 'em (Orky): Da boy needs to decide what to remember!
        # Why we need 'em (Humie): Input gate controls what new information to store
        self.da_input_gate = nn.Linear(da_input_size + da_hidden_size, da_hidden_size)

        # DA CANDIDATE VALUES - da boy's potential memories
        # Why we need 'em (Orky): Da boy needs to create potential memories!
        # Why we need 'em (Humie): Candidate values represent potential new cell state
        self.da_candidate_values = nn.Linear(da_input_size + da_hidden_size, da_hidden_size)

        # DA OUTPUT GATE - da boy decides what to share
        # Why we need 'em (Orky): Da boy needs to decide what to share!
        # Why we need 'em (Humie): Output gate controls what to output from cell state
        self.da_output_gate = nn.Linear(da_input_size + da_hidden_size, da_hidden_size)

        # DA ACTIVATION - da boy's energy surge
        # Why we need 'em (Orky): Da boy has his own way of channelin' WAAAGH energy!
        # Why we need 'em (Humie): Activation function provides non-linearity
        self.da_activation = nn.Tanh()
        self.da_sigmoid = nn.Sigmoid()

    def do_da_lstm_step(self, da_input: torch.Tensor, da_hidden: torch.Tensor, da_cell: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DO DA LSTM STEP - WHERE DA SMART MEMORY BOY SHINES!

        DIS IS WHERE DA SMART MEMORY BOY PROCESSES NEW INFORMATION WITH SMART MEMORY!

        da_input: (batch_size, da_input_size) - new information
        da_hidden: (batch_size, da_hidden_size) - previous hidden state
        da_cell: (batch_size, da_hidden_size) - previous cell state
        Returns: (new_hidden, new_cell) - new hidden and cell states

        DIS IS LIKE WATCHIN' A SMART MEMORY BOY WORK:
        1. Decide what to forget from old memory
        2. Decide what new information to remember
        3. Create potential new memories
        4. Update da cell state with forget + input
        5. Decide what to share from da new cell state
        """
        # STEP 1: COMBINE INPUT AND HIDDEN - prepare for processing
        # Why we need 'em (Orky): Da boy needs to see both new info and memory!
        # Why we need 'em (Humie): Concatenation prepares input for gate processing
        da_combined = torch.cat([da_input, da_hidden], dim=-1)

        # STEP 2: DA FORGET GATE - decide what to forget
        # Why we need 'em (Orky): Da boy needs to forget boring stuff!
        # Why we need 'em (Humie): Forget gate controls what to forget from cell state
        da_forget_gate = self.da_sigmoid(self.da_forget_gate(da_combined))

        # STEP 3: DA INPUT GATE - decide what to remember
        # Why we need 'em (Orky): Da boy needs to decide what to remember!
        # Why we need 'em (Humie): Input gate controls what new information to store
        da_input_gate = self.da_sigmoid(self.da_input_gate(da_combined))

        # STEP 4: DA CANDIDATE VALUES - create potential memories
        # Why we need 'em (Orky): Da boy needs to create potential memories!
        # Why we need 'em (Humie): Candidate values represent potential new cell state
        da_candidate = self.da_activation(self.da_candidate_values(da_combined))

        # STEP 5: UPDATE DA CELL STATE - forget + remember
        # Why we need 'em (Orky): Da boy needs to update his memory!
        # Why we need 'em (Humie): Cell state update combines forget and input operations
        da_new_cell = da_forget_gate * da_cell + da_input_gate * da_candidate

        # STEP 6: DA OUTPUT GATE - decide what to share
        # Why we need 'em (Orky): Da boy needs to decide what to share!
        # Why we need 'em (Humie): Output gate controls what to output from cell state
        da_output_gate = self.da_sigmoid(self.da_output_gate(da_combined))

        # STEP 7: DA NEW HIDDEN STATE - what to share
        # Why we need 'em (Orky): Da boy needs to share his processed memory!
        # Why we need 'em (Humie): Hidden state is the filtered cell state
        da_new_hidden = da_output_gate * self.da_activation(da_new_cell)

        return da_new_hidden, da_new_cell

class OrkyGRUCell(nn.Module):
    """
    DA GRU CELL - DA EFFICIENT MEMORY BOY!

    Dis is like an efficient Ork boy who has good memory management but
    is simpler than da LSTM boy. He's faster but still smart!

    WHY WE NEED DIS (ORKY):
    Every WAAAGH needs efficient boyz who can manage memory well but
    don't need all da complexity of da LSTM boy!

    WHY WE NEED DIS (HUMIE):
    GRU cell provides efficient memory management with reset and update gates
    for selective memory with fewer parameters than LSTM.
    """

    def __init__(self, da_input_size: int, da_hidden_size: int):
        super().__init__()
        # DA INPUT SIZE - how big da input is
        # Why we need 'em (Orky): Da boy needs to understand da input size!
        # Why we need 'em (Humie): Input size determines input dimension
        self.da_input_size = da_input_size

        # DA HIDDEN SIZE - how much memory da boy has
        # Why we need 'em (Orky): Da boy needs enough memory for his job!
        # Why we need 'em (Humie): Hidden size determines memory capacity
        self.da_hidden_size = da_hidden_size

        # DA RESET GATE - da boy resets his memory
        # Why we need 'em (Orky): Da boy needs to reset his memory sometimes!
        # Why we need 'em (Humie): Reset gate controls what to forget from hidden state
        self.da_reset_gate = nn.Linear(da_input_size + da_hidden_size, da_hidden_size)

        # DA UPDATE GATE - da boy updates his memory
        # Why we need 'em (Orky): Da boy needs to update his memory!
        # Why we need 'em (Humie): Update gate controls how much to update hidden state
        self.da_update_gate = nn.Linear(da_input_size + da_hidden_size, da_hidden_size)

        # DA CANDIDATE VALUES - da boy's potential memories
        # Why we need 'em (Orky): Da boy needs to create potential memories!
        # Why we need 'em (Humie): Candidate values represent potential new hidden state
        self.da_candidate_values = nn.Linear(da_input_size + da_hidden_size, da_hidden_size)

        # DA ACTIVATION - da boy's energy surge
        # Why we need 'em (Orky): Da boy has his own way of channelin' WAAAGH energy!
        # Why we need 'em (Humie): Activation function provides non-linearity
        self.da_activation = nn.Tanh()
        self.da_sigmoid = nn.Sigmoid()

    def do_da_gru_step(self, da_input: torch.Tensor, da_hidden: torch.Tensor) -> torch.Tensor:
        """
        DO DA GRU STEP - WHERE DA EFFICIENT MEMORY BOY SHINES!

        DIS IS WHERE DA EFFICIENT MEMORY BOY PROCESSES NEW INFORMATION WITH EFFICIENT MEMORY!

        da_input: (batch_size, da_input_size) - new information
        da_hidden: (batch_size, da_hidden_size) - previous hidden state
        Returns: (batch_size, da_hidden_size) - new hidden state

        DIS IS LIKE WATCHIN' AN EFFICIENT MEMORY BOY WORK:
        1. Decide what to reset from old memory
        2. Decide how much to update memory
        3. Create potential new memories
        4. Update da hidden state with reset + update
        """
        # STEP 1: COMBINE INPUT AND HIDDEN - prepare for processing
        # Why we need 'em (Orky): Da boy needs to see both new info and memory!
        # Why we need 'em (Humie): Concatenation prepares input for gate processing
        da_combined = torch.cat([da_input, da_hidden], dim=-1)

        # STEP 2: DA RESET GATE - decide what to reset
        # Why we need 'em (Orky): Da boy needs to reset his memory sometimes!
        # Why we need 'em (Humie): Reset gate controls what to forget from hidden state
        da_reset_gate = self.da_sigmoid(self.da_reset_gate(da_combined))

        # STEP 3: DA UPDATE GATE - decide how much to update
        # Why we need 'em (Orky): Da boy needs to decide how much to update!
        # Why we need 'em (Humie): Update gate controls how much to update hidden state
        da_update_gate = self.da_sigmoid(self.da_update_gate(da_combined))

        # STEP 4: DA CANDIDATE VALUES - create potential memories
        # Why we need 'em (Orky): Da boy needs to create potential memories!
        # Why we need 'em (Humie): Candidate values represent potential new hidden state
        da_candidate = self.da_activation(self.da_candidate_values(da_combined))

        # STEP 5: UPDATE DA HIDDEN STATE - reset + update
        # Why we need 'em (Orky): Da boy needs to update his memory!
        # Why we need 'em (Humie): Hidden state update combines reset and update operations
        da_new_hidden = (1 - da_update_gate) * da_hidden + da_update_gate * da_candidate

        return da_new_hidden

class OrkyRNNWaaagh(nn.Module):
    """
    DA RNN WAAAGH - DA COMPLETE MEMORY BOYZ SYSTEM!

    Dis is da complete RNN system with different types of memory boyz!
    You can choose between basic RNN, LSTM, or GRU memory boyz!

    WHY WE NEED DIS (ORKY):
    Dis is da ultimate memory system! It combines different types of
    memory boyz who can remember things in order and pass da memory
    down da line! Perfect for sequential processing!

    WHY WE NEED DIS (HUMIE):
    Complete RNN system with different cell types (RNN, LSTM, GRU) for
    sequential processing with various memory mechanisms.
    """

    def __init__(
        self,
        da_vocab_size: int,
        da_embedding_size: int,
        da_hidden_size: int,
        da_num_layers: int = 1,
        da_cell_type: str = "LSTM"
    ):
        super().__init__()
        # DA VOCABULARY SIZE - how many different words da Orks know
        # Why we need 'em (Orky): Da bigger da vocabulary, da more battle cries da Orks know!
        # Why we need 'em (Humie): Vocabulary size determines the number of possible tokens
        self.da_vocab_size = da_vocab_size

        # DA EMBEDDING SIZE - how big da word embeddings are
        # Why we need 'em (Orky): Da boyz need to understand words in Ork terms!
        # Why we need 'em (Humie): Embedding size determines word representation dimension
        self.da_embedding_size = da_embedding_size

        # DA HIDDEN SIZE - how much memory da boyz have
        # Why we need 'em (Orky): Da boyz need enough memory for their job!
        # Why we need 'em (Humie): Hidden size determines memory capacity
        self.da_hidden_size = da_hidden_size

        # DA NUMBER OF LAYERS - how many memory boyz we stack
        # Why we need 'em (Orky): More layers = deeper memory and better processing!
        # Why we need 'em (Humie): More layers enable more complex sequential processing
        self.da_num_layers = da_num_layers

        # DA CELL TYPE - what kind of memory boyz we use
        # Why we need 'em (Orky): Different boyz have different memory abilities!
        # Why we need 'em (Humie): Cell type determines memory mechanism (RNN, LSTM, GRU)
        self.da_cell_type = da_cell_type

        # DA TOKEN EMBEDDING - turn humie words into mighty Ork battle cries!
        # Why we need 'em (Orky): Orks don't fink in puny humie words - dey fink in WAAAGHs!
        # Why we need 'em (Humie): Token embeddings convert discrete tokens to continuous vectors
        self.da_token_embedding = nn.Embedding(da_vocab_size, da_embedding_size)

        # DA RNN LAYERS - stacked memory boyz
        # Why we need 'em (Orky): Multiple layers of memory boyz for maximum WAAAGH!
        # Why we need 'em (Humie): Stacked layers enable deep sequential processing
        if da_cell_type == "RNN":
            self.da_rnn_layers = nn.ModuleList([
                OrkyRNNCell(da_embedding_size if i == 0 else da_hidden_size, da_hidden_size)
                for i in range(da_num_layers)
            ])
        elif da_cell_type == "LSTM":
            self.da_rnn_layers = nn.ModuleList([
                OrkyLSTMCell(da_embedding_size if i == 0 else da_hidden_size, da_hidden_size)
                for i in range(da_num_layers)
            ])
        elif da_cell_type == "GRU":
            self.da_rnn_layers = nn.ModuleList([
                OrkyGRUCell(da_embedding_size if i == 0 else da_hidden_size, da_hidden_size)
                for i in range(da_num_layers)
            ])

        # DA OUTPUT PROJECTION - turn Ork thoughts back into humie words
        # Why we need 'em (Orky): Orks need to communicate their wisdom to da humies!
        # Why we need 'em (Humie): Output projection predicts next token probabilities
        self.da_output_projection = nn.Linear(da_hidden_size, da_vocab_size)

        # TIE EMBEDDINGS - save memory like a thrifty Ork
        # Why we need 'em (Orky): Smart Orks reuse their equipment to save resources!
        # Why we need 'em (Humie): Weight tying reduces parameters and improves efficiency
        # Note: We'll tie weights after initialization to avoid dimension mismatch

        self.apply(self._init_weights)
        
        # TIE EMBEDDINGS AFTER INITIALIZATION - save memory like a thrifty Ork
        # Why we need 'em (Orky): Smart Orks reuse their equipment to save resources!
        # Why we need 'em (Humie): Weight tying reduces parameters and improves efficiency
        if da_embedding_size == da_hidden_size:
            self.da_output_projection.weight = self.da_token_embedding.weight

    def _init_weights(self, module):
        """INITIALIZE DA WEIGHTS LIKE A PROPER RNN WARBOSS!"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def unleash_da_rnn_waaagh(self, da_input_tokens: torch.Tensor) -> torch.Tensor:
        """
        UNLEASH DA RNN WAAAGH - DA ULTIMATE MEMORY SYSTEM!

        DIS IS WHERE DA ENTIRE RNN SYSTEM GOES TO WAR:
        1. Convert humie words into mighty Ork battle cries
        2. Process through multiple layers of memory boyz
        3. Each boy remembers what happened before and passes it down
        4. Generate predictions based on da sequential memory!

        da_input_tokens: (batch_size, seq_len) - da input token indices
        Returns: (batch_size, seq_len, da_vocab_size) - predictions for each position
        """
        # STEP 1: DA TOKEN EMBEDDING - convert humie words to Ork battle cries!
        # Why we need 'em (Orky): Orks don't fink in puny humie words - dey fink in WAAAGHs!
        # Why we need 'em (Humie): Token embeddings convert discrete tokens to continuous vectors
        da_embedded_tokens = self.da_token_embedding(da_input_tokens)

        # STEP 2: PROCESS THROUGH DA RNN LAYERS - memory boyz in action!
        # Why we need 'em (Orky): Multiple layers of memory boyz for maximum WAAAGH!
        # Why we need 'em (Humie): Stacked RNN layers enable deep sequential processing
        da_batch_size, da_seq_len, _ = da_embedded_tokens.shape
        da_outputs = []

        # Initialize hidden states for each layer
        if self.da_cell_type == "LSTM":
            da_hidden_states = [torch.zeros(da_batch_size, self.da_hidden_size, device=da_input_tokens.device) for _ in range(self.da_num_layers)]
            da_cell_states = [torch.zeros(da_batch_size, self.da_hidden_size, device=da_input_tokens.device) for _ in range(self.da_num_layers)]
        else:
            da_hidden_states = [torch.zeros(da_batch_size, self.da_hidden_size, device=da_input_tokens.device) for _ in range(self.da_num_layers)]

        # Process each time step
        for t in range(da_seq_len):
            da_current_input = da_embedded_tokens[:, t, :]
            
            # Process through each layer
            for layer_idx, da_rnn_layer in enumerate(self.da_rnn_layers):
                if self.da_cell_type == "LSTM":
                    da_hidden_states[layer_idx], da_cell_states[layer_idx] = da_rnn_layer.do_da_lstm_step(
                        da_current_input, da_hidden_states[layer_idx], da_cell_states[layer_idx]
                    )
                elif self.da_cell_type == "GRU":
                    da_hidden_states[layer_idx] = da_rnn_layer.do_da_gru_step(
                        da_current_input, da_hidden_states[layer_idx]
                    )
                else:  # RNN
                    da_hidden_states[layer_idx] = da_rnn_layer.do_da_rnn_step(
                        da_current_input, da_hidden_states[layer_idx]
                    )
                
                da_current_input = da_hidden_states[layer_idx]

            da_outputs.append(da_current_input)

        # STEP 3: STACK DA OUTPUTS - combine all time steps
        # Why we need 'em (Orky): Da boyz need to combine all their work!
        # Why we need 'em (Humie): Stacking combines outputs from all time steps
        da_output_tensor = torch.stack(da_outputs, dim=1)

        # STEP 4: GET DA LOGITS - predict da next mighty victory!
        # Why we need 'em (Orky): Da boyz need to predict da next battle cry!
        # Why we need 'em (Humie): Output projection generates token probabilities
        da_output_logits = self.da_output_projection(da_output_tensor)

        return da_output_logits

def create_orky_rnn_waaagh(
    da_vocab_size: int = 50000,
    da_embedding_size: int = 256,
    da_hidden_size: int = 512,
    da_num_layers: int = 2,
    da_cell_type: str = "LSTM"
) -> OrkyRNNWaaagh:
    """
    CREATE A FULL-SIZED RNN WAAAGH MODEL FOR REAL MEMORY BATTLES!

    Dis creates a big RNN Waaagh model ready for serious sequential
    processing. Like fielding a complete memory system with different
    types of memory boyz!
    """
    print(f"🧠⚡ Creating Orky RNN Waaagh Model!")
    print(f"   Vocab Size: {da_vocab_size}")
    print(f"   Embedding Size: {da_embedding_size}")
    print(f"   Hidden Size: {da_hidden_size}")
    print(f"   Layers: {da_num_layers}")
    print(f"   Cell Type: {da_cell_type}")

    model = OrkyRNNWaaagh(
        da_vocab_size=da_vocab_size,
        da_embedding_size=da_embedding_size,
        da_hidden_size=da_hidden_size,
        da_num_layers=da_num_layers,
        da_cell_type=da_cell_type
    )

    # Count da parameters like countin' da horde
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Model Size: {total_params * 4 / (1024**2):.1f} MB")

    return model

if __name__ == "__main__":
    # QUICK TEST OF DA RNN WAAAGH MODEL
    print("🧠⚡ Testing da RNN Waaagh Model! ⚡🧠")

    # Create a small model for testing
    model = create_orky_rnn_waaagh(
        da_vocab_size=100,
        da_embedding_size=32,
        da_hidden_size=64,
        da_num_layers=1,
        da_cell_type="LSTM"
    )

    # Test forward pass
    batch_size, seq_len = 1, 8
    input_tokens = torch.randint(0, 100, (batch_size, seq_len))

    print(f"\nInput shape: {input_tokens.shape}")
    print(f"Input tokens: {input_tokens[0].tolist()}")

    logits = model.unleash_da_rnn_waaagh(input_tokens)
    print(f"Output shape: {logits.shape}")

    print("\n✅ RNN Waaagh model test complete! WAAAGH!")
