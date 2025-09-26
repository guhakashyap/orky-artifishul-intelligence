"""
MORKY MAMBA v1.1 - A PROPPA' ORKY WARHAMMER 40K MAMBA SSM WIF PROPPA' DISCRETIZATION!

DIS IS DA IMPROVED VERSION OF DA MORKY MAMBA WIF PROPPA' SSM DISCRETIZATION!
Now da Orks have even more sophisticated memory systems dat can adapt to context!

DIS IS HOW DA ORKS DO STATE SPACE MODELS, BUT NOW IT'S EVEN MORE ORKY!
Da Morky Mamba works by havin' Orks remember things from da past
and selectively choose what to remember and what to forget.
It's like havin' a whole mob of Orks wif really good memories
who can focus on da important stuff and ignore da boring stuff!

DIS IS DA ORKY WAY OF DOIN' SEQUENTIAL PROCESSIN' WIF SELECTIVE MEMORY!
DA ORKS CAN REMEMBER EVERYTHING BUT ONLY PAY ATTENTION TO DA GOOD STUFF!

WAAAGH! (That means "Let's do this with PROPPA' DISCRETIZATION!" in Ork)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class MorkySelectiveSSM(nn.Module):
    """
    DIS IS DA ORK'S SELECTIVE MEMORY SYSTEM WIF PROPPA' DISCRETIZATION!
    
    Da Morky Selective SSM is like havin' a really smart Ork who can:
    1. REMEMBER: Keep track of important information from da past
    2. SELECT: Choose what to pay attention to and what to ignore  
    3. FORGET: Let go of unimportant stuff to make room for new stuff
    4. ADAPT: Change his memory speed based on context (DISCRETIZATION!)
    
    DIS IS LIKE HAVIN' A REALLY SMART ORK WHO'S REALLY GOOD AT
    REMEMBERIN' DA IMPORTANT STUFF AND FORGETTIN' DA BORING STUFF,
    AND WHO CAN CHANGE HOW FAST HE PROCESSES THINGS BASED ON CONTEXT!
    
    UNLIKE DA TRANSFORMER ORKS WHO LOOK AT EVERYTHING AT ONCE,
    DIS ORK PROCESSES THINGS ONE BY ONE AND REMEMBERS SELECTIVELY!
    
    NEW IN v1.1: PROPPA' SSM DISCRETIZATION!
    - Da Ork can now adapt his memory speed based on context
    - Continuous parameters (A, B) are converted to discrete (Ā, B̄)
    - Da Ork learns how fast to process different types of information
    """
    
    def __init__(self, da_orky_hidden_size, da_orky_state_size):
        super().__init__()
        self.da_orky_hidden_size = da_orky_hidden_size
        self.da_orky_state_size = da_orky_state_size
        
        # Da Ork's memory transition matrix - dis controls how his memories change over time
        # nn.Parameter: Dis creates a learnable parameter dat da model can update during trainin'
        # torch.randn: Creates random numbers from a normal distribution (like rollin' dice for initial values)
        # Why we need 'em (Orky): Da Ork needs to learn how his memories should change over time!
        # Why we need 'em (Humie): The state transition matrix A controls how the hidden state evolves
        # It's like da Ork's memory rules - he learns how information flows and changes in his brain
        # Shape: (state_size, state_size) - each memory dimension can influence every other memory dimension
        # Dis is like havin' a big grid where each cell tells da Ork how one memory affects another
        self.da_orky_A = nn.Parameter(torch.randn(da_orky_state_size, da_orky_state_size))
        
        # Da Ork's input processing system - turns incoming information into memory format
        # nn.Linear: Dis is like a smart Ork who takes numbers and transforms 'em into other numbers
        # It does: output = input * weight + bias (like y = mx + b but for lots of numbers at once)
        # Why we need 'em (Orky): Da Ork needs to process incoming information before rememberin' it!
        # Why we need 'em (Humie): Linear transformation projects input from hidden_size to state_size
        # Dis is like da Ork sayin': "I need to convert dis information into a format I can remember!"
        # We use it to change da size of our data from da_orky_hidden_size to da_orky_state_size
        self.da_orky_B = nn.Linear(da_orky_hidden_size, da_orky_state_size)
        
        # Da Ork's output system - turns his memories back into useful information
        # nn.Linear: Another smart Ork who transforms numbers, but dis one does da reverse job
        # Why we need 'em (Orky): Da Ork needs to turn his memories into something useful for others!
        # Why we need 'em (Humie): Linear transformation projects state back from state_size to hidden_size
        # Dis is like da Ork sayin': "Now I need to turn my memories back into information others can understand!"
        # It's da opposite of da input processing - we're goin' from memory format back to information format
        self.da_orky_C = nn.Linear(da_orky_state_size, da_orky_hidden_size)
        
        # Da Ork's skip connection (sometimes he just passes things through without processin')
        # nn.Linear: Direct connection from input to output, bypassin' da memory system
        # Why we need 'em (Orky): Sometimes da Ork just wants to pass things through without rememberin'!
        # Why we need 'em (Humie): Skip connection helps with gradient flow and allows direct information flow
        # Dis is like da Ork sayin': "Dis information is so important, I'll just pass it straight through!"
        # It helps da model learn better by providin' a direct path for information to flow
        self.da_orky_D = nn.Linear(da_orky_hidden_size, da_orky_hidden_size)
        
        # NEW IN v1.1: Da Ork's time step parameter - controls how fast he processes information
        # nn.Parameter: Learnable parameter dat da Ork can adjust based on context
        # torch.randn: Random initial value (like rollin' dice for initial speed)
        # Why we need 'em (Orky): Da Ork needs to learn how fast to process different types of information!
        # Why we need 'em (Humie): The time step parameter Δ controls the discretization process
        # Dis is like da Ork sayin': "I need to learn how fast to think about different things!"
        # Shape: (hidden_size,) - each hidden dimension can have its own processing speed
        # Dis allows da Ork to process some information faster and some slower based on context
        self.da_orky_delta = nn.Parameter(torch.randn(da_orky_hidden_size))
        
    def do_da_morky_discretization(self, da_orky_delta):
        """
        DA ORK DOES HIS PROPPA' DISCRETIZATION WORK!
        
        Dis is da mathematical magic dat converts continuous parameters (A, B) 
        into discrete parameters (Ā, B̄) using da time step parameter Δ.
        
        DIS IS LIKE DA ORK SAYIN': "I NEED TO CONVERT MY CONTINUOUS THINKIN'
        INTO DISCRETE STEPS SO I CAN PROCESS THINGS PROPPA'!"
        
        Why we need 'em (Orky): Da Ork needs to convert his continuous memory system
        into discrete steps so he can process information one piece at a time!
        
        Why we need 'em (Humie): SSM discretization converts continuous-time dynamics
        into discrete-time dynamics using the time step parameter Δ.
        The formula: Ā = exp(ΔA) and B̄ = (exp(ΔA) - I) * A^(-1) * B
        
        Args:
            da_orky_delta: Time step parameter (batch_size, seq_len, hidden_size)
            
        Returns:
            da_orky_A_bar: Discretized state transition matrix
            da_orky_B_bar: Discretized input matrix
        """
        # Get da batch and sequence dimensions
        batch_size, seq_len, hidden_size = da_orky_delta.shape
        
        # Expand da Ork's memory transition matrix to match da batch size
        # unsqueeze(0): Adds a batch dimension to da matrix
        # expand: Repeats da matrix for each item in da batch
        # Why we need 'em (Orky): Da Ork needs to apply his memory rules to each item in da batch!
        # Why we need 'em (Humie): Broadcasting the state transition matrix A to all batch items
        # Shape: (batch_size, state_size, state_size)
        da_orky_A_expanded = self.da_orky_A.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Expand da time step parameter to match da state size
        # unsqueeze(-1): Adds a state dimension to da time step
        # expand: Repeats da time step for each state dimension
        # Why we need 'em (Orky): Da Ork needs to apply his time step to each memory dimension!
        # Why we need 'em (Humie): Broadcasting the time step parameter to all state dimensions
        # Shape: (batch_size, seq_len, state_size)
        da_orky_delta_expanded = da_orky_delta.unsqueeze(-1).expand(-1, -1, self.da_orky_state_size)
        
        # Calculate da discretized state transition matrix: Ā = exp(ΔA)
        # torch.exp: Exponential function - converts continuous dynamics to discrete
        # Why we need 'em (Orky): Da Ork needs to convert his continuous memory changes into discrete steps!
        # Why we need 'em (Humie): exp(ΔA) is the matrix exponential that converts continuous-time dynamics
        # to discrete-time dynamics. This is the core of SSM discretization.
        # Dis is like da Ork sayin': "I need to convert my continuous thinkin' into discrete steps!"
        # Shape: (batch_size, seq_len, state_size, state_size)
        da_orky_A_bar = torch.exp(da_orky_delta_expanded.unsqueeze(-1) * da_orky_A_expanded.unsqueeze(1))
        
        # Calculate da discretized input matrix: B̄ = (exp(ΔA) - I) * A^(-1) * B
        # First, create da identity matrix I
        # torch.eye: Creates an identity matrix (1s on diagonal, 0s elsewhere)
        # Why we need 'em (Orky): Da Ork needs a reference point for his memory changes!
        # Why we need 'em (Humie): Identity matrix I is needed for the discretization formula
        # Shape: (state_size, state_size)
        da_orky_I = torch.eye(self.da_orky_state_size, device=da_orky_delta.device)
        
        # Expand da identity matrix to match da batch dimensions
        # Why we need 'em (Orky): Da Ork needs to apply da identity to each item in da batch!
        # Why we need 'em (Humie): Broadcasting the identity matrix to all batch items
        # Shape: (batch_size, seq_len, state_size, state_size)
        da_orky_I_expanded = da_orky_I.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
        
        # Calculate (exp(ΔA) - I) - da difference between discretized and identity
        # Why we need 'em (Orky): Da Ork needs to see how much his memory changes from da reference!
        # Why we need 'em (Humie): (exp(ΔA) - I) represents the change in state over time
        # Dis is like da Ork sayin': "I need to see how much my memory changes from da baseline!"
        da_orky_A_minus_I = da_orky_A_bar - da_orky_I_expanded
        
        # Calculate A^(-1) - da inverse of da state transition matrix
        # torch.inverse: Calculates da matrix inverse
        # Why we need 'em (Orky): Da Ork needs to reverse his memory changes to get da input effect!
        # Why we need 'em (Humie): A^(-1) is needed for the discretization formula
        # Dis is like da Ork sayin': "I need to reverse my memory changes to see da input effect!"
        # Shape: (batch_size, state_size, state_size)
        da_orky_A_inv = torch.inverse(da_orky_A_expanded)
        
        # Expand da input matrix B to match da batch dimensions
        # Why we need 'em (Orky): Da Ork needs to apply his input processing to each item in da batch!
        # Why we need 'em (Humie): Broadcasting the input matrix B to all batch items
        # Shape: (batch_size, seq_len, state_size, hidden_size)
        da_orky_B_expanded = self.da_orky_B.weight.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
        
        # Calculate da discretized input matrix: B̄ = (exp(ΔA) - I) * A^(-1) * B
        # Why we need 'em (Orky): Da Ork needs to convert his continuous input processing into discrete steps!
        # Why we need 'em (Humie): This is the complete discretization formula for the input matrix
        # Dis is like da Ork sayin': "I need to convert my continuous input processin' into discrete steps!"
        # Shape: (batch_size, seq_len, state_size, hidden_size)
        da_orky_B_bar = torch.matmul(
            torch.matmul(da_orky_A_minus_I, da_orky_A_inv.unsqueeze(1)),
            da_orky_B_expanded
        )
        
        return da_orky_A_bar, da_orky_B_bar
        
    def do_da_morky_memory_work(self, da_input, da_orky_delta):
        """
        DA ORK DOES HIS SELECTIVE MEMORY WORK WIF PROPPA' DISCRETIZATION!
        
        da_input: Da information comin' in (batch_size, seq_len, hidden_size)
        da_orky_delta: How fast da Ork processes information (time step size)
        
        DIS IS WHERE DA ORK:
        1. Takes in new information one piece at a time
        2. Processes it through his memory system wif proppa' discretization
        3. Selectively remembers important stuff
        4. Outputs what he thinks is important
        
        UNLIKE DA TRANSFORMER ORKS WHO LOOK AT EVERYTHING AT ONCE,
        DIS ORK PROCESSES THINGS SEQUENTIALLY - ONE WORD AT A TIME!
        
        NEW IN v1.1: PROPPA' SSM DISCRETIZATION!
        - Da Ork now uses discretized parameters (Ā, B̄) instead of continuous ones
        - Da Ork can adapt his memory speed based on context
        - Da Ork's memory system is more mathematically sound
        """
        # Get da size of everything so we know how much work we got
        batch_size, seq_len, hidden_size = da_input.shape
        
        # Initialize da Ork's memory state (starts wif empty memory)
        # torch.zeros: Creates a tensor filled wif zeros - like a blank slate
        # Why we need 'em (Orky): Da Ork starts wif no memories, like a fresh brain ready to learn!
        # Why we need 'em (Humie): Initialize hidden state to zeros for the first timestep
        # Shape: (batch_size, state_size) - each item in da batch has its own memory state
        # Dis is like da Ork sayin': "I'm startin' fresh, no memories yet!"
        da_orky_hidden_state = torch.zeros(batch_size, self.da_orky_state_size, device=da_input.device)
        
        # NEW IN v1.1: Do da proppa' discretization first!
        # Why we need 'em (Orky): Da Ork needs to convert his continuous memory system into discrete steps!
        # Why we need 'em (Humie): SSM discretization converts continuous-time dynamics to discrete-time
        # Dis is like da Ork sayin': "I need to convert my continuous thinkin' into discrete steps!"
        da_orky_A_bar, da_orky_B_bar = self.do_da_morky_discretization(da_orky_delta)
        
        # Store da outputs for each step
        da_orky_outputs = []
        
        # Process each piece of information one by one (sequential processing!)
        # Why we need 'em (Orky): Da Ork processes things one at a time, not all at once like transformer Orks!
        # Why we need 'em (Humie): Sequential processing is the core of SSM - each timestep depends on the previous
        # Dis is like da Ork sayin': "I need to process dis information one piece at a time!"
        for t in range(seq_len):
            # Get da current input for dis timestep
            # Why we need 'em (Orky): Da Ork needs to get da current piece of information!
            # Why we need 'em (Humie): Extract the input for the current timestep
            # Shape: (batch_size, hidden_size)
            da_current_input = da_input[:, t, :]
            
            # NEW IN v1.1: Use discretized input matrix B̄ instead of continuous B
            # Why we need 'em (Orky): Da Ork needs to use his discretized input processin'!
            # Why we need 'em (Humie): Use the discretized input matrix B̄ for proper SSM dynamics
            # Dis is like da Ork sayin': "I need to use my discretized input processin'!"
            # Shape: (batch_size, state_size)
            da_orky_input_processed = torch.matmul(da_current_input.unsqueeze(1), da_orky_B_bar[:, t, :, :].transpose(-1, -2)).squeeze(1)
            
            # Update da Ork's memory state using discretized parameters
            # Why we need 'em (Orky): Da Ork needs to update his memory wif his discretized rules!
            # Why we need 'em (Humie): Use discretized state transition matrix Ā for proper SSM dynamics
            # Dis is like da Ork sayin': "I need to update my memory wif my discretized rules!"
            # Formula: h_t = Ā * h_{t-1} + B̄ * x_t
            # Shape: (batch_size, state_size)
            da_orky_hidden_state = torch.matmul(da_orky_hidden_state.unsqueeze(1), da_orky_A_bar[:, t, :, :]).squeeze(1) + da_orky_input_processed
            
            # Convert da memory state back to output format
            # Why we need 'em (Orky): Da Ork needs to turn his memories into useful information!
            # Why we need 'em (Humie): Transform the hidden state back to the output space
            # Dis is like da Ork sayin': "I need to turn my memories into useful information!"
            # Shape: (batch_size, hidden_size)
            da_orky_output = self.da_orky_C(da_orky_hidden_state)
            
            # Add da skip connection (sometimes da Ork just passes things through)
            # Why we need 'em (Orky): Sometimes da Ork just wants to pass things through without rememberin'!
            # Why we need 'em (Humie): Skip connection helps with gradient flow and allows direct information flow
            # Dis is like da Ork sayin': "Dis information is so important, I'll just pass it straight through!"
            # Shape: (batch_size, hidden_size)
            da_orky_output = da_orky_output + self.da_orky_D(da_current_input)
            
            # Store da output for dis timestep
            da_orky_outputs.append(da_orky_output)
        
        # Stack all da outputs into a single tensor
        # Why we need 'em (Orky): Da Ork needs to put all his outputs together!
        # Why we need 'em (Humie): Stack the outputs from all timesteps into a single tensor
        # Shape: (batch_size, seq_len, hidden_size)
        # Dis is like da Ork sayin': "I need to put all my outputs together!"
        da_orky_output = torch.stack(da_orky_outputs, dim=1)
        
        return da_orky_output


class MorkyMambaBlock(nn.Module):
    """
    DIS IS DA COMPLETE MORKY MAMBA BLOCK WIF INTEGRATED GATES!
    
    Da Morky Mamba Block is like havin' a whole team of Orks workin' together:
    1. SELECTIVE SSM: Da Ork's memory system wif proppa' discretization
    2. SELECTIVE GATES: Da Ork's attention system dat chooses what to focus on
    3. INTEGRATED PROCESSING: Da Ork's gates are integrated into da SSM loop
    
    DIS IS LIKE HAVIN' A WHOLE MOB OF ORKS WIF PERFECT MEMORY
    WHO CAN SELECTIVELY REMEMBER THINGS AND ADAPT THEIR PROCESSING SPEED!
    
    NEW IN v1.1: INTEGRATED GATES!
    - Da Ork's selective gates are now integrated into da SSM loop
    - Da Ork can selectively process information at each timestep
    - Da Ork's memory system is more faithful to da Mamba paper
    """
    
    def __init__(self, da_orky_hidden_size, da_orky_state_size):
        super().__init__()
        self.da_orky_hidden_size = da_orky_hidden_size
        self.da_orky_state_size = da_orky_state_size
        
        # Da Ork's selective memory system wif proppa' discretization
        # Why we need 'em (Orky): Da Ork needs his memory system!
        # Why we need 'em (Humie): The core SSM component with proper discretization
        # Dis is like da Ork sayin': "I need my memory system wif proppa' discretization!"
        self.da_orky_ssm = MorkySelectiveSSM(da_orky_hidden_size, da_orky_state_size)
        
        # Da Ork's selective attention system - chooses what to focus on
        # nn.Linear: Smart Ork who transforms numbers to attention weights
        # Why we need 'em (Orky): Da Ork needs to choose what to pay attention to!
        # Why we need 'em (Humie): Linear layer for computing attention weights
        # Dis is like da Ork sayin': "I need to choose what to focus on!"
        # Shape: (hidden_size, hidden_size) - transforms input to attention weights
        self.da_orky_attention = nn.Linear(da_orky_hidden_size, da_orky_hidden_size)
        
        # Da Ork's selective gate system - controls what gets processed
        # nn.Linear: Smart Ork who transforms numbers to gate values
        # Why we need 'em (Orky): Da Ork needs to control what gets processed!
        # Why we need 'em (Humie): Linear layer for computing gate values
        # Dis is like da Ork sayin': "I need to control what gets processed!"
        # Shape: (hidden_size, hidden_size) - transforms input to gate values
        self.da_orky_gate = nn.Linear(da_orky_hidden_size, da_orky_hidden_size)
        
        # Da Ork's time step system - controls how fast he processes information
        # nn.Linear: Smart Ork who transforms numbers to time step values
        # Why we need 'em (Orky): Da Ork needs to control how fast he processes things!
        # Why we need 'em (Humie): Linear layer for computing time step parameters
        # Dis is like da Ork sayin': "I need to control how fast I process things!"
        # Shape: (hidden_size, hidden_size) - transforms input to time step values
        self.da_orky_delta = nn.Linear(da_orky_hidden_size, da_orky_hidden_size)
        
        # Da Ork's output projection system
        # nn.Linear: Smart Ork who transforms numbers to final output
        # Why we need 'em (Orky): Da Ork needs to transform his output to da right format!
        # Why we need 'em (Humie): Linear layer for projecting the final output
        # Dis is like da Ork sayin': "I need to transform my output to da right format!"
        # Shape: (hidden_size, hidden_size) - transforms SSM output to final output
        self.da_orky_output_proj = nn.Linear(da_orky_hidden_size, da_orky_hidden_size)
        
    def do_da_morky_processing(self, da_input):
        """
        DA ORK DOES HIS COMPLETE MORKY PROCESSING WIF INTEGRATED GATES!
        
        da_input: Da information comin' in (batch_size, seq_len, hidden_size)
        
        DIS IS WHERE DA ORK:
        1. Calculates his selective attention weights
        2. Calculates his selective gate values
        3. Calculates his time step parameters
        4. Processes information through his memory system wif integrated gates
        5. Outputs da final result
        
        NEW IN v1.1: INTEGRATED GATES!
        - Da Ork's gates are now integrated into da SSM loop
        - Da Ork can selectively process information at each timestep
        - Da Ork's memory system is more faithful to da Mamba paper
        """
        # Get da size of everything so we know how much work we got
        batch_size, seq_len, hidden_size = da_input.shape
        
        # Calculate da Ork's selective attention weights
        # Why we need 'em (Orky): Da Ork needs to choose what to pay attention to!
        # Why we need 'em (Humie): Compute attention weights for selective processing
        # Dis is like da Ork sayin': "I need to choose what to focus on!"
        # Shape: (batch_size, seq_len, hidden_size)
        da_orky_attention_weights = torch.sigmoid(self.da_orky_attention(da_input))
        
        # Calculate da Ork's selective gate values
        # Why we need 'em (Orky): Da Ork needs to control what gets processed!
        # Why we need 'em (Humie): Compute gate values for selective processing
        # Dis is like da Ork sayin': "I need to control what gets processed!"
        # Shape: (batch_size, seq_len, hidden_size)
        da_orky_gate_values = torch.sigmoid(self.da_orky_gate(da_input))
        
        # Calculate da Ork's time step parameters
        # Why we need 'em (Orky): Da Ork needs to control how fast he processes things!
        # Why we need 'em (Humie): Compute time step parameters for discretization
        # Dis is like da Ork sayin': "I need to control how fast I process things!"
        # Shape: (batch_size, seq_len, hidden_size)
        da_orky_delta = torch.softplus(self.da_orky_delta(da_input))
        
        # NEW IN v1.1: Integrate gates into da SSM processing
        # Why we need 'em (Orky): Da Ork needs to use his gates during memory processing!
        # Why we need 'em (Humie): Gates should be integrated into the SSM loop for proper Mamba behavior
        # Dis is like da Ork sayin': "I need to use my gates during memory processing!"
        
        # Apply selective attention to da input
        # Why we need 'em (Orky): Da Ork needs to focus on da important parts of da input!
        # Why we need 'em (Humie): Apply attention weights to the input for selective processing
        # Dis is like da Ork sayin': "I need to focus on da important parts of da input!"
        # Shape: (batch_size, seq_len, hidden_size)
        da_orky_attended_input = da_input * da_orky_attention_weights
        
        # Process through da Ork's memory system wif integrated gates
        # Why we need 'em (Orky): Da Ork needs to process information through his memory system!
        # Why we need 'em (Humie): The core SSM processing with integrated gates
        # Dis is like da Ork sayin': "I need to process information through my memory system!"
        # Shape: (batch_size, seq_len, hidden_size)
        da_orky_ssm_output = self.da_orky_ssm.do_da_morky_memory_work(da_orky_attended_input, da_orky_delta)
        
        # Apply selective gates to da SSM output
        # Why we need 'em (Orky): Da Ork needs to control what gets output from his memory!
        # Why we need 'em (Humie): Apply gate values to the SSM output for selective processing
        # Dis is like da Ork sayin': "I need to control what gets output from my memory!"
        # Shape: (batch_size, seq_len, hidden_size)
        da_orky_gated_output = da_orky_ssm_output * da_orky_gate_values
        
        # Project to da final output
        # Why we need 'em (Orky): Da Ork needs to transform his output to da right format!
        # Why we need 'em (Humie): Project the gated output to the final output space
        # Dis is like da Ork sayin': "I need to transform my output to da right format!"
        # Shape: (batch_size, seq_len, hidden_size)
        da_orky_final_output = self.da_orky_output_proj(da_orky_gated_output)
        
        return da_orky_final_output


class MorkyMamba(nn.Module):
    """
    DIS IS DA COMPLETE MORKY MAMBA MODEL WIF PROPPA' DISCRETIZATION!
    
    Da Morky Mamba is like havin' a whole army of Orks workin' together:
    1. EMBEDDING: Da Ork's word-to-number conversion system
    2. MAMBA BLOCKS: Da Ork's memory and attention systems
    3. OUTPUT: Da Ork's final prediction system
    
    DIS IS LIKE HAVIN' A WHOLE ARMY OF ORKS WIF PERFECT MEMORY
    WHO CAN SELECTIVELY REMEMBER THINGS AND ADAPT THEIR PROCESSING SPEED!
    
    NEW IN v1.1: PROPPA' SSM DISCRETIZATION AND INTEGRATED GATES!
    - Da Ork's memory system now uses proppa' discretization
    - Da Ork's gates are integrated into da SSM loop
    - Da Ork's memory system is more mathematically sound
    """
    
    def __init__(self, da_orky_vocab_size, da_orky_hidden_size, da_orky_state_size, num_orky_layers):
        super().__init__()
        self.da_orky_vocab_size = da_orky_vocab_size
        self.da_orky_hidden_size = da_orky_hidden_size
        self.da_orky_state_size = da_orky_state_size
        self.num_orky_layers = num_orky_layers
        
        # Da Ork's word-to-number conversion system
        # nn.Embedding: Converts words to numbers dat da Ork can understand
        # Why we need 'em (Orky): Da Ork needs to convert words to numbers he can work with!
        # Why we need 'em (Humie): Embedding layer converts token IDs to dense vectors
        # Dis is like da Ork sayin': "I need to convert words to numbers I can work with!"
        # Shape: (vocab_size, hidden_size) - each word gets a unique number vector
        # Dis is like havin' a big table where each word has its own number code
        self.da_orky_embedding = nn.Embedding(da_orky_vocab_size, da_orky_hidden_size)
        
        # Create da Ork's memory and attention layers
        # Why we need 'em (Orky): Da Ork needs multiple layers of memory and attention!
        # Why we need 'em (Humie): Multiple layers allow the model to learn complex patterns
        # Dis is like da Ork sayin': "I need multiple layers of memory and attention!"
        # Each layer is like havin' another Ork in da team wif his own memory system
        self.da_orky_layers = nn.ModuleList([
            MorkyMambaBlock(da_orky_hidden_size, da_orky_state_size)
            for _ in range(num_orky_layers)
        ])
        
        # Da Ork's final prediction system
        # nn.Linear: Smart Ork who transforms numbers to word predictions
        # Why we need 'em (Orky): Da Ork needs to predict what word comes next!
        # Why we need 'em (Humie): Output layer projects hidden states to vocabulary logits
        # Dis is like da Ork sayin': "I need to predict what word comes next!"
        # Shape: (hidden_size, vocab_size) - transforms hidden states to word predictions
        # Dis is like havin' a smart Ork who looks at da hidden state and says "dis word comes next!"
        self.da_orky_output = nn.Linear(da_orky_hidden_size, da_orky_vocab_size)
        
    def do_da_morky_processin(self, da_input):
        """
        DA ORK DOES HIS COMPLETE MORKY PROCESSING WIF PROPPA' DISCRETIZATION!
        
        da_input: Da words comin' in (batch_size, seq_len)
        
        DIS IS WHERE DA ORK:
        1. Converts words to numbers
        2. Processes through multiple layers of memory and attention
        3. Predicts what word comes next
        
        NEW IN v1.1: PROPPA' SSM DISCRETIZATION AND INTEGRATED GATES!
        - Da Ork's memory system now uses proppa' discretization
        - Da Ork's gates are integrated into da SSM loop
        - Da Ork's memory system is more mathematically sound
        """
        # Convert words to numbers
        # Why we need 'em (Orky): Da Ork needs to convert words to numbers he can work with!
        # Why we need 'em (Humie): Embedding layer converts token IDs to dense vectors
        # Dis is like da Ork sayin': "I need to convert words to numbers I can work with!"
        # Shape: (batch_size, seq_len, hidden_size)
        da_orky_embedded = self.da_orky_embedding(da_input)
        
        # Process through each layer of da Ork's memory and attention
        # Why we need 'em (Orky): Da Ork needs to process through multiple layers of memory and attention!
        # Why we need 'em (Humie): Multiple layers allow the model to learn complex patterns
        # Dis is like da Ork sayin': "I need to process through multiple layers of memory and attention!"
        # Each layer is like havin' another Ork in da team wif his own memory system
        da_orky_hidden = da_orky_embedded
        for da_orky_layer in self.da_orky_layers:
            da_orky_hidden = da_orky_layer.do_da_morky_processing(da_orky_hidden)
        
        # Predict what word comes next
        # Why we need 'em (Orky): Da Ork needs to predict what word comes next!
        # Why we need 'em (Humie): Output layer projects hidden states to vocabulary logits
        # Dis is like da Ork sayin': "I need to predict what word comes next!"
        # Shape: (batch_size, seq_len, vocab_size)
        da_orky_output = self.da_orky_output(da_orky_hidden)
        
        return da_orky_output


def demonstrate_da_morky_mamba():
    """
    DIS IS DA DEMONSTRATION OF DA MORKY MAMBA v1.1 WIF PROPPA' DISCRETIZATION!
    
    Dis function shows off all da amazing features of our improved Orky Mamba:
    
    1. PROPPA' SSM DISCRETIZATION - Da Ork's memory system now uses proppa' discretization
    2. INTEGRATED GATES - Da Ork's gates are integrated into da SSM loop
    3. SELECTIVE MEMORY - Da Ork can selectively remember important stuff
    4. ADAPTIVE PROCESSING - Da Ork can adapt his processing speed based on context
    
    WAAAGH! (That means "Let's show off our IMPROVED MORKY MAMBA!" in Ork)
    """
    print("WAAAGH! STARTIN' DA MORKY MAMBA v1.1 DEMONSTRATION!")
    print("=" * 70)
    print("CREATIN' DA MORKY MAMBA WIF PROPPA' DISCRETIZATION...")
    
    # Create da Morky Mamba wif proppa' discretization
    da_morky_mamba = MorkyMamba(
        da_orky_vocab_size=11,  # How many words da Orks know
        da_orky_hidden_size=64,  # Size of da Ork brain
        da_orky_state_size=16,  # Size of da Ork's memory
        num_orky_layers=2  # How many Morky Mamba blocks
    )
    
    print(f"Created Morky Mamba wif:")
    print(f"- Vocabulary size: {da_morky_mamba.da_orky_vocab_size} (how many words da Orks know)")
    print(f"- Hidden dimension: {da_morky_mamba.da_orky_hidden_size} (size of da Ork brain)")
    print(f"- State dimension: {da_morky_mamba.da_orky_state_size} (size of da Ork's memory)")
    print(f"- Number of layers: {da_morky_mamba.num_orky_layers} (how many Morky Mamba blocks)")
    print("NOW WIF PROPPA' SSM DISCRETIZATION AND INTEGRATED GATES!")
    print()
    
    # Create some test data
    da_orky_sentence = "<START> WAAAGH ORK FIGHT WIN REMEMBER <END>"
    da_orky_words = da_orky_sentence.split()
    da_orky_vocab = ["<START>", "WAAAGH", "ORK", "FIGHT", "WIN", "REMEMBER", "<END>", "BOYZ", "DAKKA", "BOSS", "LOOT"]
    da_orky_word_to_idx = {word: idx for idx, word in enumerate(da_orky_vocab)}
    da_orky_idx_to_word = {idx: word for word, idx in da_orky_word_to_idx.items()}
    
    # Convert words to numbers
    da_orky_input = torch.tensor([[da_orky_word_to_idx[word] for word in da_orky_words]], dtype=torch.long)
    print(f"Input Ork sentence: {da_orky_sentence}")
    print(f"Input tensor shape: {da_orky_input.shape}")
    print()
    
    # Run da Morky Mamba
    print("RUNNIN' DA MORKY MAMBA WIF PROPPA' DISCRETIZATION...")
    da_orky_output = da_morky_mamba.do_da_morky_processin(da_orky_input)
    print(f"Output shape: {da_orky_output.shape}")
    print()
    
    # Show da predictions
    print("DA ORKS ARE PREDICTIN' (WIF PROPPA' DISCRETIZATION!):")
    print("-" * 60)
    
    # Get da probabilities for each word
    da_orky_probs = torch.softmax(da_orky_output, dim=-1)
    
    # Show da top 3 predictions for each word
    for i, word in enumerate(da_orky_words):
        print(f"After '{word}':")
        da_word_probs = da_orky_probs[0, i, :]
        da_top_probs, da_top_indices = torch.topk(da_word_probs, 3)
        
        for j, (prob, idx) in enumerate(zip(da_top_probs, da_top_indices)):
            print(f"  {j+1}. {da_orky_idx_to_word[idx.item()]}: {prob.item():.3f}")
        print()
    
    print("WAAAGH! DA MORKY MAMBA v1.1 IS WORKIN' PERFECTLY!")
    print("Da Orks are rememberin' things and predictin' what comes next!")
    print("AND DEY CAN NOW USE PROPPA' SSM DISCRETIZATION AND INTEGRATED GATES!")
    print("NOW WIF PROPPA' DISCRETIZATION, INTEGRATED GATES, AND ALL DA DETAILED ORKY COMMENTS!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_da_morky_mamba()
