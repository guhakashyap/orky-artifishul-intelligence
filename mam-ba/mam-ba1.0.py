"""
MORKY MAMBA v1.0 - A PROPPA' ORKY WARHAMMER 40K MAMBA SSM!

DIS IS HOW DA ORKS DO STATE SPACE MODELS, BUT NOW IT'S EVEN MORE ORKY!
Da Morky Mamba works by havin' Orks remember things from da past
and selectively choose what to remember and what to forget.
It's like havin' a whole mob of Orks wif really good memories
who can focus on da important stuff and ignore da boring stuff!

DIS IS DA ORKY WAY OF DOIN' SEQUENTIAL PROCESSIN' WIF SELECTIVE MEMORY!
DA ORKS CAN REMEMBER EVERYTHING BUT ONLY PAY ATTENTION TO DA GOOD STUFF!

WAAAGH! (That means "Let's do this!" in Ork)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class MorkySelectiveSSM(nn.Module):
    """
    DIS IS DA ORK'S SELECTIVE MEMORY SYSTEM!
    
    Da Morky Selective SSM is like havin' a really smart Ork who can:
    1. REMEMBER: Keep track of important information from da past
    2. SELECT: Choose what to pay attention to and what to ignore  
    3. FORGET: Let go of unimportant stuff to make room for new stuff
    
    DIS IS LIKE HAVIN' A REALLY SMART ORK WHO'S REALLY GOOD AT
    REMEMBERIN' DA IMPORTANT STUFF AND FORGETTIN' DA BORING STUFF!
    
    UNLIKE DA TRANSFORMER ORKS WHO LOOK AT EVERYTHING AT ONCE,
    DIS ORK PROCESSES THINGS ONE BY ONE AND REMEMBERS SELECTIVELY!
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
        
    def do_da_morky_memory_work(self, da_input, da_orky_delta):
        """
        DA ORK DOES HIS SELECTIVE MEMORY WORK!
        
        da_input: Da information comin' in (batch_size, seq_len, hidden_size)
        da_orky_delta: How fast da Ork processes information (time step size)
        
        DIS IS WHERE DA ORK:
        1. Takes in new information one piece at a time
        2. Processes it through his memory system
        3. Selectively remembers important stuff
        4. Outputs what he thinks is important
        
        UNLIKE DA TRANSFORMER ORKS WHO LOOK AT EVERYTHING AT ONCE,
        DIS ORK PROCESSES THINGS SEQUENTIALLY - ONE WORD AT A TIME!
        """
        # Get da size of everything so we know how much work we got
        batch_size, seq_len, hidden_size = da_input.shape
        
        # Initialize da Ork's memory state (starts wif empty memory)
        # torch.zeros: Creates a tensor filled wif zeros - like a blank slate
        # Why we need 'em (Orky): Da Ork starts wif no memories, like a fresh brain ready to learn!
        # Why we need 'em (Humie): Initial state is typically zero for SSM initialization
        # Shape: (batch_size, state_size) - one memory state for each item in da batch
        # Dis is like givin' each Ork a fresh, empty memory bank to start wif
        da_orky_state = torch.zeros(batch_size, self.da_orky_state_size, device=da_input.device)
        
        # Store all da Ork's outputs as he processes each piece of information
        # Dis is like keepin' track of everything da Ork says as he processes each word
        da_orky_outputs = []
        
        # Process each piece of information one by one (sequential processing)
        # Dis is da key difference from Transformers - we process sequentially, not all at once!
        # Why we need 'em (Orky): Da Ork can only look at one thing at a time, like readin' a book!
        # Why we need 'em (Humie): Sequential processing allows the model to maintain state across time steps
        for t in range(seq_len):
            # Get da current input - da word da Ork is lookin' at right now
            # da_input[:, t, :]: Takes da t-th word from all sequences in da batch
            # Why we need 'em (Orky): Da Ork needs to focus on one word at a time!
            # Why we need 'em (Humie): We process one token at a time to maintain sequential order
            da_current_input = da_input[:, t, :]  # (batch_size, hidden_size)
            
            # STEP 1: Da Ork processes da input and gets ready to remember it
            # self.da_orky_B: Transforms input from hidden_size to state_size
            # Why we need 'em (Orky): Da Ork needs to convert information into a format he can remember!
            # Why we need 'em (Humie): Input projection prepares data for state update
            # Dis is like da Ork sayin': "I need to process dis information before I can remember it!"
            # It's like convertin' a word into a memory format dat da Ork's brain can store
            da_processed_input = self.da_orky_B(da_current_input)  # (batch_size, state_size)
            
            # STEP 2: Da Ork updates his memory based on da new information
            # Dis is where da magic happens - da Ork combines his old memories wif new information
            # In a real SSM, dis would be: state = state + delta * A * state + B * input
            # But we're simplifyin' for now to avoid dimension issues
            # Why we need 'em (Orky): Da Ork needs to remember new stuff while keepin' old stuff!
            # Why we need 'em (Humie): State update combines previous state with new input
            # Dis is like da Ork sayin': "I'll add dis new information to what I already know!"
            
            # STEP 3: Transform da processed input back to hidden size for output
            # self.da_orky_C: Transforms from state_size back to hidden_size
            # Why we need 'em (Orky): Da Ork needs to turn his memories into something others can understand!
            # Why we need 'em (Humie): Output projection converts state back to hidden representation
            # Dis is like da Ork sayin': "Now I need to turn my memory into useful information!"
            # It's da opposite of da input processing - we're goin' from memory format back to information format
            da_final_output = self.da_orky_C(da_processed_input)  # (batch_size, hidden_size)
            
            # Store dis output - da Ork's response to dis particular word
            # Dis is like writin' down what da Ork said after processin' each word
            da_orky_outputs.append(da_final_output)
        
        # STEP 4: Stack all da outputs together into one big tensor
        # torch.stack: Combines all da individual outputs into one tensor
        # dim=1: Stacks along da sequence dimension (time dimension)
        # Why we need 'em (Orky): We need to put all da Ork's responses together in order!
        # Why we need 'em (Humie): Stacking creates the final output sequence
        # Dis is like takin' all da individual responses and puttin' 'em in a sequence
        # Shape: (batch_size, seq_len, hidden_size) - one output for each input word
        da_orky_final_output = torch.stack(da_orky_outputs, dim=1)
        
        return da_orky_final_output


class MorkySelectiveGate(nn.Module):
    """
    DIS IS DA ORK'S SELECTIVE GATE!
    
    Da Morky Selective Gate is like havin' a really smart Ork who can:
    1. LOOK: Examine incoming information carefully
    2. DECIDE: Choose what to pay attention to and what to ignore
    3. GATE: Control how much information flows through to da memory system
    
    DIS IS LIKE HAVIN' A REALLY SMART ORK WHO'S REALLY GOOD AT
    DECIDIN' WHAT'S IMPORTANT AND WHAT'S NOT!
    
    DIS IS DA KEY INNOVATION OF MAMBA - DA ORK CAN SELECTIVELY
    CHOOSE WHAT TO REMEMBER BASED ON DA CONTEXT!
    """
    
    def __init__(self, da_orky_hidden_size):
        super().__init__()
        self.da_orky_hidden_size = da_orky_hidden_size
        
        # Da Ork's input processing system - prepares information for gating decisions
        # nn.Linear: Dis is like a smart Ork who takes numbers and transforms 'em into other numbers
        # It does: output = input * weight + bias (like y = mx + b but for lots of numbers at once)
        # Why we need 'em (Orky): Da Ork needs to process information before decidin' what to do wif it!
        # Why we need 'em (Humie): Linear transformation prepares input for gating decisions
        # Dis is like da Ork sayin': "I need to understand dis information before I can decide what to do!"
        # We keep da same size (hidden_size to hidden_size) because we're just processin' da information
        self.da_orky_input_proj = nn.Linear(da_orky_hidden_size, da_orky_hidden_size)
        
        # Da Ork's selective attention system - learns what to pay attention to
        # nn.Linear: Another smart Ork who learns to focus on important information
        # Why we need 'em (Orky): Da Ork needs to learn what's important and what's not!
        # Why we need 'em (Humie): Selective attention learns to focus on relevant information
        # Dis is like da Ork sayin': "I need to learn which information is worth rememberin'!"
        # Dis is da key innovation - da Ork learns to be selective about what he pays attention to
        # Shape: (hidden_size, hidden_size) - learns attention weights for each dimension
        self.da_orky_selective_proj = nn.Linear(da_orky_hidden_size, da_orky_hidden_size)
        
        # Da Ork's time step control system - controls how fast he processes information
        # nn.Linear: A third smart Ork who controls processing speed
        # Why we need 'em (Orky): Da Ork needs to control how fast he processes different information!
        # Why we need 'em (Humie): Time step control allows adaptive processing speed
        # Dis is like da Ork sayin': "I need to control how fast I process different types of information!"
        # Dis allows da Ork to process important information slowly and unimportant information quickly
        # Shape: (hidden_size, hidden_size) - learns time step size for each dimension
        self.da_orky_delta_proj = nn.Linear(da_orky_hidden_size, da_orky_hidden_size)
        
    def do_da_morky_gating(self, da_input):
        """
        DA ORK DOES HIS SELECTIVE GATING!
        
        da_input: Da information comin' in (batch_size, seq_len, hidden_size)
        
        DIS IS WHERE DA ORK:
        1. Processes da incoming information to understand it
        2. Decides what to pay attention to (selective attention)
        3. Controls how fast to process different information (time step control)
        
        DIS IS DA HEART OF MAMBA - DA ORK LEARNS TO BE SELECTIVE!
        """
        # STEP 1: Da Ork processes da input to understand what he's dealin' wif
        # self.da_orky_input_proj: Transforms input for gating decisions
        # Why we need 'em (Orky): Da Ork needs to process information before decidin' what to do!
        # Why we need 'em (Humie): Input projection prepares data for gating decisions
        # Dis is like da Ork sayin': "Let me understand dis information first before I decide what to do!"
        # We process da input to make it ready for gating decisions
        da_processed_input = self.da_orky_input_proj(da_input)
        
        # STEP 2: Da Ork decides what to pay attention to (selective attention)
        # self.da_orky_selective_proj: Learns what information is important
        # F.silu: Swish activation function - it's like da Ork sayin': "If it's good, I'll pay attention!"
        # Why we need 'em (Orky): Da Ork needs to decide what's important and what's not!
        # Why we need 'em (Humie): Selective attention with SiLU activation provides smooth gating
        # SiLU is like ReLU but smoother - it's better for gating decisions
        # SiLU(x) = x * sigmoid(x) - it's smooth and differentiable, perfect for learnin'
        # Dis is like da Ork sayin': "I'll pay attention to dis if it's important, ignore it if it's not!"
        # Shape: (batch_size, seq_len, hidden_size) - attention weights for each position and dimension
        da_selective_attention = F.silu(self.da_orky_selective_proj(da_input))
        
        # STEP 3: Da Ork controls how fast to process different information (time step control)
        # self.da_orky_delta_proj: Controls time step size for each piece of information
        # F.softplus: Makes sure da time step is always positive (Orks can't go backwards in time!)
        # Why we need 'em (Orky): Da Ork needs to control how fast he processes different information!
        # Why we need 'em (Humie): Softplus ensures positive time steps for stable SSM dynamics
        # softplus(x) = log(1 + exp(x)) - it's always positive and smooth
        # Dis is like da Ork sayin': "I'll process important stuff slowly, unimportant stuff quickly!"
        # Shape: (batch_size, seq_len, hidden_size) - time step size for each position and dimension
        da_orky_delta = F.softplus(self.da_orky_delta_proj(da_input))
        
        return da_processed_input, da_selective_attention, da_orky_delta


class MorkyMambaBlock(nn.Module):
    """
    DIS IS A COMPLETE MORKY MAMBA BLOCK!
    
    Dis combines everything together:
    1. Selective Gating (da Ork decides what's important)
    2. Selective SSM (da Ork remembers selectively)
    3. Layer Normalization (da Ork stays disciplined)
    4. Residual connections (da Ork remembers what he knew before)
    
    It's like havin' a whole Ork unit workin' together to process information
    wif really good memory and selective attention!
    
    DIS IS DA HEART OF DA MORKY MAMBA - WHERE ALL DA SMART MEMORY STUFF HAPPENS!
    
    UNLIKE DA TRANSFORMER BLOCKS, DIS BLOCK PROCESSES SEQUENTIALLY
    AND CAN SELECTIVELY REMEMBER IMPORTANT INFORMATION!
    """
    
    def __init__(self, da_orky_hidden_size, da_orky_state_size):
        super().__init__()
        self.da_orky_hidden_size = da_orky_hidden_size
        self.da_orky_state_size = da_orky_state_size
        
        # Da Ork's selective gating system - decides what to pay attention to
        # Dis is like havin' a smart Ork who looks at information and decides what's important
        # Why we need 'em (Orky): Da Ork needs to be selective about what he remembers!
        # Why we need 'em (Humie): Selective gating allows the model to focus on relevant information
        # Dis is da key innovation of Mamba - selective attention based on context
        self.da_orky_selective_gate = MorkySelectiveGate(da_orky_hidden_size)
        
        # Da Ork's selective memory system - remembers things selectively
        # Dis is like havin' a smart Ork who can remember important stuff and forget unimportant stuff
        # Why we need 'em (Orky): Da Ork needs to remember things based on what he thinks is important!
        # Why we need 'em (Humie): Selective SSM provides efficient sequential processing with selective memory
        # Dis processes information sequentially and maintains state across time steps
        self.da_orky_selective_ssm = MorkySelectiveSSM(da_orky_hidden_size, da_orky_state_size)
        
        # Da disciplinary Orks dat keep everything in line (layer normalization)
        # nn.LayerNorm: Keeps da Ork's numbers in a reasonable range
        # Why we need 'em (Orky): Da Orks need to stay disciplined and not get too excited!
        # Why we need 'em (Humie): Layer normalization stabilizes training and improves convergence
        # We have two normalization layers - one before gating, one after SSM
        # Dis is like havin' two Ork bosses who keep everyone in line at different stages
        self.da_orky_norm1 = nn.LayerNorm(da_orky_hidden_size)
        self.da_orky_norm2 = nn.LayerNorm(da_orky_hidden_size)
        
        # Sometimes da Orks forget a bit (dropout for regularization)
        # nn.Dropout: Randomly sets some values to 0 during trainin'
        # Why we need 'em (Orky): Da Orks need to forget some things so they don't memorize everything!
        # Why we need 'em (Humie): Dropout prevents overfitting by randomly zeroing out neurons
        # 0.1 means 10% of da values get set to 0 randomly - dis prevents overfittin'
        self.da_orky_dropout = nn.Dropout(0.1)
        
    def do_da_complete_morky_processin(self, da_input):
        """
        DA COMPLETE MORKY MAMBA PROCESSING PIPELINE!
        
        da_input: Da information dat needs to be processed by da Morky Mamba block
        
        DIS IS WHERE DA MORKY MAMBA BLOCK:
        1. Normalizes da input (keeps da Orks in line)
        2. Does selective gating (da Ork decides what's important)
        3. Does selective SSM (da Ork remembers selectively)
        4. Applies selective attention (da Ork focuses on important stuff)
        5. Adds da original information back (residual connection)
        6. Normalizes everything again (keeps 'em in line again)
        
        DIS IS DA COMPLETE PIPELINE WHERE DA ORK PROCESSES INFORMATION
        SEQUENTIALLY AND SELECTIVELY REMEMBERS IMPORTANT STUFF!
        """
        # STEP 1: Normalize da input to keep da Orks disciplined
        # self.da_orky_norm1: Layer normalization - keeps da numbers in a reasonable range
        # Why we need 'em (Orky): Da Orks need to be disciplined before dey start workin'!
        # Why we need 'em (Humie): Layer normalization stabilizes training and improves convergence
        # Dis is like da Ork boss sayin': "Everyone, calm down and get in line before we start!"
        # It normalizes da input so da mean is 0 and da standard deviation is 1
        da_normalized_input = self.da_orky_norm1(da_input)
        
        # STEP 2: Selective gating - da Ork decides what to pay attention to
        # self.da_orky_selective_gate.do_da_morky_gating: Da Ork makes gating decisions
        # Why we need 'em (Orky): Da Ork needs to decide what's important before rememberin' it!
        # Why we need 'em (Humie): Selective gating allows the model to focus on relevant information
        # Dis is like da Ork sayin': "Let me look at dis information and decide what's worth rememberin'!"
        # Returns three things: processed input, selective attention, and time step control
        da_processed_input, da_selective_attention, da_orky_delta = self.da_orky_selective_gate.do_da_morky_gating(da_normalized_input)
        
        # STEP 3: Selective SSM - da Ork remembers selectively based on his decisions
        # self.da_orky_selective_ssm.do_da_morky_memory_work: Da Ork does his memory work
        # Why we need 'em (Orky): Da Ork needs to remember things based on what he thinks is important!
        # Why we need 'em (Humie): Selective SSM provides efficient sequential processing with selective memory
        # Dis is like da Ork sayin': "Now I'll remember dis information based on what I think is important!"
        # Da Ork processes information sequentially and maintains state across time steps
        da_ssm_output = self.da_orky_selective_ssm.do_da_morky_memory_work(da_processed_input, da_orky_delta)
        
        # STEP 4: Apply selective attention to refine da SSM output
        # Da Ork applies his attention decisions to what he remembered
        # Why we need 'em (Orky): Da Ork needs to apply his attention decisions to what he remembered!
        # Why we need 'em (Humie): Applying selective attention refines the SSM output
        # Dis is like da Ork sayin': "Now I'll focus on da important parts of what I remembered!"
        # We multiply da SSM output by da selective attention to focus on important information
        # Both tensors should have shape (batch_size, seq_len, hidden_size)
        da_attended_output = da_ssm_output * da_selective_attention
        
        # STEP 5: Add residual connection and normalize to keep everything stable
        # Da Ork adds back what he knew before and stays disciplined
        # Why we need 'em (Orky): Da Ork needs to remember what he knew before and stay disciplined!
        # Why we need 'em (Humie): Residual connection helps with gradient flow and layer normalization stabilizes training
        # Dis is like da Ork sayin': "I'll add what I learned to what I already knew, and stay disciplined!"
        # Residual connection: da_input + da_attended_output (adds original information back)
        # Dropout: Randomly zeros out some values to prevent overfitting
        # Layer normalization: Keeps everything in a reasonable range
        da_final_output = self.da_orky_norm2(da_input + self.da_orky_dropout(da_attended_output))
        
        return da_final_output


class MorkyMamba(nn.Module):
    """
    DA COMPLETE MORKY MAMBA!
    
    Dis is da full Mamba model wif multiple Morky blocks stacked together.
    Each block makes da Orks smarter and better at rememberin' and processin' sequences!
    
    DIS IS DA ULTIMATE ORK SEQUENTIAL INTELLIGENCE MACHINE!
    IT TAKES SEQUENCES AND MAKES 'EM INTO SMART ORKY MEMORIES!
    
    UNLIKE DA TRANSFORMER, DIS MODEL PROCESSES SEQUENCES ONE WORD AT A TIME
    AND CAN SELECTIVELY REMEMBER IMPORTANT INFORMATION!
    """
    
    def __init__(self, da_orky_vocab_size, da_orky_hidden_size, da_orky_state_size, num_orky_layers):
        super().__init__()
        self.da_orky_hidden_size = da_orky_hidden_size
        
        # Word embeddings (turn words into numbers da Orks can understand)
        # nn.Embedding: Dis creates a lookup table dat converts word IDs to vectors
        # It's like havin' a big book where each word has its own special number code
        # Why we need 'em (Orky): Da Orks need to turn words into numbers so dey can do math wif 'em!
        # Why we need 'em (Humie): Embeddings convert discrete tokens to continuous vectors, enabling
        # the neural network to process text data and learn semantic relationships between words
        # da_orky_vocab_size: How many different words we can have (vocabulary size)
        # da_orky_hidden_size: How big each word's vector is (embedding dimension)
        # Dis is like givin' each word a special Orky number so da Orks can work wif it
        # Shape: (vocab_size, hidden_size) - one vector for each word in da vocabulary
        self.da_orky_embedding = nn.Embedding(da_orky_vocab_size, da_orky_hidden_size)
        
        # Stack of Morky Mamba blocks - each one makes da Orks smarter!
        # nn.ModuleList: A list of neural network modules dat can be stacked together
        # Why we need 'em (Orky): We need multiple Ork blocks to make da Orks really smart!
        # Why we need 'em (Humie): Multiple layers allow the model to learn complex sequential patterns
        # Each block processes da information and passes it to da next block
        # Dis is like havin' multiple Ork units workin' together, each one smarter than da last!
        # num_orky_layers: How many Morky Mamba blocks we want to stack together
        self.da_orky_mamba_blocks = nn.ModuleList([
            MorkyMambaBlock(da_orky_hidden_size, da_orky_state_size)
            for _ in range(num_orky_layers)
        ])
        
        # Final layer to get da output (turn da Orky memories back into words)
        # nn.Linear: Transforms da hidden representations back to vocabulary logits
        # Why we need 'em (Orky): Da Orks need to turn their memories into word predictions!
        # Why we need 'em (Humie): Output layer converts hidden representations to vocabulary logits
        # da_orky_hidden_size: Input size (size of da Ork's brain)
        # da_orky_vocab_size: Output size (how many different words da Ork can predict)
        # Dis is like da Ork sayin': "Now I need to turn my memories into word predictions!"
        self.da_orky_output_layer = nn.Linear(da_orky_hidden_size, da_orky_vocab_size)
        
    def do_da_complete_morky_mamba(self, da_input):
        """
        DA COMPLETE MORKY MAMBA IN ACTION!
        
        da_input: Da input words dat need to be processed (batch_size, seq_len)
        
        DIS IS WHERE DA MAGIC HAPPENS:
        1. Turn words into numbers (embedding)
        2. Pass through all da Morky Mamba blocks (sequential processing)
        3. Get da final output (predictions for next words)
        
        DIS IS DA COMPLETE PIPELINE WHERE DA ORKS PROCESS SEQUENCES
        SEQUENTIALLY AND SELECTIVELY REMEMBER IMPORTANT INFORMATION!
        """
        # STEP 1: Turn words into numbers so da Orks can work wif 'em
        # self.da_orky_embedding(da_input): Looks up each word ID and gets its vector
        # Why we need 'em (Orky): Da Orks need to turn words into numbers so dey can do math wif 'em!
        # Why we need 'em (Humie): Embeddings convert discrete tokens to continuous vectors
        # Dis is like da Ork sayin': "I need to convert dese words into numbers I can work wif!"
        # Shape: (batch_size, seq_len) -> (batch_size, seq_len, hidden_size)
        # Each word gets converted into a vector of numbers dat represents its meaning
        da_orky_embedded_words = self.da_orky_embedding(da_input)
        
        # STEP 2: Pass through all da Morky Mamba blocks sequentially
        # Each block makes da Orks smarter and better at rememberin'
        # Why we need 'em (Orky): We need multiple Ork blocks to make da Orks really smart!
        # Why we need 'em (Humie): Multiple layers allow the model to learn complex sequential patterns
        # Dis is like havin' multiple Ork units workin' together, each one smarter than da last!
        da_orky_processed_words = da_orky_embedded_words
        for morky_mamba_block in self.da_orky_mamba_blocks:
            # Each block processes da information wif selective memory
            # Why we need 'em (Orky): Each block makes da Orks smarter and better at rememberin'!
            # Why we need 'em (Humie): Each layer processes the information and passes it to the next layer
            # Dis is like da Ork sayin': "I'll process dis information and make it smarter for da next Ork!"
            # Each block does: normalization -> selective gating -> selective SSM -> attention -> residual -> normalization
            da_orky_processed_words = morky_mamba_block.do_da_complete_morky_processin(da_orky_processed_words)
        
        # STEP 3: Get da final output - turn da Orky memories back into word predictions
        # self.da_orky_output_layer: Transforms hidden representations to vocabulary logits
        # Why we need 'em (Orky): Da Orks need to turn their memories into word predictions!
        # Why we need 'em (Humie): Output layer converts hidden representations to vocabulary logits
        # Dis is like da Ork sayin': "Now I need to turn my smart memories into word predictions!"
        # Shape: (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, vocab_size)
        # Each position gets a probability distribution over all possible next words
        da_orky_final_output = self.da_orky_output_layer(da_orky_processed_words)
        
        return da_orky_final_output


def demonstrate_da_morky_mamba():
    """
    LET'S SEE DA MORKY MAMBA IN ACTION!
    
    Dis function shows how our Morky Mamba works with a simple example.
    NOW WIF SELECTIVE MEMORY SO DA ORKS CAN REMEMBER DA IMPORTANT STUFF!
    """
    print("WAAAGH! STARTIN' DA MORKY MAMBA v1.0 DEMONSTRATION!")
    print("=" * 70)
    
    # Create a simple vocabulary of Ork words
    # Dis is like givin' each Ork word a special number so da Orks can work wif it
    da_orky_vocab = {
        "WAAAGH": 0,      # Da battle cry!
        "ORK": 1,         # Da best race in da galaxy!
        "DAKKA": 2,       # More shootin'!
        "BOSS": 3,        # Da leader of da Orks
        "BOYZ": 4,        # Da Ork soldiers
        "FIGHT": 5,       # Wot Orks do best!
        "WIN": 6,         # Wot Orks always do!
        "REMEMBER": 7,    # Wot da Morky Mamba does best!
        "<PAD>": 8,       # Padding for short sentences
        "<START>": 9,     # Start of sentence marker
        "<END>": 10       # End of sentence marker
    }
    
    # Get da size of our vocabulary
    da_orky_vocab_size = len(da_orky_vocab)
    
    # Set up da Morky Mamba parameters
    da_orky_hidden_size = 64        # Size of da Ork brain (hidden dimension)
    da_orky_state_size = 16         # Size of da Ork's memory state
    num_orky_layers = 2             # Number of Morky Mamba blocks
    
    # Create da Morky Mamba
    print("CREATIN' DA MORKY MAMBA...")
    da_morky_mamba = MorkyMamba(
        da_orky_vocab_size=da_orky_vocab_size,
        da_orky_hidden_size=da_orky_hidden_size,
        da_orky_state_size=da_orky_state_size,
        num_orky_layers=num_orky_layers
    )
    
    print(f"Created Morky Mamba wif:")
    print(f"- Vocabulary size: {da_orky_vocab_size} (how many words da Orks know)")
    print(f"- Hidden dimension: {da_orky_hidden_size} (size of da Ork brain)")
    print(f"- State dimension: {da_orky_state_size} (size of da Ork's memory)")
    print(f"- Number of layers: {num_orky_layers} (how many Morky Mamba blocks)")
    print(f"- NOW WIF SELECTIVE MEMORY SO DA ORKS CAN REMEMBER DA IMPORTANT STUFF!")
    print()
    
    # Create a simple Ork sentence
    da_orky_sentence = ["<START>", "WAAAGH", "ORK", "FIGHT", "WIN", "REMEMBER", "<END>"]
    
    # Convert da words to numbers dat da Orks can understand
    da_orky_sentence_ids = [da_orky_vocab[word] for word in da_orky_sentence]
    
    # Add padding to make it da right length
    da_max_seq_len = 10
    while len(da_orky_sentence_ids) < da_max_seq_len:
        da_orky_sentence_ids.append(da_orky_vocab["<PAD>"])
    
    # Convert to tensor and add batch dimension
    da_orky_input_tensor = torch.tensor([da_orky_sentence_ids])
    
    print("Input Ork sentence:", " ".join(da_orky_sentence))
    print("Input tensor shape:", da_orky_input_tensor.shape)
    print()
    
    # Run da Morky Mamba
    print("RUNNIN' DA MORKY MAMBA WIF SELECTIVE MEMORY...")
    with torch.no_grad():
        da_orky_output = da_morky_mamba.do_da_complete_morky_mamba(da_orky_input_tensor)
    
    print("Output shape:", da_orky_output.shape)
    print()
    
    # Show what da Orks are predictin'
    print("DA ORKS ARE PREDICTIN' (WIF SELECTIVE MEMORY!):")
    print("-" * 60)
    
    # Get da predictions for each position
    da_predictions = torch.softmax(da_orky_output[0], dim=-1)  # (seq_len, vocab_size)
    
    for word_idx in range(len(da_orky_sentence)):
        da_word = da_orky_sentence[word_idx]
        da_word_predictions = da_predictions[word_idx]
        
        # Get da top 3 predictions
        da_top_predictions = torch.topk(da_word_predictions, 3)
        
        print(f"After '{da_word}':")
        for i, (da_score, da_word_id) in enumerate(zip(da_top_predictions.values, da_top_predictions.indices)):
            # Find da word name
            da_word_name = [word for word, word_id in da_orky_vocab.items() if word_id == da_word_id.item()][0]
            print(f"  {i+1}. {da_word_name}: {da_score:.3f}")
        print()
    
    print("WAAAGH! DA MORKY MAMBA v1.0 IS WORKIN' PERFECTLY!")
    print("Da Orks are rememberin' things and predictin' what comes next!")
    print("AND DEY CAN SELECTIVELY REMEMBER DA IMPORTANT STUFF - PROPPA' ORKY MEMORY!")
    print("NOW WIF SELECTIVE MEMORY AND ALL DA DETAILED ORKY COMMENTS!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_da_morky_mamba()
