"""
GARGANT - DA ORKY TITANS TRANSFO'MA'!

DIS IS DA ULTIMATE ORK WAR MACHINE - A GARGANT!
A GARGANT IS DA ORK EQUIVALENT OF A TITAN, BUT BIGGER AND MORE ORKY!

Based on da Titans architecture, dis massive Ork war machine has:
1. SURPRISE MECHANISMS - When somethin' unexpected happens, da Orks remember it!
2. DYNAMIC MEMORY - Da Orks learn and remember during battle (test time)!
3. WAAAGH MEMORY - Da collective Ork memory dat grows stronger wif each battle!
4. ORK SURPRISE DETECTION - Da Orks get excited when somethin' unexpected happens!

DIS IS LIKE HAVIN' A WHOLE MOB OF ORKS WIF PERFECT MEMORY
WHO GET MORE EXCITED AND REMEMBER BETTER WHEN SURPRISED!

WAAAGH! (That means "Let's do this with TITAN POWER!" in Ork)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional

class OrkSurpriseDetector(nn.Module):
    """
    DIS IS DA ORK SURPRISE DETECTOR!
    
    When somethin' unexpected happens, da Orks get really excited
    and remember it better! Dis is like da Ork sayin':
    "WOT?! DAT'S NOT WOT I EXPECTED! I GOTTA REMEMBER DIS!"
    
    Da surprise mechanism helps da Orks learn from unexpected events
    and remember 'em better for future battles!
    
    DIS IS LIKE HAVIN' A REALLY EXCITABLE ORK WHO GETS SUPER SURPRISED
    WHEN SOMETHIN' UNEXPECTED HAPPENS AND REMEMBERS IT FOREVER!
    """
    
    def __init__(self, da_orky_model_size, da_surprise_threshold=0.1):
        super().__init__()
        self.da_orky_model_size = da_orky_model_size
        self.da_surprise_threshold = da_surprise_threshold
        
        # Da Ork surprise detector brain - dis is da smart part dat figures out surprises!
        # nn.Sequential: Dis is like havin' a chain of smart Orks, each one doin' his job in order
        # Dis learns to detect when somethin' unexpected happens
        # Why we need 'em (Orky): Da Ork needs a smart brain to figure out when he's surprised!
        # Why we need 'em (Humie): Sequential layers allow the model to learn complex surprise detection patterns
        # First layer: Takes da full model size and makes it smaller (like compressin' da brain)
        # ReLU: "If it's negative, ignore it!" - helps da Ork focus on da important stuff
        # Second layer: Takes da compressed brain and gives us one number (how surprised)
        # Sigmoid: Makes sure da surprise level is between 0 and 1 (0 = not surprised, 1 = very surprised)
        self.da_surprise_brain = nn.Sequential(
            nn.Linear(da_orky_model_size, da_orky_model_size // 2),  # Compress da brain
            nn.ReLU(),  # Focus on da good stuff
            nn.Linear(da_orky_model_size // 2, 1),  # Get one surprise number
            nn.Sigmoid()  # Make it between 0 and 1 (how surprised da Ork is)
        )
        
        # Da Ork surprise memory - stores how surprised da Ork was about different things
        # Dis is like a big book where da Ork writes down how surprised he was about each thing
        # Why we need 'em (Orky): Da Ork needs to remember how surprised he was about different things!
        # Why we need 'em (Humie): Memory storage allows tracking surprise levels across different inputs
        # Dictionary: Like a big book with labels - "Thing A: Very surprised", "Thing B: Not surprised"
        self.da_surprise_memory = {}
        
    def detect_da_orky_surprise(self, da_current_input, da_expected_input=None):
        """
        DETECT IF DA ORK IS SURPRISED BY SOMETHIN' UNEXPECTED!
        
        da_current_input: What da Ork is seein' right now
        da_expected_input: What da Ork was expectin' to see (optional)
        
        Returns: How surprised da Ork is (0 = not surprised, 1 = very surprised)
        
        DIS IS WHERE DA ORK FIGURES OUT IF HE'S SURPRISED!
        It's like da Ork sayin': "WOT?! DAT'S NOT WOT I EXPECTED!"
        """
        # STEP 1: Figure out if da Ork expected somethin' different
        # If we have an expected input, calculate da difference between what he expected and what he got
        # Dis is like da Ork sayin': "I expected dis, but I got dat - how different is it?"
        # Why we need 'em (Orky): Da Ork needs to compare what he expected wif what he actually got!
        # Why we need 'em (Humie): Computing the difference between expected and actual inputs
        # allows the model to quantify how surprising the current input is
        # If da Ork expected somethin' and got somethin' different, dat's a surprise!
        if da_expected_input is not None:
            # Calculate da difference between expected and actual
            # Dis tells us how different da actual input is from what da Ork expected
            da_surprise_input = da_current_input - da_expected_input
        else:
            # If da Ork didn't expect anythin' specific, just use da current input
            # Sometimes da Ork just looks at somethin' without expectin' anythin' specific
            da_surprise_input = da_current_input
        
        # STEP 2: Da Ork surprise detector brain figures out how surprised he is
        # Dis is like da Ork's brain processin' da difference and decidin': "How surprised am I?"
        # self.da_surprise_brain: Da smart Ork brain dat learned to detect surprises
        # Why we need 'em (Orky): Da Ork needs his brain to figure out how surprised he is!
        # Why we need 'em (Humie): The neural network processes the input difference to output a surprise score
        # Dis gives us a number between 0 and 1 (0 = not surprised, 1 = very surprised)
        da_surprise_level = self.da_surprise_brain(da_surprise_input)
        
        return da_surprise_level
    
    def update_da_surprise_memory(self, da_input_id, da_surprise_level):
        """
        UPDATE DA ORK'S SURPRISE MEMORY!
        
        When da Ork gets surprised, he remembers it better!
        """
        if da_input_id not in self.da_surprise_memory:
            self.da_surprise_memory[da_input_id] = 0.0
        
        # Da Ork remembers da surprise and adds it to his memory
        self.da_surprise_memory[da_input_id] += da_surprise_level.item()

class WaaaghMemoryBank(nn.Module):
    """
    DIS IS DA WAAAGH MEMORY BANK!
    
    Dis is like da collective memory of all da Orks in da WAAAGH!
    It stores important information dat all da Orks can access.
    
    Da WAAAGH memory grows stronger wif each battle and surprise!
    It's like havin' a massive Ork brain dat remembers everything!
    
    DIS IS LIKE HAVIN' A HUGE ORK BRAIN DAT ALL DA ORKS CAN ACCESS!
    WHEN ONE ORK LEARNS SOMETHIN', ALL DA OTHER ORKS KNOW IT TOO!
    """
    
    def __init__(self, da_orky_model_size, da_memory_capacity=1000):
        super().__init__()
        self.da_orky_model_size = da_orky_model_size
        self.da_memory_capacity = da_memory_capacity
        
        # Da WAAAGH memory storage - stores all da important Ork knowledge
        # nn.Parameter: Dis creates learnable parameters dat da model can update during trainin'
        # torch.randn: Creates random numbers to start wif (like random Ork thoughts)
        # Shape: (da_memory_capacity, da_orky_model_size) - one memory slot for each capacity, each wif model_size dimensions
        # Why we need 'em (Orky): Da Orks need a big brain to store all their collective knowledge!
        # Why we need 'em (Humie): Learnable parameters allow the memory to adapt and store relevant information
        # Dis is like havin' a huge Ork brain wif lots of memory slots to store important information
        self.da_waaagh_memory = nn.Parameter(torch.randn(da_memory_capacity, da_orky_model_size))
        
        # Da WAAAGH memory importance - how important each memory is
        # nn.Parameter: Another learnable parameter dat tracks how important each memory is
        # torch.ones: Starts wif all memories bein' equally important (importance = 1)
        # Why we need 'em (Orky): Da Orks need to know which memories are most important to keep!
        # Why we need 'em (Humie): Importance tracking allows the model to prioritize and manage memory efficiently
        # Dis is like havin' a rating system for each memory - "Dis memory is very important!", "Dis one is not so important"
        self.da_memory_importance = nn.Parameter(torch.ones(da_memory_capacity))
        
        # Da WAAAGH memory access - learns how to access da right memories
        # nn.Linear: A smart Ork who learns how to find da right memories based on what you're lookin' for
        # Input: da_orky_model_size (what you're lookin' for)
        # Output: da_memory_capacity (which memories to look at)
        # Why we need 'em (Orky): Da Orks need a smart way to find da right memories when dey need 'em!
        # Why we need 'em (Humie): Linear layer learns to map queries to memory attention weights
        # Dis is like havin' a smart Ork librarian who knows exactly where to find each piece of information
        self.da_memory_access = nn.Linear(da_orky_model_size, da_memory_capacity)
        
        # Da WAAAGH memory update - learns how to update memories
        # nn.Linear: A smart Ork who learns how to combine old memories wif new information
        # Input: da_orky_model_size * 2 (old memory + new information)
        # Output: da_orky_model_size (updated memory)
        # Why we need 'em (Orky): Da Orks need to learn how to update their memories wif new information!
        # Why we need 'em (Humie): Linear layer learns to combine old and new information into updated memories
        # Dis is like havin' a smart Ork who knows how to update old memories wif new learnin'
        self.da_memory_update = nn.Linear(da_orky_model_size * 2, da_orky_model_size)
        
    def access_da_waaagh_memory(self, da_query):
        """
        ACCESS DA WAAAGH MEMORY!
        
        Da Orks look through their collective memory to find relevant information.
        It's like askin' da whole WAAAGH: "DOES ANYONE REMEMBER SOMETHIN' LIKE DIS?"
        
        DIS IS WHERE DA ORKS SEARCH THROUGH THEIR COLLECTIVE MEMORY!
        It's like havin' a huge library where all da Orks can find information!
        """
        # STEP 1: Calculate how much attention to pay to each memory
        # self.da_memory_access(da_query): Da smart Ork librarian figures out which memories are relevant
        # F.softmax: Turns da raw scores into probabilities (how much attention to pay to each memory)
        # dim=-1: Apply softmax along da last dimension (across all memories)
        # Why we need 'em (Orky): Da Orks need to decide which memories are most relevant to their query!
        # Why we need 'em (Humie): Softmax normalizes attention scores into a probability distribution
        # Dis is like da Ork sayin': "I'll pay 30% attention to memory 1, 50% to memory 2, 20% to memory 3"
        da_memory_attention = F.softmax(self.da_memory_access(da_query), dim=-1)
        
        # STEP 2: Get da relevant memories based on attention
        # torch.matmul: Matrix multiplication to combine attention weights wif memories
        # da_memory_attention.unsqueeze(1): Add a dimension to make it work wif matrix multiplication
        # self.da_waaagh_memory: Da actual memory bank wif all da stored information
        # .squeeze(1): Remove da extra dimension we added
        # Why we need 'em (Orky): Da Orks need to actually get da information from da memories dey're payin' attention to!
        # Why we need 'em (Humie): Weighted combination of memories based on attention scores
        # Dis is like da Ork takin' 30% of memory 1's info, 50% of memory 2's info, 20% of memory 3's info
        da_retrieved_memory = torch.matmul(da_memory_attention.unsqueeze(1), self.da_waaagh_memory).squeeze(1)
        
        return da_retrieved_memory, da_memory_attention
    
    def update_da_waaagh_memory(self, da_new_information, da_surprise_level):
        """
        UPDATE DA WAAAGH MEMORY WIF NEW INFORMATION!
        
        When da Orks learn somethin' new (especially if it's surprising),
        dey add it to their collective WAAAGH memory!
        
        DIS IS WHERE DA ORKS ADD NEW INFORMATION TO THEIR COLLECTIVE MEMORY!
        It's like havin' a smart Ork librarian who knows how to update da library!
        """
        # STEP 1: Find da least important memory to replace
        # torch.argmin: Finds da index of da memory wif da lowest importance score
        # Why we need 'em (Orky): Da Orks need to find da least important memory to replace wif new information!
        # Why we need 'em (Humie): Memory management requires replacing less important memories with new information
        # Dis is like da Ork sayin': "Dis memory is not very important, so I can replace it wif new information!"
        da_least_important_idx = torch.argmin(self.da_memory_importance)
        
        # STEP 2: Calculate how strong da update should be based on surprise
        # Da surprise level makes da Orks remember surprising things better
        # da_surprise_level * 0.1: More surprising things get stronger updates
        # + 0.01: At least a little bit of update even for not surprising things
        # Why we need 'em (Orky): Da Orks need to remember surprising things better than boring things!
        # Why we need 'em (Humie): Surprise-based updates allow the model to prioritize important information
        # Dis is like da Ork sayin': "Dis is really surprising, so I'll remember it really well!"
        da_update_strength = da_surprise_level * 0.1 + 0.01  # At least a little bit of update
        
        # STEP 3: Make sure da_new_information has da right shape
        # Check if da_new_information has a batch dimension and remove it if needed
        # Why we need 'em (Orky): Da Orks need to make sure da new information is da right shape!
        # Why we need 'em (Humie): Tensor shape consistency is required for proper operations
        # Dis is like da Ork sayin': "I need to make sure dis new information is da right size!"
        if da_new_information.dim() == 2:
            da_new_information = da_new_information.squeeze(0)  # Remove batch dimension if present
        
        # STEP 4: Combine old memory wif new information
        # torch.cat: Sticks together da old memory and da new information
        # dim=-1: Sticks 'em together along da last dimension
        # Why we need 'em (Orky): Da Orks need to combine old memories wif new information!
        # Why we need 'em (Humie): Combining old and new information allows for incremental learning
        # Dis is like da Ork sayin': "I'll take my old memory and add dis new information to it!"
        da_combined_input = torch.cat([self.da_waaagh_memory[da_least_important_idx], da_new_information], dim=-1)
        
        # STEP 5: Update da memory using da smart update mechanism
        # self.da_memory_update: Da smart Ork who knows how to combine old and new information
        # Why we need 'em (Orky): Da Orks need a smart way to update their memories!
        # Why we need 'em (Humie): Learned update mechanism allows for sophisticated memory management
        # Dis is like da Ork sayin': "I'll use my smart brain to combine dis old memory wif dis new information!"
        da_updated_memory = self.da_memory_update(da_combined_input)
        
        # STEP 6: Update da actual memory in da bank
        # Replace da old memory wif da updated memory
        # Why we need 'em (Orky): Da Orks need to actually update their memory bank!
        # Why we need 'em (Humie): Memory updates require replacing old information with new information
        # Dis is like da Ork sayin': "I'll put dis updated memory back in da memory bank!"
        self.da_waaagh_memory.data[da_least_important_idx] = da_updated_memory
        
        # STEP 7: Update da importance (surprising things are more important!)
        # Set da importance to da surprise level plus a small base importance
        # Why we need 'em (Orky): Da Orks need to remember that surprising things are more important!
        # Why we need 'em (Humie): Importance updates help prioritize surprising information
        # Dis is like da Ork sayin': "Dis memory is really surprising, so it's really important!"
        self.da_memory_importance.data[da_least_important_idx] = da_surprise_level + 0.1

class OrkAttentionHead(nn.Module):
    """
    DIS IS A SINGLE ORK HEAD DAT LOOKS AT WORDS WIF MEMORY!
    
    Each Ork head has three jobs:
    1. QUERY: "WOT AM I LOOKIN' FOR?" (What am I interested in?)
    2. KEY: "WOT AM I?" (What am I representing?)
    3. VALUE: "WOT DO I KNOW?" (What information do I have?)
    
    BUT NOW DEY ALSO HAVE ACCESS TO DA WAAAGH MEMORY!
    
    DIS IS LIKE HAVIN' ONE SMART ORK WHO'S REALLY GOOD AT PAYIN' ATTENTION
    AND ALSO HAS ACCESS TO DA COLLECTIVE WAAAGH MEMORY!
    """
    
    def __init__(self, da_orky_model_size, da_orky_head_size):
        super().__init__()
        self.da_orky_head_size = da_orky_head_size
        
        # Da Ork's three brain bitz - each one does a different job!
        # Da lookin_fer brain bit: "Wot am I lookin' fer in dis word?"
        # nn.Linear: Dis is like a smart Ork who takes numbers and transforms 'em into other numbers
        # It does: output = input * weight + bias (like y = mx + b but for lots of numbers at once)
        # Why we need 'em (Orky): Each Ork head needs to do different finkin', so we give 'em different brain bitz!
        # Why we need 'em (Humie): Linear layers learn different linear transformations, allowing each attention head
        # to focus on different aspects of the input data (Query, Key, Value projections)
        # We use it to change da size of our data from da_orky_model_size to da_orky_head_size
        self.lookin_fer = nn.Linear(da_orky_model_size, da_orky_head_size)  # Query
        
        # Da wot_am_i brain bit: "Wot does dis word represent?" 
        # nn.Linear: Another smart Ork who transforms numbers, but dis one learns different patterns
        # Each Linear layer learns different weights, so each one becomes good at different things
        # Why we need 'em (Orky): Dis Ork learns to represent what each word IS, not what we're lookin' for!
        # Why we need 'em (Humie): The Key projection learns to encode the identity/representation of each token
        self.wot_am_i = nn.Linear(da_orky_model_size, da_orky_head_size)    # Key
        
        # Da wot_i_know brain bit: "Wot information does dis word have?"
        # nn.Linear: A third smart Ork who learns yet another way to transform da numbers
        # All three work together but learn to do different jobs (Query, Key, Value)
        # Why we need 'em (Orky): Dis Ork learns what information each word actually contains!
        # Why we need 'em (Humie): The Value projection learns to encode the actual content/information of each token
        self.wot_i_know = nn.Linear(da_orky_model_size, da_orky_head_size)   # Value
        
        # Da Ork memory integration - combines current input wif WAAAGH memory
        # nn.Linear: A smart Ork who knows how to combine current input wif collective memory
        # Input: da_orky_model_size * 2 (current input + WAAAGH memory)
        # Output: da_orky_model_size (integrated input wif memory)
        # Why we need 'em (Orky): Da Orks need to combine their current thoughts wif da collective WAAAGH memory!
        # Why we need 'em (Humie): Memory integration allows the model to leverage collective knowledge
        # Dis is like da Ork sayin': "I'll combine what I'm thinkin' now wif what all da other Orks know!"
        self.da_memory_integration = nn.Linear(da_orky_model_size * 2, da_orky_model_size)
        
    def do_da_orky_finkin_wif_memory(self, da_wordz, da_waaagh_memory):
        """
        DA ORK DOES 'IS FINKIN' WIF ACCESS TO DA WAAAGH MEMORY!
        
        da_wordz: Input words dat da Ork needs to look at
        da_waaagh_memory: Da collective WAAAGH memory
        
        DIS IS WHERE DA ORK:
        1. Combines current input wif WAAAGH memory
        2. Does his normal attention finkin'
        3. Learns from da collective Ork knowledge!
        
        DIS IS DA MAIN BRAIN WORK WHERE DA ORK:
        1. Combines his current thoughts wif da collective WAAAGH memory
        2. Asks 'imself questions about each word (wif memory enhancement!)
        3. Figures out which words are most important
        4. Combines all da important information together (wif memory knowledge!)
        """
        # Get da size of everything so we know how much work we got
        batch_size, seq_len, da_orky_model_size = da_wordz.shape
        
        # STEP 1: Combine current input wif WAAAGH memory
        # Expand da WAAAGH memory to match da sequence length
        # unsqueeze(1): Adds a dimension to make it work wif da sequence
        # expand(-1, seq_len, -1): Makes da memory match da sequence length
        # Why we need 'em (Orky): Da Ork needs to combine his current thoughts wif da collective memory!
        # Why we need 'em (Humie): Memory integration requires matching dimensions for combination
        # Dis is like da Ork sayin': "I'll take da collective WAAAGH memory and apply it to each word!"
        da_expanded_memory = da_waaagh_memory.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine current input wif WAAAGH memory
        # torch.cat: Sticks together da current input and da WAAAGH memory
        # dim=-1: Sticks 'em together along da last dimension
        # Why we need 'em (Orky): Da Ork needs to combine his current thoughts wif da collective knowledge!
        # Why we need 'em (Humie): Concatenation allows combining current input with memory information
        # Dis is like da Ork sayin': "I'll combine what I'm thinkin' now wif what all da other Orks know!"
        da_combined_input = torch.cat([da_wordz, da_expanded_memory], dim=-1)
        
        # Use da memory integration to combine everything properly
        # self.da_memory_integration: Da smart Ork who knows how to combine current input wif memory
        # Why we need 'em (Orky): Da Ork needs a smart way to combine his thoughts wif da collective memory!
        # Why we need 'em (Humie): Learned integration allows sophisticated combination of current and memory information
        # Dis is like da Ork sayin': "I'll use my smart brain to properly combine dis current input wif da WAAAGH memory!"
        da_memory_enhanced_input = self.da_memory_integration(da_combined_input)
        
        # STEP 2: Da Ork asks 'imself three questions about each word (now wif memory enhancement!)
        # Dis is like da Ork lookin' at each word and askin' (wif da help of collective memory):
        # "Wot am I lookin' fer in dis word?" (Query wif memory knowledge)
        da_orky_queries = self.lookin_fer(da_memory_enhanced_input)
        
        # "Wot is dis word representin'?" (Key wif memory knowledge)
        da_orky_keys = self.wot_am_i(da_memory_enhanced_input)
        
        # "Wot information does dis word have?" (Value wif memory knowledge)
        da_orky_values = self.wot_i_know(da_memory_enhanced_input)
        
        # STEP 3: Da Ork figures out how much each word matters to him
        # He does dis by multiplyin' his query with each word's key
        # Dis is like da Ork sayin': "Dis word matches wot I'm lookin' fer!"
        # torch.matmul: Dis is matrix multiplication - it's like da Ork multiplyin' two big grids of numbers
        # It takes da queries (what da Ork is lookin' for) and multiplies 'em wif da keys (what each word is)
        # transpose(-2, -1): Dis flips da last two dimensions of da tensor (like turnin' a grid sideways)
        # Why we need 'em (Orky): Da Ork needs to see how well each word matches what he's lookin' for!
        # Why we need 'em (Humie): Matrix multiplication computes similarity scores between queries and keys,
        # creating an attention matrix that shows how much each token should attend to every other token
        # We need dis because matrix multiplication needs da right shapes to work together
        da_orky_scores = torch.matmul(da_orky_queries, da_orky_keys.transpose(-2, -1))
        
        # Da Ork scales da scores so dey don't get too big
        # (Orks sometimes get too excited and need to calm down)
        # We divide by da square root of da head size to keep things reasonable
        # Why we need 'em (Orky): Da Ork needs to keep his excitement under control!
        # Why we need 'em (Humie): Scaling prevents attention scores from becoming too large, maintaining numerical stability
        da_orky_scores = da_orky_scores / math.sqrt(self.da_orky_head_size)
        
        # STEP 4: Da Ork turns da scores into probabilities
        # (He figures out how much attention to pay to each word)
        # F.softmax: Dis takes da raw scores and turns 'em into probabilities (numbers between 0 and 1)
        # It makes sure all da attention adds up to 1.0 (100%) - like dividin' up a pie
        # dim=-1: We apply softmax along da last dimension (across all words for each query)
        # Why we need 'em (Orky): Da Ork needs to decide how much attention to pay to each word - can't pay 200%!
        # Why we need 'em (Humie): Softmax normalizes attention scores into a probability distribution,
        # ensuring the attention weights sum to 1 and creating a proper attention mechanism
        # Dis is like da Ork decidin': "I'll pay 30% attention to word 1, 50% to word 2, 20% to word 3"
        da_orky_attention_weights = F.softmax(da_orky_scores, dim=-1)
        
        # STEP 5: Da Ork combines all da information based on his attention
        # (He takes da important bits from each word and puts 'em together)
        # torch.matmul: Another matrix multiplication! Dis time we're combinin' attention weights wif values
        # We take da attention weights (how much to pay attention to each word) and multiply 'em wif da values (da actual information)
        # Why we need 'em (Orky): Da Ork needs to actually USE da information from da words he's payin' attention to!
        # Why we need 'em (Humie): This weighted combination of values creates the final output where each token's
        # contribution is proportional to its attention weight, producing a context-aware representation
        # Dis is like da Ork sayin': "I'll take 30% of word 1's info, 50% of word 2's info, 20% of word 3's info"
        # Dis is da final answer where da Ork combines everything he learned (wif memory enhancement!)
        da_orky_answer = torch.matmul(da_orky_attention_weights, da_orky_values)
        
        return da_orky_answer, da_orky_attention_weights

class MultiHeadOrkAttention(nn.Module):
    """
    DIS IS WHERE WE GET A WHOLE MOB OF ORK HEADS WORKIN' TOGETHER WIF MEMORY!
    
    Each Ork head now has access to da WAAAGH memory,
    so dey can learn from da collective Ork knowledge!
    """
    
    def __init__(self, da_orky_model_size, num_orky_heads):
        super().__init__()
        self.da_orky_model_size = da_orky_model_size
        self.num_orky_heads = num_orky_heads
        self.da_orky_head_size = da_orky_model_size // num_orky_heads
        
        # Create all da Ork heads wif memory access
        self.da_orky_heads = nn.ModuleList([
            OrkAttentionHead(da_orky_model_size, self.da_orky_head_size) 
            for _ in range(num_orky_heads)
        ])
        
        # Da Boss Ork who combines all da answers
        self.da_boss_ork = nn.Linear(da_orky_model_size, da_orky_model_size)
        
    def do_da_mob_finkin_wif_memory(self, da_wordz, da_waaagh_memory):
        """
        ALL DA ORK HEADS WORK TOGETHER WIF ACCESS TO DA WAAAGH MEMORY!
        """
        da_orky_head_answers = []
        all_da_orky_attention_weights = []
        
        # Each Ork head does his own thinkin' wif memory access
        for ork_head in self.da_orky_heads:
            da_head_answer, da_head_attention = ork_head.do_da_orky_finkin_wif_memory(da_wordz, da_waaagh_memory)
            da_orky_head_answers.append(da_head_answer)
            all_da_orky_attention_weights.append(da_head_attention)
        
        # Combine all da Ork heads' answers
        da_combined_orky_answers = torch.cat(da_orky_head_answers, dim=-1)
        
        # Da Boss Ork makes da final decision
        da_final_orky_answer = self.da_boss_ork(da_combined_orky_answers)
        
        return da_final_orky_answer, all_da_orky_attention_weights

class OrkFeedForward(nn.Module):
    """
    DIS IS DA ORK'S BRAIN PROCESSIN' CENTER WIF MEMORY!
    
    Now da Ork brain can also process information from da WAAAGH memory,
    makin' it even smarter and more powerful!
    """
    
    def __init__(self, da_orky_model_size, da_orky_feedforward_size):
        super().__init__()
        self.make_da_brain_big = nn.Linear(da_orky_model_size, da_orky_feedforward_size)
        self.make_da_brain_right_size = nn.Linear(da_orky_feedforward_size, da_orky_model_size)
        self.da_orky_forgets = nn.Dropout(0.1)
        
    def do_da_brain_processin_wif_memory(self, da_wordz):
        """
        DA ORK BRAIN PROCESSES DA INFORMATION WIF MEMORY ENHANCEMENT!
        """
        da_expanded_brain = F.relu(self.make_da_brain_big(da_wordz))
        da_brain_wif_some_forgettin = self.da_orky_forgets(da_expanded_brain)
        da_final_brain_output = self.make_da_brain_right_size(da_brain_wif_some_forgettin)
        
        return da_final_brain_output

class OrkLayerNorm(nn.Module):
    """
    DIS IS ORK DISCIPLINE WIF MEMORY!
    
    Keeps da Orks in line even when dey have access to massive WAAAGH memory!
    """
    
    def __init__(self, da_orky_model_size, da_orky_epsilon=1e-6):
        super().__init__()
        self.da_orky_gamma = nn.Parameter(torch.ones(da_orky_model_size))
        self.da_orky_beta = nn.Parameter(torch.zeros(da_orky_model_size))
        self.da_orky_epsilon = da_orky_epsilon
        
    def forward(self, da_wordz):
        """
        KEEP DA ORKS IN LINE EVEN WIF ALL DAT MEMORY!
        """
        da_orky_mean = da_wordz.mean(-1, keepdim=True)
        da_orky_std = da_wordz.std(-1, keepdim=True)
        da_normalized_orky_values = (self.da_orky_gamma * 
                                   (da_wordz - da_orky_mean) / 
                                   (da_orky_std + self.da_orky_epsilon) + 
                                   self.da_orky_beta)
        
        return da_normalized_orky_values
    
    def keep_da_orks_in_line(self, da_wordz):
        """
        KEEP DA ORKS IN LINE EVEN WIF ALL DAT MEMORY!
        """
        return self.forward(da_wordz)

class GargantTitansBlock(nn.Module):
    """
    DIS IS A GARGANT TITANS BLOCK - DA ULTIMATE ORK WAR MACHINE!
    
    Dis combines everything together:
    1. Multi-Head Attention wif WAAAGH memory
    2. Feed-Forward Network wif memory enhancement
    3. Layer Normalization wif memory discipline
    4. Residual connections wif memory
    5. SURPRISE DETECTION AND MEMORY UPDATE!
    
    DIS IS DA HEART OF DA GARGANT - WHERE ALL DA TITAN POWER HAPPENS!
    """
    
    def __init__(self, da_orky_model_size, num_orky_heads, da_orky_feedforward_size):
        super().__init__()
        # Da Ork mob dat works together wif WAAAGH memory
        self.da_orky_attention = MultiHeadOrkAttention(da_orky_model_size, num_orky_heads)
        
        # Da Ork brain dat processes information wif memory
        self.da_orky_feed_forward = OrkFeedForward(da_orky_model_size, da_orky_feedforward_size)
        
        # Da disciplinary Orks dat keep everything in line
        self.da_orky_norm1 = OrkLayerNorm(da_orky_model_size)
        self.da_orky_norm2 = OrkLayerNorm(da_orky_model_size)
        
        # Da Ork surprise detector
        self.da_orky_surprise_detector = OrkSurpriseDetector(da_orky_model_size)
        
        # Sometimes da Orks forget a bit (dropout for regularization)
        self.da_orky_dropout = nn.Dropout(0.1)
        
    def do_da_gargant_processin(self, da_wordz, da_waaagh_memory, da_expected_input=None):
        """
        DA GARGANT TITANS BLOCK PROCESSES EVERYTHING WIF MEMORY AND SURPRISE!
        
        da_wordz: Da words dat need to be processed
        da_waaagh_memory: Da collective WAAAGH memory
        da_expected_input: What da Orks were expectin' (for surprise detection)
        
        DIS IS WHERE DA GARGANT:
        1. Detects surprises
        2. Does multi-head attention wif memory
        3. Updates memory based on surprises
        4. Does feed-forward processin' wif memory
        5. Returns enhanced output wif memory integration!
        """
        # STEP 1: Detect if da Orks are surprised
        da_surprise_level = self.da_orky_surprise_detector.detect_da_orky_surprise(
            da_wordz.mean(dim=1), da_expected_input
        )
        
        # STEP 2: Multi-Head Attention wif WAAAGH memory
        da_attention_output, da_attention_weights = self.da_orky_attention.do_da_mob_finkin_wif_memory(
            da_wordz, da_waaagh_memory
        )
        
        # Add da original information back (residual connection) and normalize
        da_wordz = self.da_orky_norm1(da_wordz + self.da_orky_dropout(da_attention_output))
        
        # STEP 3: Feed-Forward wif memory enhancement
        da_feedforward_output = self.da_orky_feed_forward.do_da_brain_processin_wif_memory(da_wordz)
        
        # Add da information back again (residual connection) and normalize
        da_wordz = self.da_orky_norm2(da_wordz + self.da_orky_dropout(da_feedforward_output))
        
        return da_wordz, da_attention_weights, da_surprise_level

class GargantTitans(nn.Module):
    """
    DA GARGANT TITANS - DA ULTIMATE ORK WAR MACHINE!
    
    Dis is da full Gargant wif multiple Titans blocks, WAAAGH memory,
    and surprise mechanisms! It's like havin' a massive Ork war machine
    dat gets smarter and remembers more wif each battle!
    
    DIS IS DA ULTIMATE ORK INTELLIGENCE MACHINE WIF TITAN POWER!
    IT TAKES WORDS, REMEMBERS EVERYTHING, AND GETS SMARTER WIF SURPRISES!
    """
    
    def __init__(self, da_orky_vocab_size, da_orky_model_size, num_orky_heads, 
                 num_orky_layers, da_orky_feedforward_size, da_max_orky_seq_len,
                 da_memory_capacity=1000):
        super().__init__()
        self.da_orky_model_size = da_orky_model_size
        
        # Word embeddings (turn words into numbers da Orks can understand)
        self.da_orky_embedding = nn.Embedding(da_orky_vocab_size, da_orky_model_size)
        
        # Positional encoding (tell da Orks where each word is in da sentence)
        self.da_orky_pos_encoding = self._create_da_orky_positional_encoding(
            da_max_orky_seq_len, da_orky_model_size)
        
        # Da WAAAGH memory bank - da collective Ork memory!
        self.da_waaagh_memory_bank = WaaaghMemoryBank(da_orky_model_size, da_memory_capacity)
        
        # Stack of Gargant Titans blocks - each one makes da Orks smarter!
        self.da_gargant_titans_blocks = nn.ModuleList([
            GargantTitansBlock(da_orky_model_size, num_orky_heads, da_orky_feedforward_size)
            for _ in range(num_orky_layers)
        ])
        
        # Final layer to get da output (turn da Orky thoughts back into words)
        self.da_orky_output_layer = nn.Linear(da_orky_model_size, da_orky_vocab_size)
        
        # Da Ork surprise detector for da whole Gargant
        self.da_gargant_surprise_detector = OrkSurpriseDetector(da_orky_model_size)
        
    def _create_da_orky_positional_encoding(self, da_max_orky_seq_len, da_orky_model_size):
        """
        CREATE POSITIONAL ENCODING FOR DA GARGANT!
        """
        da_orky_pe = torch.zeros(da_max_orky_seq_len, da_orky_model_size)
        da_orky_positions = torch.arange(0, da_max_orky_seq_len).unsqueeze(1).float()
        
        da_orky_div_term = torch.exp(torch.arange(0, da_orky_model_size, 2).float() * 
                                   -(math.log(10000.0) / da_orky_model_size))
        
        da_orky_pe[:, 0::2] = torch.sin(da_orky_positions * da_orky_div_term)
        da_orky_pe[:, 1::2] = torch.cos(da_orky_positions * da_orky_div_term)
        
        return da_orky_pe.unsqueeze(0)
        
    def do_da_gargant_titans_processin(self, da_wordz, da_expected_input=None):
        """
        DA GARGANT TITANS IN ACTION WIF MEMORY AND SURPRISE!
        
        da_wordz: Da input words dat need to be processed
        da_expected_input: What da Orks were expectin' (for surprise detection)
        
        DIS IS WHERE DA GARGANT:
        1. Accesses da WAAAGH memory
        2. Processes through all Titans blocks wif memory
        3. Detects surprises and updates memory
        4. Returns enhanced output wif memory integration!
        """
        da_orky_seq_len = da_wordz.size(1)
        
        # STEP 1: Turn words into numbers and add position info
        da_orky_embedded_words = self.da_orky_embedding(da_wordz) * math.sqrt(self.da_orky_model_size)
        da_orky_embedded_words = (da_orky_embedded_words + 
                                self.da_orky_pos_encoding[:, :da_orky_seq_len, :].to(da_wordz.device))
        
        # STEP 2: Access da WAAAGH memory
        da_waaagh_memory, da_memory_attention = self.da_waaagh_memory_bank.access_da_waaagh_memory(
            da_orky_embedded_words.mean(dim=1)  # Use average of sequence as query
        )
        
        # STEP 3: Pass through all da Gargant Titans blocks
        all_da_orky_attention_weights = []
        all_da_surprise_levels = []
        
        for gargant_titans_block in self.da_gargant_titans_blocks:
            da_orky_embedded_words, da_attention_weights, da_surprise_level = gargant_titans_block.do_da_gargant_processin(
                da_orky_embedded_words, da_waaagh_memory, da_expected_input
            )
            all_da_orky_attention_weights.append(da_attention_weights)
            all_da_surprise_levels.append(da_surprise_level)
        
        # STEP 4: Update da WAAAGH memory based on surprises
        da_average_surprise = torch.stack(all_da_surprise_levels).mean()
        self.da_waaagh_memory_bank.update_da_waaagh_memory(
            da_orky_embedded_words.mean(dim=1), da_average_surprise
        )
        
        # STEP 5: Get da final output
        da_orky_final_output = self.da_orky_output_layer(da_orky_embedded_words)
        
        return da_orky_final_output, all_da_orky_attention_weights, all_da_surprise_levels, da_waaagh_memory

def demonstrate_da_gargant_titans():
    """
    LET'S SEE DA GARGANT TITANS IN ACTION!
    
    Dis function shows how our Gargant Titans works with memory and surprise!
    """
    print("WAAAGH! STARTIN' DA GARGANT TITANS DEMONSTRATION!")
    print("DIS IS DA ULTIMATE ORK WAR MACHINE WIF TITAN POWER!")
    print("=" * 80)
    
    # Create a simple vocabulary of Ork words
    da_orky_vocab = {
        "WAAAGH": 0,      # Da battle cry!
        "ORK": 1,         # Da best race in da galaxy!
        "DAKKA": 2,       # More shootin'!
        "BOSS": 3,        # Da leader of da Orks
        "BOYZ": 4,        # Da Ork soldiers
        "FIGHT": 5,       # Wot Orks do best!
        "WIN": 6,         # Wot Orks always do!
        "SURPRISE": 7,    # When somethin' unexpected happens!
        "MEMORY": 8,      # Da Ork memory!
        "<PAD>": 9,       # Padding for short sentences
        "<START>": 10,   # Start of sentence marker
        "<END>": 11       # End of sentence marker
    }
    
    # Get da size of our vocabulary
    da_orky_vocab_size = len(da_orky_vocab)
    
    # Set up da Gargant Titans parameters
    da_orky_model_size = 128        # Size of da Ork brain (bigger for Titans!)
    num_orky_heads = 8             # Number of Ork heads (more for Titans!)
    num_orky_layers = 4            # Number of Gargant Titans blocks
    da_orky_feedforward_size = 256 # Size of da Ork brain processing center
    da_max_orky_seq_len = 15       # Maximum sentence length
    da_memory_capacity = 500       # Size of da WAAAGH memory bank
    
    # Create da Gargant Titans
    print("CREATIN' DA GARGANT TITANS...")
    da_gargant_titans = GargantTitans(
        da_orky_vocab_size=da_orky_vocab_size,
        da_orky_model_size=da_orky_model_size,
        num_orky_heads=num_orky_heads,
        num_orky_layers=num_orky_layers,
        da_orky_feedforward_size=da_orky_feedforward_size,
        da_max_orky_seq_len=da_max_orky_seq_len,
        da_memory_capacity=da_memory_capacity
    )
    
    print(f"Created Gargant Titans wif:")
    print(f"- Vocabulary size: {da_orky_vocab_size} (how many words da Orks know)")
    print(f"- Model dimension: {da_orky_model_size} (size of da Ork brain - BIGGER FOR TITANS!)")
    print(f"- Number of heads: {num_orky_heads} (how many Ork heads work together - MORE FOR TITANS!)")
    print(f"- Number of layers: {num_orky_layers} (how many Gargant Titans blocks)")
    print(f"- Feed-forward dimension: {da_orky_feedforward_size} (size of da Ork brain processing)")
    print(f"- Memory capacity: {da_memory_capacity} (size of da WAAAGH memory bank)")
    print()
    
    # Create a simple Ork sentence
    da_orky_sentence = ["<START>", "WAAAGH", "ORK", "FIGHT", "WIN", "SURPRISE", "<END>"]
    
    # Convert da words to numbers dat da Orks can understand
    da_orky_sentence_ids = [da_orky_vocab[word] for word in da_orky_sentence]
    
    # Add padding to make it da right length
    while len(da_orky_sentence_ids) < da_max_orky_seq_len:
        da_orky_sentence_ids.append(da_orky_vocab["<PAD>"])
    
    # Convert to tensor and add batch dimension
    da_orky_input_tensor = torch.tensor([da_orky_sentence_ids])
    
    print("Input Ork sentence:", " ".join(da_orky_sentence))
    print("Input tensor shape:", da_orky_input_tensor.shape)
    print()
    
    # Run da Gargant Titans
    print("RUNNIN' DA GARGANT TITANS...")
    with torch.no_grad():
        da_orky_output, all_da_orky_attention_weights, all_da_surprise_levels, da_waaagh_memory = da_gargant_titans.do_da_gargant_titans_processin(da_orky_input_tensor)
    
    print("Output shape:", da_orky_output.shape)
    print("Number of attention weight sets:", len(all_da_orky_attention_weights))
    print("Attention weights per layer:", len(all_da_orky_attention_weights[0]))
    print("Number of surprise levels:", len(all_da_surprise_levels))
    print("WAAAGH memory shape:", da_waaagh_memory.shape)
    print()
    
    # Show what da Orks are payin' attention to
    print("DA ORKS ARE PAYIN' ATTENTION TO:")
    print("-" * 60)
    
    for layer_idx, da_layer_attention in enumerate(all_da_orky_attention_weights):
        print(f"Gargant Titans Block {layer_idx + 1}:")
        for head_idx, da_head_attention in enumerate(da_layer_attention):
            print(f"  Ork Head {head_idx + 1} attention weights:")
            da_first_word_attention = da_head_attention[0, 0, :].numpy()
            for word_idx, da_attention_score in enumerate(da_first_word_attention):
                if word_idx < len(da_orky_sentence):
                    da_word = da_orky_sentence[word_idx]
                    print(f"    {da_word}: {da_attention_score:.3f}")
            print()
    
    # Show da surprise levels
    print("DA ORKS SURPRISE LEVELS:")
    print("-" * 40)
    for layer_idx, da_surprise_level in enumerate(all_da_surprise_levels):
        print(f"Gargant Titans Block {layer_idx + 1}: {da_surprise_level.item():.3f}")
    print()
    
    # Show da WAAAGH memory
    print("DA WAAAGH MEMORY IS WORKIN':")
    print("-" * 40)
    print(f"WAAAGH memory shape: {da_waaagh_memory.shape}")
    print(f"WAAAGH memory sample: {da_waaagh_memory[0, :5].numpy()}")
    print()
    
    print("WAAAGH! DA GARGANT TITANS IS WORKIN' PERFECTLY!")
    print("Da Orks are lookin' at words, rememberin' everything, and gettin' surprised!")
    print("DIS IS DA ULTIMATE ORK WAR MACHINE WIF TITAN POWER!")
    print("=" * 80)

if __name__ == "__main__":
    demonstrate_da_gargant_titans()
