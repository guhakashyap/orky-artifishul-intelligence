"""
ORKY TRANSFO'MA' v1.2 - A PROPPA' ORKY WARHAMMER 40K TRANSFORMER!


DIS IS HOW DA ORKS DO ARTIFISHUL INTELLIGENCE, BUT NOW IT'S EVEN MORE ORKY!
Da Orky Transfo'ma' works by havin' lots of Ork heads lookin' at words
and figurin' out which words is most important to each other.
It's like havin' a whole mob of Orks shoutin' at each other about
what dey fink is important, and den dey all agree on da answer!


NOW WIF MORE ORKY VARIABLES AND COMMENTS SO EVEN DA DUMBEST GROT CAN UNDERSTAND!


WAAAGH! (That means "Let's do this!" in Ork)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class OrkAttentionHead(nn.Module):
   """
   DIS IS A SINGLE ORK HEAD DAT LOOKS AT WORDS!
  
   Each Ork head has three jobs:
   1. QUERY: "WOT AM I LOOKIN' FOR?" (What am I interested in?)
   2. KEY: "WOT AM I?" (What am I representing?)
   3. VALUE: "WOT DO I KNOW?" (What information do I have?)
  
   Da Ork looks at all da other words, figures out which ones
   are most relevant to what he's lookin' for, and den combines
   all da important information together!
  
   DIS IS LIKE HAVIN' ONE SMART ORK WHO'S REALLY GOOD AT PAYIN' ATTENTION!
   """
  
   def __init__(self, da_orky_model_size, da_orky_head_size):
       super().__init__()
       # Dis is how big each Ork head's brain is
       self.da_orky_head_size = da_orky_head_size
      
       # Dese 'ere are da Ork's three brain bitz - each one does a different job!
       # Da lookin_fer brain bit: "Wot am I lookin' fer in dis word?"
       # nn.Linear: Dis is like a smart Ork who takes numbers and transforms 'em into other numbers
       # It does: output = input * weight + bias (like y = mx + b but for lots of numbers at once)
       # Why we need 'em (Orky): Each Ork head needs to do different finkin', so we give 'em different brain bitz!
       # Why we need 'em (Humie): Linear layers learn different linear transformations, allowing each attention head
       # to focus on different aspects of the input data (Query, Key, Value projections)
       # We use it to change da size of our data from da_orky_model_size to da_orky_head_size
       self.lookin_fer = nn.Linear(da_orky_model_size, da_orky_head_size)  #QUERY
      
       # Da wot_am_i brain bit: "Wot does dis word represent?"
       # nn.Linear: Another smart Ork who transforms numbers, but dis one learns different patterns
       # Each Linear layer learns different weights, so each one becomes good at different things
       # Why we need 'em (Orky): Dis Ork learns to represent what each word IS, not what we're lookin' for!
       # Why we need 'em (Humie): The Key projection learns to encode the identity/representation of each token
       self.wot_am_i = nn.Linear(da_orky_model_size, da_orky_head_size)   #KEY
      
       # Da wot_i_know brain bit: "Wot information does dis word have?"
       # nn.Linear: A third smart Ork who learns yet another way to transform da numbers
       # All three work together but learn to do different jobs (Query, Key, Value)
       # Why we need 'em (Orky): Dis Ork learns what information each word actually contains!
       # Why we need 'em (Humie): The Value projection learns to encode the actual content/information of each token
       self.wot_i_know = nn.Linear(da_orky_model_size, da_orky_head_size) #VALUE
      
   def do_da_orky_finkin(self, da_wordz):
       """
       DIS IS WHERE DA ORK DOES 'IS FINKIN'!
      
       da_wordz: Input words dat da Ork needs to look at (batch_size, seq_len, da_orky_model_size)
      
       DIS IS DA MAIN BRAIN WORK WHERE DA ORK:
       1. Asks 'imself questions about each word
       2. Figures out which words are most important
       3. Combines all da important information together
       """
       # Get da size of everything so we know how much work we got
       batch_size, seq_len, da_orky_model_size = da_wordz.shape
      
       # STEP 1: Da Ork asks 'imself three questions about each word
       # Dis is like da Ork lookin' at each word and askin':
       # "Wot am I lookin' fer in dis word?" (Query)
       da_orky_queries = self.lookin_fer(da_wordz)
      
       # "Wot is dis word representin'?" (Key)
       da_orky_keys = self.wot_am_i(da_wordz)
      
       # "Wot information does dis word have?" (Value)
       da_orky_values = self.wot_i_know(da_wordz)
      
       # STEP 2: Da Ork figures out how much each word matters to him
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
      
       # STEP 3: Da Ork scales da scores so dey don't get too big
       # (Orks sometimes get too excited and need to calm down)
       # We divide by da square root of da head size to keep things reasonable
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
       # Dis is da final answer where da Ork combines everything he learned
       da_orky_answer = torch.matmul(da_orky_attention_weights, da_orky_values)
      
       return da_orky_answer, da_orky_attention_weights


class MultiHeadOrkAttention(nn.Module):
   """
   DIS IS WHERE WE GET A WHOLE MOB OF ORK HEADS WORKIN' TOGETHER!
  
   Instead of just one Ork tryin' to figure everything out,
   we get multiple Orks (heads) to look at da problem from
   different angles. Den we combine all their answers!
  
   It's like havin' a whole mob of Orks all shoutin' their
   opinions, and den da Boss Ork (that's us) decides which
   answers are da best and combines 'em all together!
  
   EACH ORK HEAD IS LIKE A DIFFERENT SPECIALIST:
   - One might be good at lookin' for verbs
   - Another might be good at lookin' for nouns 
   - Another might be good at understandin' relationships
   """
  
   def __init__(self, da_orky_model_size, num_orky_heads):
       super().__init__()
       self.da_orky_model_size = da_orky_model_size
       self.num_orky_heads = num_orky_heads
      
       # Each Ork head gets a piece of da model to work with
       # We divide da model size by da number of heads so each head has its own space
       self.da_orky_head_size = da_orky_model_size // num_orky_heads
      
       # Create all da Ork heads - each one is a specialist!
       self.da_orky_heads = nn.ModuleList([
           OrkAttentionHead(da_orky_model_size, self.da_orky_head_size)
           for _ in range(num_orky_heads)
       ])
      
       # Dis is da Boss Ork who combines all da answers from da different heads
       # He takes all da head outputs and makes one final decision
       self.da_boss_ork = nn.Linear(da_orky_model_size, da_orky_model_size)
      
   def do_da_mob_finkin(self, da_wordz):
       """
       ALL DA ORK HEADS WORK TOGETHER TO SOLVE DA PROBLEM!
      
       da_wordz: Da words dat all da Ork heads need to look at
      
       DIS IS WHERE DA MAGIC HAPPENS:
       1. Each Ork head does his own special thinkin'
       2. We collect all their answers
       3. Da Boss Ork combines everything into one final answer
       """
       # Each Ork head does his own thinkin' and gives us his answer
       da_orky_head_answers = []
       all_da_orky_attention_weights = []
      
       # Go through each Ork head and let 'im do his work
       for ork_head in self.da_orky_heads:
           # Each head does his own finkin' and gives us his answer
           da_head_answer, da_head_attention = ork_head.do_da_orky_finkin(da_wordz)
           da_orky_head_answers.append(da_head_answer)
           all_da_orky_attention_weights.append(da_head_attention)
      
       # Combine all da Ork heads' answers by stickin' 'em together
       # Dis is like takin' all da different opinions and puttin' 'em side by side
       # torch.cat: Dis concatenates (sticks together) tensors along a specific dimension
       # dim=-1: We stick 'em together along da last dimension (like puttin' columns side by side)
       # Why we need 'em (Orky): We need to combine all da different Ork heads' opinions into one big answer!
       # Why we need 'em (Humie): Concatenation preserves all information from multiple attention heads,
       # allowing the model to capture different types of relationships simultaneously
       # If each head gives us a 64-dimensional answer, and we have 4 heads, we get a 256-dimensional answer
       da_combined_orky_answers = torch.cat(da_orky_head_answers, dim=-1)
      
       # Da Boss Ork makes da final decision by processin' all da head answers
       # He takes all da different opinions and makes one final, smart answer
       da_final_orky_answer = self.da_boss_ork(da_combined_orky_answers)
      
       return da_final_orky_answer, all_da_orky_attention_weights


class OrkFeedForward(nn.Module):
   """
   DIS IS DA ORK'S BRAIN PROCESSIN' CENTER!
  
   After da Ork heads figure out which words are important,
   dis part takes all dat information and processes it through
   da Ork's brain. It's like da Ork thinkin' really hard about
   what he learned and comin' up with a better answer.
  
   Da Ork brain has two layers:
   1. First layer: Makes da information bigger and more complex (like expandin' da brain)
   2. Second layer: Brings it back down to da right size (like compressin' it back)
  
   DIS IS LIKE HAVIN' A REALLY SMART ORK WHO TAKES SIMPLE THOUGHTS
   AND MAKES 'EM INTO COMPLEX, SMART THOUGHTS!
   """
  
   def __init__(self, da_orky_model_size, da_orky_feedforward_size):
       super().__init__()
       # Dis layer makes da information bigger and more complex
       # It's like da Ork brain expandin' to think about more things
       # nn.Linear: Takes da input (da_orky_model_size) and expands it to da_orky_feedforward_size
       # Dis gives da Ork brain more space to think about complex patterns
       # Why we need 'em (Orky): Da Ork brain needs more space to think about complex stuff before makin' a decision!
       # Why we need 'em (Humie): The expansion allows the network to learn complex non-linear transformations
       # by increasing the representational capacity in the hidden layer
       self.make_da_brain_big = nn.Linear(da_orky_model_size, da_orky_feedforward_size)
      
       # Dis layer brings it back down to da right size
       # It's like compressin' all da complex thoughts back into da right size
       # nn.Linear: Takes da expanded thoughts and compresses 'em back to da original size
       # Why we need 'em (Orky): Da Ork needs to compress all his complex thoughts back to da right size!
       # Why we need 'em (Humie): The compression projects the expanded representation back to the original
       # model dimension, maintaining the same output size as the input
       # Dis is like da Ork brain summarizin' all his complex thoughts into a simple answer
       self.make_da_brain_right_size = nn.Linear(da_orky_feedforward_size, da_orky_model_size)
      
       # Sometimes Orks forget things (dis is called dropout for regularization)
       # It helps prevent da Ork from memorizin' too much and not bein' able to learn new things
       # nn.Dropout: Randomly sets some values to 0 during trainin' (makes da Ork forget some things)
       # 0.1 means 10% of da values get set to 0 randomly - dis prevents overfittin'
       # Why we need 'em (Orky): Da Ork needs to forget some things so he doesn't memorize everything perfectly!
       # Why we need 'em (Humie): Dropout prevents overfitting by randomly zeroing out neurons during training,
       # forcing the network to learn robust features that don't depend on specific neurons
       self.da_orky_forgets = nn.Dropout(0.1)
      
   def do_da_brain_processin(self, da_wordz):
       """
       DA ORK BRAIN PROCESSES DA INFORMATION AND MAKES IT BETTER!
      
       da_wordz: Da information dat da Ork brain needs to process
      
       DIS IS WHERE DA ORK BRAIN:
       1. Expands da information to think about it more deeply
       2. Applies some forgettin' to prevent over-memorization
       3. Compresses it back to da right size
       """
       # First, da Ork brain expands da information to think about it more deeply
       # F.relu: Dis is da Rectified Linear Unit - it's like da Ork sayin': "If it's negative, ignore it!"
       # ReLU takes any negative number and makes it 0, keeps positive numbers as they are
       # Why we need 'em (Orky): Da Ork brain needs to focus on da good stuff and ignore da bad stuff!
       # Why we need 'em (Humie): ReLU introduces non-linearity, allowing the network to learn complex patterns
       # while being computationally efficient and helping with the vanishing gradient problem
       # Dis helps da Ork brain focus on da important (positive) information and ignore da bad (negative) stuff
       da_expanded_brain = F.relu(self.make_da_brain_big(da_wordz))
      
       # Sometimes da Ork forgets a bit of what he learned (prevents overfitting)
       # Dis helps da Ork learn better by not memorizin' everything perfectly
       da_brain_wif_some_forgettin = self.da_orky_forgets(da_expanded_brain)
      
       # Den, da Ork brain brings it back to da right size
       # Dis compresses all da complex thoughts back into da original size
       da_final_brain_output = self.make_da_brain_right_size(da_brain_wif_some_forgettin)
      
       return da_final_brain_output


class OrkLayerNorm(nn.Module):
   """
   DIS IS ORK DISCIPLINE!
  
   Sometimes Orks get too excited and their numbers get too big or too small.
   Dis layer keeps 'em in check by normalizin' da values so dey stay
   in a reasonable range. It's like da Ork Boss keepin' da boyz in line!
  
   DIS IS LIKE HAVIN' A DISCIPLINARY ORK WHO MAKES SURE EVERYONE
   STAYS IN THEIR PROPER PLACE AND DOESN'T GET TOO CRAZY!
   """
  
   def __init__(self, da_orky_model_size, da_orky_epsilon=1e-6):
       super().__init__()
       # Dis is da scale factor - it makes things bigger or smaller as needed
       # nn.Parameter: Dis creates a parameter dat da model can learn (like a weight dat gets updated during trainin')
       # torch.ones: Creates a tensor filled wif 1s - dis is da initial value for gamma
       # Why we need 'em (Orky): Da Ork Boss needs to learn how to scale things up or down to keep 'em in line!
       # Why we need 'em (Humie): Learnable parameters allow the model to adapt the normalization to the data,
       # providing flexibility in how the normalization is applied
       # Gamma starts at 1 (no scaling) and learns to scale up or down as needed
       self.da_orky_gamma = nn.Parameter(torch.ones(da_orky_model_size))
      
       # Dis is da shift factor - it moves things up or down as needed
       # nn.Parameter: Another learnable parameter
       # torch.zeros: Creates a tensor filled wif 0s - dis is da initial value for beta
       # Why we need 'em (Orky): Da Ork Boss also needs to learn how to shift things up or down!
       # Why we need 'em (Humie): The shift parameter allows the model to learn an optimal mean for the normalized values
       # Beta starts at 0 (no shifting) and learns to shift up or down as needed
       self.da_orky_beta = nn.Parameter(torch.zeros(da_orky_model_size))
      
       # Dis is a tiny number to prevent division by zero (Orks don't like math errors!)
       self.da_orky_epsilon = da_orky_epsilon
      
   def keep_da_orks_in_line(self, da_wordz):
       """
       KEEP DA ORKS IN LINE BY NORMALIZIN' THEIR VALUES!
      
       da_wordz: Da values dat need to be kept in line
      
       DIS IS WHERE DA DISCIPLINARY ORK:
       1. Calculates da average (mean) of all da values
       2. Calculates how spread out da values are (standard deviation)
       3. Normalizes everything so it's in a nice, controlled range
       """
       # Calculate da average of all da values (da mean)
       # mean(-1, keepdim=True): Calculates da average along da last dimension
       # -1 means da last dimension, keepdim=True keeps da same number of dimensions
       # Why we need 'em (Orky): Da Ork Boss needs to know da average so he can keep everyone in line!
       # Why we need 'em (Humie): Mean calculation is essential for normalization - we need to center the data
       # around zero before scaling it by the standard deviation
       # Dis is like findin' da average height of all da Orks in a group
       da_orky_mean = da_wordz.mean(-1, keepdim=True)
      
       # Calculate how spread out da values are (da standard deviation)
       # std(-1, keepdim=True): Calculates how spread out da values are along da last dimension
       # Why we need 'em (Orky): Da Ork Boss needs to know how different everyone is so he can normalize 'em properly!
       # Why we need 'em (Humie): Standard deviation measures the spread of data, which is crucial for proper
       # normalization - it tells us how much to scale the centered data
       # Dis tells us if da Orks are all da same height (low std) or very different heights (high std)
       da_orky_std = da_wordz.std(-1, keepdim=True)
      
       # Normalize da values: subtract da mean, divide by da std, den scale and shift
       # Dis keeps all da values in a nice, controlled range
       da_normalized_orky_values = (self.da_orky_gamma *
                                  (da_wordz - da_orky_mean) /
                                  (da_orky_std + self.da_orky_epsilon) +
                                  self.da_orky_beta)
      
       return da_normalized_orky_values


class OrkyTransformerBlock(nn.Module):
   """
   DIS IS A COMPLETE ORK TRANSFORMER BLOCK!
  
   Dis combines everything together:
   1. Multi-Head Attention (da Ork mob workin' together)
   2. Feed-Forward Network (da Ork brain processin')
   3. Layer Normalization (da Ork discipline)
   4. Residual connections (da Ork remembers what he knew before)
  
   It's like havin' a whole Ork unit workin' together to process information!
   Each block makes da Orks smarter and better at understandin' language.
  
   DIS IS DA HEART OF DA ORKY TRANSFO'MA' - WHERE ALL DA SMART STUFF HAPPENS!
   """
  
   def __init__(self, da_orky_model_size, num_orky_heads, da_orky_feedforward_size):
       super().__init__()
       # Da Ork mob dat works together to pay attention
       self.da_orky_attention = MultiHeadOrkAttention(da_orky_model_size, num_orky_heads)
      
       # Da Ork brain dat processes information
       self.da_orky_feed_forward = OrkFeedForward(da_orky_model_size, da_orky_feedforward_size)
      
       # Da disciplinary Orks dat keep everything in line
       self.da_orky_norm1 = OrkLayerNorm(da_orky_model_size)
       self.da_orky_norm2 = OrkLayerNorm(da_orky_model_size)
      
       # Sometimes da Orks forget a bit (dropout for regularization)
       self.da_orky_dropout = nn.Dropout(0.1)
      
   def do_da_complete_orky_processin(self, da_wordz):
       """
       DA COMPLETE ORK PROCESSING PIPELINE!
      
       da_wordz: Da words dat need to be processed by da Ork transformer block
      
       DIS IS WHERE DA ORK TRANSFORMER BLOCK:
       1. Does multi-head attention (da Ork mob thinks together)
       2. Adds da original information back (residual connection)
       3. Normalizes everything (keeps da Orks in line)
       4. Does feed-forward processin' (da Ork brain works)
       5. Adds da information back again (another residual connection)
       6. Normalizes everything again (keeps 'em in line again)
       """
       # STEP 1: Multi-Head Attention with residual connection
       # Da Ork mob does their thinkin' and we add back what they knew before
       da_attention_output, da_attention_weights = self.da_orky_attention.do_da_mob_finkin(da_wordz)
      
       # Add da original information back (residual connection) and normalize
       # Dis is like da Ork rememberin' what he knew before and addin' it to his new learnin'
       da_wordz = self.da_orky_norm1(da_wordz + self.da_orky_dropout(da_attention_output))
      
       # STEP 2: Feed-Forward with residual connection
       # Da Ork brain processes da information and we add back what it knew before
       da_feedforward_output = self.da_orky_feed_forward.do_da_brain_processin(da_wordz)
      
       # Add da information back again (residual connection) and normalize
       # Dis is like da Ork brain rememberin' what it learned before and addin' it to da new processin'
       da_wordz = self.da_orky_norm2(da_wordz + self.da_orky_dropout(da_feedforward_output))
      
       return da_wordz, da_attention_weights


class OrkyTransformer(nn.Module):
   """
   DA COMPLETE ORKY TRANSFO'MA'!
  
   Dis is da full transformer with multiple Ork blocks stacked together.
   Each block makes da Orks smarter and better at understandin' language!
  
   DIS IS DA ULTIMATE ORK INTELLIGENCE MACHINE!
   IT TAKES WORDS AND MAKES 'EM INTO SMART ORKY THOUGHTS!
   """
  
   def __init__(self, da_orky_vocab_size, da_orky_model_size, num_orky_heads,
                num_orky_layers, da_orky_feedforward_size, da_max_orky_seq_len):
       super().__init__()
       self.da_orky_model_size = da_orky_model_size
      
       # Word embeddings (turn words into numbers da Orks can understand)
       # Dis is like givin' each word a special Orky number so da Orks can work wif it
       # nn.Embedding: Dis creates a lookup table dat converts word IDs to vectors
       # It's like havin' a big book where each word has its own special number code
       # Why we need 'em (Orky): Da Orks need to turn words into numbers so dey can do math wif 'em!
       # Why we need 'em (Humie): Embeddings convert discrete tokens to continuous vectors, enabling
       # the neural network to process text data and learn semantic relationships between words
       # da_orky_vocab_size: How many different words we can have
       # da_orky_model_size: How big each word's vector is (how many numbers represent each word)
       self.da_orky_embedding = nn.Embedding(da_orky_vocab_size, da_orky_model_size)
      
       # Positional encoding (tell da Orks where each word is in da sentence)
       # Without dis, da Orks wouldn't know if "Ork" comes before or after "Waaagh!"
       self.da_orky_pos_encoding = self._create_da_orky_positional_encoding(
           da_max_orky_seq_len, da_orky_model_size)
      
       # Stack of Ork transformer blocks - each one makes da Orks smarter!
       self.da_orky_transformer_blocks = nn.ModuleList([
           OrkyTransformerBlock(da_orky_model_size, num_orky_heads, da_orky_feedforward_size)
           for _ in range(num_orky_layers)
       ])
      
       # Final layer to get da output (turn da Orky thoughts back into words)
       self.da_orky_output_layer = nn.Linear(da_orky_model_size, da_orky_vocab_size)
      
   def _create_da_orky_positional_encoding(self, da_max_orky_seq_len, da_orky_model_size):
       """
       CREATE POSITIONAL ENCODING FOR DA ORKS!
      
       Dis tells da Orks where each word is in da sentence.
       Without dis, da Orks wouldn't know if "Ork" comes before or after "Waaagh!"
      
       DIS USES SINE AND COSINE WAVES TO CREATE A UNIQUE PATTERN
       FOR EACH POSITION IN DA SENTENCE!
       """
       # Create a big empty space for all da positional encodings
       # torch.zeros: Creates a tensor filled wif zeros - dis is our empty canvas
       # Why we need 'em (Orky): We need a blank canvas to draw our positional patterns on!
       # Why we need 'em (Humie): We need to initialize a tensor with zeros before filling it with positional encodings
       # Shape: (da_max_orky_seq_len, da_orky_model_size) - one row for each position, one column for each dimension
       da_orky_pe = torch.zeros(da_max_orky_seq_len, da_orky_model_size)
      
       # Create positions from 0 to max_seq_len (where each word is)
       # torch.arange: Creates a sequence of numbers from 0 to da_max_orky_seq_len-1
       # Why we need 'em (Orky): We need to count da positions so each word knows where it is!
       # Why we need 'em (Humie): We need position indices to create unique positional encodings for each token position
       # unsqueeze(1): Adds a dimension to make it a column vector (like turnin' [0,1,2,3] into [[0],[1],[2],[3]])
       # Why we need 'em (Orky): We need to make it da right shape for matrix multiplication!
       # Why we need 'em (Humie): Broadcasting requires compatible tensor shapes for element-wise operations
       # float(): Converts to float numbers so we can do math wif 'em
       da_orky_positions = torch.arange(0, da_max_orky_seq_len).unsqueeze(1).float()
      
       # Create da division term for da sine and cosine patterns
       # Dis makes sure each position gets a unique pattern
       # torch.arange(0, da_orky_model_size, 2): Creates [0, 2, 4, 6, ...] - every other number
       # Why we need 'em (Orky): We need different frequencies for each dimension so each position is unique!
       # Why we need 'em (Humie): Different frequencies create unique patterns for each position and dimension
       # torch.exp: Dis is da exponential function (e^x) - it makes numbers grow really fast
       # Why we need 'em (Orky): We need to make da frequencies get bigger and bigger for each dimension!
       # Why we need 'em (Humie): Exponential scaling creates a geometric progression of frequencies
       # math.log(10000.0): Natural logarithm of 10000 - dis controls how fast da patterns change
       # Why we need 'em (Orky): Dis controls how fast da patterns change - not too fast, not too slow!
       # Why we need 'em (Humie): The 10000 factor controls the rate of frequency decay across dimensions
       da_orky_div_term = torch.exp(torch.arange(0, da_orky_model_size, 2).float() *
                                  -(math.log(10000.0) / da_orky_model_size))
      
       # Fill in da sine patterns for even positions
       # torch.sin: Dis is da sine function - it creates wavy patterns
       # Why we need 'em (Orky): We need wavy patterns so each position has a unique signature!
       # Why we need 'em (Humie): Sine waves create smooth, periodic patterns that are easy for the model to learn
       # [:, 0::2]: Takes every other column starting from 0 (even positions)
       # Why we need 'em (Orky): We alternate between sine and cosine to make each position really unique!
       # Why we need 'em (Humie): Alternating sine and cosine creates orthogonal patterns for better representation
       # Dis creates wavy patterns dat are different for each position
       da_orky_pe[:, 0::2] = torch.sin(da_orky_positions * da_orky_div_term)
      
       # Fill in da cosine patterns for odd positions
       # torch.cos: Dis is da cosine function - it creates wavy patterns dat are offset from sine
       # Why we need 'em (Orky): Cosine makes different wavy patterns dat work together wif sine!
       # Why we need 'em (Humie): Cosine provides phase-shifted patterns that complement sine waves
       # [:, 1::2]: Takes every other column starting from 1 (odd positions)
       # Why we need 'em (Orky): We fill da odd positions wif cosine to make 'em different from sine!
       # Why we need 'em (Humie): This creates a complete set of orthogonal basis functions
       # Dis creates different wavy patterns for da odd positions
       da_orky_pe[:, 1::2] = torch.cos(da_orky_positions * da_orky_div_term)
      
       # Add batch dimension so it works wif our batch processing
       # unsqueeze(0): Adds a dimension at position 0 (da first dimension)
       # Why we need 'em (Orky): We need to add a batch dimension so our model can work wif batches!
       # Why we need 'em (Humie): Neural networks expect batch dimensions for efficient processing
       # Dis changes da shape from (seq_len, model_size) to (1, seq_len, model_size)
       # We need dis because our model expects a batch dimension (even if it's just 1)
       return da_orky_pe.unsqueeze(0)
      
   def do_da_complete_orky_transfo_ma(self, da_wordz):
       """
       DA COMPLETE ORKY TRANSFO'MA' IN ACTION!
      
       da_wordz: Da input words dat need to be processed (batch_size, seq_len)
      
       DIS IS WHERE DA MAGIC HAPPENS:
       1. Turn words into numbers and add position info
       2. Pass through all da Ork transformer blocks
       3. Get da final output (predictions for next words)
       """
       # Get da length of da sequence so we know how many words we're workin' wif
       da_orky_seq_len = da_wordz.size(1)
      
       # STEP 1: Turn words into numbers and add position info
       # First, turn each word into a vector of numbers (embedding)
       # self.da_orky_embedding(da_wordz): Looks up each word ID and gets its vector
       # * math.sqrt(self.da_orky_model_size): Scales da embeddings by da square root of model size
       # Dis scaling helps keep da values in a good range for da neural network
       da_orky_embedded_words = self.da_orky_embedding(da_wordz) * math.sqrt(self.da_orky_model_size)
      
       # Den, add da positional encoding so da Orks know where each word is
       # [:, :da_orky_seq_len, :]: Takes only da first da_orky_seq_len positions (in case our sequence is shorter than max)
       # .to(da_wordz.device): Makes sure da positional encoding is on da same device (CPU or GPU) as our input
       da_orky_embedded_words = (da_orky_embedded_words +
                               self.da_orky_pos_encoding[:, :da_orky_seq_len, :].to(da_wordz.device))
      
       # STEP 2: Pass through all da Ork transformer blocks
       # Each block makes da Orks smarter and better at understandin'
       all_da_orky_attention_weights = []
       for orky_transformer_block in self.da_orky_transformer_blocks:
           da_orky_embedded_words, da_attention_weights = orky_transformer_block.do_da_complete_orky_processin(da_orky_embedded_words)
           all_da_orky_attention_weights.append(da_attention_weights)
      
       # STEP 3: Get da final output
       # Turn da Orky thoughts back into word predictions
       da_orky_final_output = self.da_orky_output_layer(da_orky_embedded_words)
      
       return da_orky_final_output, all_da_orky_attention_weights


def demonstrate_da_orky_transfo_ma():
   """
   LET'S SEE DA ORKY TRANSFO'MA' IN ACTION!
  
   Dis function shows how our Orky Transformer works with a simple example.
   NOW WIF EVEN MORE ORKY VARIABLES AND COMMENTS!
   """
   print("WAAAGH! STARTIN' DA ORKY TRANSFO'MA' v1.2 DEMONSTRATION!")
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
       "<PAD>": 7,       # Padding for short sentences
       "<START>": 8,     # Start of sentence marker
       "<END>": 9        # End of sentence marker
   }
  
   # Get da size of our vocabulary
   da_orky_vocab_size = len(da_orky_vocab)
  
   # Set up da Orky Transformer parameters
   da_orky_model_size = 64        # Size of da Ork brain (model dimension)
   num_orky_heads = 4             # Number of Ork heads (attention heads)
   num_orky_layers = 2            # Number of Ork transformer blocks
   da_orky_feedforward_size = 128 # Size of da Ork brain processing center
   da_max_orky_seq_len = 10       # Maximum sentence length
  
   # Create da Orky Transformer
   print("CREATIN' DA ORKY TRANSFO'MA'...")
   da_orky_transformer = OrkyTransformer(
       da_orky_vocab_size=da_orky_vocab_size,
       da_orky_model_size=da_orky_model_size,
       num_orky_heads=num_orky_heads,
       num_orky_layers=num_orky_layers,
       da_orky_feedforward_size=da_orky_feedforward_size,
       da_max_orky_seq_len=da_max_orky_seq_len
   )
  
   print(f"Created Orky Transformer wif:")
   print(f"- Vocabulary size: {da_orky_vocab_size} (how many words da Orks know)")
   print(f"- Model dimension: {da_orky_model_size} (size of da Ork brain)")
   print(f"- Number of heads: {num_orky_heads} (how many Ork heads work together)")
   print(f"- Number of layers: {num_orky_layers} (how many Ork transformer blocks)")
   print(f"- Feed-forward dimension: {da_orky_feedforward_size} (size of da Ork brain processing)")
   print()
  
   # Create a simple Ork sentence
   da_orky_sentence = ["<START>", "WAAAGH", "ORK", "FIGHT", "WIN", "<END>"]
  
   # Convert da words to numbers dat da Orks can understand
   da_orky_sentence_ids = [da_orky_vocab[word] for word in da_orky_sentence]
  
   # Add padding to make it da right length
   while len(da_orky_sentence_ids) < da_max_orky_seq_len:
       da_orky_sentence_ids.append(da_orky_vocab["<PAD>"])
  
   # Convert to tensor and add batch dimension
       # torch.tensor: Converts our list of numbers into a PyTorch tensor (a fancy array dat can do math)
       # Why we need 'em (Orky): We need to turn our numbers into fancy Orky tensors so dey can do math!
       # Why we need 'em (Humie): PyTorch tensors enable GPU acceleration and automatic differentiation
       # [da_orky_sentence_ids]: Wraps our list in another list to create a batch dimension
       # Why we need 'em (Orky): We need a batch dimension even if we only have one sentence!
       # Why we need 'em (Humie): Neural networks expect batched input for efficient processing
       # Dis makes our shape (seq_len,) into (1, seq_len) - one sentence in a batch
   da_orky_input_tensor = torch.tensor([da_orky_sentence_ids])
  
   print("Input Ork sentence:", " ".join(da_orky_sentence))
   print("Input tensor shape:", da_orky_input_tensor.shape)
   print()
  
   # Run da Orky Transformer
   print("RUNNIN' DA ORKY TRANSFO'MA'...")
   with torch.no_grad():
       da_orky_output, all_da_orky_attention_weights = da_orky_transformer.do_da_complete_orky_transfo_ma(da_orky_input_tensor)
  
   print("Output shape:", da_orky_output.shape)
   print("Number of attention weight sets:", len(all_da_orky_attention_weights))
   print("Attention weights per layer:", len(all_da_orky_attention_weights[0]))
   print()
  
   # Show what da Orks are payin' attention to
   print("DA ORKS ARE PAYIN' ATTENTION TO:")
   print("-" * 50)
  
   for layer_idx, da_layer_attention in enumerate(all_da_orky_attention_weights):
       print(f"Layer {layer_idx + 1} (Ork Transformer Block {layer_idx + 1}):")
       for head_idx, da_head_attention in enumerate(da_layer_attention):
           print(f"  Head {head_idx + 1} (Ork Head {head_idx + 1}) attention weights:")
           # Show attention for first word
           # [0, 0, :]: Takes da first batch (0), first word (0), and all attention weights (:)
           # Why we need 'em (Orky): We need to get da attention weights for da first word so we can see wot da Ork is lookin' at!
           # Why we need 'em (Humie): We extract attention weights for visualization and analysis
           # .numpy(): Converts from PyTorch tensor to NumPy array so we can print it nicely
           # Why we need 'em (Orky): We need to convert to NumPy so we can print da numbers nicely!
           # Why we need 'em (Humie): NumPy arrays are easier to work with for printing and basic operations
           da_first_word_attention = da_head_attention[0, 0, :].numpy()
           for word_idx, da_attention_score in enumerate(da_first_word_attention):
               if word_idx < len(da_orky_sentence):
                   da_word = da_orky_sentence[word_idx]
                   print(f"    {da_word}: {da_attention_score:.3f}")
           print()
  
   print("WAAAGH! DA ORKY TRANSFO'MA' v1.2 IS WORKIN' PERFECTLY!")
   print("Da Orks are lookin' at words and figurin' out which ones matter!")
   print("NOW WIF EVEN MORE ORKY VARIABLES AND COMMENTS!")
   print("=" * 70)


if __name__ == "__main__":
   demonstrate_da_orky_transfo_ma()






