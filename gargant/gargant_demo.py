"""
GARGANT TITANS DEMONSTRATION SCRIPT

DIS IS DA ULTIMATE DEMONSTRATION OF DA GARGANT TITANS!
Dis script shows off all da amazing features of our Orky Titans transformer:

1. WAAAGH MEMORY - Da collective Ork memory dat grows stronger!
2. SURPRISE DETECTION - When somethin' unexpected happens, da Orks remember it!
3. DYNAMIC MEMORY - Da Orks learn and remember during battle (test time)!
4. TITAN POWER - Massive Ork war machine wif multiple heads and layers!

WAAAGH! (That means "Let's show off our TITAN POWER!" in Ork)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gargant_titans import GargantTitans, demonstrate_da_gargant_titans

def create_da_orky_vocabulary():
    """
    CREATE DA ORKY VOCABULARY FOR DA GARGANT TITANS!
    
    Dis creates a comprehensive vocabulary of Ork words dat da Gargant can use.
    """
    da_orky_vocab = {
        # Basic Ork words
        "WAAAGH": 0,        # Da battle cry!
        "ORK": 1,           # Da best race in da galaxy!
        "DAKKA": 2,         # More shootin'!
        "BOSS": 3,          # Da leader of da Orks
        "BOYZ": 4,          # Da Ork soldiers
        "FIGHT": 5,         # Wot Orks do best!
        "WIN": 6,           # Wot Orks always do!
        
        # Memory and surprise words
        "SURPRISE": 7,      # When somethin' unexpected happens!
        "MEMORY": 8,        # Da Ork memory!
        "REMEMBER": 9,      # Da Ork remembers!
        "LEARN": 10,        # Da Ork learns!
        "FORGET": 11,       # Da Ork forgets (sometimes)!
        
        # Titan and Gargant words
        "GARGANT": 12,      # Da massive Ork war machine!
        "TITAN": 13,        # Da ultimate war machine!
        "POWER": 14,        # Da Ork power!
        "WAR": 15,          # Wot Orks love!
        "DESTROY": 16,      # Wot Orks do to enemies!
        
        # Special tokens
        "<PAD>": 17,        # Padding for short sentences
        "<START>": 18,      # Start of sentence marker
        "<END>": 19,        # End of sentence marker
        "<UNK>": 20         # Unknown word marker
    }
    
    return da_orky_vocab

def create_da_orky_sentences():
    """
    CREATE DA ORKY SENTENCES FOR TESTIN' DA GARGANT TITANS!
    
    Dis creates various Ork sentences to test da Gargant's memory and surprise detection.
    """
    da_orky_sentences = [
        # Basic Ork sentences
        ["<START>", "WAAAGH", "ORK", "FIGHT", "WIN", "<END>"],
        ["<START>", "BOSS", "BOYZ", "DAKKA", "DESTROY", "<END>"],
        ["<START>", "GARGANT", "POWER", "WAR", "WIN", "<END>"],
        
        # Memory and surprise sentences
        ["<START>", "SURPRISE", "MEMORY", "LEARN", "REMEMBER", "<END>"],
        ["<START>", "ORK", "SURPRISE", "GARGANT", "POWER", "<END>"],
        ["<START>", "TITAN", "MEMORY", "WAAAGH", "WIN", "<END>"],
        
        # Complex sentences
        ["<START>", "GARGANT", "TITAN", "POWER", "WAR", "DESTROY", "WIN", "<END>"],
        ["<START>", "BOSS", "ORK", "SURPRISE", "MEMORY", "LEARN", "WIN", "<END>"],
        ["<START>", "WAAAGH", "BOYZ", "DAKKA", "FIGHT", "POWER", "WIN", "<END>"]
    ]
    
    return da_orky_sentences

def demonstrate_da_gargant_memory():
    """
    DEMONSTRATE DA GARGANT'S MEMORY CAPABILITIES!
    
    Dis shows how da Gargant remembers information and gets surprised.
    """
    print("WAAAGH! DEMONSTRATIN' DA GARGANT'S MEMORY!")
    print("=" * 60)
    
    # Create da Gargant Titans
    da_orky_vocab = create_da_orky_vocabulary()
    da_orky_vocab_size = len(da_orky_vocab)
    
    da_gargant_titans = GargantTitans(
        da_orky_vocab_size=da_orky_vocab_size,
        da_orky_model_size=128,
        num_orky_heads=8,
        num_orky_layers=4,
        da_orky_feedforward_size=256,
        da_max_orky_seq_len=20,
        da_memory_capacity=1000
    )
    
    # Test da Gargant wif different sentences
    da_orky_sentences = create_da_orky_sentences()
    
    print("TESTIN' DA GARGANT WIF DIFFERENT SENTENCES:")
    print("-" * 50)
    
    for sentence_idx, da_sentence in enumerate(da_orky_sentences):
        print(f"\nSentence {sentence_idx + 1}: {' '.join(da_sentence)}")
        
        # Convert to tensor
        da_sentence_ids = [da_orky_vocab[word] for word in da_sentence]
        while len(da_sentence_ids) < 20:
            da_sentence_ids.append(da_orky_vocab["<PAD>"])
        
        da_input_tensor = torch.tensor([da_sentence_ids])
        
        # Run da Gargant
        with torch.no_grad():
            da_output, da_attention, da_surprise, da_memory = da_gargant_titans.do_da_gargant_titans_processin(da_input_tensor)
        
        # Show da results
        da_avg_surprise = torch.stack(da_surprise).mean().item()
        print(f"  Average surprise level: {da_avg_surprise:.3f}")
        print(f"  Memory shape: {da_memory.shape}")
        print(f"  Output shape: {da_output.shape}")
        
        # Show da most likely next words
        da_next_word_probs = F.softmax(da_output[0, -1, :], dim=-1)
        da_top_words = torch.topk(da_next_word_probs, 5)
        
        print("  Most likely next words:")
        for i, (prob, word_id) in enumerate(zip(da_top_words.values, da_top_words.indices)):
            da_word = list(da_orky_vocab.keys())[word_id.item()]
            print(f"    {i+1}. {da_word}: {prob.item():.3f}")

def demonstrate_da_gargant_surprise():
    """
    DEMONSTRATE DA GARGANT'S SURPRISE DETECTION!
    
    Dis shows how da Gargant detects surprises and remembers 'em better.
    """
    print("\nWAAAGH! DEMONSTRATIN' DA GARGANT'S SURPRISE DETECTION!")
    print("=" * 60)
    
    # Create da Gargant Titans
    da_orky_vocab = create_da_orky_vocabulary()
    da_orky_vocab_size = len(da_orky_vocab)
    
    da_gargant_titans = GargantTitans(
        da_orky_vocab_size=da_orky_vocab_size,
        da_orky_model_size=128,
        num_orky_heads=8,
        num_orky_layers=4,
        da_orky_feedforward_size=256,
        da_max_orky_seq_len=20,
        da_memory_capacity=1000
    )
    
    # Test wif expected vs unexpected inputs
    da_expected_sentence = ["<START>", "WAAAGH", "ORK", "FIGHT", "WIN", "<END>"]
    da_unexpected_sentence = ["<START>", "SURPRISE", "MEMORY", "LEARN", "REMEMBER", "<END>"]
    
    print("TESTIN' WIF EXPECTED INPUT:")
    print(f"Sentence: {' '.join(da_expected_sentence)}")
    
    # Convert expected sentence to tensor
    da_expected_ids = [da_orky_vocab[word] for word in da_expected_sentence]
    while len(da_expected_ids) < 20:
        da_expected_ids.append(da_orky_vocab["<PAD>"])
    
    da_expected_tensor = torch.tensor([da_expected_ids])
    
    # Run wif expected input
    with torch.no_grad():
        da_output1, da_attention1, da_surprise1, da_memory1 = da_gargant_titans.do_da_gargant_titans_processin(da_expected_tensor)
    
    da_avg_surprise1 = torch.stack(da_surprise1).mean().item()
    print(f"Average surprise level: {da_avg_surprise1:.3f}")
    
    print("\nTESTIN' WIF UNEXPECTED INPUT:")
    print(f"Sentence: {' '.join(da_unexpected_sentence)}")
    
    # Convert unexpected sentence to tensor
    da_unexpected_ids = [da_orky_vocab[word] for word in da_unexpected_sentence]
    while len(da_unexpected_ids) < 20:
        da_unexpected_ids.append(da_orky_vocab["<PAD>"])
    
    da_unexpected_tensor = torch.tensor([da_unexpected_ids])
    
    # Run wif unexpected input
    with torch.no_grad():
        da_output2, da_attention2, da_surprise2, da_memory2 = da_gargant_titans.do_da_gargant_titans_processin(da_unexpected_tensor)
    
    da_avg_surprise2 = torch.stack(da_surprise2).mean().item()
    print(f"Average surprise level: {da_avg_surprise2:.3f}")
    
    print(f"\nSURPRISE DIFFERENCE: {da_avg_surprise2 - da_avg_surprise1:.3f}")
    if da_avg_surprise2 > da_avg_surprise1:
        print("DA GARGANT IS MORE SURPRISED BY DA UNEXPECTED INPUT!")
    else:
        print("DA GARGANT IS NOT VERY SURPRISED...")

def demonstrate_da_gargant_attention():
    """
    DEMONSTRATE DA GARGANT'S ATTENTION MECHANISMS!
    
    Dis shows how da Gargant pays attention to different words.
    """
    print("\nWAAAGH! DEMONSTRATIN' DA GARGANT'S ATTENTION!")
    print("=" * 60)
    
    # Create da Gargant Titans
    da_orky_vocab = create_da_orky_vocabulary()
    da_orky_vocab_size = len(da_orky_vocab)
    
    da_gargant_titans = GargantTitans(
        da_orky_vocab_size=da_orky_vocab_size,
        da_orky_model_size=128,
        num_orky_heads=8,
        num_orky_layers=4,
        da_orky_feedforward_size=256,
        da_max_orky_seq_len=20,
        da_memory_capacity=1000
    )
    
    # Test wif a complex sentence
    da_test_sentence = ["<START>", "GARGANT", "TITAN", "POWER", "WAR", "DESTROY", "WIN", "<END>"]
    print(f"Test sentence: {' '.join(da_test_sentence)}")
    
    # Convert to tensor
    da_sentence_ids = [da_orky_vocab[word] for word in da_test_sentence]
    while len(da_sentence_ids) < 20:
        da_sentence_ids.append(da_orky_vocab["<PAD>"])
    
    da_input_tensor = torch.tensor([da_sentence_ids])
    
    # Run da Gargant
    with torch.no_grad():
        da_output, da_attention, da_surprise, da_memory = da_gargant_titans.do_da_gargant_titans_processin(da_input_tensor)
    
    # Show attention patterns
    print("\nATTENTION PATTERNS:")
    print("-" * 40)
    
    for layer_idx, da_layer_attention in enumerate(da_attention):
        print(f"\nGargant Titans Block {layer_idx + 1}:")
        for head_idx, da_head_attention in enumerate(da_layer_attention):
            print(f"  Ork Head {head_idx + 1}:")
            
            # Show attention for first word
            da_first_word_attention = da_head_attention[0, 0, :].numpy()
            da_attention_pairs = [(da_test_sentence[i], da_first_word_attention[i]) 
                                 for i in range(len(da_test_sentence))]
            da_attention_pairs.sort(key=lambda x: x[1], reverse=True)
            
            print("    Top attention words:")
            for word, attention in da_attention_pairs[:3]:
                print(f"      {word}: {attention:.3f}")

def main():
    """
    DA MAIN DEMONSTRATION OF DA GARGANT TITANS!
    
    Dis runs all da demonstrations to show off da Gargant's capabilities.
    """
    print("WAAAGH! WELCOME TO DA GARGANT TITANS DEMONSTRATION!")
    print("DIS IS DA ULTIMATE ORK WAR MACHINE WIF TITAN POWER!")
    print("=" * 80)
    
    # Run da basic demonstration
    demonstrate_da_gargant_titans()
    
    # Run da memory demonstration
    demonstrate_da_gargant_memory()
    
    # Run da surprise demonstration
    demonstrate_da_gargant_surprise()
    
    # Run da attention demonstration
    demonstrate_da_gargant_attention()
    
    print("\nWAAAGH! DA GARGANT TITANS DEMONSTRATION IS COMPLETE!")
    print("DIS IS DA ULTIMATE ORK WAR MACHINE WIF TITAN POWER!")
    print("DA ORKS HAVE MEMORY, SURPRISE DETECTION, AND TITAN STRENGTH!")
    print("=" * 80)

if __name__ == "__main__":
    main()
