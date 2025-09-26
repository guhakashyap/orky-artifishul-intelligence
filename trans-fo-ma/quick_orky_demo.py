"""
QUICK ORKY DEMO - A FAST VERSION DAT DOESN'T TAKE FOREVER!

DIS USES A SMALLER MODEL AND LESS TRAININ' SO IT'S FAST!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random

# Import the transformer with the correct method
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("orky_transformer1_2", "orky_transformer1.2.py")
orky_transformer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orky_transformer_module)
OrkyTransformer = orky_transformer_module.OrkyTransformer

def get_quick_orky_phrases():
    """
    GET A SMALL SET OF ORKY PHRASES FOR QUICK TRAININ'!
    """
    return [
        "<START> GREEN IS BEST <END>",
        "<START> BIGGA IS BETTA <END>",
        "<START> MORE DAKKA MORE FUN <END>",
        "<START> WAAAGH IS DA ANSWER <END>",
        "<START> ORKS ARE MADE FOR FIGHTIN <END>",
        "<START> AND WINNIN <END>",
        "<START> RED WUNZ GO FASTA <END>",
        "<START> KRUMP EM ALL <END>",
        "<START> SMASH DA GITZ <END>",
        "<START> GET DA CHOPPA <END>",
        "<START> WE NEED MORE DAKKA <END>",
        "<START> DA BOSS HAS A PLAN <END>",
        "<START> FOLLOW DA BIGGEST ORK <END>",
        "<START> LOOT DA WRECKAGE <END>",
        "<START> BUILD IT BIGGA <END>",
        "<START> TEEF ARE FOR BUYIN MORE DAKKA <END>",
        "<START> A PROPA CHOPPA IS A HAPPY CHOPPA <END>",
        "<START> NEVER TRUST A GIT DAT WHISPERS <END>",
        "<START> DA MORE BOYZ DA BIGGER DA WAAAGH <END>",
        "<START> GORK IS BRUTAL BUT KUNNIN <END>",
        "<START> MORK IS KUNNIN BUT BRUTAL <END>",
        "<START> A RED WUN ALWAYS GOES FASTA <END>",
        "<START> YELLOW MAKES BIGGA BOOMS <END>",
        "<START> BLUE IS LUCKY <END>",
        "<START> BLACK IS DED ARD <END>",
        "<START> DAKKA DAKKA DAKKA <END>",
        "<START> CHARGE DA GUN LINE <END>",
        "<START> GET STUCK IN LADS <END>",
        "<START> FOR DA WAAAGH <END>",
        "<START> FOR DA BOSS <END>",
        "<START> LETZ GO <END>",
        "<START> HUMIES ARE SQUISHY <END>",
        "<START> DEY HIDE IN METAL BOXES <END>",
        "<START> DAT HUMIE TANK NEEDS LOOTIN <END>",
        "<START> DEY SHOUT A LOT BUT RUN FAST <END>",
        "<START> A HUMIE GUN IS BORIN <END>",
        "<START> DEY GOT NO DAKKA <END>",
        "<START> DA ELDAR ARE SNEAKY GITZ <END>",
        "<START> DEY DANCE INSTEAD OF FIGHTIN PROPPA <END>",
        "<START> DEIR ARMOR IS THIN LIKE PAPER <END>",
        "<START> A GOOD CHOPPA GOES RIGHT THROUGH <END>",
        "<START> DEY FINK DEY ARE SO KUNNIN <END>",
        "<START> BUT ORKS ARE BRUTALLY KUNNIN <END>",
        "<START> DAT SQUIG LOOKS TASTY <END>",
        "<START> DA SQUIGS ARE HUNGRY <END>",
        "<START> FEED EM DA GITZ <END>",
        "<START> GORK AND MORK ARE WATCHIN <END>",
        "<START> DA GROTS ARE REVOLTIN <END>",
        "<START> ZOGGIN GROTS <END>",
        "<START> A GOOD STOMP SORTS EM OUT <END>",
        "<START> DIS SCRAP IS A GOOD SCRAP <END>",
        "<START> I LIKE DA NOISE <END>",
        "<START> IT SOUNDS LIKE VICTORY <END>",
        "<START> MEKBOY FIX DA GUBBINZ <END>",
        "<START> DA WEIRDBOY IS ZAPPIN <END>",
        "<START> DA PAINBOY HAS DA SYRINGE <END>",
        "<START> DA DOK IS KRAZY <END>",
        "<START> FIRE DA BIG GUNZ <END>",
        "<START> DA BATTLEWAGON IS READY <END>",
        "<START> MORE ARMOR PLATES <END>",
        "<START> AND MORE GUNZ <END>",
        "<START> PAINT IT RED FOR SPEED <END>",
        "<START> DRIVE DA WAGON KLOSE <END>",
        "<START> BUILD DA STOMPA TALLA <END>",
        "<START> MORE BOYZ MORE FIGHTIN <END>",
        "<START> SHOOTAS OR CHOPPAS <END>",
        "<START> WHY NOT BOTH <END>",
        "<START> DAT HUMIE LOOKS WEAK <END>",
        "<START> DEY GOT POINTY EARS AND POINTY STIKKS <END>",
        "<START> NOT ENUFF DAKKA <END>",
        "<START> DEY CRY WHEN DEIR TOYS BREAK <END>",
        "<START> WE BREAK ALL DEIR TOYS <END>",
        "<START> DA MEKBOY IS BUILDIN SOMEFING <END>",
        "<START> IT WILL BE LOUD AND SHOOTY <END>",
        "<START> DAT KUSTOM JOB IS A BEAUTY <END>",
        "<START> MORE PLATES MORE RIVETS <END>",
        "<START> DA BIGGEST GUNS ARE ALWAYS KUSTOM <END>",
        "<START> IF IT AINT BROKEN ADD MORE DAKKA <END>",
        "<START> DA WEIRDBOY IS GLOWIN GREEN <END>",
        "<START> HIS HEAD IS GONNA POP <END>",
        "<START> WAAAGH ENERGY IS FLOWIN <END>",
        "<START> HE ZAPPED A GIT INTO A SQUIG <END>",
        "<START> DA WEIRDBOY IS MUMBLIN TO HIMSELF <END>",
        "<START> STAY AWAY FROM DA WEIRDBOY <END>",
        "<START> UNLESS YOU WANT GREEN LIGHTNIN IN YA FACE <END>",
        "<START> HE CAN STOMP GITZ WITH HIS BRAIN <END>",
        "<START> GORK SPEAKS THROUGH DA WEIRDBOY <END>",
        "<START> OR MAYBE ITS MORK <END>"
    ]

def create_quick_vocab(phrases):
    """Create vocabulary from phrases"""
    all_words = set()
    for phrase in phrases:
        words = phrase.split()
        all_words.update(words)
    
    vocab = {"<PAD>": 0, "<START>": 1, "<END>": 2}
    word_list = sorted(list(all_words))
    
    for word in word_list:
        if word not in vocab:
            vocab[word] = len(vocab)
    
    reverse_vocab = {v: k for k, v in vocab.items()}
    return vocab, reverse_vocab

def words_to_numbers_quick(sentences, vocab, max_seq_len=20):
    """Convert words to numbers - quick version"""
    all_sentences_as_numbers = []
    
    for sentence in sentences:
        words = sentence.split()
        numbered_sentence = [vocab.get(word, vocab["<PAD>"]) for word in words]
        
        while len(numbered_sentence) < max_seq_len:
            numbered_sentence.append(vocab["<PAD>"])
        
        all_sentences_as_numbers.append(numbered_sentence)
    
    return torch.tensor(all_sentences_as_numbers)

def create_target_data_quick(input_data, vocab):
    """Create target data - quick version"""
    target_data = torch.cat([
        input_data[:, 1:],
        torch.full((input_data.shape[0], 1), vocab["<PAD>"])
    ], dim=1)
    
    return target_data

def quick_train(orky_transformer, input_data, target_data, vocab, num_epochs=50):
    """Quick training - much faster!"""
    optimizer = optim.Adam(orky_transformer.parameters(), lr=0.01)  # Higher learning rate
    loss_function = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])
    
    print(f"QUICK TRAININ' FOR {num_epochs} ROUNDS...")
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        output, _ = orky_transformer.do_da_complete_orky_transfo_ma(input_data)
        loss = loss_function(output.view(-1, len(vocab)), target_data.view(-1))
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Round {epoch + 1}/{num_epochs} - Loss: {loss.item():.4f}")
    
    print(f"QUICK TRAININ' DONE! Final Loss: {loss.item():.4f}")
    return orky_transformer

def generate_quick_orky(orky_transformer, vocab, reverse_vocab, max_length=15):
    """Generate Orky speech - quick version"""
    orky_transformer.eval()
    
    with torch.no_grad():
        current_sequence = [vocab["<START>"]]
        
        for _ in range(max_length - 1):
            input_tensor = torch.tensor([current_sequence + [vocab["<PAD>"]] * (max_length - len(current_sequence))])
            
            output, _ = orky_transformer.do_da_complete_orky_transfo_ma(input_tensor)
            last_word_logits = output[0, len(current_sequence) - 1, :]
            
            # Simple greedy selection for speed
            next_token = torch.argmax(last_word_logits).item()
            current_sequence.append(next_token)
            
            if next_token == vocab["<END>"]:
                break
        
        words = [reverse_vocab[token] for token in current_sequence if token != vocab["<PAD>"]]
        return " ".join(words)

def quick_demo():
    """Quick demo that doesn't take forever!"""
    print("WAAAGH! QUICK ORKY DEMO - NO MORE WAITIN' FOREVER!")
    print("=" * 60)
    
    # Get quick phrases
    print("LOADIN' QUICK ORKY PHRASES...")
    orky_phrases = get_quick_orky_phrases()
    print(f"Loaded {len(orky_phrases)} phrases")
    
    # Create vocabulary
    print("CREATIN' VOCABULARY...")
    vocab, reverse_vocab = create_quick_vocab(orky_phrases)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Convert to numbers
    print("TURNIN' WORDS INTO NUMBERS...")
    max_seq_len = 15  # Shorter sequences
    input_data = words_to_numbers_quick(orky_phrases, vocab, max_seq_len)
    target_data = create_target_data_quick(input_data, vocab)
    print(f"Data shape: {input_data.shape}")
    
    # Create small model
    print("CREATIN' SMALL ORKY TRANSFO'MA'...")
    d_model = 32      # Much smaller
    num_heads = 2     # Fewer heads
    num_layers = 1    # Only one layer
    d_ff = 64         # Smaller feed-forward
    
    orky_transformer = OrkyTransformer(
        da_orky_vocab_size=len(vocab),
        da_orky_model_size=d_model,
        num_orky_heads=num_heads,
        num_orky_layers=num_layers,
        da_orky_feedforward_size=d_ff,
        da_max_orky_seq_len=max_seq_len
    )
    
    print(f"Small model created: {d_model} dim, {num_heads} heads, {num_layers} layers")
    
    # Quick training
    print("STARTIN' QUICK TRAININ'...")
    trained_transformer = quick_train(
        orky_transformer, 
        input_data, 
        target_data, 
        vocab, 
        num_epochs=30  # Much fewer epochs
    )
    
    # Generate some speech
    print("\n" + "=" * 60)
    print("GENERATIN' ORKY SPEECH:")
    print("=" * 60)
    
    for i in range(15):
        orky_speech = generate_quick_orky(trained_transformer, vocab, reverse_vocab)
        print(f"{i+1:2d}. {orky_speech}")
    
    print("\nWAAAGH! QUICK DEMO DONE!")
    print("Dat was much faster, right?")
    print("=" * 60)

if __name__ == "__main__":
    quick_demo()

