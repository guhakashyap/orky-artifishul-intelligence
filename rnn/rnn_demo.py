#!/usr/bin/env python3
"""
RNN DEMO - DA MEMORY BOYZ IN ACTION! 🧠⚡

Dis is where we see da RNN Waaagh in action! We'll show how da different
types of memory boyz work and compare their performance on sequential tasks.

WHAT WE'LL DO:
- Test basic RNN, LSTM, and GRU memory boyz
- Show how they handle sequential data
- Compare their memory capabilities
- Demonstrate text generation with memory

FOR DA BOYZ: Dis is like watchin' different types of Ork boyz remember
things and pass information down da line! Each type has different memory abilities!

FOR HUMANS: This demonstrates RNN variants (RNN, LSTM, GRU) with practical
examples showing their sequential processing capabilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict, Any
import random

# Import our Orky RNN implementation
from rnn_waaagh import OrkyRNNWaaagh, create_orky_rnn_waaagh

class OrkyRNNTrainer:
    """
    DA RNN TRAINER - DA WARBOSS WHO TEACHES DA MEMORY BOYZ!
    
    Dis is like a Warboss who teaches da memory boyz how to remember
    things better by showing 'em examples and correcting their mistakes!
    
    WHY WE NEED DIS (ORKY):
    Every WAAAGH needs a Warboss to teach da boyz how to fight better!
    Dis is like havin' a Warboss who teaches da memory boyz!
    
    WHY WE NEED DIS (HUMIE):
    Trainer class handles model training with loss computation, optimization,
    and performance monitoring for RNN models.
    """
    
    def __init__(self, da_model: OrkyRNNWaaagh, da_learning_rate: float = 0.001):
        self.da_model = da_model
        self.da_optimizer = optim.Adam(da_model.parameters(), lr=da_learning_rate)
        self.da_criterion = nn.CrossEntropyLoss()
        self.da_training_losses = []
        self.da_validation_losses = []
    
    def train_da_memory_boyz(self, da_input_sequences: torch.Tensor, da_target_sequences: torch.Tensor, 
                           da_epochs: int = 10, da_batch_size: int = 32) -> Dict[str, List[float]]:
        """
        TRAIN DA MEMORY BOYZ - TEACH 'EM HOW TO REMEMBER BETTER!
        
        DIS IS WHERE DA WARBOSS TEACHES DA MEMORY BOYZ:
        1. Show 'em examples of sequences
        2. Let 'em make predictions
        3. Correct their mistakes
        4. Repeat until they get good!
        
        da_input_sequences: (batch_size, seq_len) - input sequences
        da_target_sequences: (batch_size, seq_len) - target sequences
        da_epochs: number of training rounds
        da_batch_size: how many sequences to process at once
        """
        print(f"🧠⚡ Training da {self.da_model.da_cell_type} memory boyz for {da_epochs} epochs!")
        
        da_model = self.da_model
        da_optimizer = self.da_optimizer
        da_criterion = self.da_criterion
        
        # Convert to batches
        num_batches = len(da_input_sequences) // da_batch_size
        
        for epoch in range(da_epochs):
            da_model.train()
            epoch_loss = 0.0
            
            # Shuffle data for better training
            indices = torch.randperm(len(da_input_sequences))
            da_input_sequences = da_input_sequences[indices]
            da_target_sequences = da_target_sequences[indices]
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * da_batch_size
                end_idx = start_idx + da_batch_size
                
                batch_input = da_input_sequences[start_idx:end_idx]
                batch_target = da_target_sequences[start_idx:end_idx]
                
                # Forward pass - let da memory boyz work!
                da_optimizer.zero_grad()
                da_logits = da_model.unleash_da_rnn_waaagh(batch_input)
                
                # Compute loss - how wrong were da boyz?
                da_loss = da_criterion(da_logits.view(-1, da_logits.size(-1)), batch_target.view(-1))
                
                # Backward pass - teach 'em their mistakes!
                da_loss.backward()
                
                # Gradient clipping - don't let da boyz get too excited!
                torch.nn.utils.clip_grad_norm_(da_model.parameters(), max_norm=1.0)
                
                # Update weights - make 'em better!
                da_optimizer.step()
                
                epoch_loss += da_loss.item()
            
            avg_loss = epoch_loss / num_batches
            self.da_training_losses.append(avg_loss)
            
            if epoch % 2 == 0:
                print(f"   Epoch {epoch+1}/{da_epochs}: Loss = {avg_loss:.4f}")
        
        print(f"✅ Training complete! Final loss: {avg_loss:.4f}")
        return {
            'training_losses': self.da_training_losses,
            'final_loss': avg_loss
        }
    
    def test_da_memory_boyz(self, da_test_sequences: torch.Tensor, da_test_targets: torch.Tensor) -> Dict[str, float]:
        """
        TEST DA MEMORY BOYZ - SEE HOW GOOD DEY ARE!
        
        DIS IS WHERE WE TEST DA MEMORY BOYZ ON NEW SEQUENCES:
        1. Give 'em new sequences they haven't seen
        2. Let 'em make predictions
        3. See how accurate they are!
        """
        print(f"🧠⚡ Testing da {self.da_model.da_cell_type} memory boyz!")
        
        self.da_model.eval()
        with torch.no_grad():
            da_logits = self.da_model.unleash_da_rnn_waaagh(da_test_sequences)
            da_predictions = torch.argmax(da_logits, dim=-1)
            
            # Calculate accuracy
            da_correct = (da_predictions == da_test_targets).float()
            da_accuracy = da_correct.mean().item()
            
            # Calculate loss
            da_loss = self.da_criterion(da_logits.view(-1, da_logits.size(-1)), da_test_targets.view(-1)).item()
            
            print(f"   Test Accuracy: {da_accuracy:.4f}")
            print(f"   Test Loss: {da_loss:.4f}")
            
            return {
                'accuracy': da_accuracy,
                'loss': da_loss,
                'predictions': da_predictions
            }

def create_sequential_data(da_vocab_size: int = 50, da_seq_len: int = 10, da_num_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CREATE SEQUENTIAL DATA - DA TRAINING EXAMPLES FOR DA MEMORY BOYZ!
    
    Dis creates simple sequential patterns for da memory boyz to learn.
    Like teaching Orks to count or follow simple patterns!
    
    WHY WE NEED DIS (ORKY):
    Da memory boyz need examples to learn from! Dis is like showing
    'em how to count or follow patterns!
    
    WHY WE NEED DIS (HUMIE):
    Creates synthetic sequential data for training and testing RNN models.
    """
    print(f"🧠⚡ Creating sequential data: {da_num_samples} samples of length {da_seq_len}")
    
    # Create simple patterns
    da_input_sequences = []
    da_target_sequences = []
    
    for _ in range(da_num_samples):
        # Create a simple pattern: input[i] -> target[i] = input[i] + 1 (mod vocab_size)
        da_input = torch.randint(0, da_vocab_size, (da_seq_len,))
        da_target = (da_input + 1) % da_vocab_size
        
        da_input_sequences.append(da_input)
        da_target_sequences.append(da_target)
    
    return torch.stack(da_input_sequences), torch.stack(da_target_sequences)

def compare_rnn_variants():
    """
    COMPARE RNN VARIANTS - SEE WHICH MEMORY BOYZ ARE BEST!
    
    DIS IS WHERE WE TEST ALL DA DIFFERENT TYPES OF MEMORY BOYZ:
    1. Basic RNN boyz
    2. LSTM boyz (smart memory)
    3. GRU boyz (efficient memory)
    
    We'll see which ones learn fastest and remember best!
    """
    print("🧠⚡ COMPARING RNN VARIANTS - WHICH MEMORY BOYZ ARE BEST? ⚡🧠")
    print("=" * 70)
    
    # Create training data
    da_vocab_size = 20
    da_seq_len = 8
    da_num_samples = 500
    
    print(f"Creating training data: {da_num_samples} samples of length {da_seq_len}")
    da_input_data, da_target_data = create_sequential_data(da_vocab_size, da_seq_len, da_num_samples)
    
    # Split data
    da_train_size = int(0.8 * len(da_input_data))
    da_train_input = da_input_data[:da_train_size]
    da_train_target = da_target_data[:da_train_size]
    da_test_input = da_input_data[da_train_size:]
    da_test_target = da_target_data[da_train_size:]
    
    print(f"Training samples: {len(da_train_input)}")
    print(f"Test samples: {len(da_test_input)}")
    
    # Test different RNN variants
    da_variants = ["RNN", "LSTM", "GRU"]
    da_results = {}
    
    for variant in da_variants:
        print(f"\n{'='*20} TESTING {variant} MEMORY BOYZ {'='*20}")
        
        # Create model
        da_model = create_orky_rnn_waaagh(
            da_vocab_size=da_vocab_size,
            da_embedding_size=32,
            da_hidden_size=64,
            da_num_layers=1,
            da_cell_type=variant
        )
        
        # Create trainer
        da_trainer = OrkyRNNTrainer(da_model, da_learning_rate=0.01)
        
        # Train the model
        start_time = time.time()
        da_training_results = da_trainer.train_da_memory_boyz(
            da_train_input, da_train_target, da_epochs=20, da_batch_size=32
        )
        training_time = time.time() - start_time
        
        # Test the model
        da_test_results = da_trainer.test_da_memory_boyz(da_test_input, da_test_target)
        
        # Store results
        da_results[variant] = {
            'training_time': training_time,
            'final_training_loss': da_training_results['final_loss'],
            'test_accuracy': da_test_results['accuracy'],
            'test_loss': da_test_results['loss'],
            'training_losses': da_training_results['training_losses']
        }
        
        print(f"✅ {variant} Results:")
        print(f"   Training Time: {training_time:.2f}s")
        print(f"   Final Training Loss: {da_training_results['final_loss']:.4f}")
        print(f"   Test Accuracy: {da_test_results['accuracy']:.4f}")
        print(f"   Test Loss: {da_test_results['loss']:.4f}")
    
    # Compare results
    print(f"\n{'='*20} COMPARISON RESULTS {'='*20}")
    print(f"{'Variant':<8} {'Time(s)':<8} {'Train Loss':<12} {'Test Acc':<10} {'Test Loss':<10}")
    print("-" * 60)
    
    for variant in da_variants:
        results = da_results[variant]
        print(f"{variant:<8} {results['training_time']:<8.2f} {results['final_training_loss']:<12.4f} "
              f"{results['test_accuracy']:<10.4f} {results['test_loss']:<10.4f}")
    
    # Find best performer
    best_accuracy = max(da_results[variant]['test_accuracy'] for variant in da_variants)
    best_variant = [v for v in da_variants if da_results[v]['test_accuracy'] == best_accuracy][0]
    
    print(f"\n🏆 BEST MEMORY BOYZ: {best_variant} with {best_accuracy:.4f} accuracy!")
    
    return da_results

def demonstrate_text_generation():
    """
    DEMONSTRATE TEXT GENERATION - DA MEMORY BOYZ WRITE STORIES!
    
    DIS IS WHERE WE SHOW HOW DA MEMORY BOYZ CAN GENERATE TEXT:
    1. Train 'em on simple text patterns
    2. Let 'em generate new text
    3. See how well they remember patterns!
    """
    print("\n🧠⚡ TEXT GENERATION DEMO - DA MEMORY BOYZ WRITE STORIES! ⚡🧠")
    print("=" * 70)
    
    # Create simple text data
    da_text = "WAAAGH! ORKS FIGHT! ORKS WIN! ORKS STRONG! ORKS BRAVE! ORKS SMART! ORKS FAST! ORKS BIG! ORKS LOUD! ORKS GREEN! ORKS TOUGH!"
    da_words = da_text.split()
    da_vocab = list(set(da_words))
    da_word_to_idx = {word: idx for idx, word in enumerate(da_vocab)}
    da_idx_to_word = {idx: word for word, idx in da_word_to_idx.items()}
    da_vocab_size = len(da_vocab)
    
    print(f"Vocabulary: {da_vocab}")
    print(f"Vocabulary size: {da_vocab_size}")
    
    # Create sequences
    da_seq_len = 4
    da_sequences = []
    da_targets = []
    
    for i in range(len(da_words) - da_seq_len):
        da_sequence = [da_word_to_idx[da_words[i+j]] for j in range(da_seq_len)]
        da_target = da_word_to_idx[da_words[i + da_seq_len]]
        da_sequences.append(da_sequence)
        da_targets.append(da_target)
    
    da_input_tensor = torch.tensor(da_sequences)
    da_target_tensor = torch.tensor(da_targets)
    
    print(f"Created {len(da_sequences)} sequences of length {da_seq_len}")
    
    # Train LSTM model for text generation
    da_model = create_orky_rnn_waaagh(
        da_vocab_size=da_vocab_size,
        da_embedding_size=16,
        da_hidden_size=32,
        da_num_layers=1,
        da_cell_type="LSTM"
    )
    
    da_trainer = OrkyRNNTrainer(da_model, da_learning_rate=0.01)
    da_trainer.train_da_memory_boyz(da_input_tensor, da_target_tensor, da_epochs=50, da_batch_size=8)
    
    # Generate text
    print("\n🧠⚡ GENERATING TEXT WITH DA MEMORY BOYZ! ⚡🧠")
    
    da_model.eval()
    with torch.no_grad():
        # Start with a seed sequence
        da_seed = [da_word_to_idx["WAAAGH!"], da_word_to_idx["ORKS"], da_word_to_idx["FIGHT!"], da_word_to_idx["ORKS"]]
        da_generated = da_seed.copy()
        
        # Generate next words
        for _ in range(10):
            da_input_seq = torch.tensor([da_generated[-da_seq_len:]]).long()
            da_logits = da_model.unleash_da_rnn_waaagh(da_input_seq)
            da_next_word_logits = da_logits[0, -1, :]
            da_next_word_idx = torch.argmax(da_next_word_logits).item()
            da_generated.append(da_next_word_idx)
        
        # Convert back to words
        da_generated_words = [da_idx_to_word[idx] for idx in da_generated]
        da_generated_text = " ".join(da_generated_words)
        
        print(f"Generated text: {da_generated_text}")
    
    print("✅ Text generation demo complete!")

def plot_training_curves(da_results: Dict[str, Any]):
    """
    PLOT TRAINING CURVES - SEE HOW DA MEMORY BOYZ LEARN!
    
    DIS SHOWS HOW DA DIFFERENT MEMORY BOYZ LEARN OVER TIME:
    1. Plot training loss curves
    2. Compare learning speeds
    3. See which ones learn fastest!
    """
    print("\n🧠⚡ PLOTTING TRAINING CURVES! ⚡🧠")
    
    plt.figure(figsize=(12, 8))
    
    # Plot training losses
    plt.subplot(2, 2, 1)
    for variant in da_results:
        plt.plot(da_results[variant]['training_losses'], label=f'{variant} Memory Boyz', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curves - How Fast Da Memory Boyz Learn!')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot final accuracies
    plt.subplot(2, 2, 2)
    variants = list(da_results.keys())
    accuracies = [da_results[v]['test_accuracy'] for v in variants]
    colors = ['red', 'blue', 'green']
    bars = plt.bar(variants, accuracies, color=colors, alpha=0.7)
    plt.ylabel('Test Accuracy')
    plt.title('Final Test Accuracy - How Good Da Memory Boyz Are!')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Plot training times
    plt.subplot(2, 2, 3)
    times = [da_results[v]['training_time'] for v in variants]
    bars = plt.bar(variants, times, color=colors, alpha=0.7)
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time - How Fast Da Memory Boyz Train!')
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{time:.1f}s', ha='center', va='bottom')
    
    # Plot final losses
    plt.subplot(2, 2, 4)
    losses = [da_results[v]['test_loss'] for v in variants]
    bars = plt.bar(variants, losses, color=colors, alpha=0.7)
    plt.ylabel('Test Loss')
    plt.title('Final Test Loss - How Much Da Memory Boyz Struggle!')
    
    # Add value labels on bars
    for bar, loss in zip(bars, losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{loss:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/Users/guha.skashyap/llms/orky-artifishul-intelligence/rnn/rnn_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Training curves saved as 'rnn_comparison.png'!")

def main():
    """
    MAIN FUNCTION - DA ULTIMATE RNN DEMO!
    
    DIS IS WHERE WE RUN ALL DA RNN DEMOS:
    1. Compare different RNN variants
    2. Show text generation
    3. Plot training curves
    4. Show which memory boyz are best!
    """
    print("🧠⚡ RNN DEMO - DA MEMORY BOYZ IN ACTION! ⚡🧠")
    print("=" * 70)
    print("Dis is where we see da RNN Waaagh in action!")
    print("We'll test different types of memory boyz and see which ones are best!")
    print("=" * 70)
    
    # Compare RNN variants
    da_results = compare_rnn_variants()
    
    # Demonstrate text generation
    demonstrate_text_generation()
    
    # Plot training curves
    plot_training_curves(da_results)
    
    print("\n" + "=" * 70)
    print("✅ RNN DEMO COMPLETE! WAAAGH!")
    print("Da memory boyz have shown their skills!")
    print("=" * 70)

if __name__ == "__main__":
    main()
