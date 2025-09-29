#!/usr/bin/env python3
"""
QUICK RNN DEMO - DA MEMORY BOYZ IN ACTION! 🧠⚡

Dis is a quick demo of da RNN Waaagh! We'll show how da different
types of memory boyz work and compare their performance.

WHAT WE'LL DO:
- Test basic RNN, LSTM, and GRU memory boyz
- Show how they handle sequential data
- Compare their memory capabilities

FOR DA BOYZ: Dis is like watchin' different types of Ork boyz remember
things and pass information down da line! Each type has different memory abilities!

FOR HUMANS: This demonstrates RNN variants (RNN, LSTM, GRU) with practical
examples showing their sequential processing capabilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from typing import List, Tuple, Dict, Any

# Import our Orky RNN implementation
from rnn_waaagh import OrkyRNNWaaagh, create_orky_rnn_waaagh

class OrkyRNNTrainer:
    """
    DA RNN TRAINER - DA WARBOSS WHO TEACHES DA MEMORY BOYZ!
    """
    
    def __init__(self, da_model: OrkyRNNWaaagh, da_learning_rate: float = 0.001):
        self.da_model = da_model
        self.da_optimizer = optim.Adam(da_model.parameters(), lr=da_learning_rate)
        self.da_criterion = nn.CrossEntropyLoss()
        self.da_training_losses = []
    
    def train_da_memory_boyz(self, da_input_sequences: torch.Tensor, da_target_sequences: torch.Tensor, 
                           da_epochs: int = 10, da_batch_size: int = 32) -> Dict[str, List[float]]:
        """
        TRAIN DA MEMORY BOYZ - TEACH 'EM HOW TO REMEMBER BETTER!
        """
        print(f"🧠⚡ Training da {self.da_model.da_cell_type} memory boyz for {da_epochs} epochs!")
        
        da_model = self.da_model
        da_optimizer = self.da_optimizer
        da_criterion = self.da_criterion
        
        # Convert to batches
        num_batches = max(1, len(da_input_sequences) // da_batch_size)
        
        for epoch in range(da_epochs):
            da_model.train()
            epoch_loss = 0.0
            
            # Shuffle data for better training
            indices = torch.randperm(len(da_input_sequences))
            da_input_shuffled = da_input_sequences[indices]
            da_target_shuffled = da_target_sequences[indices]
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * da_batch_size
                end_idx = min(start_idx + da_batch_size, len(da_input_sequences))
                
                batch_input = da_input_shuffled[start_idx:end_idx]
                batch_target = da_target_shuffled[start_idx:end_idx]
                
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

def demonstrate_simple_generation():
    """
    DEMONSTRATE SIMPLE GENERATION - DA MEMORY BOYZ MAKE PREDICTIONS!
    """
    print("\n🧠⚡ SIMPLE GENERATION DEMO - DA MEMORY BOYZ MAKE PREDICTIONS! ⚡🧠")
    print("=" * 70)
    
    # Create a simple model
    da_model = create_orky_rnn_waaagh(
        da_vocab_size=10,
        da_embedding_size=16,
        da_hidden_size=32,
        da_num_layers=1,
        da_cell_type="LSTM"
    )
    
    # Create simple training data
    da_input = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    da_target = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]])
    
    print("Training on simple sequence: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]")
    print("Target sequence: [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]")
    
    # Train the model
    da_trainer = OrkyRNNTrainer(da_model, da_learning_rate=0.01)
    da_trainer.train_da_memory_boyz(da_input, da_target, da_epochs=50, da_batch_size=1)
    
    # Test generation
    print("\n🧠⚡ TESTING GENERATION! ⚡🧠")
    
    da_model.eval()
    with torch.no_grad():
        # Test with the same input
        da_logits = da_model.unleash_da_rnn_waaagh(da_input)
        da_predictions = torch.argmax(da_logits, dim=-1)
        
        print(f"Input:  {da_input[0].tolist()}")
        print(f"Target: {da_target[0].tolist()}")
        print(f"Pred:   {da_predictions[0].tolist()}")
        
        # Calculate accuracy
        da_correct = (da_predictions == da_target).float()
        da_accuracy = da_correct.mean().item()
        print(f"Accuracy: {da_accuracy:.4f}")
    
    print("✅ Simple generation demo complete!")

def main():
    """
    MAIN FUNCTION - DA ULTIMATE RNN DEMO!
    """
    print("🧠⚡ QUICK RNN DEMO - DA MEMORY BOYZ IN ACTION! ⚡🧠")
    print("=" * 70)
    print("Dis is where we see da RNN Waaagh in action!")
    print("We'll test different types of memory boyz and see which ones are best!")
    print("=" * 70)
    
    # Compare RNN variants
    da_results = compare_rnn_variants()
    
    # Demonstrate simple generation
    demonstrate_simple_generation()
    
    print("\n" + "=" * 70)
    print("✅ QUICK RNN DEMO COMPLETE! WAAAGH!")
    print("Da memory boyz have shown their skills!")
    print("=" * 70)

if __name__ == "__main__":
    main()
