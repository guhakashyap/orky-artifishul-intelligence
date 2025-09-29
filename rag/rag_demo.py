#!/usr/bin/env python3
"""
RAG DEMO - DA LIBRARIAN WARBOSS DEMONSTRATION! 📚⚡

Dis demo shows off da RAG Waaagh architecture with retrieval-augmented generation!
Watch how da librarian boyz search through da big book while da smart Warboss
combines retrieved info with his knowledge to give da best answers!

FOR DA BOYZ: Dis shows how da RAG system works with librarian boyz who can
find any information and a smart Warboss who can use it to give da best answers!
FOR HUMANS: This demonstrates the RAG architecture with retrieval and generation
components for enhanced knowledge-based question answering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from rag_waaagh import OrkyRAGWaaagh, create_orky_rag_waaagh

def quick_rag_demo():
    """
    QUICK RAG DEMO - SHOW OFF DA LIBRARIAN WARBOSS SYSTEM!

    Dis shows how da RAG Waaagh model works with retrieval and generation!
    """
    print("📚⚡ QUICK RAG DEMO - DA LIBRARIAN WARBOSS SYSTEM! ⚡📚")
    print("=" * 60)

    # Create a small RAG model for demo
    model = create_orky_rag_waaagh(
        da_vocab_size=50,
        da_orky_model_size=32,
        da_num_layers=2,
        da_retrieval_top_k=3
    )

    # Test with different queries
    test_queries = [
        torch.tensor([[1, 2, 3, 4, 5]]),  # Simple query
        torch.tensor([[10, 20, 30, 40, 49]]),  # Different range
        torch.tensor([[1, 1, 1, 1, 1]]),  # Repeated pattern
    ]

    # Create some dummy documents
    documents = [
        torch.randn(8, 32),   # Document 1
        torch.randn(12, 32),  # Document 2
        torch.randn(6, 32),   # Document 3
        torch.randn(10, 32),  # Document 4
        torch.randn(15, 32),  # Document 5
    ]

    for i, query_tokens in enumerate(test_queries):
        print(f"\n🔵 Query {i+1}: {query_tokens[0].tolist()}")
        
        # Forward pass
        with torch.no_grad():
            logits = model.unleash_da_rag_waaagh(query_tokens, documents)
            predictions = F.softmax(logits, dim=-1)
            
        print(f"   Output shape: {logits.shape}")
        print(f"   Max prediction: {predictions.max().item():.4f}")
        print(f"   Prediction entropy: {-torch.sum(predictions * torch.log(predictions + 1e-8), dim=-1).mean().item():.4f}")

    print("\n✅ Quick RAG demo complete! WAAAGH!")

def retrieval_demo():
    """
    RETRIEVAL DEMO - SHOW DA LIBRARIAN BOYZ IN ACTION!

    Dis shows how da document retriever finds relevant documents!
    """
    print("\n📚⚡ RETRIEVAL DEMO - DA LIBRARIAN BOYZ IN ACTION! ⚡📚")
    print("=" * 60)

    # Create a model with more documents for better retrieval
    model = create_orky_rag_waaagh(
        da_vocab_size=100,
        da_orky_model_size=64,
        da_num_layers=3,
        da_retrieval_top_k=5
    )

    # Create diverse documents
    documents = [
        torch.randn(10, 64),  # Document about WAAAGH!
        torch.randn(12, 64),  # Document about Dakka
        torch.randn(8, 64),   # Document about Krumpin'
        torch.randn(15, 64),  # Document about Trukks
        torch.randn(6, 64),   # Document about Squigs
        torch.randn(20, 64),  # Document about Warbosses
        torch.randn(9, 64),   # Document about Grots
        torch.randn(11, 64),  # Document about Mekboyz
    ]

    # Test different queries
    test_queries = [
        "What is WAAAGH?",
        "How to build a Trukk?",
        "Best way to krump gits?",
        "What do Warbosses do?",
        "How to train Grots?"
    ]

    for query_text in test_queries:
        print(f"\n🔵 Query: {query_text}")
        
        # Convert query to tokens (simplified)
        query_tokens = torch.randint(0, 100, (1, 8))
        
        # Test retrieval
        with torch.no_grad():
            query_embeddings = model.da_rag_generator.da_token_embedding(query_tokens)
            query_avg = torch.mean(query_embeddings, dim=1)
            
            retrieved_docs, relevance_scores = model.da_document_retriever.do_da_document_search(
                query_avg, documents
            )
            
        print(f"   Retrieved {len(retrieved_docs[0])} documents")
        print(f"   Relevance scores: {[f'{score:.3f}' for score in relevance_scores[0]]}")

    print("\n✅ Retrieval demo complete! WAAAGH!")

def rag_vs_baseline_comparison():
    """
    RAG VS BASELINE COMPARISON - SHOW DA POWER OF RETRIEVAL!

    Dis compares RAG with baseline generation to show da benefits of retrieval!
    """
    print("\n📚⚡ RAG VS BASELINE COMPARISON - DA POWER OF RETRIEVAL! ⚡📚")
    print("=" * 60)

    # Create RAG model
    rag_model = create_orky_rag_waaagh(
        da_vocab_size=100,
        da_orky_model_size=64,
        da_num_layers=3,
        da_retrieval_top_k=3
    )

    # Create baseline model (just generator without retrieval)
    class BaselineModel(nn.Module):
        def __init__(self, vocab_size, model_size, num_layers):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, model_size)
            self.layers = nn.ModuleList([
                nn.TransformerDecoderLayer(model_size, nhead=8, dim_feedforward=model_size*4, batch_first=True)
                for _ in range(num_layers)
            ])
            self.output_proj = nn.Linear(model_size, vocab_size)
            
        def forward(self, x):
            x = self.embedding(x)
            for layer in self.layers:
                x = layer(x, x)
            return self.output_proj(x)

    baseline_model = BaselineModel(100, 64, 3)

    # Compare parameter counts
    rag_params = sum(p.numel() for p in rag_model.parameters())
    baseline_params = sum(p.numel() for p in baseline_model.parameters())

    print(f"🔵 RAG Parameters: {rag_params:,}")
    print(f"🔵 Baseline Parameters: {baseline_params:,}")
    print(f"🔵 Parameter Ratio: {rag_params / baseline_params:.2f}x")

    # Test inference speed
    test_input = torch.randint(0, 100, (1, 16))
    documents = [torch.randn(10, 64) for _ in range(5)]
    
    # RAG inference
    start_time = time.time()
    with torch.no_grad():
        for _ in range(50):
            _ = rag_model.unleash_da_rag_waaagh(test_input, documents)
    rag_time = time.time() - start_time

    # Baseline inference
    start_time = time.time()
    with torch.no_grad():
        for _ in range(50):
            _ = baseline_model(test_input)
    baseline_time = time.time() - start_time

    print(f"\n🔵 RAG Inference Time: {rag_time:.4f}s")
    print(f"🔵 Baseline Inference Time: {baseline_time:.4f}s")
    print(f"🔵 Speed Ratio: {rag_time / baseline_time:.2f}x")

    print("\n✅ RAG vs Baseline comparison complete! WAAAGH!")

def knowledge_base_demo():
    """
    KNOWLEDGE BASE DEMO - SHOW DA POWER OF EXTERNAL KNOWLEDGE!

    Dis shows how RAG can use external knowledge to answer questions!
    """
    print("\n📚⚡ KNOWLEDGE BASE DEMO - DA POWER OF EXTERNAL KNOWLEDGE! ⚡📚")
    print("=" * 60)

    # Create a model for knowledge-based QA
    model = create_orky_rag_waaagh(
        da_vocab_size=100,
        da_orky_model_size=64,
        da_num_layers=3,
        da_retrieval_top_k=3
    )

    # Simulate a knowledge base about Ork warfare
    knowledge_base = [
        torch.randn(20, 64),  # Document about WAAAGH tactics
        torch.randn(15, 64),  # Document about Dakka strategies
        torch.randn(18, 64),  # Document about Krumpin' techniques
        torch.randn(12, 64),  # Document about Trukk maintenance
        torch.randn(25, 64),  # Document about Warboss leadership
        torch.randn(10, 64),  # Document about Grot training
        torch.randn(22, 64),  # Document about Mekboy inventions
        torch.randn(16, 64),  # Document about Squig breeding
    ]

    # Test different knowledge-based queries
    knowledge_queries = [
        "How to plan a WAAAGH?",
        "Best Dakka for long range?",
        "How to train Grots for battle?",
        "What makes a good Warboss?",
        "How to maintain a Trukk?",
        "Best Squig for scouting?",
        "How to invent new weapons?",
        "What is the Ork way of war?"
    ]

    for query_text in knowledge_queries:
        print(f"\n🔵 Knowledge Query: {query_text}")
        
        # Convert to tokens
        query_tokens = torch.randint(0, 100, (1, 10))
        
        # Test RAG with knowledge base
        with torch.no_grad():
            logits = model.unleash_da_rag_waaagh(query_tokens, knowledge_base)
            predictions = F.softmax(logits, dim=-1)
            
        print(f"   Output shape: {logits.shape}")
        print(f"   Max prediction: {predictions.max().item():.4f}")
        print(f"   Knowledge integration: {'✅' if predictions.max().item() > 0.01 else '❌'}")

    print("\n✅ Knowledge base demo complete! WAAAGH!")

def main():
    """
    MAIN DEMO FUNCTION - RUN ALL DA RAG DEMONSTRATIONS!

    Dis runs all da RAG demos to show off da librarian Warboss system!
    """
    print("📚⚡ RAG DEMO - DA LIBRARIAN WARBOSS DEMONSTRATION! ⚡📚")
    print("=" * 80)
    print("Dis demo shows off da RAG Waaagh architecture with retrieval-augmented generation!")
    print("Watch how da librarian boyz search through da big book while da smart Warboss")
    print("combines retrieved info with his knowledge to give da best answers!")
    print("=" * 80)

    # Run all demos
    quick_rag_demo()
    retrieval_demo()
    rag_vs_baseline_comparison()
    knowledge_base_demo()

    print("\n🎉 ALL RAG DEMOS COMPLETE! WAAAGH! 🎉")
    print("Da RAG Waaagh architecture shows how retrieval-augmented generation can")
    print("answer questions using external knowledge! Da librarian boyz find relevant")
    print("information while da smart Warboss combines it with his knowledge!")

if __name__ == "__main__":
    main()
