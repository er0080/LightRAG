#!/usr/bin/env python3
"""
Test script for Qwen3 reranker integration in LightRAG
"""

import asyncio
import os
from lightrag.rerank import qwen3_rerank, create_qwen3_rerank_model

async def test_qwen3_reranker():
    """Test the Qwen3 reranker functionality"""
    
    # Test documents
    docs = [
        {"content": "The capital of France is Paris, located in the north-central part of the country."},
        {"content": "Tokyo is the capital city of Japan and one of the most populous metropolitan areas in the world."},
        {"content": "London is the capital of England and the United Kingdom, situated on the River Thames."},
        {"content": "Berlin is the capital and largest city of Germany."},
        {"content": "Paris is famous for the Eiffel Tower, Louvre Museum, and its romantic atmosphere."},
    ]
    
    query = "What is the capital of France?"
    
    print("Testing Qwen3 Reranker")
    print("=" * 50)
    print(f"Query: {query}")
    print(f"Number of documents: {len(docs)}")
    print()
    
    # Test configuration (modify these for your setup)
    base_url = os.getenv("QWEN3_BASE_URL", "http://localhost:8000/v1/score")
    api_key = os.getenv("QWEN3_API_KEY", None)  # Optional
    model = os.getenv("QWEN3_MODEL", "Qwen/Qwen3-Reranker-0.6B")
    
    print(f"Configuration:")
    print(f"  Base URL: {base_url}")
    print(f"  API Key: {'Set' if api_key else 'Not set'}")
    print(f"  Model: {model}")
    print()
    
    try:
        # Test direct function call
        print("Testing direct qwen3_rerank function...")
        result = await qwen3_rerank(
            query=query,
            documents=docs,
            model=model,
            base_url=base_url,
            api_key=api_key,
            top_n=3
        )
        
        print("✅ Direct function call successful!")
        print("Top 3 reranked results:")
        for i, doc in enumerate(result):
            score = doc.get("rerank_score", "N/A")
            content = doc.get("content", "")[:100] + "..." if len(doc.get("content", "")) > 100 else doc.get("content", "")
            print(f"  {i+1}. Score: {score:.4f} - {content}")
        print()
        
        # Test RerankModel wrapper
        print("Testing RerankModel wrapper...")
        rerank_model = create_qwen3_rerank_model(
            model=model,
            base_url=base_url,
            api_key=api_key
        )
        
        wrapper_result = await rerank_model.rerank(
            query=query,
            documents=docs,
            top_n=3
        )
        
        print("✅ RerankModel wrapper successful!")
        print("Top 3 reranked results (wrapper):")
        for i, doc in enumerate(wrapper_result):
            score = doc.get("rerank_score", "N/A")
            content = doc.get("content", "")[:100] + "..." if len(doc.get("content", "")) > 100 else doc.get("content", "")
            print(f"  {i+1}. Score: {score:.4f} - {content}")
        print()
        
        print("🎉 All tests passed! Qwen3 reranker is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Qwen3 reranker: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your vLLM server is running with Qwen3-Reranker model")
        print("2. Check that the base URL is correct (should end with /v1/score)")
        print("3. Verify your API key if authentication is required")
        print("4. Ensure the model name matches what's loaded in vLLM")
        return False

def test_environment_config():
    """Test environment configuration for LightRAG server"""
    
    print("\nTesting Environment Configuration")
    print("=" * 50)
    
    # Simulate environment variables that would be used by LightRAG server
    env_vars = {
        "RERANK_PROVIDER": "qwen3",
        "RERANK_BINDING_HOST": "http://localhost:8000/v1/score",
        "RERANK_BINDING_API_KEY": "optional-api-key",
        "RERANK_MODEL": "Qwen/Qwen3-Reranker-0.6B",
        "QWEN3_RERANK_INSTRUCTION": "Given a web search query, retrieve relevant passages that answer the query"
    }
    
    print("Example environment variables for LightRAG server:")
    for key, value in env_vars.items():
        print(f"  {key}={value}")
    
    print("\nTo use with LightRAG server, add these to your .env file:")
    print("  export RERANK_PROVIDER=qwen3")
    print("  export RERANK_BINDING_HOST=http://your-vllm-server:8000/v1/score")
    print("  export RERANK_MODEL=Qwen/Qwen3-Reranker-0.6B")
    print("  # Optional: export RERANK_BINDING_API_KEY=your-api-key")
    print("  # Optional: export QWEN3_RERANK_INSTRUCTION='custom instruction'")

async def main():
    """Main test function"""
    
    print("LightRAG Qwen3 Reranker Integration Test")
    print("=" * 60)
    print()
    
    # Test environment configuration
    test_environment_config()
    
    # Test actual reranker functionality
    success = await test_qwen3_reranker()
    
    if success:
        print("\n✅ Integration test completed successfully!")
        print("The Qwen3 reranker is ready to use with LightRAG server.")
    else:
        print("\n❌ Integration test failed.")
        print("Please check your vLLM server setup and configuration.")

if __name__ == "__main__":
    asyncio.run(main())
