from __future__ import annotations

import os
import aiohttp
from typing import Callable, Any, List, Dict, Optional
from pydantic import BaseModel, Field

from .utils import logger


class RerankModel(BaseModel):
    """
    Wrapper for rerank functions that can be used with LightRAG.

    Example usage:
    ```python
    from lightrag.rerank import RerankModel, jina_rerank

    # Create rerank model
    rerank_model = RerankModel(
        rerank_func=jina_rerank,
        kwargs={
            "model": "BAAI/bge-reranker-v2-m3",
            "api_key": "your_api_key_here",
            "base_url": "https://api.jina.ai/v1/rerank"
        }
    )

    # Use in LightRAG
    rag = LightRAG(
        rerank_model_func=rerank_model.rerank,
        # ... other configurations
    )

    # Query with rerank enabled (default)
    result = await rag.aquery(
        "your query",
        param=QueryParam(enable_rerank=True)
    )
    ```

    Or define a custom function directly:
    ```python
    async def my_rerank_func(query: str, documents: list, top_n: int = None, **kwargs):
        return await jina_rerank(
            query=query,
            documents=documents,
            model="BAAI/bge-reranker-v2-m3",
            api_key="your_api_key_here",
            top_n=top_n or 10,
            **kwargs
        )

    rag = LightRAG(
        rerank_model_func=my_rerank_func,
        # ... other configurations
    )

    # Control rerank per query
    result = await rag.aquery(
        "your query",
        param=QueryParam(enable_rerank=True)  # Enable rerank for this query
    )
    ```
    """

    rerank_func: Callable[[Any], List[Dict]]
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: Optional[int] = None,
        **extra_kwargs,
    ) -> List[Dict[str, Any]]:
        """Rerank documents using the configured model function."""
        # Merge extra kwargs with model kwargs
        kwargs = {**self.kwargs, **extra_kwargs}
        return await self.rerank_func(
            query=query, documents=documents, top_n=top_n, **kwargs
        )


class MultiRerankModel(BaseModel):
    """Multiple rerank models for different modes/scenarios."""

    # Primary rerank model (used if mode-specific models are not defined)
    rerank_model: Optional[RerankModel] = None

    # Mode-specific rerank models
    entity_rerank_model: Optional[RerankModel] = None
    relation_rerank_model: Optional[RerankModel] = None
    chunk_rerank_model: Optional[RerankModel] = None

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        mode: str = "default",
        top_n: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Rerank using the appropriate model based on mode."""

        # Select model based on mode
        if mode == "entity" and self.entity_rerank_model:
            model = self.entity_rerank_model
        elif mode == "relation" and self.relation_rerank_model:
            model = self.relation_rerank_model
        elif mode == "chunk" and self.chunk_rerank_model:
            model = self.chunk_rerank_model
        elif self.rerank_model:
            model = self.rerank_model
        else:
            logger.warning(f"No rerank model available for mode: {mode}")
            return documents

        return await model.rerank(query, documents, top_n, **kwargs)


async def generic_rerank_api(
    query: str,
    documents: List[Dict[str, Any]],
    model: str,
    base_url: str,
    api_key: str,
    top_n: Optional[int] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Generic rerank function that works with Jina/Cohere compatible APIs.

    Args:
        query: The search query
        documents: List of documents to rerank
        model: Model identifier
        base_url: API endpoint URL
        api_key: API authentication key
        top_n: Number of top results to return
        **kwargs: Additional API-specific parameters

    Returns:
        List of reranked documents with relevance scores
    """
    if not api_key:
        logger.warning("No API key provided for rerank service")
        return documents

    if not documents:
        return documents

    # Prepare documents for reranking - handle both text and dict formats
    prepared_docs = []
    for doc in documents:
        if isinstance(doc, dict):
            # Use 'content' field if available, otherwise use 'text' or convert to string
            text = doc.get("content") or doc.get("text") or str(doc)
        else:
            text = str(doc)
        prepared_docs.append(text)

    # Prepare request
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    data = {"model": model, "query": query, "documents": prepared_docs, **kwargs}

    if top_n is not None:
        data["top_n"] = min(top_n, len(prepared_docs))

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(base_url, headers=headers, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Rerank API error {response.status}: {error_text}")
                    return documents

                result = await response.json()

                # Extract reranked results
                if "results" in result:
                    # Standard format: results contain index and relevance_score
                    reranked_docs = []
                    for item in result["results"]:
                        if "index" in item:
                            doc_idx = item["index"]
                            if 0 <= doc_idx < len(documents):
                                reranked_doc = documents[doc_idx].copy()
                                if "relevance_score" in item:
                                    reranked_doc["rerank_score"] = item[
                                        "relevance_score"
                                    ]
                                reranked_docs.append(reranked_doc)
                    return reranked_docs
                else:
                    logger.warning("Unexpected rerank API response format")
                    return documents

    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        return documents


async def jina_rerank(
    query: str,
    documents: List[Dict[str, Any]],
    model: str = "BAAI/bge-reranker-v2-m3",
    top_n: Optional[int] = None,
    base_url: str = "https://api.jina.ai/v1/rerank",
    api_key: Optional[str] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Rerank documents using Jina AI API.

    Args:
        query: The search query
        documents: List of documents to rerank
        model: Jina rerank model name
        top_n: Number of top results to return
        base_url: Jina API endpoint
        api_key: Jina API key
        **kwargs: Additional parameters

    Returns:
        List of reranked documents with relevance scores
    """
    if api_key is None:
        api_key = os.getenv("JINA_API_KEY") or os.getenv("RERANK_API_KEY")

    return await generic_rerank_api(
        query=query,
        documents=documents,
        model=model,
        base_url=base_url,
        api_key=api_key,
        top_n=top_n,
        **kwargs,
    )


async def cohere_rerank(
    query: str,
    documents: List[Dict[str, Any]],
    model: str = "rerank-english-v2.0",
    top_n: Optional[int] = None,
    base_url: str = "https://api.cohere.ai/v1/rerank",
    api_key: Optional[str] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Rerank documents using Cohere API.

    Args:
        query: The search query
        documents: List of documents to rerank
        model: Cohere rerank model name
        top_n: Number of top results to return
        base_url: Cohere API endpoint
        api_key: Cohere API key
        **kwargs: Additional parameters

    Returns:
        List of reranked documents with relevance scores
    """
    if api_key is None:
        api_key = os.getenv("COHERE_API_KEY") or os.getenv("RERANK_API_KEY")

    return await generic_rerank_api(
        query=query,
        documents=documents,
        model=model,
        base_url=base_url,
        api_key=api_key,
        top_n=top_n,
        **kwargs,
    )


async def qwen3_rerank(
    query: str,
    documents: List[Dict[str, Any]],
    model: str = "Qwen/Qwen3-Reranker-0.6B",
    top_n: Optional[int] = None,
    base_url: str = "http://localhost:8000/v1/score",
    api_key: Optional[str] = None,
    instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Rerank documents using Qwen3 Reranker via vLLM's OpenAI-compatible API.

    Args:
        query: The search query
        documents: List of documents to rerank
        model: Qwen3 reranker model name
        top_n: Number of top results to return
        base_url: vLLM API endpoint (should end with /v1/score)
        api_key: API key if required
        instruction: The instruction for the reranking task
        **kwargs: Additional parameters

    Returns:
        List of reranked documents with relevance scores
    """
    if not documents:
        return documents

    # Qwen3 Reranker prompt templates from vLLM documentation
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    
    query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
    document_template = "<Document>: {doc}{suffix}"

    # Prepare documents for reranking
    prepared_docs = []
    for doc in documents:
        if isinstance(doc, dict):
            text = doc.get("content") or doc.get("text") or str(doc)
        else:
            text = str(doc)
        prepared_docs.append(text)

    # Format query according to Qwen3 template
    formatted_query = query_template.format(
        prefix=prefix, 
        instruction=instruction, 
        query=query
    )
    
    # Format documents according to Qwen3 template
    formatted_documents = [
        document_template.format(doc=doc, suffix=suffix) 
        for doc in prepared_docs
    ]

    # Prepare request headers
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Prepare request data for vLLM's scoring endpoint
    data = {
        "model": model,
        "text_1": [formatted_query] * len(formatted_documents),
        "text_2": formatted_documents,
        **kwargs
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(base_url, headers=headers, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Qwen3 rerank API error {response.status}: {error_text}")
                    return documents

                result = await response.json()

                # Extract scores from vLLM response
                if "data" in result:
                    scores = []
                    for item in result["data"]:
                        if "score" in item:
                            scores.append(item["score"])
                        else:
                            logger.warning("Score not found in response item")
                            scores.append(0.0)
                else:
                    logger.warning("Unexpected vLLM API response format")
                    return documents

                # Combine documents with their scores
                doc_scores = list(zip(documents, scores))
                
                # Sort by relevance score (higher is better)
                doc_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Apply top_n limit if specified
                if top_n is not None:
                    doc_scores = doc_scores[:top_n]

                # Add rerank scores to documents
                reranked_docs = []
                for doc, score in doc_scores:
                    reranked_doc = doc.copy() if isinstance(doc, dict) else {"content": str(doc)}
                    reranked_doc["rerank_score"] = score
                    reranked_docs.append(reranked_doc)

                return reranked_docs

    except Exception as e:
        logger.error(f"Error during Qwen3 reranking: {e}")
        return documents


def create_qwen3_rerank_model(
    model: str = "Qwen/Qwen3-Reranker-0.6B",
    base_url: str = "http://localhost:8000/v1/score",
    api_key: Optional[str] = None,
    instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
    **kwargs
) -> RerankModel:
    """
    Create a RerankModel configured for Qwen3 Reranker via vLLM API.
    
    Args:
        model: Qwen3 reranker model name
        base_url: vLLM API endpoint (should end with /v1/score)
        api_key: API key if required
        instruction: The instruction for the reranking task
        **kwargs: Additional parameters
    
    Returns:
        Configured RerankModel instance
    
    Example usage:
    ```python
    from lightrag.rerank import create_qwen3_rerank_model
    from lightrag import LightRAG

    # Create Qwen3 rerank model
    rerank_model = create_qwen3_rerank_model(
        base_url="http://your-vllm-server:8000/v1/score",
        api_key="your-api-key-if-needed"
    )

    # Use in LightRAG
    rag = LightRAG(
        rerank_model_func=rerank_model.rerank,
        # ... other configurations
    )
    ```
    """
    return RerankModel(
        rerank_func=qwen3_rerank,
        kwargs={
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "instruction": instruction,
            **kwargs
        }
    )


# Convenience function for custom API endpoints
async def custom_rerank(
    query: str,
    documents: List[Dict[str, Any]],
    model: str,
    base_url: str,
    api_key: str,
    top_n: Optional[int] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Rerank documents using a custom API endpoint.
    This is useful for self-hosted or custom rerank services.
    """
    return await generic_rerank_api(
        query=query,
        documents=documents,
        model=model,
        base_url=base_url,
        api_key=api_key,
        top_n=top_n,
        **kwargs,
    )


if __name__ == "__main__":
    import asyncio

    async def main():
        # Example usage
        docs = [
            {"content": "The capital of France is Paris."},
            {"content": "Tokyo is the capital of Japan."},
            {"content": "London is the capital of England."},
        ]

        query = "What is the capital of France?"

        # Test Jina reranker
        print("Testing Jina reranker...")
        result = await jina_rerank(
            query=query, documents=docs, top_n=2, api_key="your-api-key-here"
        )
        print("Jina result:", result)

        # Test Qwen3 reranker
        print("\nTesting Qwen3 reranker...")
        qwen3_result = await qwen3_rerank(
            query=query, 
            documents=docs, 
            top_n=2,
            base_url="http://localhost:8000/v1/score",
            api_key="your-api-key-if-needed"  # Optional
        )
        print("Qwen3 result:", qwen3_result)

        # Test with RerankModel wrapper
        print("\nTesting with RerankModel wrapper...")
        qwen3_model = create_qwen3_rerank_model(
            base_url="http://localhost:8000/v1/score",
            api_key="your-api-key-if-needed"
        )
        wrapper_result = await qwen3_model.rerank(
            query=query,
            documents=docs,
            top_n=2
        )
        print("RerankModel wrapper result:", wrapper_result)

    asyncio.run(main())
