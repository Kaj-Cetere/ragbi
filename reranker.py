"""
Reranker Module for Sefaria RAG
Uses ZeroEntropy's zerank-2 model to rerank retrieved chunks.
"""

import os
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
from zeroentropy import ZeroEntropy

load_dotenv()

logger = logging.getLogger(__name__)

# Initialize ZeroEntropy client
ZEROENTROPY_API_KEY = os.getenv("ZEROENTROPY_API_KEY")

def init_reranker_client() -> Optional[ZeroEntropy]:
    """Initialize ZeroEntropy client for reranking."""
    try:
        if not ZEROENTROPY_API_KEY:
            logger.warning("⚠️ ZEROENTROPY_API_KEY not found. Reranking will be disabled.")
            return None
        
        client = ZeroEntropy(api_key=ZEROENTROPY_API_KEY)
        logger.info("✅ ZeroEntropy reranker client initialized")
        return client
    except Exception as e:
        logger.error(f"❌ Failed to initialize ZeroEntropy client: {e}")
        return None


def rerank_chunks(
    query: str,
    chunks: List[Dict],
    model: str = "zerank-2",
    top_n: Optional[int] = None
) -> List[Dict]:
    """
    Rerank chunks using ZeroEntropy's zerank-2 model.
    
    Args:
        query: The user's search query
        chunks: List of chunk dictionaries with 'content' field
        model: Reranker model to use (default: zerank-2)
        top_n: Number of top results to return (None = return all reranked)
    
    Returns:
        List of reranked chunks with updated 'rerank_score' field
    """
    if not chunks:
        logger.warning("⚠️ No chunks provided for reranking")
        return []
    
    try:
        client = init_reranker_client()
        if not client:
            logger.warning("⚠️ Reranker client not available, returning original chunks")
            return chunks
        
        logger.info(f"🔍 Starting reranking with {model} for {len(chunks)} chunks")
        
        # Extract document texts for reranking
        documents = [chunk.get("content", "") for chunk in chunks]
        
        # Call ZeroEntropy rerank API
        rerank_response = client.models.rerank(
            model=model,
            query=query,
            documents=documents,
            top_n=top_n
        )
        
        # Map reranked results back to original chunks
        reranked_chunks = []
        for result in rerank_response.results:
            original_chunk = chunks[result.index].copy()
            original_chunk["rerank_score"] = result.relevance_score
            original_chunk["original_rank"] = result.index + 1
            reranked_chunks.append(original_chunk)
        
        logger.info(f"✅ Reranking completed, returned {len(reranked_chunks)} chunks")
        
        # Log top rerank scores
        if reranked_chunks:
            top_scores = [chunk.get('rerank_score', 0) for chunk in reranked_chunks[:3]]
            logger.info(f"📊 Top 3 rerank scores: {[f'{s:.3f}' for s in top_scores]}")
        
        return reranked_chunks
        
    except Exception as e:
        logger.error(f"❌ Reranking failed: {e}")
        logger.warning("⚠️ Falling back to original chunk order")
        return chunks


def rerank_with_fallback(
    query: str,
    chunks: List[Dict],
    model: str = "zerank-2",
    top_n: Optional[int] = None,
    enable_reranking: bool = True
) -> tuple[List[Dict], bool]:
    """
    Rerank chunks with fallback to original order if reranking fails.
    
    Returns:
        Tuple of (reranked_chunks, was_reranked)
    """
    if not enable_reranking:
        logger.info("ℹ️ Reranking disabled by user")
        return chunks, False
    
    reranked = rerank_chunks(query, chunks, model, top_n)
    
    # Check if reranking actually happened
    was_reranked = any('rerank_score' in chunk for chunk in reranked)
    
    return reranked, was_reranked
