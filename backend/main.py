"""
Kesher AI - FastAPI Backend
Exposes the Sefaria RAG pipeline as a REST API with streaming support.
"""

import os
import sys
import json
import asyncio
import logging
import time
from typing import Optional, AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import requests
import httpx

from dotenv import load_dotenv
from supabase import create_client, Client
from openrouter import OpenRouter
from xai_sdk import Client as XAIClient

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_commentaries import MAIN_COMMENTATORS
from enricher import fetch_commentaries_for_ref
from reranker import rerank_with_fallback
from citation_agent import generate_response_with_citations_stream

load_dotenv()

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Default to Gemini Contextual embeddings
EMBEDDING_MODEL = "google/gemini-embedding-001"
EMBEDDING_DIMENSIONS = 3072
RPC_FUNCTION = "match_sefaria_chunks_gemini_contextual"

# Chat Model (using xAI directly)
CHAT_MODEL = "grok-4-1-fast-non-reasoning"

# RAG Settings
TOP_K_RETRIEVE_FOR_RERANK = 50
TOP_K_AFTER_RERANK = 10
TOP_K_HYDRATE = 3

# --- GLOBAL CLIENTS ---
openrouter_client: OpenRouter = None
xai_client: XAIClient = None
supabase_client: Client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize clients on startup."""
    global openrouter_client, xai_client, supabase_client

    openrouter_client = OpenRouter(api_key=OPENROUTER_API_KEY)
    xai_client = XAIClient(api_key=XAI_API_KEY)
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

    logger.info("✅ Initialized API clients (OpenRouter for embeddings, xAI for chat)")
    yield
    logger.info("👋 Shutting down")


# --- FASTAPI APP ---
app = FastAPI(
    title="Kesher AI API",
    description="Sefaria RAG API with streaming support",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- REQUEST/RESPONSE MODELS ---

class QueryRequest(BaseModel):
    """Request model for chat queries."""
    query: str = Field(..., min_length=1, max_length=2000, description="User's question")
    use_agent: bool = Field(default=True, description="Use quotation agent for structured responses")


class SourceChunk(BaseModel):
    """A source chunk from the RAG pipeline."""
    rank: int
    ref: str
    content: str
    context_text: Optional[str] = None
    similarity: float
    rerank_score: Optional[float] = None
    book: str
    siman: Optional[int] = None
    seif: Optional[int] = None
    commentaries: list = []
    hydrated: bool = False


class QueryResponse(BaseModel):
    """Response model for non-streaming queries."""
    answer: str
    sources: list[SourceChunk]


# --- CORE RAG FUNCTIONS ---

def get_query_embedding(query: str) -> list[float]:
    """Generate embedding for user query using Gemini."""
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://kesher.ai",
                "X-Title": "Kesher AI"
            },
            json={
                "model": EMBEDDING_MODEL,
                "input": query
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]
    except Exception as e:
        logger.error(f"❌ Embedding error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")


def search_chunks(embedding: list[float], match_count: int = TOP_K_RETRIEVE_FOR_RERANK) -> list[dict]:
    """Search for similar chunks using Supabase RPC."""
    try:
        result = supabase_client.rpc(
            RPC_FUNCTION,
            {
                "query_embedding": embedding,
                "match_threshold": 0.3,
                "match_count": match_count
            }
        ).execute()
        return result.data if result.data else []
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Vector search failed: {e}")


def hydrate_chunks(chunks: list[dict], hydrate_count: int = TOP_K_HYDRATE) -> list[dict]:
    """Hydrate top chunks with commentaries (concurrent)."""
    import concurrent.futures
    
    hydrated = []
    chunks_to_hydrate = chunks[:hydrate_count]
    
    def fetch_for_chunk(chunk_data):
        i, chunk = chunk_data
        sefaria_ref = chunk.get("sefaria_ref", "")
        metadata = chunk.get("metadata", {})
        book_title = metadata.get("book", "")
        
        commentaries = fetch_commentaries_for_ref(sefaria_ref, book_title)
        
        return {
            "rank": i + 1,
            "ref": sefaria_ref,
            "content": chunk.get("content", ""),
            "context_text": chunk.get("context_text", ""),
            "similarity": chunk.get("similarity", 0),
            "rerank_score": chunk.get("rerank_score"),
            "original_rank": chunk.get("original_rank"),
            "book": book_title,
            "siman": metadata.get("siman"),
            "seif": metadata.get("seif"),
            "commentaries": commentaries,
            "hydrated": True
        }
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=hydrate_count) as executor:
        future_to_chunk = {
            executor.submit(fetch_for_chunk, (i, chunk)): chunk 
            for i, chunk in enumerate(chunks_to_hydrate)
        }
        
        for future in concurrent.futures.as_completed(future_to_chunk):
            try:
                result = future.result()
                hydrated.append(result)
            except Exception as e:
                logger.error(f"❌ Hydration error: {e}")
    
    hydrated.sort(key=lambda x: x["rank"])
    
    # Add remaining chunks without hydration
    for i, chunk in enumerate(chunks[hydrate_count:], start=hydrate_count):
        metadata = chunk.get("metadata", {})
        hydrated.append({
            "rank": i + 1,
            "ref": chunk.get("sefaria_ref", ""),
            "content": chunk.get("content", ""),
            "context_text": chunk.get("context_text", ""),
            "similarity": chunk.get("similarity", 0),
            "rerank_score": chunk.get("rerank_score"),
            "original_rank": chunk.get("original_rank"),
            "book": metadata.get("book", ""),
            "siman": metadata.get("siman"),
            "seif": metadata.get("seif"),
            "commentaries": [],
            "hydrated": False
        })
    
    return hydrated


def build_context_prompt(hydrated_chunks: list[dict]) -> str:
    """Build the context section for the LLM prompt."""
    context_parts = []
    
    for chunk in hydrated_chunks:
        section = f"## {chunk['ref']} (Similarity: {chunk['similarity']:.2f})\n"
        section += f"**Base Text:**\n{chunk['content']}\n"

        if chunk.get("context_text"):
            section += f"\n**Contextual Notes:**\n{chunk['context_text']}\n"
        
        if chunk['hydrated'] and chunk['commentaries']:
            section += "\n**Commentaries:**\n"
            for comm in chunk['commentaries']:
                section += f"- **{comm['commentator']}:** {comm['text']}\n"
        
        context_parts.append(section)
    
    return "\n---\n".join(context_parts)


async def generate_response_stream(query: str, context: str) -> AsyncIterator[str]:
    """Generate streaming LLM response using xAI SDK."""
    from xai_sdk.chat import user, system

    system_prompt = """You are a knowledgeable assistant specializing in Torah texts, particularly the Shulchan Arukh and its classical commentaries.

Your role is to:
1. Answer questions based on the provided context from authentic Torah sources
2. Cite specific references (e.g., "Shulchan Arukh, Choshen Mishpat 1:1")
3. Explain the main text and relevant commentaries when available
4. Be accurate and scholarly, but accessible
5. If the context doesn't contain enough information, say so clearly

Important: Base your answers primarily on the provided context. Do not make up sources or rulings."""

    user_prompt = f"""Here is relevant context from Torah sources:

{context}

---

User's Question: {query}

Please provide a helpful, accurate response based on the sources above."""

    try:
        # Create chat with xAI
        chat = xai_client.chat.create(
            model=CHAT_MODEL,
            temperature=0.3,
            max_tokens=2000
        )

        chat.append(system(system_prompt))
        chat.append(user(user_prompt))

        # Stream response (note: chat.stream() is a sync generator)
        for response, chunk in chat.stream():
            if chunk.content:
                yield chunk.content

    except Exception as e:
        logger.error(f"❌ LLM error: {e}")
        yield f"\n\nError generating response: {e}"


# --- API ENDPOINTS ---

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "kesher-api"}


@app.post("/api/chat/stream")
async def chat_stream(request: QueryRequest):
    """
    Stream a response to the user's query.
    Uses SSE (Server-Sent Events) for streaming.
    """
    query = request.query
    use_agent = request.use_agent
    
    logger.info(f"🚀 New query: '{query[:50]}...' (agent={use_agent})")
    
    async def event_generator():
        # Performance metrics tracking
        metrics = {
            "query_length": len(query),
            "timestamps": {},
            "durations": {},
            "counts": {},
            "metadata": {}
        }
        pipeline_start = time.time()
        metrics["timestamps"]["pipeline_start"] = pipeline_start

        try:
            # Step 1: Embed query
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating embedding...'})}\n\n"
            embed_start = time.time()
            embedding = get_query_embedding(query)
            embed_end = time.time()
            metrics["durations"]["embedding"] = round((embed_end - embed_start) * 1000, 2)
            metrics["counts"]["embedding_dimensions"] = len(embedding)

            # Emit embedding metrics
            yield f"data: {json.dumps({'type': 'metrics', 'stage': 'embedding', 'duration_ms': metrics['durations']['embedding'], 'details': {'dimensions': len(embedding)}})}\n\n"

            # Step 2: Vector search
            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching sources...'})}\n\n"
            search_start = time.time()
            chunks = search_chunks(embedding)
            search_end = time.time()
            metrics["durations"]["vector_search"] = round((search_end - search_start) * 1000, 2)
            metrics["counts"]["chunks_retrieved"] = len(chunks) if chunks else 0

            # Calculate similarity distribution
            if chunks:
                similarities = [c.get("similarity", 0) for c in chunks]
                metrics["metadata"]["similarity_max"] = round(max(similarities), 4)
                metrics["metadata"]["similarity_min"] = round(min(similarities), 4)
                metrics["metadata"]["similarity_avg"] = round(sum(similarities) / len(similarities), 4)

            # Emit search metrics
            yield f"data: {json.dumps({'type': 'metrics', 'stage': 'vector_search', 'duration_ms': metrics['durations']['vector_search'], 'details': {'chunks_found': metrics['counts']['chunks_retrieved'], 'similarity_max': metrics['metadata'].get('similarity_max'), 'similarity_avg': metrics['metadata'].get('similarity_avg')}})}\n\n"

            if not chunks:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No relevant sources found'})}\n\n"
                return

            # Step 3: Rerank
            yield f"data: {json.dumps({'type': 'status', 'message': 'Reranking results...'})}\n\n"
            rerank_start = time.time()
            chunks, was_reranked = rerank_with_fallback(
                query=query,
                chunks=chunks,
                model="zerank-2",
                top_n=TOP_K_AFTER_RERANK,
                enable_reranking=True
            )
            rerank_end = time.time()
            metrics["durations"]["reranking"] = round((rerank_end - rerank_start) * 1000, 2)
            metrics["metadata"]["reranking_applied"] = was_reranked
            metrics["counts"]["chunks_after_rerank"] = len(chunks)

            # Calculate rerank score distribution
            if was_reranked and chunks:
                rerank_scores = [c.get("rerank_score", 0) for c in chunks if c.get("rerank_score")]
                if rerank_scores:
                    metrics["metadata"]["rerank_score_max"] = round(max(rerank_scores), 4)
                    metrics["metadata"]["rerank_score_avg"] = round(sum(rerank_scores) / len(rerank_scores), 4)

            # Emit rerank metrics
            yield f"data: {json.dumps({'type': 'metrics', 'stage': 'reranking', 'duration_ms': metrics['durations']['reranking'], 'details': {'was_reranked': was_reranked, 'chunks_after': metrics['counts']['chunks_after_rerank'], 'top_rerank_score': metrics['metadata'].get('rerank_score_max')}})}\n\n"

            # Step 4: Hydrate with commentaries
            yield f"data: {json.dumps({'type': 'status', 'message': 'Fetching commentaries...'})}\n\n"
            hydrate_start = time.time()
            hydrated = hydrate_chunks(chunks)
            hydrate_end = time.time()
            metrics["durations"]["hydration"] = round((hydrate_end - hydrate_start) * 1000, 2)

            # Count commentaries fetched
            total_commentaries = sum(len(h.get("commentaries", [])) for h in hydrated if h.get("hydrated"))
            metrics["counts"]["chunks_hydrated"] = sum(1 for h in hydrated if h.get("hydrated"))
            metrics["counts"]["commentaries_fetched"] = total_commentaries

            # Emit hydration metrics
            yield f"data: {json.dumps({'type': 'metrics', 'stage': 'hydration', 'duration_ms': metrics['durations']['hydration'], 'details': {'chunks_hydrated': metrics['counts']['chunks_hydrated'], 'commentaries_fetched': total_commentaries}})}\n\n"

            # Send sources to frontend
            sources_data = [
                {
                    "rank": h["rank"],
                    "ref": h["ref"],
                    "content": h["content"][:300] + "..." if len(h["content"]) > 300 else h["content"],
                    "similarity": h["similarity"],
                    "rerank_score": h.get("rerank_score"),
                    "book": h["book"],
                    "siman": h.get("siman"),
                    "seif": h.get("seif"),
                    "commentaries": h["commentaries"][:3] if h["commentaries"] else [],
                    "hydrated": h["hydrated"]
                }
                for h in hydrated[:5]  # Top 5 sources
            ]
            yield f"data: {json.dumps({'type': 'sources', 'data': sources_data})}\n\n"

            # Step 5: Generate response
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating response...'})}\n\n"
            context = build_context_prompt(hydrated)
            metrics["counts"]["context_length"] = len(context)

            llm_start = time.time()
            actual_llm_start = None  # Track when LLM actually starts (after source cache)
            first_token_time = None
            token_count = 0
            paragraph_count = 0
            citation_count = 0

            if use_agent:
                # Use citation agent with paragraph-based streaming
                try:
                    async for event in generate_response_with_citations_stream(
                        query=query,
                        context=context,
                        retrieved_chunks=hydrated,
                        xai_api_key=XAI_API_KEY,
                        openrouter_api_key=OPENROUTER_API_KEY
                    ):
                        # Handle source cache timing event
                        if event["type"] == "source_cache_built":
                            source_cache_time = event.get("duration_ms", 0)
                            metrics["durations"]["source_cache"] = source_cache_time
                            yield f"data: {json.dumps({'type': 'metrics', 'stage': 'source_cache', 'duration_ms': source_cache_time, 'details': {'note': 'Fetching translations from Sefaria'}})}\n\n"
                            actual_llm_start = time.time()  # LLM starts now
                            continue

                        if first_token_time is None and event["type"] in ("paragraph", "citation"):
                            first_token_time = time.time()
                            # Calculate time to first token from actual LLM start
                            if actual_llm_start:
                                metrics["durations"]["time_to_first_token"] = round((first_token_time - actual_llm_start) * 1000, 2)
                            else:
                                metrics["durations"]["time_to_first_token"] = round((first_token_time - llm_start) * 1000, 2)
                            yield f"data: {json.dumps({'type': 'metrics', 'stage': 'llm_first_token', 'duration_ms': metrics['durations']['time_to_first_token']})}\n\n"

                        if event["type"] == "paragraph":
                            paragraph_count += 1
                            token_count += len(event.get("content", "").split())  # Rough word count
                            yield f"data: {json.dumps({'type': 'paragraph', 'content': event['content']})}\n\n"
                        elif event["type"] == "citation":
                            citation_count += 1
                            yield f"data: {json.dumps({
                                'type': 'citation',
                                'ref': event['ref'],
                                'context': event['context'],
                                'hebrew': event['hebrew'],
                                'english': event['english'],
                                'book': event.get('book', ''),
                                'hebrew_excerpt': event.get('hebrew_excerpt'),
                                'translation_success': event.get('translation_success', True)
                            })}\n\n"
                        # Skip 'done' events - we handle that separately
                except Exception as e:
                    logger.error(f"❌ Citation agent failed, falling back: {e}")
                    async for chunk in generate_response_stream(query, context):
                        if first_token_time is None:
                            first_token_time = time.time()
                            metrics["durations"]["time_to_first_token"] = round((first_token_time - llm_start) * 1000, 2)
                        token_count += len(chunk.split())
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            else:
                async for chunk in generate_response_stream(query, context):
                    if first_token_time is None:
                        first_token_time = time.time()
                        metrics["durations"]["time_to_first_token"] = round((first_token_time - llm_start) * 1000, 2)
                    token_count += len(chunk.split())
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"

            llm_end = time.time()
            # Use actual_llm_start if available (excludes source_cache time)
            if use_agent and actual_llm_start:
                metrics["durations"]["llm_generation"] = round((llm_end - actual_llm_start) * 1000, 2)
            else:
                metrics["durations"]["llm_generation"] = round((llm_end - llm_start) * 1000, 2)
            metrics["counts"]["paragraphs"] = paragraph_count
            metrics["counts"]["citations"] = citation_count
            metrics["counts"]["approx_words"] = token_count

            # Calculate total pipeline time
            pipeline_end = time.time()
            metrics["durations"]["total_pipeline"] = round((pipeline_end - pipeline_start) * 1000, 2)

            # Calculate throughput
            if metrics["durations"]["llm_generation"] > 0:
                metrics["metadata"]["words_per_second"] = round(token_count / (metrics["durations"]["llm_generation"] / 1000), 2)

            # Emit final LLM and summary metrics
            yield f"data: {json.dumps({'type': 'metrics', 'stage': 'llm_complete', 'duration_ms': metrics['durations']['llm_generation'], 'details': {'paragraphs': paragraph_count, 'citations': citation_count, 'approx_words': token_count, 'words_per_second': metrics['metadata'].get('words_per_second')}})}\n\n"

            # Emit complete metrics summary
            yield f"data: {json.dumps({'type': 'metrics_summary', 'total_duration_ms': metrics['durations']['total_pipeline'], 'breakdown': metrics['durations'], 'counts': metrics['counts'], 'metadata': metrics['metadata']})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

            logger.info(f"✅ Query completed in {metrics['durations']['total_pipeline']:.0f}ms - Embed: {metrics['durations']['embedding']:.0f}ms, Search: {metrics['durations']['vector_search']:.0f}ms, Rerank: {metrics['durations']['reranking']:.0f}ms, Hydrate: {metrics['durations']['hydration']:.0f}ms, LLM: {metrics['durations']['llm_generation']:.0f}ms")

        except Exception as e:
            logger.error(f"❌ Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Content-Type": "text/event-stream",
        }
    )


@app.get("/api/sefaria/text/{ref:path}")
async def get_sefaria_text(ref: str, context: int = 1):
    """
    Proxy endpoint to fetch text from Sefaria API.
    Returns the chapter/section with the target verse highlighted.
    """
    try:
        # Extract chapter ref (remove verse number for context)
        parts = ref.split(':')
        chapter_ref = parts[0] if len(parts) > 1 else ref
        verse_num = int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else None
        
        url = f"https://www.sefaria.org/api/texts/{chapter_ref}?context={context}"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        he_texts = data.get("he", [])
        en_texts = data.get("text", [])
        
        if isinstance(he_texts, str):
            he_texts = [he_texts]
        if isinstance(en_texts, str):
            en_texts = [en_texts]
        
        verses = []
        for i in range(max(len(he_texts), len(en_texts))):
            verses.append({
                "number": i + 1,
                "he": he_texts[i] if i < len(he_texts) else "",
                "en": en_texts[i] if i < len(en_texts) else "",
                "isTarget": (i + 1) == verse_num if verse_num else False
            })
        
        return {
            "ref": chapter_ref,
            "title": data.get("indexTitle", chapter_ref),
            "heTitle": data.get("heIndexTitle", ""),
            "verses": verses,
            "targetVerse": verse_num
        }
        
    except Exception as e:
        logger.error(f"❌ Sefaria API error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch from Sefaria: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
