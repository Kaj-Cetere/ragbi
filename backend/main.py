"""
Kesher AI - FastAPI Backend
Exposes the Sefaria RAG pipeline as a REST API with streaming support.
"""

import os
import sys
import json
import asyncio
import logging
from typing import Optional, AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import requests

from dotenv import load_dotenv
from supabase import create_client, Client
from openrouter import OpenRouter

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
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Default to Gemini Contextual embeddings
EMBEDDING_MODEL = "google/gemini-embedding-001"
EMBEDDING_DIMENSIONS = 3072
RPC_FUNCTION = "match_sefaria_chunks_gemini_contextual"

# Chat Model
CHAT_MODEL = "google/gemini-2.5-flash-lite-preview-09-2025"

# RAG Settings
TOP_K_RETRIEVE_FOR_RERANK = 60
TOP_K_AFTER_RERANK = 10
TOP_K_HYDRATE = 5

# --- GLOBAL CLIENTS ---
openrouter_client: OpenRouter = None
supabase_client: Client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize clients on startup."""
    global openrouter_client, supabase_client
    
    openrouter_client = OpenRouter(api_key=OPENROUTER_API_KEY)
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    logger.info("✅ Initialized API clients")
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
        "https://ragbi.vercel.app"
    ],
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
    """Generate streaming LLM response."""
    system_prompt = """You are a knowledgeable assistant specializing in Torah texts, particularly the Shulchan Arukh and its classical commentaries.

Your role is to:
1. Answer questions based on the provided context from authentic Torah sources
2. Cite specific references (e.g., "Shulchan Arukh, Choshen Mishpat 1:1")
3. Explain the main text and relevant commentaries when available
4. Be accurate and scholarly, but accessible
5. If the context doesn't contain enough information, say so clearly

Important: Base your answers primarily on the provided context. Do not make up sources or rulings."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""Here is relevant context from Torah sources:

{context}

---

User's Question: {query}

Please provide a helpful, accurate response based on the sources above."""}
    ]
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://kesher.ai",
                "X-Title": "Kesher AI"
            },
            json={
                "model": CHAT_MODEL,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 2000,
                "stream": True
            },
            stream=True,
            timeout=120
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta:
                                yield delta['content']
                    except json.JSONDecodeError:
                        continue
                        
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
        try:
            # Step 1: Embed query
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating embedding...'})}\n\n"
            embedding = get_query_embedding(query)
            
            # Step 2: Vector search
            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching sources...'})}\n\n"
            chunks = search_chunks(embedding)
            
            if not chunks:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No relevant sources found'})}\n\n"
                return
            
            # Step 3: Rerank
            yield f"data: {json.dumps({'type': 'status', 'message': 'Reranking results...'})}\n\n"
            chunks, was_reranked = rerank_with_fallback(
                query=query,
                chunks=chunks,
                model="zerank-2",
                top_n=TOP_K_AFTER_RERANK,
                enable_reranking=True
            )
            
            # Step 4: Hydrate with commentaries
            yield f"data: {json.dumps({'type': 'status', 'message': 'Fetching commentaries...'})}\n\n"
            hydrated = hydrate_chunks(chunks)
            
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
            
            if use_agent:
                # Use citation agent with paragraph-based streaming
                try:
                    async for event in generate_response_with_citations_stream(
                        query=query,
                        context=context,
                        retrieved_chunks=hydrated,
                        openrouter_api_key=OPENROUTER_API_KEY
                    ):
                        if event["type"] == "paragraph":
                            yield f"data: {json.dumps({'type': 'paragraph', 'content': event['content']})}\n\n"
                        elif event["type"] == "citation":
                            yield f"data: {json.dumps({'type': 'citation', 'ref': event['ref'], 'context': event['context'], 'hebrew': event['hebrew'], 'english': event['english'], 'book': event.get('book', '')})}\n\n"
                        # Skip 'done' events - we handle that separately
                except Exception as e:
                    logger.error(f"❌ Citation agent failed, falling back: {e}")
                    async for chunk in generate_response_stream(query, context):
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            else:
                async for chunk in generate_response_stream(query, context):
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            logger.error(f"❌ Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
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
