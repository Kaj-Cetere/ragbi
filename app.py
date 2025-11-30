"""
Sefaria RAG Chat Application
A Streamlit-based chat interface for querying Torah texts with commentary hydration.
"""

import os
import streamlit as st
import requests
import time
import logging
from dotenv import load_dotenv
from openrouter import OpenRouter
from supabase import create_client, Client

from config_commentaries import MAIN_COMMENTATORS
from enricher import fetch_commentaries_for_ref

load_dotenv()

# --- DEBUGGING LOGGING CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Embedding Models
EMBEDDING_MODELS = {
    "Qwen-3 (1536d)": {
        "model": "qwen/qwen3-embedding-8b",
        "dimensions": 1536,
        "rpc_function": "match_sefaria_chunks"
    },
    "Gemini (3072d)": {
        "model": "google/gemini-embedding-001",
        "dimensions": 3072,
        "rpc_function": "match_sefaria_chunks_gemini"
    }
}

# Chat Model
CHAT_MODEL = "google/gemini-2.5-flash-lite-preview-09-2025"  # Fast and capable model

# RAG Settings
TOP_K_RETRIEVE = 10  # Total chunks to retrieve
TOP_K_HYDRATE = 0    #Top chunks to hydrate with commentaries

# --- CLIENTS ---
@st.cache_resource
def init_clients():
    """Initialize API clients (cached for performance)."""
    openrouter = OpenRouter(api_key=OPENROUTER_API_KEY)
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return openrouter, supabase

openrouter_client, supabase_client = init_clients()


# --- CORE RAG FUNCTIONS ---

def get_query_embedding(query: str, model_config: dict) -> list[float]:
    """Generate embedding for user query using the specified model."""
    start_time = time.time()
    model_name = model_config["model"]
    
    # Determine which model type for clearer logging
    if "qwen" in model_name:
        model_type = "Qwen-3"
    elif "gemini" in model_name:
        model_type = "Gemini"
    else:
        model_type = model_name
    
    logger.info(f"🔍 Starting {model_type} embedding generation for query: '{query[:50]}...'")
    
    dimensions = model_config.get("dimensions")
    
    try:
        # Use OpenRouter SDK for Qwen (supports dimensions param)
        if "qwen" in model_name:
            api_start = time.time()
            response = openrouter_client.embeddings.generate(
                model=model_name,
                input=query,
                dimensions=dimensions
            )
            api_time = time.time() - api_start
            logger.info(f"✅ {model_type} embedding API call took {api_time:.3f}s")
            embedding = response.data[0].embedding
        else:
            # Use raw API for Gemini (no dimensions param needed)
            api_start = time.time()
            response = requests.post(
                url="https://openrouter.ai/api/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://sefaria-rag.local",
                    "X-Title": "Sefaria RAG"
                },
                json={
                    "model": model_name,
                    "input": query
                },
                timeout=60
            )
            api_time = time.time() - api_start
            logger.info(f"✅ {model_type} embedding API call took {api_time:.3f}s")
            response.raise_for_status()
            data = response.json()
            embedding = data["data"][0]["embedding"]
        
        total_time = time.time() - start_time
        logger.info(f"🎯 {model_type} embedding generation completed in {total_time:.3f}s (dimensions: {len(embedding)})")
        return embedding
            
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"❌ {model_type} embedding generation failed after {error_time:.3f}s: {e}")
        st.error(f"Embedding error: {e}")
        return None


def search_chunks(embedding: list[float], rpc_function: str, match_count: int = TOP_K_RETRIEVE) -> list[dict]:
    """Search for similar chunks using Supabase RPC function."""
    start_time = time.time()
    logger.info(f"🔍 Starting vector search using {rpc_function} (match_count: {match_count})")
    
    try:
        rpc_start = time.time()
        result = supabase_client.rpc(
            rpc_function,
            {
                "query_embedding": embedding,
                "match_threshold": 0.3,
                "match_count": match_count
            }
        ).execute()
        rpc_time = time.time() - rpc_start
        
        chunks = result.data if result.data else []
        total_time = time.time() - start_time
        
        logger.info(f"✅ RPC call took {rpc_time:.3f}s, returned {len(chunks)} chunks")
        logger.info(f"🎯 Vector search completed in {total_time:.3f}s")
        
        # Log similarity scores of top results
        if chunks:
            top_similarities = [chunk.get('similarity', 0) for chunk in chunks[:3]]
            logger.info(f"📊 Top 3 similarities: {[f'{s:.3f}' for s in top_similarities]}")
        
        return chunks
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"❌ Vector search failed after {error_time:.3f}s: {e}")
        st.error(f"Search error: {e}")
        return []


def hydrate_chunks(chunks: list[dict], hydrate_count: int = TOP_K_HYDRATE) -> list[dict]:
    """
    Hydrate top chunks with commentaries from Sefaria API.
    Only the top N chunks get commentary hydration.
    """
    start_time = time.time()
    logger.info(f"🔍 Starting commentary hydration for top {hydrate_count} chunks")
    
    hydrated = []
    
    for i, chunk in enumerate(chunks[:hydrate_count]):
        chunk_start = time.time()
        sefaria_ref = chunk.get("sefaria_ref", "")
        metadata = chunk.get("metadata", {})
        book_title = metadata.get("book", "")
        
        logger.info(f"📖 Fetching commentaries for chunk {i+1}/{hydrate_count}: {sefaria_ref}")
        
        # Fetch commentaries for this chunk
        commentaries = fetch_commentaries_for_ref(sefaria_ref, book_title)
        chunk_time = time.time() - chunk_start
        
        logger.info(f"✅ Chunk {i+1} hydration took {chunk_time:.3f}s, found {len(commentaries)} commentaries")
        
        hydrated.append({
            "rank": i + 1,
            "ref": sefaria_ref,
            "content": chunk.get("content", ""),
            "similarity": chunk.get("similarity", 0),
            "book": book_title,
            "siman": metadata.get("siman"),
            "seif": metadata.get("seif"),
            "commentaries": commentaries,
            "hydrated": True
        })
    
    # Add remaining chunks without hydration (background context)
    for i, chunk in enumerate(chunks[hydrate_count:], start=hydrate_count):
        metadata = chunk.get("metadata", {})
        hydrated.append({
            "rank": i + 1,
            "ref": chunk.get("sefaria_ref", ""),
            "content": chunk.get("content", ""),
            "similarity": chunk.get("similarity", 0),
            "book": metadata.get("book", ""),
            "siman": metadata.get("siman"),
            "seif": metadata.get("seif"),
            "commentaries": [],
            "hydrated": False
        })
    
    total_time = time.time() - start_time
    total_commentaries = sum(len(chunk.get("commentaries", [])) for chunk in hydrated)
    logger.info(f"🎯 Hydration completed in {total_time:.3f}s, fetched {total_commentaries} total commentaries")
    
    return hydrated


def build_context_prompt(hydrated_chunks: list[dict]) -> str:
    """Build the context section for the LLM prompt."""
    context_parts = []
    
    for chunk in hydrated_chunks:
        section = f"## {chunk['ref']} (Similarity: {chunk['similarity']:.2f})\n"
        section += f"**Base Text:**\n{chunk['content']}\n"
        
        if chunk['hydrated'] and chunk['commentaries']:
            section += "\n**Commentaries:**\n"
            for comm in chunk['commentaries']:
                section += f"- **{comm['commentator']}:** {comm['text']}\n"
        
        context_parts.append(section)
    
    return "\n---\n".join(context_parts)


def generate_response(query: str, context: str, chat_history: list[dict]) -> str:
    """Generate LLM response using OpenRouter chat API."""
    
    system_prompt = """You are a knowledgeable assistant specializing in Torah texts, particularly the Shulchan Arukh and its classical commentaries.

Your role is to:
1. Answer questions based on the provided context from authentic Torah sources
2. Cite specific references (e.g., "Shulchan Arukh, Choshen Mishpat 1:1")
3. Explain the main text and relevant commentaries when available
4. Be accurate and scholarly, but accessible
5. If the context doesn't contain enough information, say so clearly

Important: Base your answers primarily on the provided context. Do not make up sources or rulings."""

    # Build messages for chat completion
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
                "HTTP-Referer": "https://sefaria-rag.local",
                "X-Title": "Sefaria RAG"
            },
            json={
                "model": CHAT_MODEL,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 2000
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error generating response: {e}"


def generate_response_stream(query: str, context: str):
    """Generate streaming LLM response using OpenRouter chat API."""
    start_time = time.time()
    logger.info(f"🔍 Starting LLM response generation for query: '{query[:50]}...'")
    logger.info(f"📝 Context length: {len(context)} characters")
    
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
        api_start = time.time()
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://sefaria-rag.local",
                "X-Title": "Sefaria RAG"
            },
            json={
                "model": CHAT_MODEL,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 2000,
                "stream": True
            },
            stream=True,
            timeout=60
        )
        response.raise_for_status()
        
        api_time = time.time() - api_start
        logger.info(f"✅ LLM API call initiated in {api_time:.3f}s")
        
        first_chunk_time = None
        total_chars = 0
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    try:
                        import json
                        chunk = json.loads(data)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta:
                                if first_chunk_time is None:
                                    first_chunk_time = time.time() - api_start
                                    logger.info(f"⚡ First token received after {first_chunk_time:.3f}s")
                                
                                content = delta['content']
                                total_chars += len(content)
                                yield content
                    except:
                        continue
        
        total_time = time.time() - start_time
        logger.info(f"🎯 LLM response generation completed in {total_time:.3f}s, generated {total_chars} characters")
        if first_chunk_time:
            logger.info(f"📊 Generation speed: {total_chars/(total_time-first_chunk_time):.1f} chars/sec after first token")
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"❌ LLM response generation failed after {error_time:.3f}s: {e}")
        yield f"Error generating response: {e}"


# --- STREAMLIT UI ---

st.set_page_config(
    page_title="Sefaria RAG",
    page_icon="📜",
    layout="wide"
)

st.title("📜 Sefaria RAG Chat")
st.caption("Ask questions about Halakha from the Shulchan Arukh with classical commentaries")

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This application uses **Retrieval-Augmented Generation (RAG)** to answer questions about Torah texts.
    
    **How it works:**
    1. Your question is embedded using Qwen-3
    2. Similar text chunks are retrieved from Supabase
    3. Top 3 results are hydrated with commentaries from Sefaria API
    4. An LLM generates a response based on the context
    
    **Available Commentaries:**
    - **Orach Chayim**: Magen Avraham, Taz
    - **Yoreh De'ah**: Shach, Taz  
    - **Choshen Mishpat**: Shach, Sma
    - **Even HaEzer**: Chelkat Mechokek, Beit Shmuel
    """)
    
    st.divider()
    
    # Settings
    st.header("Settings")
    
    # Embedding model selection
    embedding_choice = st.selectbox(
        "Embedding Model",
        options=list(EMBEDDING_MODELS.keys()),
        index=0,
        help="Choose which embedding model to use for semantic search"
    )
    selected_model = EMBEDDING_MODELS[embedding_choice]
    
    st.divider()
    
    show_sources = st.checkbox("Show source chunks", value=True)
    show_commentaries = st.checkbox("Show commentaries detail", value=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retrieved_chunks" not in st.session_state:
    st.session_state.retrieved_chunks = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about Halakha..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Start overall query timing
    query_start_time = time.time()
    logger.info(f"🚀 Starting new query: '{prompt[:50]}...' using {embedding_choice}")
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner(f"Searching with {embedding_choice}..."):
            # Step 1: Generate query embedding
            embedding_start = time.time()
            embedding = get_query_embedding(prompt, selected_model)
            embedding_time = time.time() - embedding_start
            
            if embedding:
                # Step 2: Search for similar chunks
                search_start = time.time()
                chunks = search_chunks(embedding, selected_model["rpc_function"])
                search_time = time.time() - search_start
                
                if chunks:
                    # Step 3: Skip commentary hydration for faster testing
                    # hydrated = hydrate_chunks(chunks)
                    
                    # Use raw chunks directly
                    hydration_start = time.time()
                    hydrated = []
                    for i, chunk in enumerate(chunks):
                        metadata = chunk.get("metadata", {})
                        hydrated.append({
                            "rank": i + 1,
                            "ref": chunk.get("sefaria_ref", ""),
                            "content": chunk.get("content", ""),
                            "similarity": chunk.get("similarity", 0),
                            "book": metadata.get("book", ""),
                            "siman": metadata.get("siman"),
                            "seif": metadata.get("seif"),
                            "commentaries": [],  # No commentaries
                            "hydrated": False     # Mark as not hydrated
                        })
                    hydration_time = time.time() - hydration_start
                    
                    st.session_state.retrieved_chunks = hydrated
                    
                    # Step 4: Build context (without commentaries)
                    context_start = time.time()
                    context = build_context_prompt(hydrated)
                    context_time = time.time() - context_start
                    
                    # Step 5: Generate response (streaming)
                    generation_start = time.time()
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    for chunk in generate_response_stream(prompt, context):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    
                    response_placeholder.markdown(full_response)
                    generation_time = time.time() - generation_start
                    
                    # Store assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response
                    })
                    
                    # Overall performance summary
                    total_query_time = time.time() - query_start_time
                    logger.info(f"📊 QUERY PERFORMANCE SUMMARY:")
                    logger.info(f"  🔍 Embedding generation: {embedding_time:.3f}s ({embedding_time/total_query_time*100:.1f}%)")
                    logger.info(f"  🔎 Vector search: {search_time:.3f}s ({search_time/total_query_time*100:.1f}%)")
                    logger.info(f"  📝 Context building: {context_time:.3f}s ({context_time/total_query_time*100:.1f}%)")
                    logger.info(f"  ⚡ LLM generation: {generation_time:.3f}s ({generation_time/total_query_time*100:.1f}%)")
                    logger.info(f"  🔄 Chunk processing: {hydration_time:.3f}s ({hydration_time/total_query_time*100:.1f}%)")
                    logger.info(f"  🎯 TOTAL QUERY TIME: {total_query_time:.3f}s")
                    logger.info(f"  📈 Retrieved {len(chunks)} chunks, generated {len(full_response)} chars response")
                    
                    # Show sources if enabled
                    if show_sources and hydrated:
                        with st.expander("📚 View Sources", expanded=False):
                            for chunk in hydrated:
                                icon = "📄"  # Always show as regular chunk (no hydration)
                                st.markdown(f"**{icon} {chunk['ref']}** (Similarity: {chunk['similarity']:.2f})")
                                st.text(chunk['content'][:300] + "..." if len(chunk['content']) > 300 else chunk['content'])
                                
                                # Commentaries section commented out for faster testing
                                # if show_commentaries and chunk['commentaries']:
                                #     for comm in chunk['commentaries']:
                                #         st.markdown(f"  - *{comm['commentator']}*: {comm['text'][:200]}...")
                                st.divider()
                else:
                    st.warning("No matching sources found. Try rephrasing your question.")
            else:
                st.error("Failed to generate query embedding.")

# Footer
st.divider()
st.caption("Powered by Qwen-3 & Gemini Embeddings, Supabase pgvector (halfvec), and OpenRouter")
