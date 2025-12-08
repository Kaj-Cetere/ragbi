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
from reranker import rerank_with_fallback
from quotation_agent import generate_response_with_quotations_sync, generate_response_with_quotations_stream

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
    },
    "Gemini Contextual (3072d)": {
        "model": "google/gemini-embedding-001",
        "dimensions": 3072,
        "rpc_function": "match_sefaria_chunks_gemini_contextual"
    }
}

# Chat Model
CHAT_MODEL = "google/gemini-2.5-flash-lite-preview-09-2025"  # Fast and capable model

# RAG Settings
TOP_K_RETRIEVE = 10  # Total chunks to retrieve (without reranking)
TOP_K_RETRIEVE_FOR_RERANK = 60  # Chunks to retrieve when reranking is enabled
TOP_K_AFTER_RERANK = 10  # Top chunks to keep after reranking
TOP_K_HYDRATE = 5    # Top chunks to hydrate with commentaries

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
    Uses concurrent requests for 5x faster performance.
    """
    import concurrent.futures
    
    start_time = time.time()
    logger.info(f"🔍 Starting concurrent commentary hydration for top {hydrate_count} chunks")
    
    hydrated = []
    
    # Prepare data for concurrent fetching
    chunks_to_hydrate = chunks[:hydrate_count]
    
    def fetch_for_chunk(chunk_data):
        """Helper function to fetch commentaries for a single chunk"""
        i, chunk = chunk_data
        chunk_start = time.time()
        sefaria_ref = chunk.get("sefaria_ref", "")
        metadata = chunk.get("metadata", {})
        book_title = metadata.get("book", "")
        
        logger.info(f"📖 Fetching commentaries for chunk {i+1}/{hydrate_count}: {sefaria_ref}")
        
        # Fetch commentaries for this chunk
        commentaries = fetch_commentaries_for_ref(sefaria_ref, book_title)
        chunk_time = time.time() - chunk_start
        
        logger.info(f"✅ Chunk {i+1} hydration took {chunk_time:.3f}s, found {len(commentaries)} commentaries")
        
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
    
    # Fetch commentaries concurrently using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=hydrate_count) as executor:
        # Submit all tasks
        future_to_chunk = {
            executor.submit(fetch_for_chunk, (i, chunk)): chunk 
            for i, chunk in enumerate(chunks_to_hydrate)
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_chunk):
            try:
                result = future.result()
                hydrated.append(result)
            except Exception as e:
                logger.error(f"❌ Error hydrating chunk: {e}")
    
    # Sort by original rank to maintain order
    hydrated.sort(key=lambda x: x["rank"])
    
    # Add remaining chunks without hydration (background context)
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
    
    total_time = time.time() - start_time
    total_commentaries = sum(len(chunk.get("commentaries", [])) for chunk in hydrated)
    logger.info(f"🎯 Concurrent hydration completed in {total_time:.3f}s, fetched {total_commentaries} total commentaries")
    logger.info(f"⚡ Speedup: ~{hydrate_count}x faster than sequential")
    
    return hydrated


def build_context_prompt(hydrated_chunks: list[dict]) -> str:
    """Build the context section for the LLM prompt."""
    context_parts = []
    
    for chunk in hydrated_chunks:
        section = f"## {chunk['ref']} (Similarity: {chunk['similarity']:.2f})\n"
        section += f"**Base Text:**\n{chunk['content']}\n"

        # Optional contextual embedding text (for contextual models)
        if chunk.get("context_text"):
            section += f"\n**Contextual Notes:**\n{chunk['context_text']}\n"
        
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
    3. Top chunks are hydrated with commentaries from Sefaria API
    4. An LLM generates a response based on the context
    
    **Available Commentaries:**
    - **Orach Chayim**: Mishnah Berurah
    - **Yoreh De'ah**: Ba'er Hetev on Shulchan Arukh, Yoreh De'ah  
    - **Choshen Mishpat**: Ba'er Hetev on Shulchan Arukh, Choshen Mishpat
    - **Even HaEzer**: Ba'er Hetev on Shulchan Arukh, Even HaEzer
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
    
    # Reranking settings
    st.header("Reranking")
    enable_reranking = st.toggle(
        "🎯 Enable Reranking",
        value=True,
        help="Use zerank-2 to rerank retrieved chunks for better relevance"
    )
    
    if enable_reranking:
        rerank_retrieve_count = st.slider(
            "Initial Retrieval Count",
            min_value=20,
            max_value=100,
            value=60,
            step=10,
            help="Number of chunks to retrieve before reranking"
        )
        rerank_top_k = st.slider(
            "Top K After Rerank",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            help="Number of top chunks to keep after reranking"
        )
        rerank_hydrate_count = st.slider(
            "Chunks to Hydrate",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Number of top reranked chunks to hydrate with commentaries"
        )
    else:
        rerank_retrieve_count = TOP_K_RETRIEVE
        rerank_top_k = TOP_K_RETRIEVE
        rerank_hydrate_count = TOP_K_HYDRATE
    
    st.divider()
    
    # Comparison mode toggle
    comparison_mode = st.toggle(
        "🔬 Comparison Mode",
        value=False,
        help="Compare regular vs contextual embeddings side-by-side"
    )
    
    show_sources = st.checkbox("Show source chunks", value=True)
    show_commentaries = st.checkbox("Show commentaries detail", value=True)
    show_llm_prompt = st.checkbox("Show full LLM prompt (debug)", value=False,
                                  help="Display the exact prompt (system + user message) sent to the LLM.")
    
    st.divider()
    
    # Quotation Agent Settings
    st.header("🤖 AI Agent")
    use_quotation_agent = st.toggle(
        "📜 Enable Quotation Agent",
        value=False,
        help="Use pydantic-ai agent to intelligently quote specific seifim and commentaries in responses"
    )
    
    if use_quotation_agent:
        st.info("The agent will use tools to quote specific sources directly in its response.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retrieved_chunks" not in st.session_state:
    st.session_state.retrieved_chunks = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Helper function to process chunks into hydrated format
def process_chunks_to_hydrated(chunks: list[dict]) -> list[dict]:
    """Convert raw chunks to hydrated format (without actual hydration)."""
    hydrated = []
    for i, chunk in enumerate(chunks):
        metadata = chunk.get("metadata", {})
        hydrated.append({
            "rank": i + 1,
            "ref": chunk.get("sefaria_ref", ""),
            "content": chunk.get("content", ""),
            "context_text": chunk.get("context_text", ""),  # Include context if available
            "similarity": chunk.get("similarity", 0),
            "book": metadata.get("book", ""),
            "siman": metadata.get("siman"),
            "seif": metadata.get("seif"),
            "commentaries": [],
            "hydrated": False
        })
    return hydrated

# Chat input
if prompt := st.chat_input("Ask a question about Halakha..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # COMPARISON MODE: Side-by-side comparison
    if comparison_mode:
        st.markdown("### 🔬 Comparison: Regular vs Contextual Embeddings")
        
        # Generate embedding once (same for both searches)
        gemini_model = EMBEDDING_MODELS["Gemini (3072d)"]
        with st.spinner("Generating query embedding..."):
            embedding = get_query_embedding(prompt, gemini_model)
        
        if embedding:
            col1, col2 = st.columns(2)
            
            # LEFT COLUMN: Regular Gemini
            with col1:
                st.markdown("#### 📊 Regular Gemini")
                with st.spinner("Searching..."):
                    chunks_regular = search_chunks(embedding, "match_sefaria_chunks_gemini")
                
                if chunks_regular:
                    hydrated_regular = process_chunks_to_hydrated(chunks_regular)
                    context_regular = build_context_prompt(hydrated_regular)
                    
                    response_placeholder_1 = st.empty()
                    full_response_1 = ""
                    for chunk in generate_response_stream(prompt, context_regular):
                        full_response_1 += chunk
                        response_placeholder_1.markdown(full_response_1 + "▌")
                    response_placeholder_1.markdown(full_response_1)
                    
                    if show_sources:
                        with st.expander("📚 Sources", expanded=False):
                            for chunk in hydrated_regular:
                                st.markdown(f"**{chunk['ref']}** ({chunk['similarity']:.2f})")
                                st.text(chunk['content'][:200] + "...")
                                st.divider()
                else:
                    st.warning("No results found")
            
            # RIGHT COLUMN: Contextual Gemini
            with col2:
                st.markdown("#### 🧠 Contextual Gemini")
                with st.spinner("Searching..."):
                    chunks_contextual = search_chunks(embedding, "match_sefaria_chunks_gemini_contextual")
                
                if chunks_contextual:
                    hydrated_contextual = process_chunks_to_hydrated(chunks_contextual)
                    context_contextual = build_context_prompt(hydrated_contextual)
                    
                    response_placeholder_2 = st.empty()
                    full_response_2 = ""
                    for chunk in generate_response_stream(prompt, context_contextual):
                        full_response_2 += chunk
                        response_placeholder_2.markdown(full_response_2 + "▌")
                    response_placeholder_2.markdown(full_response_2)
                    
                    if show_sources:
                        with st.expander("📚 Sources (with context)", expanded=False):
                            for chunk in hydrated_contextual:
                                st.markdown(f"**{chunk['ref']}** ({chunk['similarity']:.2f})")
                                if chunk.get('context_text'):
                                    st.info(f"🧠 Context: {chunk['context_text']}")
                                st.text(chunk['content'][:200] + "...")
                                st.divider()
                else:
                    st.warning("No results (contextual embeddings may not be populated yet)")
        else:
            st.error("Failed to generate query embedding.")
    
    # NORMAL MODE: Single model search
    else:
        query_start_time = time.time()
        logger.info(f"🚀 Starting new query: '{prompt[:50]}...' using {embedding_choice}")
        logger.info(f"⚙️ Reranking: {'enabled' if enable_reranking else 'disabled'}")
        
        with st.chat_message("assistant"):
            with st.spinner(f"Searching with {embedding_choice}..."):
                # Step 1: Generate query embedding
                embedding_start = time.time()
                embedding = get_query_embedding(prompt, selected_model)
                embedding_time = time.time() - embedding_start
                
                if embedding:
                    # Step 2: Search for similar chunks
                    # Retrieve more chunks if reranking is enabled
                    retrieve_count = rerank_retrieve_count if enable_reranking else TOP_K_RETRIEVE
                    search_start = time.time()
                    chunks = search_chunks(embedding, selected_model["rpc_function"], match_count=retrieve_count)
                    search_time = time.time() - search_start
                    
                    if chunks:
                        # Step 3: Rerank chunks (if enabled)
                        rerank_time = 0
                        was_reranked = False
                        if enable_reranking:
                            rerank_start = time.time()
                            chunks, was_reranked = rerank_with_fallback(
                                query=prompt,
                                chunks=chunks,
                                model="zerank-2",
                                top_n=rerank_top_k,
                                enable_reranking=True
                            )
                            rerank_time = time.time() - rerank_start
                            logger.info(f"🎯 Reranking {'succeeded' if was_reranked else 'failed/skipped'}, took {rerank_time:.3f}s")
                        
                        # Step 4: Hydrate chunks with commentaries from Sefaria
                        hydrate_count = rerank_hydrate_count if enable_reranking else TOP_K_HYDRATE
                        hydration_start = time.time()
                        hydrated = hydrate_chunks(chunks, hydrate_count=hydrate_count)
                        hydration_time = time.time() - hydration_start
                        
                        st.session_state.retrieved_chunks = hydrated
                        
                        # Step 4: Build context (with commentaries & optional contextual notes)
                        context_start = time.time()
                        context = build_context_prompt(hydrated)
                        context_time = time.time() - context_start
                        
                        # Optional: Show and log full LLM prompt
                        system_prompt_debug = """You are a knowledgeable assistant specializing in Torah texts, particularly the Shulchan Arukh and its classical commentaries.

Your role is to:
1. Answer questions based on the provided context from authentic Torah sources
2. Cite specific references (e.g., "Shulchan Arukh, Choshen Mishpat 1:1")
3. Explain the main text and relevant commentaries when available
4. Be accurate and scholarly, but accessible
5. If the context doesn't contain enough information, say so clearly

Important: Base your answers primarily on the provided context. Do not make up sources or rulings."""

                        user_message_debug = f"""Here is relevant context from Torah sources:

{context}

---

User's Question: {prompt}

Please provide a helpful, accurate response based on the sources above."""

                        messages_debug = [
                            {"role": "system", "content": system_prompt_debug},
                            {"role": "user", "content": user_message_debug},
                        ]

                        logger.info(f"🧪 LLM PROMPT MESSAGES: {messages_debug}")

                        if show_llm_prompt:
                            with st.expander("🔎 View LLM Prompt (debug)", expanded=False):
                                st.markdown("**System Prompt:**")
                                st.code(system_prompt_debug)
                                st.markdown("**User Message (Context + Question):**")
                                st.code(user_message_debug)

                        # Step 5: Generate response (with or without quotation agent)
                        generation_start = time.time()
                        response_placeholder = st.empty()
                        full_response = ""
                        
                        if use_quotation_agent:
                            # Use pydantic-ai agent with quotation tools (streaming)
                            logger.info("🤖 Using quotation agent for streaming response generation")
                            try:
                                import asyncio
                                import nest_asyncio
                                nest_asyncio.apply()
                                
                                async def stream_quotation_response():
                                    """Async generator wrapper for streaming."""
                                    full_resp = ""
                                    async for chunk in generate_response_with_quotations_stream(
                                        query=prompt,
                                        context=context,
                                        retrieved_chunks=hydrated,
                                        supabase=supabase_client,
                                        openrouter_api_key=OPENROUTER_API_KEY
                                    ):
                                        full_resp += chunk
                                        response_placeholder.markdown(full_resp + "▌")
                                    response_placeholder.markdown(full_resp)
                                    return full_resp
                                
                                # Run the async streaming in the event loop
                                full_response = asyncio.run(stream_quotation_response())
                                
                                # Log success
                                logger.info(f"📊 Agent generated streaming response with quotations")
                            except Exception as e:
                                logger.error(f"❌ Quotation agent failed: {e}")
                                st.error(f"Agent error: {e}. Falling back to standard generation.")
                                # Fallback to standard streaming
                                for chunk in generate_response_stream(prompt, context):
                                    full_response += chunk
                                    response_placeholder.markdown(full_response + "▌")
                                response_placeholder.markdown(full_response)
                        else:
                            # Standard streaming response
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
                        if enable_reranking and rerank_time > 0:
                            logger.info(f"  🎯 Reranking: {rerank_time:.3f}s ({rerank_time/total_query_time*100:.1f}%)")
                        logger.info(f"  🔄 Chunk hydration: {hydration_time:.3f}s ({hydration_time/total_query_time*100:.1f}%)")
                        logger.info(f"  📝 Context building: {context_time:.3f}s ({context_time/total_query_time*100:.1f}%)")
                        logger.info(f"  ⚡ LLM generation: {generation_time:.3f}s ({generation_time/total_query_time*100:.1f}%)")
                        logger.info(f"  🎯 TOTAL QUERY TIME: {total_query_time:.3f}s")
                        logger.info(f"  📈 Retrieved {retrieve_count} chunks, {f'reranked to {len(chunks)}' if was_reranked else 'no reranking'}, generated {len(full_response)} chars response")
                        
                        # Show sources if enabled
                        if show_sources and hydrated:
                            with st.expander("📚 View Sources", expanded=False):
                                for chunk in hydrated:
                                    icon = "📄"
                                    hydrated_label = " 💧" if chunk.get('hydrated') else ""
                                    
                                    # Show both similarity and rerank score if available
                                    score_display = f"Similarity: {chunk['similarity']:.2f}"
                                    if 'rerank_score' in chunk and chunk['rerank_score'] is not None:
                                        score_display += f" | Rerank: {chunk['rerank_score']:.3f}"
                                        if 'original_rank' in chunk and chunk['original_rank'] is not None:
                                            score_display += f" (was #{chunk['original_rank']})"
                                    
                                    st.markdown(f"**{icon} {chunk['ref']}{hydrated_label}** ({score_display})")
                                    st.text(chunk['content'][:300] + "..." if len(chunk['content']) > 300 else chunk['content'])
                                    # Optionally show attached commentaries for hydrated chunks
                                    if show_commentaries and chunk.get('hydrated') and chunk.get('commentaries'):
                                        st.markdown("**Commentaries:**")
                                        for comm in chunk['commentaries']:
                                            st.markdown(f"- **{comm['commentator']}**: {comm['text']}")
                                    st.divider()
                    else:
                        st.warning("No matching sources found. Try rephrasing your question.")
                else:
                    st.error("Failed to generate query embedding.")

# Footer
st.divider()
st.caption("Powered by Qwen-3 & Gemini Embeddings, zerank-2 Reranker, Supabase pgvector (halfvec), and OpenRouter")
