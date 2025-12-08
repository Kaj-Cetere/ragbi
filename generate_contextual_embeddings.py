"""
Generate Contextual Embeddings for Shulchan Aruch Seifim

This script:
1. Fetches chunks from the database (first 300 for testing)
2. Groups them by book and siman
3. Generates contextual descriptions using Grok (xAI direct API)
4. Re-embeds with Gemini
5. Updates the database with new embeddings

Uses prompt caching optimization:
- Small simanim (≤15 seifim): entire siman as context
- Large simanim (>15 seifim): windowed groups of 5
- Prompt structured for xAI automatic caching (prefix stays same, suffix varies)
"""

import os
import time
import requests
import httpx
import json
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI
from tqdm import tqdm
from collections import defaultdict

load_dotenv()

# --- CONFIGURATION ---
XAI_API_KEY = os.getenv("XAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Models
CONTEXT_MODEL = "grok-4-1-fast-non-reasoning"  # xAI direct (non-reasoning for speed)
GEMINI_EMBEDDING_MODEL = "google/gemini-embedding-001"  # For embeddings via OpenRouter

# Settings
CONTEXT_TEMPERATURE = 0.3
CONTEXT_MAX_TOKENS = 150
EMBEDDING_BATCH_SIZE = 50  # Batch size for Gemini embeddings
TEST_CHUNK_LIMIT = 4000 # Total chunks to process per run (will paginate through Supabase's 1000 limit)
SUPABASE_PAGE_SIZE = 1000  # Supabase max rows per query
PROGRESS_FILE = "contextual_embeddings_progress.json"  # Checkpoint file

# Separator for combining context with content
SEPARATOR = "\n\n---\n\n"

# --- CLIENTS ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# xAI client using OpenAI SDK
xai_client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
    timeout=httpx.Timeout(120.0)
)

# --- PROMPT ---
CONTEXTUAL_PROMPT_HEBREW = """
<הסימן>
{SIMAN_CONTEXT}
</הסימן>

<סעיף_נוכחי>
{CURRENT_SEIF}
</סעיף_נוכחי>

תן הסבר קצר וענייני (משפט אחד עד שניים) המתאר את תפקיד הסעיף הנוכחי בהקשר ההלכתי הרחב יותר של הסימן. 
ההסבר צריך לעזור בחיפוש ואחזור המידע - ציין את הנושא ההלכתי, ואם רלוונטי, את הקשר לסעיפים סביבו.
ענה רק עם ההסבר הקצר ללא שום הקדמה או סיום.

דוגמה לפלט רצוי:
"סעיף זה עוסק בדיני קידוש בליל שבת, ומפרט את נוסח הברכה על היין לפני הקידוש עצמו."
"""


def load_progress() -> dict:
    """Load progress from checkpoint file."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load progress file: {e}")
    return {"completed_simanim": [], "last_updated": None}


def save_progress(book: str, siman: int):
    """Save progress to checkpoint file."""
    progress = load_progress()
    siman_key = f"{book}:{siman}"
    
    if siman_key not in progress["completed_simanim"]:
        progress["completed_simanim"].append(siman_key)
    
    progress["last_updated"] = datetime.now().isoformat()
    
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save progress: {e}")


def is_siman_completed(book: str, siman: int) -> bool:
    """Check if a siman has already been completed."""
    progress = load_progress()
    siman_key = f"{book}:{siman}"
    return siman_key in progress["completed_simanim"]


def fetch_chunks_for_testing(limit: int = TEST_CHUNK_LIMIT, skip_processed: bool = True) -> list[dict]:
    """
    Fetch chunks that need contextual embeddings, ordered by chunk_index.
    Handles Supabase's 1000-row limit by paginating through results.
    """
    print(f"📥 Fetching up to {limit} chunks (skip_processed={skip_processed})...")
    
    all_chunks = []
    offset = 0
    
    while len(all_chunks) < limit:
        # Calculate how many rows to fetch in this batch
        remaining = limit - len(all_chunks)
        batch_size = min(SUPABASE_PAGE_SIZE, remaining)
        
        query = supabase.table("sefaria_text_chunks") \
            .select("id, chunk_index, content, sefaria_ref, metadata")
        
        # Skip chunks that already have contextual embeddings
        if skip_processed:
            query = query.is_("embedding_gemini_contextual", "null")
        
        result = query.order("chunk_index") \
            .range(offset, offset + batch_size - 1) \
            .execute()
        
        batch_chunks = result.data if result.data else []
        
        if not batch_chunks:
            # No more chunks available
            break
        
        all_chunks.extend(batch_chunks)
        offset += len(batch_chunks)
        
        print(f"   Fetched batch: {len(batch_chunks)} chunks (total: {len(all_chunks)}/{limit})")
        
        # If we got fewer rows than requested, we've reached the end
        if len(batch_chunks) < batch_size:
            break
    
    print(f"✅ Fetched {len(all_chunks)} chunks total")
    return all_chunks


def group_chunks_by_book_and_siman(chunks: list[dict]) -> dict:
    """
    Group chunks by book and siman for efficient context generation.
    
    Returns:
        {
            "Shulchan Arukh, Orach Chayim": {
                1: [chunk1, chunk2, ...],  # Siman 1
                2: [chunk3, chunk4, ...],  # Siman 2
            },
            ...
        }
    """
    grouped = defaultdict(lambda: defaultdict(list))
    
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        book = metadata.get("book", "Unknown")
        siman = metadata.get("siman", 0)
        
        grouped[book][siman].append(chunk)
    
    # Sort seifim within each siman by seif number
    for book in grouped:
        for siman in grouped[book]:
            grouped[book][siman].sort(key=lambda x: x.get("metadata", {}).get("seif", 0))
    
    return grouped


def format_siman_context(seifim: list[dict]) -> str:
    """Format a list of seifim as context text."""
    context_parts = []
    for seif in seifim:
        metadata = seif.get("metadata", {})
        seif_num = metadata.get("seif", "?")
        content = seif.get("content", "")
        context_parts.append(f"סעיף {seif_num}: {content}")
    
    return "\n\n".join(context_parts)


def get_context_window(seif_index: int, all_seifim: list[dict], total_seifim: int) -> list[dict]:
    """
    Get the context window for a seif based on the caching strategy.
    
    - Small simanim (≤15): use entire siman
    - Large simanim (>15): use windowed context based on group of 5
    """
    if total_seifim <= 15:
        # Small siman: use entire siman
        return all_seifim
    else:
        # Large siman: use windowed context
        # seif_index is 0-based
        group_index = seif_index // 5  # Which group of 5 this seif belongs to
        window_start = max(0, group_index * 5 - 2)  # 2 seifim padding before
        window_end = min(total_seifim, (group_index + 1) * 5 + 2)  # 2 seifim padding after
        return all_seifim[window_start:window_end]


def generate_context_for_seif(current_seif: dict, siman_context_text: str) -> tuple[str | None, dict]:
    """
    Generate contextual description using Grok via xAI direct API.
    
    Args:
        current_seif: The current seif chunk
        siman_context_text: Pre-formatted siman context (for caching efficiency)
    
    Returns:
        Tuple of (context_text, usage_stats)
    """
    current_content = current_seif.get("content", "")
    
    # Build prompt with siman context first (cached) + current seif (varies)
    prompt = CONTEXTUAL_PROMPT_HEBREW.format(
        SIMAN_CONTEXT=siman_context_text,
        CURRENT_SEIF=current_content
    )
    
    try:
        completion = xai_client.chat.completions.create(
            model=CONTEXT_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=CONTEXT_TEMPERATURE,
            max_tokens=CONTEXT_MAX_TOKENS
        )
        
        context_text = completion.choices[0].message.content.strip()
        
        # Remove quotes if the model wrapped its response in them
        if context_text.startswith('"') and context_text.endswith('"'):
            context_text = context_text[1:-1]
        
        # Extract usage stats for cache monitoring
        usage = {}
        if completion.usage:
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "cached_tokens": 0
            }
            # Try to get cached tokens from different possible locations
            if hasattr(completion.usage, 'prompt_tokens_details') and completion.usage.prompt_tokens_details:
                if hasattr(completion.usage.prompt_tokens_details, 'cached_tokens'):
                    usage["cached_tokens"] = completion.usage.prompt_tokens_details.cached_tokens
        
        return context_text, usage
        
    except Exception as e:
        print(f"⚠️ Context generation error: {e}")
        return None, {}


def get_gemini_embeddings_batch(texts: list[str]) -> list[list[float]] | None:
    """Generate embeddings using Gemini via OpenRouter API."""
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://sefaria-rag.local",
                "X-Title": "Sefaria RAG Contextual Embeddings"
            },
            json={
                "model": GEMINI_EMBEDDING_MODEL,
                "input": texts
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings
        
    except Exception as e:
        print(f"⚠️ Gemini embedding error: {e}")
        return None


def update_chunk_with_context(chunk_id: str, context_text: str, content_with_context: str, embedding: list[float]):
    """Update a chunk with its contextual data."""
    try:
        supabase.table("sefaria_text_chunks") \
            .update({
                "context_text": context_text,
                "content_with_context": content_with_context,
                "embedding_gemini_contextual": embedding
            }) \
            .eq("id", chunk_id) \
            .execute()
    except Exception as e:
        print(f"⚠️ Database update error for {chunk_id}: {e}")


def update_chunks_batch(chunks_with_embeddings: list[dict]) -> tuple[int, int]:
    """Update multiple chunks with their contextual data in one operation."""
    success_count = 0
    fail_count = 0
    
    for chunk_data in chunks_with_embeddings:
        try:
            supabase.table("sefaria_text_chunks") \
                .update({
                    "context_text": chunk_data["context_text"],
                    "content_with_context": chunk_data["content_with_context"],
                    "embedding_gemini_contextual": chunk_data["embedding"]
                }) \
                .eq("id", chunk_data["id"]) \
                .execute()
            success_count += 1
        except Exception as e:
            print(f"⚠️ Database update error for {chunk_data['id']}: {e}")
            fail_count += 1
    
    return success_count, fail_count


def process_siman(book: str, siman: int, seifim: list[dict]) -> tuple[list[dict], dict]:
    """
    Process all seifim in a siman and generate contexts.
    
    Returns:
        Tuple of (processed_chunks, cache_stats)
    """
    total_seifim = len(seifim)
    processed = []
    
    # Track cache statistics
    cache_stats = {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_cached_tokens": 0,
        "requests": 0
    }
    
    # Track current context window for cache efficiency
    current_context_key = None
    current_context_text = None
    
    for i, seif in enumerate(seifim):
        # Determine context window
        context_seifim = get_context_window(i, seifim, total_seifim)
        
        # Create a cache key based on context window
        if total_seifim <= 15:
            context_key = f"{book}:{siman}:full"
        else:
            group_index = i // 5
            context_key = f"{book}:{siman}:group_{group_index}"
        
        # Format context text (reuse if same context window)
        if context_key != current_context_key:
            current_context_key = context_key
            current_context_text = format_siman_context(context_seifim)
        
        # Generate context using xAI
        context_text, usage = generate_context_for_seif(seif, current_context_text)
        
        # Track usage
        if usage:
            cache_stats["total_prompt_tokens"] += usage.get("prompt_tokens", 0)
            cache_stats["total_completion_tokens"] += usage.get("completion_tokens", 0)
            cache_stats["total_cached_tokens"] += usage.get("cached_tokens", 0)
            cache_stats["requests"] += 1
        
        if context_text:
            content_with_context = context_text + SEPARATOR + seif.get("content", "")
            processed.append({
                "id": seif["id"],
                "context_text": context_text,
                "content_with_context": content_with_context,
                "sefaria_ref": seif.get("sefaria_ref", "")
            })
        else:
            print(f"⚠️ Failed to generate context for {seif.get('sefaria_ref', 'unknown')}")
        
        # Small delay to avoid rate limiting
        time.sleep(0.1)
    
    return processed, cache_stats


def run_contextual_embedding_pipeline():
    """Main pipeline for generating contextual embeddings."""
    print("🚀 Starting Contextual Embedding Pipeline")
    print(f"   Model: {CONTEXT_MODEL}")
    print(f"   Temperature: {CONTEXT_TEMPERATURE}")
    print(f"   Test limit: {TEST_CHUNK_LIMIT} chunks")
    print(f"   Progress file: {PROGRESS_FILE}")
    
    # Load existing progress
    progress = load_progress()
    completed_count = len(progress["completed_simanim"])
    if completed_count > 0:
        print(f"   Resuming: {completed_count} simanim already completed")
    print()
    
    # Step 1: Fetch chunks
    chunks = fetch_chunks_for_testing(TEST_CHUNK_LIMIT)
    if not chunks:
        print("❌ No chunks to process")
        return
    
    # Step 2: Group by book and siman
    print("📊 Grouping chunks by book and siman...")
    grouped = group_chunks_by_book_and_siman(chunks)
    
    # Count simanim
    total_simanim = sum(len(simanim) for simanim in grouped.values())
    print(f"   Found {len(grouped)} books, {total_simanim} simanim")
    
    for book, simanim in grouped.items():
        print(f"   - {book}: {len(simanim)} simanim")
    print()
    
    # Step 3: Generate contexts for each siman
    success_count = 0
    fail_count = 0
    total_cache_stats = {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_cached_tokens": 0,
        "requests": 0
    }
    
    for book, simanim in grouped.items():
        print(f"📖 Processing {book}...")
        
        for siman, seifim in tqdm(sorted(simanim.items()), desc=f"  Simanim"):
            # Skip if already completed (resume capability)
            if is_siman_completed(book, siman):
                print(f"⏭️ Skipping {book} siman {siman} (already completed)")
                continue
            processed, cache_stats = process_siman(book, siman, seifim)
            
            if processed:
                # Generate embeddings for this siman immediately
                texts_to_embed = [p["content_with_context"] for p in processed]
                
                siman_embeddings = []
                for i in range(0, len(texts_to_embed), EMBEDDING_BATCH_SIZE):
                    batch_texts = texts_to_embed[i:i + EMBEDDING_BATCH_SIZE]
                    
                    # Retry logic
                    max_retries = 3
                    embeddings = None
                    
                    for retry in range(max_retries):
                        embeddings = get_gemini_embeddings_batch(batch_texts)
                        if embeddings:
                            break
                        elif retry < max_retries - 1:
                            print(f"⚠️ Retry {retry + 1} for {book} siman {siman} batch {i // EMBEDDING_BATCH_SIZE + 1}")
                            time.sleep(3)
                    
                    if embeddings:
                        siman_embeddings.extend(embeddings)
                    else:
                        print(f"❌ Failed to generate embeddings for {book} siman {siman} batch {i // EMBEDDING_BATCH_SIZE + 1}")
                        siman_embeddings.extend([None] * len(batch_texts))
                    
                    time.sleep(0.2)
                
                # Prepare chunks with embeddings for database update
                chunks_with_embeddings = []
                for i, chunk in enumerate(processed):
                    if i < len(siman_embeddings) and siman_embeddings[i]:
                        chunks_with_embeddings.append({
                            "id": chunk["id"],
                            "context_text": chunk["context_text"],
                            "content_with_context": chunk["content_with_context"],
                            "embedding": siman_embeddings[i]
                        })
                
                # Update database immediately for this siman
                if chunks_with_embeddings:
                    success, fail = update_chunks_batch(chunks_with_embeddings)
                    print(f"✅ {book} siman {siman}: {success} updated, {fail} failed")
                    
                    # Update counters
                    success_count += success
                    fail_count += fail
                    
                    # Save progress checkpoint
                    if success > 0:
                        save_progress(book, siman)
                        print(f"💾 Progress saved for {book} siman {siman}")
            
            # Aggregate cache stats
            total_cache_stats["total_prompt_tokens"] += cache_stats.get("total_prompt_tokens", 0)
            total_cache_stats["total_completion_tokens"] += cache_stats.get("total_completion_tokens", 0)
            total_cache_stats["total_cached_tokens"] += cache_stats.get("total_cached_tokens", 0)
            total_cache_stats["requests"] += cache_stats.get("requests", 0)
    
    print(f"\n✅ Processing complete!")
    
    # Display cache statistics
    if total_cache_stats["requests"] > 0:
        cache_ratio = (total_cache_stats["total_cached_tokens"] / total_cache_stats["total_prompt_tokens"] * 100) if total_cache_stats["total_prompt_tokens"] > 0 else 0
        print(f"\n📊 Cache Statistics:")
        print(f"   Total requests: {total_cache_stats['requests']}")
        print(f"   Total prompt tokens: {total_cache_stats['total_prompt_tokens']:,}")
        print(f"   Total cached tokens: {total_cache_stats['total_cached_tokens']:,}")
        print(f"   Cache hit ratio: {cache_ratio:.1f}%")
        print(f"   Avg completion tokens: {total_cache_stats['total_completion_tokens'] / total_cache_stats['requests']:.0f}")
    
    # Skip if nothing was processed
    if success_count == 0 and fail_count == 0:
        print("❌ No chunks processed")
        return
    
    
    
    print(f"\n✅ Pipeline complete!")
    print(f"   Successfully processed: {success_count}")
    print(f"   Failed: {fail_count}")
    
    # Show final progress
    final_progress = load_progress()
    print(f"   Total simanim completed: {len(final_progress['completed_simanim'])}")
    if final_progress.get('last_updated'):
        print(f"   Last updated: {final_progress['last_updated']}")
    
    # Verification query hint
    print("\n📊 To verify, run in Supabase SQL Editor:")
    print("   SELECT COUNT(*) FROM sefaria_text_chunks WHERE embedding_gemini_contextual IS NOT NULL;")


def clear_progress():
    """Clear the progress file (for fresh start)."""
    if os.path.exists(PROGRESS_FILE):
        try:
            os.remove(PROGRESS_FILE)
            print(f"🗑️ Progress file {PROGRESS_FILE} cleared")
        except Exception as e:
            print(f"⚠️ Could not clear progress file: {e}")
    else:
        print(f"ℹ️ No progress file to clear")


if __name__ == "__main__":
    run_contextual_embedding_pipeline()
