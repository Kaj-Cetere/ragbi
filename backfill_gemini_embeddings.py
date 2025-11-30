"""
Backfill Gemini embeddings for existing rows in sefaria_text_chunks.
Uses google/gemini-embedding-001 via OpenRouter (3072 dimensions).
"""

import os
import time
import requests
from dotenv import load_dotenv
from supabase import create_client, Client
from tqdm import tqdm

load_dotenv()

# --- CONFIGURATION ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Gemini embedding model (3072 dimensions by default)
GEMINI_EMBEDDING_MODEL = "google/gemini-embedding-001"

# Batch sizes (optimized for speed)
EMBEDDING_BATCH_SIZE = 100  # OpenRouter batch size for embeddings (increased)
DB_BATCH_SIZE = 200         # Number of rows to update at once (increased)
FETCH_BATCH_SIZE = 1000    # Number of rows to fetch at once (increased)

# --- CLIENTS ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_gemini_embeddings_batch(texts: list[str]) -> list[list[float]] | None:
    """Generate embeddings using Gemini via OpenRouter API."""
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://sefaria-rag.local",
                "X-Title": "Sefaria RAG Gemini Backfill"
            },
            json={
                "model": GEMINI_EMBEDDING_MODEL,
                "input": texts
                # No dimensions param needed - Gemini defaults to 3072
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        # Extract embeddings from response
        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings
        
    except Exception as e:
        print(f"⚠️ Gemini Embedding Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        time.sleep(2)
        return None


def fetch_chunks_without_gemini_embedding(offset: int = 0, limit: int = FETCH_BATCH_SIZE):
    """Fetch chunks that don't have Gemini embeddings yet."""
    result = supabase.table("sefaria_text_chunks") \
        .select("id, content") \
        .is_("embedding_gemini", "null") \
        .order("chunk_index") \
        .range(offset, offset + limit - 1) \
        .execute()
    return result.data


def update_gemini_embedding(chunk_id: str, embedding: list[float]):
    """Update a single chunk with its Gemini embedding."""
    supabase.table("sefaria_text_chunks") \
        .update({"embedding_gemini": embedding}) \
        .eq("id", chunk_id) \
        .execute()


def batch_update_embeddings(updates: list[dict]):
    """Update multiple chunks with their Gemini embeddings."""
    for update in updates:
        try:
            supabase.table("sefaria_text_chunks") \
                .update({"embedding_gemini": update["embedding"]}) \
                .eq("id", update["id"]) \
                .execute()
        except Exception as e:
            print(f"⚠️ Update error for {update['id']}: {e}")


def count_chunks_needing_embedding():
    """Count how many chunks still need Gemini embeddings."""
    result = supabase.table("sefaria_text_chunks") \
        .select("id", count="exact") \
        .is_("embedding_gemini", "null") \
        .execute()
    return result.count


def run_backfill():
    """Main backfill function."""
    print("🚀 Starting Gemini embedding backfill...")
    
    # Count total chunks needing embedding
    total_needed = count_chunks_needing_embedding()
    print(f"📊 Chunks needing Gemini embeddings: {total_needed}")
    
    if total_needed == 0:
        print("✅ All chunks already have Gemini embeddings!")
        return
    
    processed = 0
    failed = 0
    
    with tqdm(total=total_needed, desc="Backfilling") as pbar:
        while True:
            # Fetch batch of chunks without Gemini embeddings
            chunks = fetch_chunks_without_gemini_embedding(offset=0)
            
            if not chunks:
                break
            
            # Process in embedding batches
            for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
                batch = chunks[i:i + EMBEDDING_BATCH_SIZE]
                texts = [chunk["content"] for chunk in batch]
                ids = [chunk["id"] for chunk in batch]
                
                # Get Gemini embeddings
                max_retries = 3
                embeddings = None
                
                for retry in range(max_retries):
                    embeddings = get_gemini_embeddings_batch(texts)
                    if embeddings:
                        break
                    elif retry < max_retries - 1:
                        print(f"⚠️ Retry {retry + 1} for batch...")
                        time.sleep(3)
                
                if not embeddings:
                    print(f"❌ Failed to get embeddings for batch of {len(batch)}")
                    failed += len(batch)
                    pbar.update(len(batch))
                    continue
                
                # Update database with embeddings
                updates = [
                    {"id": ids[j], "embedding": embeddings[j]}
                    for j in range(len(embeddings))
                ]
                
                # Update in smaller sub-batches to avoid timeouts
                for k in range(0, len(updates), 50):
                    sub_batch = updates[k:k+50]
                    batch_update_embeddings(sub_batch)
                processed += len(updates)
                pbar.update(len(updates))
                
                # Minimal delay - Gemini has generous rate limits
                time.sleep(0.1)
    
    print(f"\n✅ Backfill complete!")
    print(f"   Processed: {processed}")
    print(f"   Failed: {failed}")
    
    # Verify
    remaining = count_chunks_needing_embedding()
    print(f"   Remaining without embeddings: {remaining}")


if __name__ == "__main__":
    run_backfill()
