import os
from dotenv import load_dotenv
from openrouter import OpenRouter
from supabase import create_client
from reranker import rerank_chunks

load_dotenv()

# --- SETUP ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

client = OpenRouter(api_key=OPENROUTER_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
TARGET_DIMENSIONS = 1536

# Reranking settings
ENABLE_RERANKING = True
RETRIEVE_COUNT = 50  # Retrieve more chunks for reranking
RERANK_TOP_K = 10    # Keep top 10 after reranking

# --- TEST DATA (The "Answer Key") ---
# You define these manually. It's the best way to start.
TEST_CASES = [
    {
        "question": "What does hashem's name mean?",
        # We expect this specific Seif to appear in the results
        "expected_ref": "Shulchan Arukh, Orach Chayim 5:1" 
        # (Note: Verify the exact ref string in your DB first!)
    },
    {
        "question": "If my mother is a divorcee, can I do brichas kohanim?",
        "expected_ref": "Shulchan Arukh, Orach Chayim 128:42"
    },
    {
        "question": "Can I leave a pot in a non-jew's home?",
        "expected_ref": "Shulchan Arukh, Yoreh De'ah 122:9" 
    }
]

# --- HELPER ---
def get_embedding(text):
    res = client.embeddings.generate(
        model=EMBEDDING_MODEL,
        input=text,
        dimensions=TARGET_DIMENSIONS
    )
    return res.data[0].embedding

# --- THE TEST LOOP ---
def run_test():
    score_without_rerank = 0
    score_with_rerank = 0
    print(f"🚀 Starting Test on {len(TEST_CASES)} questions...\n")
    print(f"⚙️ Reranking: {'ENABLED' if ENABLE_RERANKING else 'DISABLED'}")
    print(f"📊 Retrieve Count: {RETRIEVE_COUNT}, Rerank Top K: {RERANK_TOP_K}\n")

    for case in TEST_CASES:
        query = case["question"]
        target = case["expected_ref"]
        
        print(f"❓ Question: {query}")
        
        # 1. Get Embedding
        vector = get_embedding(query)
        
        # 2. Search Supabase (retrieve more for reranking)
        retrieve_count = RETRIEVE_COUNT if ENABLE_RERANKING else 10
        response = supabase.rpc("match_documents", {
            "query_embedding": vector,
            "match_threshold": 0.3,  # Lower threshold to catch more potential matches
            "match_count": retrieve_count
        }).execute()
        
        chunks = response.data if response.data else []
        
        # 3a. Check WITHOUT reranking (top 10 from vector search)
        found_without_rerank = False
        retrieved_refs_no_rerank = []
        
        for hit in chunks[:10]:
            ref = hit['sefaria_ref']
            retrieved_refs_no_rerank.append(ref)
            if target in ref:
                found_without_rerank = True
        
        if found_without_rerank:
            score_without_rerank += 1
            print(f"✅ WITHOUT RERANK: Found '{target}' in top 10")
        else:
            print(f"❌ WITHOUT RERANK: Expected '{target}' not in top 10")
            print(f"   --> Got: {retrieved_refs_no_rerank[:5]}")
        
        # 3b. Check WITH reranking (if enabled)
        if ENABLE_RERANKING and chunks:
            reranked_chunks = rerank_chunks(
                query=query,
                chunks=chunks,
                model="zerank-2",
                top_n=RERANK_TOP_K
            )
            
            found_with_rerank = False
            retrieved_refs_rerank = []
            
            for chunk in reranked_chunks:
                ref = chunk.get('sefaria_ref', '')
                retrieved_refs_rerank.append(ref)
                if target in ref:
                    found_with_rerank = True
            
            if found_with_rerank:
                score_with_rerank += 1
                print(f"✅ WITH RERANK: Found '{target}' in top {RERANK_TOP_K}")
            else:
                print(f"❌ WITH RERANK: Expected '{target}' not in top {RERANK_TOP_K}")
                print(f"   --> Got: {retrieved_refs_rerank[:5]}")
        
        print("-" * 60)

    print(f"\n🎯 RESULTS:")
    print(f"   WITHOUT Reranking: {score_without_rerank}/{len(TEST_CASES)}")
    if ENABLE_RERANKING:
        print(f"   WITH Reranking:    {score_with_rerank}/{len(TEST_CASES)}")
        improvement = score_with_rerank - score_without_rerank
        print(f"   Improvement:       {'+' if improvement >= 0 else ''}{improvement}")

if __name__ == "__main__":
    run_test()