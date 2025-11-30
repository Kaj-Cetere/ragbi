import os
from dotenv import load_dotenv
from openrouter import OpenRouter
from supabase import create_client

load_dotenv()

# --- SETUP ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

client = OpenRouter(api_key=OPENROUTER_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
TARGET_DIMENSIONS = 1536

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
    score = 0
    print(f"🚀 Starting Test on {len(TEST_CASES)} questions...\n")

    for case in TEST_CASES:
        query = case["question"]
        target = case["expected_ref"]
        
        print(f"❓ Question: {query}")
        
        # 1. Get Embedding
        vector = get_embedding(query)
        
        # 2. Search Supabase (Top 5)
        response = supabase.rpc("match_documents", {
            "query_embedding": vector,
            "match_threshold": 0.5, # Lower threshold to catch more potential matches
            "match_count": 5
        }).execute()
        
        # 3. Check if the "Right Answer" is in the results
        found = False
        retrieved_refs = []
        
        for hit in response.data:
            ref = hit['sefaria_ref']
            retrieved_refs.append(ref)
            
            # Loose match check (e.g. if target is "1:1" and hit is "1:1")
            # You might need to adjust logic depending on exact string format in DB
            if target in ref: 
                found = True
                break
        
        if found:
            print(f"✅ PASS: Found '{target}'")
            score += 1
        else:
            print(f"❌ FAIL: Expected '{target}'")
            print(f"   --> Got these instead: {retrieved_refs}")
        
        print("-" * 30)

    print(f"\n🎯 Final Score: {score}/{len(TEST_CASES)}")

if __name__ == "__main__":
    run_test()