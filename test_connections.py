import os
import json
from dotenv import load_dotenv
from openrouter import OpenRouter
from supabase import create_client, Client

load_dotenv()

# Test OpenRouter
print("Testing OpenRouter API...")
try:
    client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))
    response = client.embeddings.generate(
        model="qwen/qwen3-embedding-8b",
        input=["Test text for embedding"],
        dimensions=1536
    )
    embedding = response.data[0].embedding
    print(f"✓ OpenRouter API working (embedding dimension: {len(embedding)})")
except Exception as e:
    print(f"✗ OpenRouter API error: {e}")

# Test Supabase
print("\nTesting Supabase connection...")
try:
    supabase: Client = create_client(
        os.getenv("SUPABASE_URL"), 
        os.getenv("SUPABASE_KEY")
    )
    
    # Test a simple query
    result = supabase.table("sefaria_text_chunks").select("count").execute()
    print(f"✓ Supabase connection working (current count: {result.data})")
except Exception as e:
    print(f"✗ Supabase connection error: {e}")
