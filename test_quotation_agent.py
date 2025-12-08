"""
Test script for the quotation agent

This script demonstrates how the pydantic-ai quotation agent works
by running a simple query and showing the structured output.
"""

import os
import asyncio
from dotenv import load_dotenv
from supabase import create_client
from quotation_agent import (
    generate_response_with_quotations,
    format_quotation_response,
    QuotationResponse
)

load_dotenv()

# Setup
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


async def test_quotation_agent():
    """Test the quotation agent with a sample query."""
    
    # Sample query
    query = "What does the Shulchan Arukh say about saying Hashem's name?"
    
    # Mock retrieved chunks (in real usage, these come from RAG retrieval)
    # For testing, we'll fetch a few chunks from the database
    print("🔍 Fetching sample chunks from database...")
    result = supabase.table('sefaria_text_chunks') \
        .select('content, sefaria_ref, metadata') \
        .eq('metadata->>book', 'Shulchan Arukh, Orach Chayim') \
        .limit(5) \
        .execute()
    
    retrieved_chunks = []
    for chunk in result.data:
        metadata = chunk.get('metadata', {})
        retrieved_chunks.append({
            'ref': chunk.get('sefaria_ref'),
            'sefaria_ref': chunk.get('sefaria_ref'),
            'content': chunk.get('content'),
            'book': metadata.get('book', ''),
            'siman': metadata.get('siman'),
            'seif': metadata.get('seif'),
            'commentaries': [],
            'hydrated': False
        })
    
    # Build simple context
    context = "\n\n".join([
        f"**{chunk['ref']}**\n{chunk['content']}"
        for chunk in retrieved_chunks
    ])
    
    print(f"\n📝 Query: {query}")
    print(f"\n📚 Retrieved {len(retrieved_chunks)} chunks")
    print("\n🤖 Running quotation agent...\n")
    
    # Run the agent
    try:
        response = await generate_response_with_quotations(
            query=query,
            context=context,
            retrieved_chunks=retrieved_chunks,
            supabase=supabase,
            openrouter_api_key=OPENROUTER_API_KEY
        )
        
        print("=" * 80)
        print("AGENT RESPONSE")
        print("=" * 80)
        print()
        
        # Show structured output
        print("📊 STRUCTURED OUTPUT:")
        print(f"  - Main answer length: {len(response.answer)} chars")
        print(f"  - Seif quotations: {len(response.seif_quotations)}")
        print(f"  - Commentary quotations: {len(response.commentary_quotations)}")
        print()
        
        # Show formatted response
        print("📜 FORMATTED RESPONSE:")
        print("-" * 80)
        formatted = format_quotation_response(response)
        print(formatted)
        print("-" * 80)
        
        # Show raw quotations
        if response.seif_quotations:
            print("\n📖 SEIF QUOTATIONS (RAW):")
            for i, quote in enumerate(response.seif_quotations, 1):
                print(f"\n  {i}. {quote.ref}")
                print(f"     Book: {quote.book}")
                print(f"     Siman: {quote.siman}, Seif: {quote.seif}")
                print(f"     Text: {quote.text[:100]}...")
        
        if response.commentary_quotations:
            print("\n💬 COMMENTARY QUOTATIONS (RAW):")
            for i, quote in enumerate(response.commentary_quotations, 1):
                print(f"\n  {i}. {quote.commentator} on {quote.ref}")
                print(f"     Text: {quote.text[:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Testing Pydantic-AI Quotation Agent")
    print("=" * 80)
    asyncio.run(test_quotation_agent())
