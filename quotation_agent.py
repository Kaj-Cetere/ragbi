"""
Pydantic-AI Agent for Quoting Shulchan Aruch and Commentaries

This module provides an AI agent that produces structured responses with
embedded quotations from Shulchan Aruch and commentaries, allowing perfect
control over quote placement within the response.
"""

import os
from typing import Optional, Literal
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from supabase import Client
from enricher import fetch_commentaries_for_ref


# --- STRUCTURED OUTPUT MODELS ---

class ResponseSegment(BaseModel):
    """A segment of the response - either text or a quotation reference."""
    segment_type: Literal["text", "quote_seif", "quote_commentary"] = Field(
        description="Type of segment: 'text' for prose, 'quote_seif' for Shulchan Aruch quote, 'quote_commentary' for commentary quote"
    )
    content: str = Field(
        description="For 'text': the prose content. For quotes: the Sefaria reference (e.g., 'Shulchan Arukh, Orach Chayim 229:1')"
    )
    commentator: Optional[str] = Field(
        default=None,
        description="For 'quote_commentary' only: the commentator name (e.g., 'Mishnah Berurah')"
    )


class StructuredQuotationResponse(BaseModel):
    """A structured response with interleaved text and quotation references."""
    segments: list[ResponseSegment] = Field(
        description="Ordered list of response segments (text and quotes interleaved)"
    )


# --- AGENT DEFINITION ---

# Create the agent with structured output
quotation_agent = Agent(
    'openrouter:x-ai/grok-4.1-fast',
    output_type=StructuredQuotationResponse,
    system_prompt="""You are a Torah scholar assistant. Create responses with quotations woven naturally into the text.

OUTPUT FORMAT:
Return a list of segments, alternating between text and quotes. Each segment is either:
- {"segment_type": "text", "content": "Your explanation here..."}
- {"segment_type": "quote_seif", "content": "Shulchan Arukh, Orach Chayim 229:1"}
- {"segment_type": "quote_commentary", "content": "Shulchan Arukh, Orach Chayim 229:1", "commentator": "Mishnah Berurah"}

EXAMPLE STRUCTURE:
[
  {"segment_type": "text", "content": "The Shulchan Arukh specifies the correct blessing for seeing a rainbow:"},
  {"segment_type": "quote_seif", "content": "Shulchan Arukh, Orach Chayim 229:1"},
  {"segment_type": "text", "content": "The Mishnah Berurah adds important practical guidance:"},
  {"segment_type": "quote_commentary", "content": "Shulchan Arukh, Orach Chayim 229:1", "commentator": "Mishnah Berurah"},
  {"segment_type": "text", "content": "In practice, recite the blessing immediately upon seeing a clear rainbow, without staring excessively."}
]

RULES:
- Use EXACT references from the provided context (e.g., "Shulchan Arukh, Orach Chayim 229:1")
- Each quote should be preceded and followed by explanatory text
- Keep text segments concise - let the sources speak
- Quote each source only ONCE
- Always start and end with text segments"""
)


# --- HELPER FUNCTIONS ---

def fetch_quote_text(
    ref: str,
    retrieved_chunks: list[dict],
    supabase: Client
) -> str:
    """Fetch the text content for a Shulchan Arukh reference."""
    # First, check if this ref is in our retrieved chunks
    for chunk in retrieved_chunks:
        if chunk.get('ref') == ref or chunk.get('sefaria_ref') == ref:
            return chunk.get('content', '')
    
    # If not in chunks, try to fetch from database
    try:
        result = supabase.table('sefaria_text_chunks') \
            .select('content, sefaria_ref, metadata') \
            .eq('sefaria_ref', ref) \
            .limit(1) \
            .execute()
        
        if result.data and len(result.data) > 0:
            return result.data[0].get('content', '')
    except Exception:
        pass
    
    return f"[Text for {ref} not available]"


def fetch_commentary_text(
    base_ref: str,
    commentator_name: str,
    retrieved_chunks: list[dict]
) -> tuple[str, str]:
    """
    Fetch commentary text for a reference.
    Returns (commentary_text, actual_commentator_name).
    """
    # First, check if this commentary is in our retrieved chunks
    for chunk in retrieved_chunks:
        chunk_ref = chunk.get('ref') or chunk.get('sefaria_ref', '')
        if chunk_ref == base_ref or base_ref in chunk_ref:
            # Check if commentaries are already hydrated
            commentaries = chunk.get('commentaries', [])
            for comm in commentaries:
                if commentator_name.lower() in comm.get('commentator', '').lower():
                    return comm.get('text', ''), comm.get('commentator', commentator_name)
    
    # If not in chunks, try to fetch from Sefaria API
    try:
        book_title = ""
        if "Orach Chayim" in base_ref:
            book_title = "Shulchan Arukh, Orach Chayim"
        elif "Yoreh De'ah" in base_ref:
            book_title = "Shulchan Arukh, Yoreh De'ah"
        elif "Choshen Mishpat" in base_ref:
            book_title = "Shulchan Arukh, Choshen Mishpat"
        elif "Even HaEzer" in base_ref:
            book_title = "Shulchan Arukh, Even HaEzer"
        
        commentaries = fetch_commentaries_for_ref(base_ref, book_title)
        
        for comm in commentaries:
            if commentator_name.lower() in comm.get('commentator', '').lower():
                return comm.get('text', ''), comm.get('commentator', commentator_name)
    except Exception:
        pass
    
    return f"[Commentary from {commentator_name} on {base_ref} not available]", commentator_name


def render_segment(
    segment: ResponseSegment,
    retrieved_chunks: list[dict],
    supabase: Client
) -> str:
    """Render a single segment to formatted text."""
    if segment.segment_type == "text":
        return segment.content
    
    elif segment.segment_type == "quote_seif":
        text = fetch_quote_text(segment.content, retrieved_chunks, supabase)
        return f"\n\n**{segment.content}**\n\n> {text}\n\n"
    
    elif segment.segment_type == "quote_commentary":
        text, commentator = fetch_commentary_text(
            segment.content,
            segment.commentator or "Commentary",
            retrieved_chunks
        )
        return f"\n\n**{commentator}** on {segment.content}\n\n> {text}\n\n"
    
    return ""


# --- MAIN GENERATION FUNCTIONS ---

async def generate_response_with_quotations(
    query: str,
    context: str,
    retrieved_chunks: list[dict],
    supabase: Client,
    openrouter_api_key: str
) -> str:
    """
    Generate a response with embedded quotations using structured output.
    
    Args:
        query: The user's question
        context: The RAG context (formatted chunks with commentaries)
        retrieved_chunks: The raw retrieved chunks from vector search
        supabase: Supabase client for database access
        openrouter_api_key: OpenRouter API key for the model
    
    Returns:
        A string response with inline quotation blocks
    """
    # Set OpenRouter API key
    os.environ['OPENROUTER_API_KEY'] = openrouter_api_key
    
    # Build the user prompt with context
    user_prompt = f"""Here is relevant context from Torah sources:

{context}

---

User's Question: {query}

Create a response with quotations woven naturally into the text. Use the exact references from the context above."""
    
    # Run the agent
    result = await quotation_agent.run(user_prompt)
    
    # Render the structured response
    structured_response: StructuredQuotationResponse = result.output
    
    rendered_parts = []
    for segment in structured_response.segments:
        rendered_parts.append(render_segment(segment, retrieved_chunks, supabase))
    
    return "".join(rendered_parts)


async def generate_response_with_quotations_stream(
    query: str,
    context: str,
    retrieved_chunks: list[dict],
    supabase: Client,
    openrouter_api_key: str
):
    """
    Generate a response with embedded quotations, yielding segments as they're rendered.
    
    Note: The model response itself is not streamed (structured output requires full response),
    but the rendering of segments is streamed for a responsive UI.
    
    Yields:
        String chunks of the rendered response
    """
    # Set OpenRouter API key
    os.environ['OPENROUTER_API_KEY'] = openrouter_api_key
    
    # Build the user prompt with context
    user_prompt = f"""Here is relevant context from Torah sources:

{context}

---

User's Question: {query}

Create a response with quotations woven naturally into the text. Use the exact references from the context above."""
    
    # Run the agent (not streamed - structured output needs full response)
    result = await quotation_agent.run(user_prompt)
    
    # Get the structured response
    structured_response: StructuredQuotationResponse = result.output
    
    # Yield each rendered segment for a streaming effect
    for segment in structured_response.segments:
        rendered = render_segment(segment, retrieved_chunks, supabase)
        yield rendered


# --- SYNCHRONOUS WRAPPER FOR STREAMLIT ---

def generate_response_with_quotations_sync(
    query: str,
    context: str,
    retrieved_chunks: list[dict],
    supabase: Client,
    openrouter_api_key: str
) -> str:
    """
    Synchronous wrapper for generate_response_with_quotations.
    
    Returns:
        A string response with inline quotation blocks
    """
    import asyncio
    import nest_asyncio
    
    nest_asyncio.apply()
    return asyncio.run(generate_response_with_quotations(
        query, context, retrieved_chunks, supabase, openrouter_api_key
    ))
