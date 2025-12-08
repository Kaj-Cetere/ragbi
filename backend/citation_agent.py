"""
Citation Agent for Torah AI

A cleaner approach to citations that:
1. Pre-fetches seifim with Hebrew text and English translation
2. Uses XML-style <cite> tags in the LLM response
3. Streams paragraph-by-paragraph (on line breaks) for smooth animations
4. Parses and hydrates citations with actual source text and translation
"""

import os
import re
import json
import logging
import requests
from typing import AsyncIterator, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
CHAT_MODEL = "google/gemini-2.5-flash-lite-preview-09-2025"


@dataclass
class SourceText:
    """Pre-fetched source text with Hebrew and English."""
    ref: str
    hebrew: str
    english: str
    book: str
    siman: Optional[int] = None
    seif: Optional[int] = None


def fetch_sefaria_text(ref: str) -> Optional[dict]:
    """Fetch text from Sefaria API with Hebrew and English translation."""
    try:
        # Clean the reference for API call
        api_ref = ref.replace(" ", "_")
        url = f"https://www.sefaria.org/api/texts/{api_ref}"
        
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Handle various response formats
        hebrew = data.get("he", "")
        english = data.get("text", "")
        
        # If it's a list, join it
        if isinstance(hebrew, list):
            hebrew = " ".join(str(h) for h in hebrew if h)
        if isinstance(english, list):
            english = " ".join(str(e) for e in english if e)
        
        return {
            "hebrew": hebrew,
            "english": english,
            "ref": ref,
            "title": data.get("indexTitle", ref)
        }
    except Exception as e:
        logger.warning(f"Failed to fetch from Sefaria for {ref}: {e}")
        return None


def build_source_cache(hydrated_chunks: list[dict]) -> dict[str, SourceText]:
    """
    Pre-fetch all source texts from retrieved chunks.
    Returns a dict mapping ref -> SourceText with Hebrew and English.
    """
    cache: dict[str, SourceText] = {}
    
    for chunk in hydrated_chunks:
        ref = chunk.get("ref", "")
        if not ref or ref in cache:
            continue
        
        # First, try to use content from the chunk itself
        hebrew = chunk.get("content", "")
        english = ""
        
        # Try to fetch English translation from Sefaria
        sefaria_data = fetch_sefaria_text(ref)
        if sefaria_data:
            # Prefer Sefaria Hebrew if available (might be cleaner)
            if sefaria_data.get("hebrew"):
                hebrew = sefaria_data["hebrew"]
            english = sefaria_data.get("english", "")
        
        cache[ref] = SourceText(
            ref=ref,
            hebrew=hebrew,
            english=english,
            book=chunk.get("book", ""),
            siman=chunk.get("siman"),
            seif=chunk.get("seif")
        )
        
        # Also cache commentaries
        for comm in chunk.get("commentaries", []):
            comm_ref = f"{ref}:{comm.get('commentator', 'Commentary')}"
            if comm_ref not in cache:
                cache[comm_ref] = SourceText(
                    ref=comm_ref,
                    hebrew=comm.get("text", ""),
                    english="",  # Commentaries often don't have translations
                    book=comm.get("commentator", "")
                )
    
    logger.info(f"Built source cache with {len(cache)} entries")
    return cache


def build_citation_prompt(query: str, context: str, source_refs: list[str]) -> tuple[str, str]:
    """
    Build system and user prompts that instruct the LLM to use <cite> tags.
    Returns (system_prompt, user_prompt).
    """
    # Format available references for the prompt
    refs_list = "\n".join(f"- {ref}" for ref in source_refs[:10])
    
    system_prompt = """You are a Torah scholar assistant specializing in Shulchan Arukh and its commentaries.

CITATION FORMAT:
When you want to cite a source, use this XML tag format:
<cite ref="EXACT_REFERENCE">your brief explanation of this source</cite>

Example:
The Shulchan Arukh teaches that <cite ref="Shulchan Arukh, Orach Chaim 1:1">one should arise with vigor to serve the Creator</cite>. This applies every morning.

IMPORTANT RULES:
1. Use the EXACT reference strings provided in the context (copy-paste them)
2. The text inside <cite> tags is YOUR explanation - I will display the actual source text separately
3. Keep your explanations clear and accessible
4. Each source should be cited at most once
5. Write in flowing paragraphs with line breaks between ideas
6. Always cite sources when stating halachic rulings
7. If the context doesn't contain enough information, say so clearly

STRUCTURE:
- Start with a brief introduction
- Weave citations naturally into your explanation
- Use line breaks (double newlines) between paragraphs for readability
- Conclude with practical guidance if applicable"""

    user_prompt = f"""Here are the relevant Torah sources for this question:

{context}

---

AVAILABLE REFERENCES (use these exact strings for citations):
{refs_list}

---

USER'S QUESTION: {query}

Please provide a helpful, accurate response with citations using the <cite ref="...">explanation</cite> format."""

    return system_prompt, user_prompt


async def stream_with_citations(
    query: str,
    context: str,
    hydrated_chunks: list[dict],
    openrouter_api_key: str
) -> AsyncIterator[dict]:
    """
    Stream LLM response with paragraph-based buffering and citation parsing.
    
    Yields events:
    - {"type": "paragraph", "content": "text..."} - A complete paragraph
    - {"type": "citation", "ref": "...", "context": "...", "hebrew": "...", "english": "..."}
    - {"type": "done"}
    """
    # Step 1: Pre-fetch all source texts
    source_cache = build_source_cache(hydrated_chunks)
    source_refs = list(source_cache.keys())
    
    # Step 2: Build prompts
    system_prompt, user_prompt = build_citation_prompt(query, context, source_refs)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Step 3: Stream from OpenRouter
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://kesher.ai",
                "X-Title": "Kesher AI"
            },
            json={
                "model": CHAT_MODEL,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 3000,
                "stream": True
            },
            stream=True,
            timeout=120
        )
        response.raise_for_status()
        
        # Paragraph buffer for streaming
        buffer = ""
        
        for line in response.iter_lines():
            if not line:
                continue
                
            line = line.decode('utf-8')
            if not line.startswith('data: '):
                continue
                
            data = line[6:]
            if data == '[DONE]':
                break
                
            try:
                chunk = json.loads(data)
                if 'choices' not in chunk or len(chunk['choices']) == 0:
                    continue
                    
                delta = chunk['choices'][0].get('delta', {})
                content = delta.get('content', '')
                
                if not content:
                    continue
                
                buffer += content
                
                # Check for paragraph breaks (double newline or single newline for simpler streaming)
                # We'll emit on single newlines for responsiveness
                while '\n' in buffer:
                    # Find the line break
                    idx = buffer.index('\n')
                    paragraph = buffer[:idx].strip()
                    buffer = buffer[idx + 1:]
                    
                    if paragraph:
                        # Parse this paragraph for citations and emit
                        async for event in parse_and_emit_paragraph(paragraph, source_cache):
                            yield event
                            
            except json.JSONDecodeError:
                continue
        
        # Emit any remaining content in buffer
        if buffer.strip():
            async for event in parse_and_emit_paragraph(buffer.strip(), source_cache):
                yield event
        
        yield {"type": "done"}
        
    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield {"type": "paragraph", "content": f"Error generating response: {e}"}
        yield {"type": "done"}


async def parse_and_emit_paragraph(
    text: str,
    source_cache: dict[str, SourceText]
) -> AsyncIterator[dict]:
    """
    Parse a paragraph for <cite> tags and emit appropriate events.
    
    For text segments: yields {"type": "paragraph", "content": "..."}
    For citations: yields {"type": "citation", "ref": "...", "context": "...", "hebrew": "...", "english": "..."}
    """
    # Regex to find <cite ref="...">...</cite> tags
    cite_pattern = re.compile(r'<cite\s+ref="([^"]+)">(.*?)</cite>', re.DOTALL)
    
    last_end = 0
    
    for match in cite_pattern.finditer(text):
        # Emit any text before this citation
        before_text = text[last_end:match.start()].strip()
        if before_text:
            yield {"type": "paragraph", "content": before_text}
        
        # Extract citation details
        ref = match.group(1)
        context_text = match.group(2).strip()
        
        # Look up the source in cache
        source = source_cache.get(ref)
        
        if source:
            yield {
                "type": "citation",
                "ref": ref,
                "context": context_text,
                "hebrew": source.hebrew,
                "english": source.english,
                "book": source.book
            }
        else:
            # Source not in cache - try to find a partial match
            matched_source = None
            for cached_ref, cached_source in source_cache.items():
                if ref in cached_ref or cached_ref in ref:
                    matched_source = cached_source
                    break
            
            if matched_source:
                yield {
                    "type": "citation",
                    "ref": ref,
                    "context": context_text,
                    "hebrew": matched_source.hebrew,
                    "english": matched_source.english,
                    "book": matched_source.book
                }
            else:
                # Fallback: emit as a paragraph with the citation inline
                yield {"type": "paragraph", "content": f"{context_text} ({ref})"}
        
        last_end = match.end()
    
    # Emit any remaining text after the last citation
    remaining = text[last_end:].strip()
    if remaining:
        yield {"type": "paragraph", "content": remaining}


# --- MAIN EXPORT FUNCTION ---

async def generate_response_with_citations_stream(
    query: str,
    context: str,
    retrieved_chunks: list[dict],
    openrouter_api_key: str
) -> AsyncIterator[dict]:
    """
    Main entry point for citation-aware response generation.
    
    Args:
        query: User's question
        context: Formatted RAG context
        retrieved_chunks: Hydrated chunks from vector search
        openrouter_api_key: API key for OpenRouter
    
    Yields:
        Events for frontend rendering:
        - {"type": "paragraph", "content": "..."}
        - {"type": "citation", "ref": "...", "context": "...", "hebrew": "...", "english": "..."}
        - {"type": "done"}
    """
    async for event in stream_with_citations(
        query=query,
        context=context,
        hydrated_chunks=retrieved_chunks,
        openrouter_api_key=openrouter_api_key
    ):
        yield event
