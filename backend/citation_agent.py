"""
Citation Agent for Torah AI

A cleaner approach to citations that:
1. Uses XML-style <cite> tags with optional excerpt attribute for partial translations
2. Streams paragraph-by-paragraph (on line breaks) for smooth animations
3. Parses and hydrates citations with actual source text and translation
4. Supports highlighting specific Hebrew excerpts that correspond to translations
"""

import re
import logging
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
CHAT_MODEL = "grok-4-1-fast-non-reasoning"


def build_citation_prompt(query: str, context: str, source_refs: list[str]) -> tuple[str, str]:
    """
    Build system and user prompts that instruct the LLM to use <cite> tags.
    LLM now only selects sources and excerpts - NO translation required.
    """
    refs_list = "\n".join(f"- {ref}" for ref in source_refs[:25])

    system_prompt = """You are a Torah scholar assistant specializing in Shulchan Arukh and its commentaries.

CITATION FORMAT:
When citing a source, use this self-closing XML tag format:

<cite ref="EXACT_REFERENCE" highlight='RELEVANT_HEBREW_TEXT'/>

The highlight attribute pinpoints the SPECIFIC Hebrew text that supports your point.
This is the PREFERRED citation style - it shows scholarly precision.

OMIT highlight ONLY when the ENTIRE source text is directly relevant (less common).

HIGHLIGHT GUIDELINES:
- Use SINGLE QUOTES (') around Hebrew text
- Include 8-30 Hebrew words - enough for context
- Copy the EXACT Hebrew from the source
- Think: "What specific phrase answers the user's question?"

EXAMPLES:

1. Highlighting the relevant portion:
   Regarding the custom of eating dairy <cite ref="Mishnah Berurah 187:4" highlight='אוכלים מאכלי חלב ואח״כ מאכול בשר'/>

2. Another highlighted citation (commentary on Shulchan Arukh):
   The Ba'er Hetev explains <cite ref="Ba'er Hetev on Shulchan Arukh, Orach Chayim 1:5" highlight='צריך להזהר מאוד בזה'/>

3. Full source citation (use sparingly - only when ALL text is relevant):
   <cite ref="Shulchan Arukh, Orach Chayim 1:1"/>

CRITICAL RULES:
1. Use EXACT reference strings from the AVAILABLE REFERENCES list
2. Copy reference strings EXACTLY as listed - do not modify or construct them
3. DEFAULT to using highlight - ask yourself "which specific words support this?"
4. Use SINGLE QUOTES (') not double quotes (") for highlight
5. Each source should be cited at most once
6. Cite no more than 5 sources unless clearly necessary
7. Write in flowing paragraphs with line breaks between ideas
8. Always cite sources when stating halachic rulings
9. If the context doesn't contain enough information, say so clearly

STRUCTURE:
- Start with a brief introduction
- Weave citations naturally into your explanation
- Use line breaks between paragraphs for readability
- Conclude with practical guidance if applicable"""

    user_prompt = f"""Here are the relevant Torah sources for this question:

{context}

---

AVAILABLE REFERENCES (use these exact strings):
{refs_list}

---

USER'S QUESTION: {query}

Please provide a helpful, accurate response with citations. Remember: just use <cite ref="..."/> tags - translations will be added automatically."""

    return system_prompt, user_prompt


async def stream_with_citations(
    query: str,
    context: str,
    hydrated_chunks: list[dict],
    xai_api_key: str,
    openrouter_api_key: str
) -> AsyncIterator[dict]:
    """
    Stream LLM response with paragraph-based buffering and citation parsing.

    Yields events:
    - {"type": "paragraph", "content": "text..."} - A complete paragraph
    - {"type": "citation", "ref": "...", "context": "...", "hebrew": "...", "english": "...", "hebrew_excerpt": "..."}
    - {"type": "done"}
    """
    from xai_sdk import Client as XAIClient
    from xai_sdk.chat import user, system

    # Step 1: Extract source refs from chunks
    source_refs = []
    for chunk in hydrated_chunks:
        ref = chunk.get("ref", "")
        if ref and ref not in source_refs:
            source_refs.append(ref)
        # Also collect commentary refs
        for comm in chunk.get("commentaries", []):
            comm_ref = comm.get("ref", "")
            if comm_ref and comm_ref not in source_refs:
                source_refs.append(comm_ref)

    # Step 2: Build prompts
    system_prompt, user_prompt = build_citation_prompt(query, context, source_refs)

    # Step 3: Stream from xAI
    try:
        # Create xAI client and chat
        xai_client = XAIClient(api_key=xai_api_key)
        chat = xai_client.chat.create(
            model=CHAT_MODEL,
            temperature=0.3,
            max_tokens=3000
        )

        chat.append(system(system_prompt))
        chat.append(user(user_prompt))

        # Paragraph buffer for streaming
        buffer = ""

        # Note: chat.stream() is a sync generator, use regular for loop
        for response, chunk in chat.stream():
            if not chunk.content:
                continue

            buffer += chunk.content

            # Check for paragraph breaks (double newline or single newline for simpler streaming)
            # We'll emit on single newlines for responsiveness
            while '\n' in buffer:
                # Find the line break
                idx = buffer.index('\n')
                paragraph = buffer[:idx].strip()
                buffer = buffer[idx + 1:]

                if paragraph:
                    # Parse this paragraph for citations and emit
                    async for event in parse_and_emit_paragraph(paragraph, hydrated_chunks, openrouter_api_key):
                        yield event

        # Emit any remaining content in buffer
        if buffer.strip():
            async for event in parse_and_emit_paragraph(buffer.strip(), hydrated_chunks, openrouter_api_key):
                yield event

        yield {"type": "done"}

    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield {"type": "paragraph", "content": f"Error generating response: {e}"}
        yield {"type": "done"}


def find_best_source_match(ref: str, hydrated_chunks: list[dict]) -> Optional[dict]:
    """
    Find the best matching chunk for a reference that isn't in the chunks list.
    Handles various reference formats intelligently.

    Priority:
    1. Look for similar Sefaria refs (e.g., "Mishnah Berurah 1:1" for "Mishnah Berurah 1:2")
    2. Match by book name for any seif in that siman
    3. Fall back to base SA seif
    """
    # Try to extract the book name and numbers from the ref
    # Common formats:
    # - "Mishnah Berurah 1:3"
    # - "Ba'er Hetev on Shulchan Arukh, Orach Chayim 1:3"
    # - "Shulchan Arukh, Orach Chayim 1:1"

    # Strategy 1: Find any ref from the same book in the same siman
    # Extract siman number from ref
    siman_match = re.search(r'(\d+):\d+$', ref)
    if siman_match:
        siman = siman_match.group(1)
        # Find book name (everything before the numbers)
        book_part = re.sub(r'\s*\d+:\d+$', '', ref)

        # Look for any chunk ref from the same book and siman
        for chunk in hydrated_chunks:
            chunk_ref = chunk.get("ref", "")
            if chunk_ref.startswith(book_part) and f" {siman}:" in chunk_ref:
                return chunk

    # Strategy 2: Match by book title alone (first available)
    for chunk in hydrated_chunks:
        chunk_ref = chunk.get("ref", "")
        # Check if the book name matches
        if "Mishnah Berurah" in ref and "Mishnah Berurah" in chunk_ref:
            return chunk
        if "Ba'er Hetev" in ref and "Ba'er Hetev" in chunk_ref:
            return chunk
        if "Shulchan Arukh" in ref and chunk_ref.startswith("Shulchan Arukh"):
            return chunk

    # Strategy 3: Try partial match
    for chunk in hydrated_chunks:
        chunk_ref = chunk.get("ref", "")
        if ref in chunk_ref or chunk_ref in ref:
            return chunk

    return None


async def parse_and_emit_paragraph(
    text: str,
    hydrated_chunks: list[dict],
    openrouter_api_key: str
) -> AsyncIterator[dict]:
    """
    Parse a paragraph for <cite> tags, translate them, and emit events.
    Citations are emitted AFTER translation completes.
    """
    from translator import translate_source, TranslationRequest

    # Updated regex for self-closing tags
    cite_pattern = re.compile(
        r'''<cite\s+ref="([^"]+)"(?:\s+highlight=(?:'((?:(?!'\s*/>).)*)'|"((?:(?!"\s*/>).)*)")?)?\s*/>''',
        re.DOTALL
    )

    last_end = 0

    for match in cite_pattern.finditer(text):
        # Emit any text before this citation
        before_text = text[last_end:match.start()].strip()
        if before_text:
            yield {"type": "paragraph", "content": before_text}

        # Extract citation details
        ref = match.group(1)
        hebrew_highlight = match.group(2) or match.group(3)

        # Look up the source chunk
        source_chunk = None

        # First, try to find exact match in main chunks
        for chunk in hydrated_chunks:
            if chunk.get("ref") == ref:
                source_chunk = chunk
                break

        # If not found, check commentaries
        if not source_chunk:
            for chunk in hydrated_chunks:
                for comm in chunk.get("commentaries", []):
                    if comm.get("ref") == ref:
                        # Create a pseudo-chunk for the commentary
                        source_chunk = {
                            "ref": comm.get("ref", ""),
                            "content": comm.get("text", ""),
                            "book": comm.get("commentator", "Commentary"),
                            "context_text": None
                        }
                        break
                if source_chunk:
                    break

        # If still not found, try fallback matching
        if not source_chunk:
            source_chunk = find_best_source_match(ref, hydrated_chunks)

        if source_chunk:
            # Create translation request
            translation_request = TranslationRequest(
                hebrew_text=source_chunk.get("content", ""),
                hebrew_highlight=hebrew_highlight,
                source_ref=ref,
                book=source_chunk.get("book", ""),
                context_text=source_chunk.get("context_text")
            )

            # Get translation (non-streaming)
            translation_result = await translate_source(
                translation_request,
                openrouter_api_key
            )

            # Emit citation with translation
            yield {
                "type": "citation",
                "ref": ref,
                "context": translation_result.translation if translation_result.success else "",
                "hebrew": source_chunk.get("content", ""),
                "english": "",  # No Sefaria translation anymore
                "book": source_chunk.get("book", ""),
                "hebrew_highlight": hebrew_highlight,
                "translation_success": translation_result.success
            }
        else:
            logger.warning(f"Citation ref not found in chunks: {ref}")
            yield {"type": "paragraph", "content": f"[Citation: {ref}]"}

        last_end = match.end()

    # Emit any remaining text
    remaining = text[last_end:].strip()
    if remaining:
        yield {"type": "paragraph", "content": remaining}


# --- MAIN EXPORT FUNCTION ---

async def generate_response_with_citations_stream(
    query: str,
    context: str,
    retrieved_chunks: list[dict],
    xai_api_key: str,
    openrouter_api_key: str
) -> AsyncIterator[dict]:
    """
    Main entry point for citation-aware response generation.

    Args:
        query: User's question
        context: Formatted RAG context
        retrieved_chunks: Hydrated chunks from vector search
        xai_api_key: API key for xAI
        openrouter_api_key: API key for OpenRouter (used for translations)

    Yields:
        Events for frontend rendering:
        - {"type": "paragraph", "content": "..."}
        - {"type": "citation", "ref": "...", "context": "...", "hebrew": "...", "english": "...", "hebrew_excerpt": "..."}
        - {"type": "done"}
    """
    async for event in stream_with_citations(
        query=query,
        context=context,
        hydrated_chunks=retrieved_chunks,
        xai_api_key=xai_api_key,
        openrouter_api_key=openrouter_api_key
    ):
        yield event
