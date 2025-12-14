"""
Citation Agent for Torah AI

A cleaner approach to citations that:
1. Pre-fetches seifim with Hebrew text and English translation
2. Uses XML-style <cite> tags with optional excerpt attribute for partial translations
3. Streams paragraph-by-paragraph (on line breaks) for smooth animations
4. Parses and hydrates citations with actual source text and translation
5. Supports highlighting specific Hebrew excerpts that correspond to translations
"""

import os
import re
import json
import logging
import requests
import httpx
from typing import AsyncIterator, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
CHAT_MODEL = "grok-4-1-fast-non-reasoning"


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

        # Extract seif number from ref if it exists (e.g., "Shulchan Arukh, Orach Chayim 1:2" -> seif 2)
        seif_index = None
        if ":" in ref:
            try:
                # Get the part after the last space and extract seif number after the colon
                ref_parts = ref.rsplit(" ", 1)[-1]  # Get "1:2"
                if ":" in ref_parts:
                    seif_num = int(ref_parts.split(":")[-1])  # Get 2
                    seif_index = seif_num - 1  # Convert to 0-based index
            except (ValueError, IndexError):
                pass

        # If it's a list, extract the specific seif if we have an index, otherwise join all
        if isinstance(hebrew, list):
            if seif_index is not None and 0 <= seif_index < len(hebrew):
                hebrew = str(hebrew[seif_index]) if hebrew[seif_index] else ""
            else:
                hebrew = " ".join(str(h) for h in hebrew if h)
        if isinstance(english, list):
            if seif_index is not None and 0 <= seif_index < len(english):
                english = str(english[seif_index]) if english[seif_index] else ""
            else:
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


def build_source_cache(hydrated_chunks: list[dict], parallel: bool = True) -> dict[str, SourceText]:
    """
    Pre-fetch all source texts from retrieved chunks.
    Returns a dict mapping ref -> SourceText with Hebrew and English.

    Args:
        hydrated_chunks: List of chunks from retrieval
        parallel: If True, fetch Sefaria texts in parallel (much faster)
    """
    import concurrent.futures
    import time

    cache_start = time.time()
    cache: dict[str, SourceText] = {}
    refs_to_fetch: list[tuple[str, dict]] = []

    # First pass: collect refs and cache what we already have
    for chunk in hydrated_chunks:
        ref = chunk.get("ref", "")
        if not ref or ref in cache:
            continue

        refs_to_fetch.append((ref, chunk))

        # Also cache commentaries (these don't need Sefaria fetch)
        for comm in chunk.get("commentaries", []):
            commentator = comm.get('commentator', 'Commentary')
            comm_text = comm.get("text", "")
            comm_ref = comm.get("ref", "")

            if comm_ref and comm_ref not in cache:
                cache[comm_ref] = SourceText(
                    ref=comm_ref,
                    hebrew=comm_text,
                    english="",
                    book=commentator
                )

            fallback_ref = f"{ref}:{commentator}"
            if fallback_ref not in cache:
                cache[fallback_ref] = SourceText(
                    ref=comm_ref or fallback_ref,
                    hebrew=comm_text,
                    english="",
                    book=commentator
                )

    # Parallel fetch from Sefaria for main refs only
    if parallel and refs_to_fetch:
        def fetch_one(ref_chunk_tuple):
            ref, chunk = ref_chunk_tuple
            hebrew = chunk.get("content", "")
            english = ""

            sefaria_data = fetch_sefaria_text(ref)
            if sefaria_data:
                if sefaria_data.get("hebrew"):
                    hebrew = sefaria_data["hebrew"]
                english = sefaria_data.get("english", "")

            return ref, SourceText(
                ref=ref,
                hebrew=hebrew,
                english=english,
                book=chunk.get("book", ""),
                siman=chunk.get("siman"),
                seif=chunk.get("seif")
            )

        # Fetch in parallel with max 5 workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(fetch_one, refs_to_fetch))
            for ref, source in results:
                cache[ref] = source
    else:
        # Sequential fallback
        for ref, chunk in refs_to_fetch:
            hebrew = chunk.get("content", "")
            english = ""

            sefaria_data = fetch_sefaria_text(ref)
            if sefaria_data:
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

    cache_time = (time.time() - cache_start) * 1000
    logger.info(f"Built source cache with {len(cache)} entries in {cache_time:.0f}ms (parallel={parallel})")
    return cache


def build_citation_prompt(query: str, context: str, source_refs: list[str]) -> tuple[str, str]:
    """
    Build system and user prompts that instruct the LLM to use <cite> tags.
    Now includes excerpt attribute for partial translations.
    Returns (system_prompt, user_prompt).
    """
    # Format available references for the prompt (include more to cover commentaries)
    refs_list = "\n".join(f"- {ref}" for ref in source_refs[:25])

    system_prompt = """You are a Torah scholar assistant specializing in Shulchan Arukh and its commentaries.

CITATION FORMAT:
When citing a source, use this XML tag format:
<cite ref="EXACT_REFERENCE">ENGLISH_TRANSLATION</cite>
OR for partial excerpts:
<cite ref="EXACT_REFERENCE" excerpt='HEBREW_EXCERPT'>ENGLISH_TRANSLATION</cite>

IMPORTANT - USE SINGLE QUOTES FOR EXCERPT:
When including an excerpt, use SINGLE QUOTES (') around the Hebrew text, not double quotes (")
This is critical because Hebrew text often contains quotation marks (gershayim) as punctuation.

TRANSLATION PHILOSOPHY - TRANSLATE WHAT'S RELEVANT:
- GOAL: Translate the ENTIRE portion of the source that is RELEVANT to your point
- Sometimes the entire source is relevant - translate it all (no excerpt)
- Often only PART of the source is relevant - use excerpt to translate just that portion
- Be FLEXIBLE: The key question is "what part of this source supports my point?"
- When using excerpt, be LIBERAL and GENEROUS - include the full relevant portion
- Excerpts should be substantial (typically 10-20+ words) to give proper context
- NEVER use tiny excerpts (under 8 words) - either quote more generously or cite the whole source

GUIDING PRINCIPLE - RELEVANCE OVER LENGTH:
Don't base your decision on the total length of the source.
Base it on how much of the source is actually relevant to your answer.

EXAMPLES OF GOOD EXCERPT USAGE:
1. A seif discusses 5 different laws, but you only need 1 law → Use excerpt for that specific law
2. A Mishnah Berurah has a long explanation, but only the middle portion relates to your point → Excerpt that portion generously
3. You need to highlight a specific ruling within a longer passage → Excerpt the ruling with surrounding context
4. Half of a source is relevant, half is not → Excerpt the relevant half

WHEN TO TRANSLATE THE FULL SOURCE (no excerpt):
1. The ENTIRE source is relevant to your point - don't artificially limit it
2. Most or all of the source supports what you're saying - include it all
3. The source is inherently short (a brief statement) - no need to excerpt
4. You're giving the full context of a ruling or explanation

EXAMPLES:

1. FULL translation when entire source is relevant (NO EXCERPT):
   <cite ref="Shulchan Arukh, Orach Chayim 1:1">One should strengthen himself like a lion to arise in the morning for the service of his Creator</cite>

2. EXCERPT when only part is relevant (note: generous 15+ word excerpt with SINGLE QUOTES):
   <cite ref="Mishnah Berurah 494:14" excerpt='אוכלים מאכלי חלב ואח״כ מאכול בשר וצריכין להביא עמהם שתי לחמים ולהמתין בין חלב לבשר כדי שיעשה קינוח והדחה'>People eat dairy foods and then meat foods, and must bring two breads with them and wait between dairy and meat in order to perform rinsing and cleansing</cite>

3. FULL translation when the entire commentary is relevant (NO EXCERPT):
   <cite ref="Mishnah Berurah 1:1">The reason is so that one should not be embarrassed before people who mock him for his piety</cite>

4. EXCERPT when half the source is relevant (use a generous, complete portion):
   <cite ref="Shulchan Arukh, Orach Chayim 156:1" excerpt='אם אכל דבר שאינו קובע אכילה די בנטילת ידיו או במשמוש מפה ואכילת פת או מאכלי חלב'>If one ate something that is not a fixed meal, hand washing or wiping with a cloth is sufficient, along with eating bread or dairy foods</cite>

CRITICAL RULES:
1. Use EXACT reference strings from the AVAILABLE REFERENCES list
2. When using excerpt: Use SINGLE QUOTES (') not double quotes (")
3. PREFER full translations - only use excerpt for genuinely long sources
4. Excerpts must be substantial (8-10 words minimum) and meaningful
5. Provide complete, accurate ENGLISH TRANSLATIONS of the Hebrew
6. Each source should be cited at most once
7. Be RELATIVELY CONCISE - cite no more than 5 sources unless clearly necessary
8. Write in flowing paragraphs with line breaks between ideas
9. Always cite sources when stating halachic rulings
10. If the context doesn't contain enough information, say so clearly

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

Please provide a helpful, accurate response with citations.

REMINDER:
- Translate the ENTIRE portion of each source that is RELEVANT to your point
- Use excerpt when only PART of a source is relevant (often half or a few sentences)
- Be LIBERAL and GENEROUS with excerpts - include full context (typically 10-20+ words)
- When using excerpt, use SINGLE QUOTES: excerpt='...' not excerpt="..."
- If the entire source is relevant, translate it all (no excerpt needed)"""

    return system_prompt, user_prompt


async def stream_with_citations(
    query: str,
    context: str,
    hydrated_chunks: list[dict],
    xai_api_key: str
) -> AsyncIterator[dict]:
    """
    Stream LLM response with paragraph-based buffering and citation parsing.

    Yields events:
    - {"type": "source_cache_built", "duration_ms": ...} - Timing for source cache
    - {"type": "paragraph", "content": "text..."} - A complete paragraph
    - {"type": "citation", "ref": "...", "context": "...", "hebrew": "...", "english": "...", "hebrew_excerpt": "..."}
    - {"type": "done"}
    """
    import time
    from xai_sdk import Client as XAIClient
    from xai_sdk.chat import user, system

    # Step 1: Pre-fetch all source texts (now parallelized)
    cache_start = time.time()
    source_cache = build_source_cache(hydrated_chunks, parallel=True)
    cache_duration = (time.time() - cache_start) * 1000
    source_refs = list(source_cache.keys())

    # Emit source cache timing
    yield {"type": "source_cache_built", "duration_ms": round(cache_duration, 2)}

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
                    async for event in parse_and_emit_paragraph(paragraph, source_cache):
                        yield event

        # Emit any remaining content in buffer
        if buffer.strip():
            async for event in parse_and_emit_paragraph(buffer.strip(), source_cache):
                yield event

        yield {"type": "done"}

    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield {"type": "paragraph", "content": f"Error generating response: {e}"}
        yield {"type": "done"}


def find_best_source_match(ref: str, source_cache: dict[str, SourceText]) -> Optional[SourceText]:
    """
    Find the best matching source for a reference that isn't in the cache.
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

        # Look for any cached ref from the same book and siman
        for cached_ref, cached_source in source_cache.items():
            if cached_ref.startswith(book_part) and f" {siman}:" in cached_ref:
                return cached_source

    # Strategy 2: Match by book title alone (first available)
    for cached_ref, cached_source in source_cache.items():
        # Check if the book name matches
        if "Mishnah Berurah" in ref and "Mishnah Berurah" in cached_ref:
            return cached_source
        if "Ba'er Hetev" in ref and "Ba'er Hetev" in cached_ref:
            return cached_source
        if "Shulchan Arukh" in ref and cached_ref.startswith("Shulchan Arukh"):
            return cached_source

    # Strategy 3: Try partial match
    for cached_ref, cached_source in source_cache.items():
        if ref in cached_ref or cached_ref in ref:
            return cached_source

    return None


async def parse_and_emit_paragraph(
    text: str,
    source_cache: dict[str, SourceText]
) -> AsyncIterator[dict]:
    """
    Parse a paragraph for <cite> tags and emit appropriate events.
    Now supports excerpt attribute for Hebrew highlighting.

    For text segments: yields {"type": "paragraph", "content": "..."}
    For citations: yields {"type": "citation", "ref": "...", "context": "...", "hebrew": "...", "english": "...", "hebrew_excerpt": "..."}
    """
    # Updated regex to capture optional excerpt attribute with proper quote handling
    # Handles Hebrew gershayim (") and geresh (') within the excerpt text
    # Uses negative lookahead to match quotes that aren't the closing delimiter
    # Pattern: excerpt='...' or excerpt="..." where ... can contain any characters
    cite_pattern = re.compile(
        r'''<cite\s+ref="([^"]+)"(?:\s+excerpt=(?:'((?:(?!'\s*>).)*)'|"((?:(?!"\s*>).)*)")?)?\s*>(.*?)</cite>''',
        re.DOTALL
    )

    last_end = 0

    for match in cite_pattern.finditer(text):
        # Emit any text before this citation
        before_text = text[last_end:match.start()].strip()
        if before_text:
            yield {"type": "paragraph", "content": before_text}

        # Extract citation details
        # Group 1: ref
        # Group 2: excerpt with single quotes (or None)
        # Group 3: excerpt with double quotes (or None)
        # Group 4: context text
        ref = match.group(1)
        hebrew_excerpt = match.group(2) or match.group(3)  # Take whichever is not None
        context_text = match.group(4).strip()

        # Look up the source in cache
        source = source_cache.get(ref)

        if source:
            yield {
                "type": "citation",
                "ref": ref,
                "context": context_text,
                "hebrew": source.hebrew,
                "english": source.english,
                "book": source.book,
                "hebrew_excerpt": hebrew_excerpt  # NEW: Pass excerpt to frontend
            }
        else:
            # Source not in cache - try smart matching for commentary subsections
            matched_source = find_best_source_match(ref, source_cache)

            if matched_source:
                yield {
                    "type": "citation",
                    "ref": ref,
                    "context": context_text,
                    "hebrew": matched_source.hebrew,
                    "english": matched_source.english,
                    "book": matched_source.book,
                    "hebrew_excerpt": hebrew_excerpt  # NEW: Pass excerpt to frontend
                }
            else:
                # Fallback: emit as a paragraph with the citation inline
                logger.warning(f"Citation ref not found in cache: {ref}")
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
    xai_api_key: str
) -> AsyncIterator[dict]:
    """
    Main entry point for citation-aware response generation.

    Args:
        query: User's question
        context: Formatted RAG context
        retrieved_chunks: Hydrated chunks from vector search
        xai_api_key: API key for xAI

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
        xai_api_key=xai_api_key
    ):
        yield event
