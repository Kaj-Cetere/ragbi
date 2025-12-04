# Hybrid Search with PGroonga + RRF Implementation Plan

## Overview

This plan implements **Hybrid Search** combining:
1. **Semantic Search** (existing) - Gemini contextual embeddings
2. **Keyword Search** (new) - PGroonga full-text search

Results are merged using **Reciprocal Rank Fusion (RRF)** for optimal relevance.

---

## 1. Current State Analysis

### Database Schema
```sql
-- Existing table: sefaria_text_chunks
- id uuid PRIMARY KEY
- chunk_index BIGSERIAL
- content text                          -- Original Hebrew seif
- context_text text                     -- LLM-generated context (Hebrew)
- content_with_context text             -- Combined: context + separator + content
- sefaria_ref text                      -- e.g., "Shulchan Arukh, Orach Chayim 1:1"
- metadata jsonb                        -- { "book": "...", "siman": 1, "seif": 1 }
- embedding vector(1536)                -- Qwen embeddings
- embedding_gemini halfvec(3072)        -- Gemini embeddings
- embedding_gemini_contextual halfvec(3072) -- Contextual Gemini embeddings
```

### Current Search Flow (app.py)
1. User query (English) → Gemini embedding
2. Vector search via `match_sefaria_chunks_gemini_contextual` RPC
3. Results hydrated with commentaries
4. LLM generates response

### Key Design Decision: Search Over `content_with_context`
The `content_with_context` column contains:
- LLM-generated Hebrew context (topic, relationships)
- Original Hebrew text

Searching this column enables keyword matches on:
- Explicit terms in the original text
- Topic keywords from the context (e.g., "קידוש", "שבת")

---

## 2. Why PGroonga for Hebrew

### Problem with Standard PostgreSQL FTS
PostgreSQL's `tsvector` uses a "simple" parser for Hebrew that:
- Treats each word as an exact string
- Cannot handle Hebrew prefixes: ב, ה, ו, ל, מ, כ, ש
- Search for "אברהם" won't find "ואברהם" or "לאברהם"

### PGroonga Solution
- Uses **N-gram tokenization** (breaks words into character sequences)
- Automatically handles prefixes/suffixes
- Search for "שבת" will match "בשבת", "השבת", "ושבת", etc.
- Zero configuration required on Supabase
- Supported natively in Supabase extensions

### Operators
| Operator | Description | Example |
|----------|-------------|---------|
| `&@~` | Full-text search with query syntax | `content &@~ 'שבת קידוש'` |
| `&@` | Simple match | `content &@ 'שבת'` |

### Query Syntax
- **AND**: `word1 word2` (space-separated)
- **OR**: `word1 OR word2`
- **NOT**: `word1 -word2`

---

## 3. English → Hebrew Query Translation

### Challenge
Users query in English, but the database contains Hebrew text.

### Solution: LLM Keyword Extraction
Before searching, pass the English query through an LLM to extract Hebrew keywords.

### Prompt Design
```
Given the following question about Jewish law (Halakha), extract the key Hebrew terms that would appear in the Shulchan Arukh or its commentaries.

Return ONLY a space-separated list of Hebrew words/phrases.
Do NOT include any English, explanations, or formatting.
Include:
- Hebrew transliterations of proper nouns (e.g., "Shabbat" → "שבת")
- Halakhic technical terms (e.g., "forbidden work" → "מלאכה")
- Relevant concepts from the context

User query: "{query}"

Hebrew keywords:
```

### Example Translations
| English Query | Hebrew Keywords |
|---------------|-----------------|
| "What are the laws of Kiddush on Shabbat?" | `קידוש שבת יין ברכה` |
| "Can I carry on Yom Tov?" | `יום טוב הוצאה טלטול מלאכה` |
| "Laws of honoring parents" | `כיבוד אב אם הורים` |

### Implementation Notes
- Use a fast, cheap model (Gemini Flash, GPT-4o-mini)
- Cache translations for repeated queries
- Set `temperature=0` for deterministic output
- Max tokens: 50-100 (should be a short list)

### Edge Cases
1. **Query already in Hebrew**: Detect Hebrew characters and skip translation
2. **Mixed language query**: Extract Hebrew from query + translate English parts
3. **Empty translation result**: Fall back to vector-only search
4. **Transliterated Hebrew in English**: Handle common patterns (Shabbat, Torah, etc.)

---

## 4. Implementation Plan

### Phase 1: Database Schema Changes

#### 4.1 Enable PGroonga Extension
```sql
-- Enable PGroonga extension (run in Supabase SQL Editor)
CREATE EXTENSION IF NOT EXISTS pgroonga;
```

#### 4.2 Create PGroonga Index
```sql
-- Create full-text search index on content_with_context
-- This enables searching the contextualized text for keywords
CREATE INDEX IF NOT EXISTS idx_pgroonga_content_with_context 
ON sefaria_text_chunks 
USING pgroonga (content_with_context);

-- Alternative: Also index original content for comparison
CREATE INDEX IF NOT EXISTS idx_pgroonga_content 
ON sefaria_text_chunks 
USING pgroonga (content);
```

**Index Considerations:**
- PGroonga indexes are fast to create (~seconds for 10k rows)
- Index size is typically 2-3x the text size
- N-gram tokenizer handles Hebrew without configuration

### Phase 2: Hybrid Search RPC Function

#### 4.3 Create Hybrid Search Function with RRF
```sql
-- Drop existing function if exists
DROP FUNCTION IF EXISTS hybrid_search_sefaria(text, halfvec(3072), int, float, float, int);

CREATE OR REPLACE FUNCTION hybrid_search_sefaria(
    query_text text,                    -- Hebrew keywords (space-separated)
    query_embedding halfvec(3072),      -- Gemini contextual embedding
    match_count int DEFAULT 10,         -- Number of results to return
    full_text_weight float DEFAULT 1.0, -- Weight for keyword results
    semantic_weight float DEFAULT 1.0,  -- Weight for vector results
    rrf_k int DEFAULT 60                -- RRF smoothing constant
)
RETURNS TABLE (
    id uuid,
    content text,
    context_text text,
    content_with_context text,
    sefaria_ref text,
    metadata jsonb,
    semantic_score float,
    keyword_score float,
    combined_score float,
    match_source text                   -- 'semantic', 'keyword', or 'both'
)
LANGUAGE plpgsql STABLE
AS $$
BEGIN
    -- Set ef_search for better recall with high-dimensional vectors
    SET LOCAL hnsw.ef_search = 60;
    
    RETURN QUERY
    WITH 
    -- Step 1: Keyword search using PGroonga
    keyword_results AS (
        SELECT
            stc.id,
            pgroonga_score(stc.tableoid, stc.ctid) AS score,
            ROW_NUMBER() OVER (ORDER BY pgroonga_score(stc.tableoid, stc.ctid) DESC) AS rank
        FROM sefaria_text_chunks stc
        WHERE stc.content_with_context &@~ query_text
          AND stc.content_with_context IS NOT NULL
        ORDER BY pgroonga_score(stc.tableoid, stc.ctid) DESC
        LIMIT match_count * 3  -- Fetch extra for fusion
    ),
    
    -- Step 2: Semantic search using contextual embeddings
    semantic_results AS (
        SELECT
            stc.id,
            (1 - (stc.embedding_gemini_contextual <=> query_embedding)) AS score,
            ROW_NUMBER() OVER (ORDER BY stc.embedding_gemini_contextual <=> query_embedding ASC) AS rank
        FROM sefaria_text_chunks stc
        WHERE stc.embedding_gemini_contextual IS NOT NULL
        ORDER BY stc.embedding_gemini_contextual <=> query_embedding ASC
        LIMIT match_count * 3  -- Fetch extra for fusion
    ),
    
    -- Step 3: RRF Fusion
    fused_results AS (
        SELECT
            COALESCE(kr.id, sr.id) AS id,
            -- RRF Score: 1 / (rank + k)
            COALESCE(full_text_weight / (kr.rank + rrf_k), 0.0) AS keyword_rrf,
            COALESCE(semantic_weight / (sr.rank + rrf_k), 0.0) AS semantic_rrf,
            COALESCE(full_text_weight / (kr.rank + rrf_k), 0.0) + 
                COALESCE(semantic_weight / (sr.rank + rrf_k), 0.0) AS combined_rrf,
            kr.score AS kw_score,
            sr.score AS sem_score,
            CASE 
                WHEN kr.id IS NOT NULL AND sr.id IS NOT NULL THEN 'both'
                WHEN kr.id IS NOT NULL THEN 'keyword'
                ELSE 'semantic'
            END AS source
        FROM keyword_results kr
        FULL OUTER JOIN semantic_results sr ON kr.id = sr.id
    )
    
    -- Step 4: Join back to get full chunk data
    SELECT
        stc.id,
        stc.content,
        stc.context_text,
        stc.content_with_context,
        stc.sefaria_ref,
        stc.metadata,
        fr.sem_score::float AS semantic_score,
        fr.kw_score::float AS keyword_score,
        fr.combined_rrf::float AS combined_score,
        fr.source AS match_source
    FROM fused_results fr
    JOIN sefaria_text_chunks stc ON fr.id = stc.id
    ORDER BY fr.combined_rrf DESC
    LIMIT match_count;
END;
$$;

-- Grant permissions
GRANT EXECUTE ON FUNCTION hybrid_search_sefaria(text, halfvec(3072), int, float, float, int) 
TO authenticated, anon;
```

#### 4.4 Keyword-Only Search Function (for testing/fallback)
```sql
CREATE OR REPLACE FUNCTION keyword_search_sefaria(
    query_text text,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id uuid,
    content text,
    context_text text,
    sefaria_ref text,
    metadata jsonb,
    score float
)
LANGUAGE sql STABLE
AS $$
    SELECT
        stc.id,
        stc.content,
        stc.context_text,
        stc.sefaria_ref,
        stc.metadata,
        pgroonga_score(stc.tableoid, stc.ctid)::float AS score
    FROM sefaria_text_chunks stc
    WHERE stc.content_with_context &@~ query_text
      AND stc.content_with_context IS NOT NULL
    ORDER BY pgroonga_score(stc.tableoid, stc.ctid) DESC
    LIMIT match_count;
$$;

GRANT EXECUTE ON FUNCTION keyword_search_sefaria(text, int) TO authenticated, anon;
```

### Phase 3: Python Application Changes

#### 4.5 New File: `query_translator.py`
```python
"""
Translate English queries to Hebrew keywords for hybrid search.
"""

import os
import re
import requests
from functools import lru_cache

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TRANSLATION_MODEL = "google/gemini-2.0-flash-lite-001"  # Fast and cheap

TRANSLATION_PROMPT = '''Given the following question about Jewish law (Halakha), extract the key Hebrew terms that would appear in the Shulchan Arukh or its commentaries.

Return ONLY a space-separated list of Hebrew words/phrases.
Do NOT include any English, explanations, or formatting.
Include:
- Hebrew transliterations of proper nouns (e.g., "Shabbat" → "שבת")
- Halakhic technical terms (e.g., "forbidden work" → "מלאכה")
- Relevant concepts from the context

User query: "{query}"

Hebrew keywords:'''


def contains_hebrew(text: str) -> bool:
    """Check if text contains Hebrew characters."""
    return bool(re.search(r'[\u0590-\u05FF]', text))


def extract_hebrew_from_mixed(text: str) -> str:
    """Extract Hebrew words from a mixed-language query."""
    hebrew_pattern = r'[\u0590-\u05FF]+'
    hebrew_words = re.findall(hebrew_pattern, text)
    return ' '.join(hebrew_words)


@lru_cache(maxsize=256)
def translate_query_to_hebrew(query: str) -> str | None:
    """
    Translate an English query to Hebrew keywords.
    Returns None if translation fails.
    
    Uses LRU cache to avoid repeated API calls for same queries.
    """
    # If query is already mostly Hebrew, extract and return Hebrew words
    if contains_hebrew(query):
        hebrew_words = extract_hebrew_from_mixed(query)
        if hebrew_words:
            return hebrew_words
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": TRANSLATION_MODEL,
                "messages": [
                    {"role": "user", "content": TRANSLATION_PROMPT.format(query=query)}
                ],
                "temperature": 0,
                "max_tokens": 100,
            },
            timeout=10
        )
        response.raise_for_status()
        
        result = response.json()
        hebrew_keywords = result["choices"][0]["message"]["content"].strip()
        
        # Validate: should contain Hebrew characters
        if not contains_hebrew(hebrew_keywords):
            return None
        
        # Clean up: remove any English, punctuation, extra spaces
        hebrew_keywords = ' '.join(re.findall(r'[\u0590-\u05FF]+', hebrew_keywords))
        
        return hebrew_keywords if hebrew_keywords else None
        
    except Exception as e:
        print(f"⚠️ Query translation error: {e}")
        return None
```

#### 4.6 Updates to `app.py`

```python
# Add import at top
from query_translator import translate_query_to_hebrew

# Add new search function
def hybrid_search_chunks(
    embedding: list[float], 
    hebrew_keywords: str | None,
    match_count: int = TOP_K_RETRIEVE,
    full_text_weight: float = 1.0,
    semantic_weight: float = 1.0
) -> list[dict]:
    """
    Perform hybrid search combining semantic and keyword search.
    Falls back to semantic-only if no Hebrew keywords provided.
    """
    start_time = time.time()
    
    # If no keywords, fall back to semantic-only search
    if not hebrew_keywords:
        logger.info("⚠️ No Hebrew keywords, falling back to semantic-only search")
        return search_chunks(embedding, "match_sefaria_chunks_gemini_contextual", match_count)
    
    logger.info(f"🔍 Starting hybrid search with keywords: {hebrew_keywords}")
    
    try:
        result = supabase_client.rpc(
            "hybrid_search_sefaria",
            {
                "query_text": hebrew_keywords,
                "query_embedding": embedding,
                "match_count": match_count,
                "full_text_weight": full_text_weight,
                "semantic_weight": semantic_weight,
                "rrf_k": 60
            }
        ).execute()
        
        chunks = result.data if result.data else []
        total_time = time.time() - start_time
        
        # Log match sources
        sources = [c.get('match_source', 'unknown') for c in chunks]
        source_counts = {s: sources.count(s) for s in set(sources)}
        logger.info(f"✅ Hybrid search returned {len(chunks)} results in {total_time:.3f}s")
        logger.info(f"📊 Match sources: {source_counts}")
        
        return chunks
        
    except Exception as e:
        logger.error(f"❌ Hybrid search failed: {e}")
        # Fall back to semantic-only
        return search_chunks(embedding, "match_sefaria_chunks_gemini_contextual", match_count)
```

### Phase 4: UI Updates

#### 4.7 Add Hybrid Search Toggle in Sidebar
```python
# In sidebar settings section
st.divider()
st.header("Search Mode")

search_mode = st.radio(
    "Search Strategy",
    options=["Semantic Only", "Hybrid (Semantic + Keywords)"],
    index=1,  # Default to hybrid
    help="Hybrid combines meaning-based search with keyword matching"
)

if search_mode == "Hybrid (Semantic + Keywords)":
    col1, col2 = st.columns(2)
    with col1:
        semantic_weight = st.slider("Semantic Weight", 0.0, 2.0, 1.0, 0.1)
    with col2:
        keyword_weight = st.slider("Keyword Weight", 0.0, 2.0, 1.0, 0.1)
    
    show_keywords = st.checkbox("Show extracted Hebrew keywords", value=True)
```

#### 4.8 Update Query Processing
```python
# In the query processing section
if search_mode == "Hybrid (Semantic + Keywords)":
    # Translate query to Hebrew keywords
    with st.spinner("Extracting Hebrew keywords..."):
        hebrew_keywords = translate_query_to_hebrew(prompt)
    
    if show_keywords and hebrew_keywords:
        st.info(f"🔤 Hebrew keywords: {hebrew_keywords}")
    
    # Hybrid search
    chunks = hybrid_search_chunks(
        embedding, 
        hebrew_keywords,
        match_count=TOP_K_RETRIEVE,
        full_text_weight=keyword_weight,
        semantic_weight=semantic_weight
    )
else:
    # Semantic-only search
    chunks = search_chunks(embedding, selected_model["rpc_function"])
```

---

## 5. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        HYBRID SEARCH PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────┘

User Query (English)
        │
        ├──────────────────┬────────────────────┐
        │                  │                    │
        ▼                  ▼                    │
   ┌─────────┐      ┌──────────────┐           │
   │ Gemini  │      │ LLM Query    │           │
   │Embedding│      │ Translation  │           │
   └────┬────┘      └──────┬───────┘           │
        │                  │                    │
        │                  ▼                    │
        │           Hebrew Keywords             │
        │           "שבת קידוש יין"            │
        │                  │                    │
        ▼                  ▼                    │
   ┌─────────────────────────────────────┐     │
   │        Supabase RPC                 │     │
   │     hybrid_search_sefaria()         │     │
   │                                     │     │
   │  ┌──────────────┐ ┌──────────────┐ │     │
   │  │  PGroonga    │ │   pgvector   │ │     │
   │  │  Keyword     │ │   Semantic   │ │     │
   │  │  Search      │ │   Search     │ │     │
   │  └──────┬───────┘ └──────┬───────┘ │     │
   │         │                │          │     │
   │         └────────┬───────┘          │     │
   │                  │                  │     │
   │         ┌────────▼────────┐         │     │
   │         │  RRF Fusion     │         │     │
   │         │  k=60           │         │     │
   │         └────────┬────────┘         │     │
   │                  │                  │     │
   └──────────────────┼──────────────────┘     │
                      │                         │
                      ▼                         │
              Ranked Results                    │
              + Match Sources                   │
                      │                         │
                      ▼                         │
              ┌───────────────┐                 │
              │ Commentary    │                 │
              │ Hydration     │                 │
              └───────┬───────┘                 │
                      │                         │
                      ▼                         │
              ┌───────────────┐                 │
              │ LLM Response  │◄────────────────┘
              │ Generation    │   (uses original query)
              └───────────────┘
```

---

## 6. Edge Cases & Quirks

### 6.1 Hebrew Text Edge Cases

| Case | Example | Handling |
|------|---------|----------|
| **Prefixes** | ב, ה, ו, ל, מ, כ, ש | PGroonga N-grams handle naturally |
| **Nikud (vowels)** | שָׁבָּת vs שבת | PGroonga matches both |
| **Final letters** | ם, ן, ץ, ף, ך | Standard Hebrew, no special handling needed |
| **Gematria/numbers** | א׳, ב׳ | May not match well, rely on semantic |
| **Abbreviations** | רמב״ם, ש״ע | Index as-is, user should search expanded form |

### 6.2 Query Edge Cases

| Case | Handling |
|------|----------|
| **Empty query** | Skip hybrid, use semantic only |
| **Query with only stopwords** | Fall back to semantic only |
| **Very long query** | LLM extracts key terms, limits to ~5-7 keywords |
| **Query in Hebrew** | Skip translation, use directly |
| **Query with typos** | N-grams provide some fuzzy matching |
| **Query with transliteration** | LLM converts "Shabbat" → "שבת" |

### 6.3 Data Quirks

| Issue | Mitigation |
|-------|------------|
| **Rema text in `<small>` tags** | Already cleaned in ingestion, appears in content |
| **Missing `content_with_context`** | RPC function has NULL check, falls back |
| **Empty context_text** | Still indexes original content |
| **Very short seifim** | May have low keyword scores, semantic compensates |

### 6.4 PGroonga Quirks

1. **pgroonga_score returns 0.0** if:
   - Index not used (sequential scan instead)
   - Query doesn't match any documents
   - The index wasn't built with scoring support

2. **Performance**: For very large tables (>1M rows), consider:
   - Partitioning by book
   - More aggressive LIMIT in CTEs

3. **NULL handling**: The RPC function must handle NULL content_with_context rows

---

## 7. Performance Considerations

### 7.1 Expected Latencies
| Component | Estimated Time |
|-----------|----------------|
| Query translation (LLM) | 100-300ms |
| PGroonga keyword search | 10-50ms |
| pgvector semantic search | 200-500ms |
| RRF fusion (SQL) | <5ms |
| **Total hybrid search** | 300-800ms |

### 7.2 Optimization Strategies

1. **Parallel search**: PGroonga and pgvector queries run in same transaction
2. **Query caching**: LRU cache for query translations
3. **Result limiting**: Fetch 3x match_count, then fuse and limit
4. **Index tuning**: 
   - PGroonga: No tuning needed (N-gram default)
   - pgvector: Already optimized (m=24, ef_construction=96)

### 7.3 Monitoring Metrics
Add logging for:
- Translation latency
- Keyword search latency
- Semantic search latency
- Match source distribution (keyword/semantic/both)
- Cache hit rate for translations

---

## 8. Testing Plan

### 8.1 Unit Tests

```python
# test_query_translator.py

def test_hebrew_detection():
    assert contains_hebrew("שבת") == True
    assert contains_hebrew("Shabbat") == False
    assert contains_hebrew("What is שבת?") == True

def test_mixed_extraction():
    assert extract_hebrew_from_mixed("What is שבת?") == "שבת"
    assert extract_hebrew_from_mixed("שבת and קידוש") == "שבת קידוש"

def test_translation_caching():
    # First call hits API
    result1 = translate_query_to_hebrew("laws of Shabbat")
    # Second call hits cache
    result2 = translate_query_to_hebrew("laws of Shabbat")
    assert result1 == result2
```

### 8.2 Integration Tests

```sql
-- Test PGroonga index works
SELECT * FROM sefaria_text_chunks 
WHERE content_with_context &@~ 'שבת'
LIMIT 5;

-- Test keyword-only search RPC
SELECT * FROM keyword_search_sefaria('שבת קידוש', 5);

-- Test hybrid search RPC
SELECT * FROM hybrid_search_sefaria(
    'שבת קידוש',
    '[0.1, 0.2, ...]'::halfvec(3072),  -- Actual embedding
    10,
    1.0,
    1.0,
    60
);
```

### 8.3 Quality Tests

| Query (English) | Expected Top Results | Metric |
|-----------------|---------------------|--------|
| "Laws of Kiddush on Friday night" | OC 271 (קידוש) | Should include siman 271 |
| "Can I work on Shabbat?" | OC 301-340 (מלאכות שבת) | Should include work-related simanim |
| "Honoring parents" | YD 240 (כיבוד אב ואם) | Should include siman 240 |
| "Damages and liability" | CM 378-427 (נזיקין) | Should include damages simanim |

### 8.4 A/B Comparison

Compare retrieval quality:
| Query | Semantic Only (Top-3) | Hybrid (Top-3) | Better? |
|-------|----------------------|----------------|---------|
| ... | ... | ... | ... |

---

## 9. Implementation Files

### New Files to Create
1. **`keyword_fusion.sql`** - Schema migration and RPC functions
2. **`query_translator.py`** - English → Hebrew translation logic
3. **`test_hybrid_search.py`** - Test suite

### Files to Modify
1. **`app.py`** - Add hybrid search mode, UI controls
2. **`requirements.txt`** - No new dependencies needed

---

## 10. SQL Migration Script

Save this as `keyword_fusion.sql`:

```sql
-- =============================================================================
-- Migration: Hybrid Search with PGroonga + RRF
-- Run this in the Supabase SQL Editor
-- =============================================================================

-- Step 1: Enable PGroonga extension
CREATE EXTENSION IF NOT EXISTS pgroonga;

-- Step 2: Create PGroonga index on content_with_context
CREATE INDEX IF NOT EXISTS idx_pgroonga_content_with_context 
ON sefaria_text_chunks 
USING pgroonga (content_with_context);

-- Optional: Also index original content
CREATE INDEX IF NOT EXISTS idx_pgroonga_content 
ON sefaria_text_chunks 
USING pgroonga (content);

-- Step 3: Create keyword-only search function
DROP FUNCTION IF EXISTS keyword_search_sefaria(text, int);

CREATE OR REPLACE FUNCTION keyword_search_sefaria(
    query_text text,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id uuid,
    content text,
    context_text text,
    sefaria_ref text,
    metadata jsonb,
    score float
)
LANGUAGE sql STABLE
AS $$
    SELECT
        stc.id,
        stc.content,
        stc.context_text,
        stc.sefaria_ref,
        stc.metadata,
        pgroonga_score(stc.tableoid, stc.ctid)::float AS score
    FROM sefaria_text_chunks stc
    WHERE stc.content_with_context &@~ query_text
      AND stc.content_with_context IS NOT NULL
    ORDER BY pgroonga_score(stc.tableoid, stc.ctid) DESC
    LIMIT match_count;
$$;

GRANT EXECUTE ON FUNCTION keyword_search_sefaria(text, int) TO authenticated, anon;

-- Step 4: Create hybrid search function with RRF
DROP FUNCTION IF EXISTS hybrid_search_sefaria(text, halfvec(3072), int, float, float, int);

CREATE OR REPLACE FUNCTION hybrid_search_sefaria(
    query_text text,
    query_embedding halfvec(3072),
    match_count int DEFAULT 10,
    full_text_weight float DEFAULT 1.0,
    semantic_weight float DEFAULT 1.0,
    rrf_k int DEFAULT 60
)
RETURNS TABLE (
    id uuid,
    content text,
    context_text text,
    content_with_context text,
    sefaria_ref text,
    metadata jsonb,
    semantic_score float,
    keyword_score float,
    combined_score float,
    match_source text
)
LANGUAGE plpgsql STABLE
AS $$
BEGIN
    SET LOCAL hnsw.ef_search = 60;
    
    RETURN QUERY
    WITH 
    keyword_results AS (
        SELECT
            stc.id,
            pgroonga_score(stc.tableoid, stc.ctid) AS score,
            ROW_NUMBER() OVER (ORDER BY pgroonga_score(stc.tableoid, stc.ctid) DESC) AS rank
        FROM sefaria_text_chunks stc
        WHERE stc.content_with_context &@~ query_text
          AND stc.content_with_context IS NOT NULL
        ORDER BY pgroonga_score(stc.tableoid, stc.ctid) DESC
        LIMIT match_count * 3
    ),
    semantic_results AS (
        SELECT
            stc.id,
            (1 - (stc.embedding_gemini_contextual <=> query_embedding)) AS score,
            ROW_NUMBER() OVER (ORDER BY stc.embedding_gemini_contextual <=> query_embedding ASC) AS rank
        FROM sefaria_text_chunks stc
        WHERE stc.embedding_gemini_contextual IS NOT NULL
        ORDER BY stc.embedding_gemini_contextual <=> query_embedding ASC
        LIMIT match_count * 3
    ),
    fused_results AS (
        SELECT
            COALESCE(kr.id, sr.id) AS id,
            COALESCE(full_text_weight / (kr.rank + rrf_k), 0.0) AS keyword_rrf,
            COALESCE(semantic_weight / (sr.rank + rrf_k), 0.0) AS semantic_rrf,
            COALESCE(full_text_weight / (kr.rank + rrf_k), 0.0) + 
                COALESCE(semantic_weight / (sr.rank + rrf_k), 0.0) AS combined_rrf,
            kr.score AS kw_score,
            sr.score AS sem_score,
            CASE 
                WHEN kr.id IS NOT NULL AND sr.id IS NOT NULL THEN 'both'
                WHEN kr.id IS NOT NULL THEN 'keyword'
                ELSE 'semantic'
            END AS source
        FROM keyword_results kr
        FULL OUTER JOIN semantic_results sr ON kr.id = sr.id
    )
    SELECT
        stc.id,
        stc.content,
        stc.context_text,
        stc.content_with_context,
        stc.sefaria_ref,
        stc.metadata,
        fr.sem_score::float AS semantic_score,
        fr.kw_score::float AS keyword_score,
        fr.combined_rrf::float AS combined_score,
        fr.source AS match_source
    FROM fused_results fr
    JOIN sefaria_text_chunks stc ON fr.id = stc.id
    ORDER BY fr.combined_rrf DESC
    LIMIT match_count;
END;
$$;

GRANT EXECUTE ON FUNCTION hybrid_search_sefaria(text, halfvec(3072), int, float, float, int) 
TO authenticated, anon;

-- =============================================================================
-- VERIFICATION QUERIES
-- =============================================================================

-- Check PGroonga extension is enabled
SELECT * FROM pg_extension WHERE extname = 'pgroonga';

-- Check indexes were created
SELECT indexname, indexdef 
FROM pg_indexes 
WHERE tablename = 'sefaria_text_chunks' 
  AND indexname LIKE '%pgroonga%';

-- Test keyword search (replace with actual Hebrew)
-- SELECT * FROM keyword_search_sefaria('שבת', 5);

-- =============================================================================
```

---

## 11. Weight Tuning Guidelines

### Default Weights
- `semantic_weight = 1.0`
- `full_text_weight = 1.0`

### When to Increase Keyword Weight
- User searches for specific terms (names, exact phrases)
- Query contains rare Hebrew words
- Semantic search returns too many tangential results

### When to Increase Semantic Weight
- User asks conceptual questions
- Query uses paraphrases or synonyms
- Translation produces weak/generic keywords

### Suggested Presets
| Query Type | Semantic | Keyword |
|------------|----------|---------|
| Conceptual ("What are the laws of...") | 1.2 | 0.8 |
| Specific ("Siman 271 about Kiddush") | 0.8 | 1.2 |
| Mixed | 1.0 | 1.0 |

---

## 12. Rollback Plan

If hybrid search causes issues:

```sql
-- Remove PGroonga indexes
DROP INDEX IF EXISTS idx_pgroonga_content_with_context;
DROP INDEX IF EXISTS idx_pgroonga_content;

-- Remove RPC functions
DROP FUNCTION IF EXISTS hybrid_search_sefaria(text, halfvec(3072), int, float, float, int);
DROP FUNCTION IF EXISTS keyword_search_sefaria(text, int);

-- Keep PGroonga extension (may be used by other features)
-- DROP EXTENSION IF EXISTS pgroonga;
```

In `app.py`:
- Remove hybrid search toggle
- Revert to semantic-only search

---

## 13. Future Enhancements

### 13.1 Query Expansion
- Use LLM to generate synonyms for Hebrew terms
- Example: "קידוש" → "קידוש OR הבדלה OR יין"

### 13.2 Personalized Weights
- Learn optimal weights from user feedback
- Adjust per-user or per-query-type

### 13.3 Phrase Matching
- PGroonga supports phrase queries: `"exact phrase"`
- Boost results that match multi-word sequences

### 13.4 Book-Specific Tuning
- Different weights for different sections (OC vs CM)
- Different keyword extraction prompts

### 13.5 Caching Layer
- Cache hybrid search results for common queries
- Invalidate on new data ingestion

---

## 14. Approval Checklist

- [ ] PGroonga extension approved
- [ ] Schema migration reviewed
- [ ] RRF parameters approved (k=60, default weights)
- [ ] Query translation prompt reviewed
- [ ] Performance targets acceptable
- [ ] Testing plan approved
- [ ] Ready to implement

---

*Created: December 2, 2025*
*Based on: ARCHITECTURE.md, contextualized_embeddings_plan.md, app.py analysis*
