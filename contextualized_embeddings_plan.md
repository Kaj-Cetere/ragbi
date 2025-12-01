# Contextualized Embeddings Plan

## Overview

This plan implements **Contextual Retrieval** (as pioneered by Anthropic) for the Sefaria RAG system. Each Shulchan Aruch seif will be enriched with LLM-generated contextual explanations, then re-embedded for improved retrieval accuracy.

---

## 1. Current State Analysis

### Database Schema
```sql
-- Existing table: sefaria_text_chunks
- id uuid PRIMARY KEY
- chunk_index BIGSERIAL         -- Auto-increment, preserves order
- content text                  -- The cleaned Hebrew seif text
- embedding vector(1536)        -- Qwen embeddings
- embedding_gemini halfvec(3072) -- Gemini embeddings
- sefaria_ref text              -- e.g., "Shulchan Arukh, Orach Chayim 1:1"
- metadata jsonb                -- { "book": "...", "siman": 1, "seif": 1, "section": "..." }
```

### Key Points
- **chunk_index** enables neighbor lookups (5 before, 2 after)
- **metadata.siman** allows us to check siman boundaries
- Gemini embeddings already use `halfvec(3072)` for better performance

---

## 2. Implementation Plan

### Phase 1: Schema Changes

Add new columns to store:
1. **`context_text`** - The LLM-generated Hebrew contextual explanation
2. **`content_with_context`** - Combined: `context_text + "\n\n" + content`
3. **`embedding_gemini_contextual`** - New Gemini embedding of the contextualized content

```sql
-- Add columns for contextual embeddings
ALTER TABLE sefaria_text_chunks
ADD COLUMN IF NOT EXISTS context_text text;           -- LLM-generated context

ALTER TABLE sefaria_text_chunks
ADD COLUMN IF NOT EXISTS content_with_context text;   -- Combined for embedding

ALTER TABLE sefaria_text_chunks
ADD COLUMN IF NOT EXISTS embedding_gemini_contextual halfvec(3072);  -- New embedding

-- Create HNSW index for contextual embeddings
CREATE INDEX IF NOT EXISTS idx_embedding_gemini_contextual 
ON sefaria_text_chunks 
USING hnsw (embedding_gemini_contextual halfvec_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### Phase 2: Context Generation Logic

#### 2.1 Gathering Surrounding Context

For each seif, gather:
- **5 chunks before** (if available and same siman)
- **2 chunks after** (if available and same siman)
- **Current chunk** (the target seif)

**Siman boundary rules:**
- If a surrounding chunk is from a different siman, exclude it (it's unrelated)
- If surrounding chunks span multiple simanim, only include chunks from the same siman
- This ensures the LLM receives only relevant halakhic context

**Implementation:**
```python
def get_surrounding_context(chunk_index: int, siman: int, book: str) -> dict:
    """
    Fetch surrounding chunks for context, respecting siman boundaries.
    Returns: { "before": [...], "current": {...}, "after": [...] }
    """
    # Query chunks with chunk_index in range [current-5, current+2]
    # Filter to same book and same siman
    pass
```

#### 2.2 Context Generation Prompt (Hebrew)

Since the output must be in Hebrew, the prompt will be in Hebrew:

```python
CONTEXTUAL_PROMPT_HEBREW = """
<הקשר>
{SURROUNDING_CHUNKS}
</הקשר>

<סעיף_נוכחי>
{CURRENT_SEIF}
</סעיף_נוכחי>

תן הסבר קצר וענייני (משפט אחד עד שניים) המתאר את תפקיד הסעיף הנוכחי בהקשר ההלכתי הרחב יותר של הסימן. 
ההסבר צריך לעזור בחיפוש ואחזור המידע - ציין את הנושא ההלכתי, ואם רלוונטי, את הקשר לסעיפים סביבו.
ענה רק עם ההסבר הקצר ללא שום הקדמה או סיום.
"""
```

**Example output:**
> סעיף זה עוסק בדיני קידוש בליל שבת, ומפרט את נוסח הברכה על היין לפני הקידוש עצמו.

#### 2.3 LLM Configuration

- **Model:** `x-ai/grok-4.1-fast` via OpenRouter
- **Temperature:** 0.3 (low for consistency)
- **Max tokens:** 150 (context should be 50-100 tokens)

### Phase 3: Re-Embedding with Gemini

After context generation:
1. Combine: `context_text + "\n\n---\n\n" + content`
2. Generate new Gemini embedding using `google/gemini-embedding-001`
3. Store in `embedding_gemini_contextual`

**Separator choice:** `\n\n---\n\n` (double newline + horizontal rule + double newline)
- Clear visual separator
- Allows easy splitting when displaying to users

### Phase 4: Search Function

Create new RPC function for contextual search:

```sql
CREATE OR REPLACE FUNCTION match_sefaria_chunks_gemini_contextual (
    query_embedding halfvec(3072),
    match_threshold float DEFAULT 0.5,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id uuid,
    content text,               -- Original content (for display)
    context_text text,          -- Context (for debugging/display)
    sefaria_ref text,
    metadata jsonb,
    similarity float
)
LANGUAGE sql STABLE
AS $$
    SELECT
        sefaria_text_chunks.id,
        sefaria_text_chunks.content,
        sefaria_text_chunks.context_text,
        sefaria_text_chunks.sefaria_ref,
        sefaria_text_chunks.metadata,
        1 - (sefaria_text_chunks.embedding_gemini_contextual <=> query_embedding) AS similarity
    FROM sefaria_text_chunks
    WHERE sefaria_text_chunks.embedding_gemini_contextual IS NOT NULL
      AND 1 - (sefaria_text_chunks.embedding_gemini_contextual <=> query_embedding) > match_threshold
    ORDER BY (sefaria_text_chunks.embedding_gemini_contextual <=> query_embedding) ASC
    LIMIT match_count;
$$;
```

---

## 3. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CONTEXTUAL EMBEDDING PIPELINE                    │
└─────────────────────────────────────────────────────────────────────┘

Step 1: Gather Context
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  5 chunks   │    │   Current   │    │  2 chunks   │
│   before    │ ─▶ │    Seif     │ ◀─ │   after     │
│ (same siman)│    │             │    │ (same siman)│
└─────────────┘    └─────────────┘    └─────────────┘
                          │
                          ▼
Step 2: Generate Context (Grok 4.1 Fast)
┌─────────────────────────────────────────────────────────────────────┐
│  Prompt (Hebrew):                                                    │
│  "תן הסבר קצר... על תפקיד הסעיף בהקשר ההלכתי..."                      │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
Step 3: Combine & Store
┌─────────────────────────────────────────────────────────────────────┐
│  context_text:         "סעיף זה עוסק בדיני קידוש..."                 │
│  content_with_context: context_text + "\n\n---\n\n" + content        │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
Step 4: Embed with Gemini
┌─────────────────────────────────────────────────────────────────────┐
│  google/gemini-embedding-001 → embedding_gemini_contextual           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Test Phase (First 300 Chunks)

### 4.1 Test Scope
- Process only `chunk_index <= 300`
- This covers approximately the first few simanim of the first book

### 4.2 Comparison Metrics
After processing, compare retrieval quality:

| Query | Gemini Original | Gemini Contextual | Notes |
|-------|-----------------|-------------------|-------|
| "דיני קידוש" | ... | ... | |
| "הלכות שבת" | ... | ... | |
| ... | ... | ... | |

### 4.3 Success Criteria
- Contextual embeddings should show improved relevance for:
  - Queries that lack specific terminology in the seif
  - Queries about broader topics where context helps
  - Cross-referential queries

---

## 5. Cost & Performance Estimates

### 5.1 Context Generation (Grok 4.1 Fast)
- **Input:** ~500 tokens per chunk (surrounding context + current seif)
- **Output:** ~100 tokens (context explanation)
- **For 300 chunks:** ~180K input tokens, ~30K output tokens

### 5.2 Embedding (Gemini)
- **Input:** ~400 tokens per contextualized chunk
- **For 300 chunks:** ~120K tokens

### 5.3 Time Estimate
- Context generation: ~1-2 seconds per chunk × 300 = ~5-10 minutes
- Embedding: ~0.5 seconds per batch of 100 × 3 batches = ~1.5 minutes
- **Total estimated time:** ~10-15 minutes for 300 chunks

---

## 6. Implementation Files

### New Files to Create:
1. **`add_contextual_embedding_schema.sql`** - Schema migration
2. **`generate_contextual_embeddings.py`** - Main processing script

### Files to Modify:
1. **`app.py`** - Add new embedding model option for contextual search

---

## 7. Rollback Plan

If contextual embeddings don't improve retrieval:
1. Keep original columns (`embedding_gemini`) intact
2. Drop contextual columns with:
   ```sql
   ALTER TABLE sefaria_text_chunks DROP COLUMN IF EXISTS context_text;
   ALTER TABLE sefaria_text_chunks DROP COLUMN IF EXISTS content_with_context;
   ALTER TABLE sefaria_text_chunks DROP COLUMN IF EXISTS embedding_gemini_contextual;
   ```

---

## 8. Future Considerations

### 8.1 Full Dataset Processing
After successful testing with 300 chunks:
- Process remaining ~X chunks
- Estimated time: proportional to total chunks

### 8.2 Caching/Optimization
- Consider prompt caching if OpenRouter/Grok supports it
- Batch embedding requests (already implemented for Gemini)

### 8.3 Display Considerations
When presenting to users:
- Show only `content` (original seif text)
- Optionally show `context_text` in a "Context" expandable section
- Never mix context into the actual source display

---

## 9. Questions for Review

1. **Separator choice:** Is `\n\n---\n\n` appropriate, or prefer something else?
2. **Context length:** 5 before + 2 after - should this be adjusted?
3. **Siman boundary:** Strict (only same siman) or allow 1 chunk from adjacent siman?
4. **Temperature:** 0.3 for context generation - should it be lower (0.1) for more deterministic output?

---

## Approval Checklist

- [ ] Schema changes approved
- [ ] Hebrew prompt reviewed
- [ ] Cost estimates acceptable
- [ ] Test scope (300 chunks) approved
- [ ] Ready to proceed with implementation
