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
- **chunk_index** enables neighbor lookups
- **metadata.siman** and **metadata.book** allow us to check siman boundaries within each of the 4 books
- Gemini embeddings already use `halfvec(3072)` for better performance
- **4 Books:** Orach Chayim, Yoreh Deah, Choshen Mishpat, Even HaEzer (each has its own simanim)

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
-- Optimized parameters for large 3072-dimension embeddings:
--   m = 24 (more connections per node for high-dim vectors)
--   ef_construction = 96 (higher quality graph construction)
CREATE INDEX IF NOT EXISTS idx_embedding_gemini_contextual 
ON sefaria_text_chunks 
USING hnsw (embedding_gemini_contextual halfvec_cosine_ops)
WITH (m = 24, ef_construction = 96);

-- Note: At query time, set ef_search for better recall:
-- SET LOCAL hnsw.ef_search = 60;
```

### Phase 2: Context Generation Logic

#### 2.1 Context Strategy (Prompt Caching Optimized)

**Key Insight:** To maximize prompt caching benefits, neighboring seifim should share the same context. This reduces API costs significantly.

**Strategy:**

| Siman Size | Context Strategy | Cache Benefit |
|------------|------------------|---------------|
| ≤ 15 seifim | Use **entire siman** as context for all seifim | 100% cache hit after first seif |
| > 15 seifim | **Group seifim into batches of 5**, each batch shares a context window | ~80% cache hit |

**For Large Simanim (> 15 seifim) - Two Options:**

**Option A: Fixed Groups of 5 (Recommended)**
- Seifim 1-5 share context: [seifim 1-7]
- Seifim 6-10 share context: [seifim 4-12]  
- Seifim 11-15 share context: [seifim 9-17]
- etc.

**Option B: Sliding Window with Batch Reuse**
- Process in order, reuse same prompt for 5 consecutive seifim
- Advance window by 5 seifim at a time

**Book-Aware Processing:**
Each of the 4 books (Orach Chayim, Yoreh Deah, Choshen Mishpat, Even HaEzer) must be processed independently:
- Siman 1 in Orach Chayim ≠ Siman 1 in Yoreh Deah
- Context must NEVER cross book boundaries
- Use `metadata.book` to filter correctly

**Implementation:**
```python
def get_context_for_seif(book: str, siman: int, seif: int, all_seifim_in_siman: list) -> str:
    """
    Get context for a seif, optimized for prompt caching.
    
    Args:
        book: e.g., "Shulchan Arukh, Orach Chayim"
        siman: Siman number
        seif: Seif number within siman
        all_seifim_in_siman: List of all seifim in this siman
    
    Returns:
        Context string (entire siman or windowed subset)
    """
    total_seifim = len(all_seifim_in_siman)
    
    if total_seifim <= 15:
        # Small siman: use entire siman as context
        return format_siman_context(all_seifim_in_siman)
    else:
        # Large siman: use windowed context based on seif's group
        group_index = (seif - 1) // 5  # Which group of 5 this seif belongs to
        window_start = max(0, group_index * 5 - 2)  # 2 seifim padding before
        window_end = min(total_seifim, (group_index + 1) * 5 + 2)  # 2 seifim padding after
        return format_siman_context(all_seifim_in_siman[window_start:window_end])
```

#### 2.2 Context Generation Prompt (Hebrew)

Since the output must be in Hebrew, the prompt will be in Hebrew:

```python
CONTEXTUAL_PROMPT_HEBREW = """
<הסימן>
{SIMAN_CONTEXT}
</הסימן>

<סעיף_נוכחי>
{CURRENT_SEIF}
</סעיף_נוכחי>

תן הסבר קצר וענייני (משפט אחד עד שניים) המתאר את תפקיד הסעיף הנוכחי בהקשר ההלכתי הרחב יותר של הסימן. 
ההסבר צריך לעזור בחיפוש ואחזור המידע - ציין את הנושא ההלכתי, ואם רלוונטי, את הקשר לסעיפים סביבו.
ענה רק עם ההסבר הקצר ללא שום הקדמה או סיום.

דוגמה לפלט רצוי:
"סעיף זה עוסק בדיני קידוש בליל שבת, ומפרט את נוסח הברכה על היין לפני הקידוש עצמו."
"""
```

**Note on Prompt Caching:**
- The `<הסימן>` section remains constant for all seifim in same context group
- Only `<סעיף_נוכחי>` changes between requests
- This enables significant cache hits on the context portion

#### 2.3 LLM Configuration

- **Model:** `x-ai/grok-4.1-fast` via OpenRouter
- **Temperature:** 0.3 (balanced for consistency with slight flexibility)
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
│                  (Per Book: OC, YD, CM, EH separately)               │
└─────────────────────────────────────────────────────────────────────┘

Step 0: Group by Book → Siman
┌─────────────────────────────────────────────────────────────────────┐
│  For each book (OC, YD, CM, EH):                                     │
│    For each siman in book:                                           │
│      Determine context strategy (full siman vs windowed)             │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
Step 1: Gather Context (Cache-Optimized)
┌─────────────────────────────────────────────────────────────────────┐
│  Small Siman (≤15):     Use entire siman for all seifim             │
│  Large Siman (>15):     Use windowed context, shared per 5 seifim   │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
Step 2: Generate Context (Grok 4.1 Fast) - WITH PROMPT CACHING
┌─────────────────────────────────────────────────────────────────────┐
│  <הסימן> ... </הסימן>           ← CACHED (same for seif group)       │
│  <סעיף_נוכחי> ... </סעיף_נוכחי>  ← VARIES (unique per seif)          │
│  "תן הסבר קצר..."                                                    │
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

### 5.1 Context Generation (Grok 4.1 Fast) - With Prompt Caching

**Without caching:**
- Input: ~500 tokens per chunk (siman context + current seif)
- Output: ~100 tokens (context explanation)
- For 300 chunks: ~150K input tokens, ~30K output tokens

**With prompt caching (estimated savings):**
- Small simanim (≤15 seifim): ~90% cache hit on context portion
- Large simanim (>15 seifim): ~80% cache hit per group of 5
- **Estimated effective input:** ~30-50K tokens (vs 150K without caching)

### 5.2 Embedding (Gemini)
- **Input:** ~400 tokens per contextualized chunk
- **For 300 chunks:** ~120K tokens

### 5.3 Time Estimate
- Context generation: ~0.5-1 second per chunk (faster with cache) × 300 = ~3-5 minutes
- Embedding: ~0.5 seconds per batch of 100 × 3 batches = ~1.5 minutes
- **Total estimated time:** ~5-7 minutes for 300 chunks (with caching)

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
- **Prompt caching:** Implemented via shared context strategy
- **Batch embedding requests:** Already implemented for Gemini
- **Processing order:** Process simanim sequentially to maximize cache locality

### 8.3 Display Considerations
When presenting to users:
- Show only `content` (original seif text)
- Optionally show `context_text` in a "Context" expandable section
- Never mix context into the actual source display

---

## 9. Resolved Questions

1. ✅ **Separator choice:** `\n\n---\n\n` approved
2. ✅ **Context strategy:** Cache-optimized (full siman for ≤15, windowed groups of 5 for >15)
3. ✅ **Book boundaries:** Strict (never cross books, process OC/YD/CM/EH separately)
4. ✅ **Index optimization:** m=24, ef_construction=96, ef_search=60

## 10. Remaining Questions

1. ✅ **Large siman strategy:** Option A (fixed groups) - better context at boundaries
2. ✅ **Temperature:** Keep at 0.3 for balanced consistency and slight flexibility

---

## Approval Checklist

- [ ] Schema changes approved
- [ ] Hebrew prompt reviewed
- [ ] Cost estimates acceptable
- [ ] Test scope (300 chunks) approved
- [ ] Large siman strategy chosen (Option A)
- [ ] Ready to proceed with implementation
