-- =============================================================================
-- Migration: Add Contextual Embedding Columns and Index
-- Run this in the Supabase SQL Editor
-- =============================================================================

-- Step 1: Add column for LLM-generated context
ALTER TABLE sefaria_text_chunks
ADD COLUMN IF NOT EXISTS context_text text;

-- Step 2: Add column for combined content (context + original)
ALTER TABLE sefaria_text_chunks
ADD COLUMN IF NOT EXISTS content_with_context text;

-- Step 3: Add column for contextual Gemini embeddings
ALTER TABLE sefaria_text_chunks
ADD COLUMN IF NOT EXISTS embedding_gemini_contextual halfvec(3072);

-- Step 4: Create HNSW index optimized for 3072-dimension embeddings
-- Parameters: m=24 (more connections), ef_construction=96 (higher quality)
CREATE INDEX IF NOT EXISTS idx_embedding_gemini_contextual 
ON sefaria_text_chunks 
USING hnsw (embedding_gemini_contextual halfvec_cosine_ops)
WITH (m = 24, ef_construction = 96);

-- Step 5: Drop existing function if it exists
DROP FUNCTION IF EXISTS match_sefaria_chunks_gemini_contextual(halfvec(3072), float, int);

-- Step 6: Create RPC function for contextual semantic search
CREATE OR REPLACE FUNCTION match_sefaria_chunks_gemini_contextual (
    query_embedding halfvec(3072),
    match_threshold float DEFAULT 0.5,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id uuid,
    content text,
    context_text text,
    sefaria_ref text,
    metadata jsonb,
    similarity float
)
LANGUAGE sql STABLE
AS $$
    -- Set ef_search for better recall with high-dimensional vectors
    -- Note: SET LOCAL only works in plpgsql, so we rely on session settings
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

-- Step 7: Grant execute permission
GRANT EXECUTE ON FUNCTION match_sefaria_chunks_gemini_contextual(halfvec(3072), float, int) TO authenticated, anon;

-- =============================================================================
-- VERIFICATION QUERIES:
-- =============================================================================

-- Check columns were added:
-- SELECT column_name, data_type, udt_name 
-- FROM information_schema.columns 
-- WHERE table_name = 'sefaria_text_chunks' 
--   AND column_name IN ('context_text', 'content_with_context', 'embedding_gemini_contextual');

-- Check index was created:
-- SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'sefaria_text_chunks';

-- =============================================================================
-- QUERY TIME OPTIMIZATION:
-- Run this before queries for better recall:
-- SET LOCAL hnsw.ef_search = 60;
-- =============================================================================
