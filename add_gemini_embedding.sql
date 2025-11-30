-- =============================================================================
-- Migration: Add Gemini Embedding Column (halfvec for 3072 dimensions)
-- Requires: pgvector 0.7.0+ (Supabase has this by default on new projects)
-- =============================================================================

-- Step 1: Add the new halfvec column for Gemini embeddings (3072 dimensions)
-- halfvec uses 16-bit floats, allowing up to 4000 dimensions (vs 2000 for vector)
ALTER TABLE sefaria_text_chunks
ADD COLUMN IF NOT EXISTS embedding_gemini halfvec(3072);

-- Step 2: Create an HNSW index for the Gemini embeddings
-- Using halfvec_cosine_ops for cosine similarity
CREATE INDEX IF NOT EXISTS idx_embedding_gemini 
ON sefaria_text_chunks 
USING hnsw (embedding_gemini halfvec_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Step 3: Drop existing Gemini search function if it exists
DROP FUNCTION IF EXISTS match_sefaria_chunks_gemini(halfvec(3072), float, int);

-- Step 4: Create new RPC function for Gemini-based semantic search
CREATE OR REPLACE FUNCTION match_sefaria_chunks_gemini (
    query_embedding halfvec(3072),
    match_threshold float DEFAULT 0.5,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id uuid,
    content text,
    sefaria_ref text,
    metadata jsonb,
    similarity float
)
LANGUAGE sql STABLE
AS $$
    SELECT
        sefaria_text_chunks.id,
        sefaria_text_chunks.content,
        sefaria_text_chunks.sefaria_ref,
        sefaria_text_chunks.metadata,
        1 - (sefaria_text_chunks.embedding_gemini <=> query_embedding) AS similarity
    FROM sefaria_text_chunks
    WHERE sefaria_text_chunks.embedding_gemini IS NOT NULL
      AND 1 - (sefaria_text_chunks.embedding_gemini <=> query_embedding) > match_threshold
    ORDER BY (sefaria_text_chunks.embedding_gemini <=> query_embedding) ASC
    LIMIT match_count;
$$;

-- Step 5: Grant execute permission to authenticated and anon users
GRANT EXECUTE ON FUNCTION match_sefaria_chunks_gemini(halfvec(3072), float, int) TO authenticated, anon;

-- =============================================================================
-- VERIFICATION QUERIES:
-- =============================================================================

-- Check column was added:
-- SELECT column_name, data_type, udt_name 
-- FROM information_schema.columns 
-- WHERE table_name = 'sefaria_text_chunks' AND column_name = 'embedding_gemini';

-- Check how many rows have Gemini embeddings:
-- SELECT COUNT(*) FROM sefaria_text_chunks WHERE embedding_gemini IS NOT NULL;

-- Test the function (requires embedding vector):
-- SELECT * FROM match_sefaria_chunks_gemini(
--     '[0.1, 0.2, ...]'::halfvec(3072),
--     0.3,
--     5
-- );
