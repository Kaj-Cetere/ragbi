-- =============================================================================
-- Vector Search Performance Optimization Script
-- Based on analysis of 8-14 second query times in production
-- Target: Reduce total query time from 8-14s to 3-6s (50-60% improvement)
-- 
-- Performance Issues Identified:
-- 1. Vector search: 2-3s (15-26% of total time) - RPC calls consistently slow
-- 2. Embedding generation: 1-5s (20-37% of total time) - Gemini slower than Qwen  
-- 3. LLM generation: 4-7s (48-54% of total time) - Biggest bottleneck
-- =============================================================================

-- =============================================================================
-- STEP 1: OPTIMIZE QWEN EMBEDDINGS INDEX (1536 dimensions)
-- Current: IVFFLAT index with lists=100 (slower for vector search)
-- Target: HNSW index for 70% performance improvement
-- =============================================================================

-- Drop old IVFFLAT index (slower for vector search)
DROP INDEX IF EXISTS idx_embedding;

-- Create optimized HNSW index for Qwen embeddings
-- m=16: Good balance for 1536d vectors (original paper: 5-48 range)
-- ef_construction=64: Standard default, good balance of build time vs quality
CREATE INDEX idx_embedding_hnsw ON sefaria_text_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- =============================================================================
-- STEP 2: OPTIMIZE GEMINI EMBEDDINGS INDEX (3072 dimensions)  
-- Current: HNSW with m=16, ef_construction=64 (suboptimal for high dimensions)
-- Target: Higher m value for better 3072d performance
-- =============================================================================

-- Drop existing Gemini index (suboptimal settings)
DROP INDEX IF EXISTS idx_embedding_gemini;

-- Create optimized HNSW index for Gemini embeddings
-- m=24: Higher value better for 3072d vectors (original paper: bigger M for high dimensional)
-- ef_construction=96: 2x m value as required, better quality for high dimensions
CREATE INDEX idx_embedding_gemini ON sefaria_text_chunks 
USING hnsw (embedding_gemini halfvec_cosine_ops)
WITH (m = 24, ef_construction = 96);

-- =============================================================================
-- STEP 3: OPTIMIZE QWEN RPC FUNCTION WITH QUERY-TIME TUNING
-- Current: No ef_search optimization (uses default 40)
-- Target: Set optimal ef_search for balance of speed/accuracy
-- =============================================================================

DROP FUNCTION IF EXISTS match_sefaria_chunks(vector(1536), float, int);

CREATE OR REPLACE FUNCTION match_sefaria_chunks (
    query_embedding vector(1536),
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
LANGUAGE plpgsql STABLE
AS $$
BEGIN
    -- Set ef_search for optimal balance: higher than default for better accuracy
    -- but not too high to maintain speed. 40 is good baseline, can tune per needs
    SET LOCAL hnsw.ef_search = 40;
    
    RETURN QUERY
    SELECT
        sefaria_text_chunks.id,
        sefaria_text_chunks.content,
        sefaria_text_chunks.sefaria_ref,
        sefaria_text_chunks.metadata,
        1 - (sefaria_text_chunks.embedding <=> query_embedding) AS similarity
    FROM sefaria_text_chunks
    WHERE 1 - (sefaria_text_chunks.embedding <=> query_embedding) > match_threshold
    ORDER BY (sefaria_text_chunks.embedding <=> query_embedding) ASC
    LIMIT match_count;
END;
$$;

-- =============================================================================
-- STEP 4: OPTIMIZE GEMINI RPC FUNCTION WITH QUERY-TIME TUNING
-- Current: No ef_search optimization (uses default 40)
-- Target: Higher ef_search for 3072d vectors (more complex search space)
-- =============================================================================

DROP FUNCTION IF EXISTS match_sefaria_chunks_gemini(halfvec(3072), float, int);

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
LANGUAGE plpgsql STABLE
AS $$
BEGIN
    -- Higher ef_search for 3072d vectors to maintain accuracy
    -- 60 provides better results for high-dimensional space
    SET LOCAL hnsw.ef_search = 60;
    
    RETURN QUERY
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
END;
$$;

-- =============================================================================
-- STEP 5: DATABASE CONFIGURATION OPTIMIZATIONS
-- These require superuser access - request from Supabase support if needed
-- =============================================================================

-- Note: These commands require superuser privileges
-- Contact Supabase support to apply these database-wide settings:

-- ALTER SYSTEM SET shared_preload_libraries = 'vector';
-- ALTER SYSTEM SET maintenance_work_mem = '1GB';  -- For faster index builds
-- ALTER SYSTEM SET hnsw.ef_search = 40;  -- Global default
-- SELECT pg_reload_conf();  -- Reload configuration

-- =============================================================================
-- STEP 6: PERFORMANCE VERIFICATION QUERIES
-- Run these after applying optimizations to verify improvements
-- =============================================================================

-- Check index creation and sizes
SELECT 
    indexname, 
    tablename, 
    indexdef,
    pg_size_pretty(pg_relation_size(indexname::regclass)) as index_size
FROM pg_indexes 
WHERE tablename = 'sefaria_text_chunks' 
  AND indexname LIKE '%embedding%';

-- Test Qwen search performance (run multiple times to average)
-- EXPLAIN ANALYZE SELECT * FROM match_sefaria_chunks(
--     '[0.1, 0.2, 0.3]'::vector(1536),  -- Replace with actual embedding
--     0.5,
--     10
-- );

-- Test Gemini search performance (run multiple times to average)  
-- EXPLAIN ANALYZE SELECT * FROM match_sefaria_chunks_gemini(
--     '[0.1, 0.2, 0.3]'::halfvec(3072),  -- Replace with actual embedding
--     0.5,
--     10
-- );

-- =============================================================================
-- EXPECTED PERFORMANCE IMPROVEMENTS
-- =============================================================================
-- 
-- Vector Search: 2-3s → 0.3-0.8s (70% improvement)
--   - HNSW indexes reduce distance comparisons
--   - Optimized m/ef_construction for dimension sizes  
--   - Query-time ef_search tuning
--
-- Embedding Generation: 1-5s → 1-2s (20-40% improvement)
--   - Prefer Qwen over Gemini for speed
--   - Consider caching repeated embeddings
--
-- LLM Generation: 4-7s → 2-4s (30-40% improvement)  
--   - Reduce max_tokens: 2000 → 1000
--   - Lower temperature: 0.3 → 0.1
--   - Faster model selection
--
-- TOTAL QUERY TIME: 8-14s → 3-6s (50-60% improvement)
--
-- =============================================================================
-- IMPLEMENTATION NOTES:
-- =============================================================================
-- 1. Run this script during low-traffic period (index rebuilds take time)
-- 2. Monitor query times after deployment using the enhanced logging in app.py
-- 3. Consider implementing result caching for frequently asked questions
-- 4. Test with both Qwen and Gemini embeddings to compare real-world performance
-- 5. If performance is still insufficient, consider reducing match_count from 10 to 5
--
-- =============================================================================

-- Script created: November 28 2025
-- Based on production performance analysis and pgvector optimization research
