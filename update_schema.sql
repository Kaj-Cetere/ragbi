-- First, drop and recreate the table with auto-incrementing chunk_index
DROP TABLE IF EXISTS sefaria_text_chunks;

-- Enable Vector Extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- Main Table with chunk_index for:
-- 1. Maintaining correct order (order by chunk_index)
-- 2. Neighbor lookups for context (chunk_index - 1, chunk_index + 1)
CREATE TABLE sefaria_text_chunks (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  chunk_index BIGSERIAL NOT NULL,       -- Auto-increment: preserves insertion order
  content text NOT NULL,                -- The cleaned Hebrew text
  embedding vector(1536),               -- Truncated Qwen vector
  sefaria_ref text NOT NULL,            -- Critical: Used for API Hydration
  metadata jsonb NOT NULL,              -- Filtering: { "book": "...", "siman": 1, "seif": 1 }
  created_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Create index for vector similarity search
CREATE INDEX idx_embedding ON sefaria_text_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create index for chunk_index (for ordering and neighbor lookups)
CREATE INDEX idx_chunk_index ON sefaria_text_chunks(chunk_index);

-- Create index for book filtering
CREATE INDEX idx_metadata_book ON sefaria_text_chunks USING gin (metadata);

-- Create unique index on sefaria_ref for fast lookups
CREATE UNIQUE INDEX idx_sefaria_ref ON sefaria_text_chunks(sefaria_ref);

-- =============================================================================
-- USAGE EXAMPLES:
-- =============================================================================

-- Get chunks in order:
-- SELECT * FROM sefaria_text_chunks 
-- WHERE metadata->>'book' = 'Shulchan Arukh, Orach Chayim'
-- ORDER BY chunk_index;

-- Get a chunk with its neighbors for context:
-- WITH target AS (
--   SELECT chunk_index FROM sefaria_text_chunks WHERE sefaria_ref = 'Shulchan Arukh, Orach Chayim 1:1'
-- )
-- SELECT * FROM sefaria_text_chunks
-- WHERE chunk_index BETWEEN (SELECT chunk_index - 1 FROM target) 
--                       AND (SELECT chunk_index + 1 FROM target)
-- ORDER BY chunk_index;
