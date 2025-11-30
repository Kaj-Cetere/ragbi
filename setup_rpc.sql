-- Supabase RPC Function for Semantic Search
-- Run this in the Supabase SQL Editor: https://supabase.com/dashboard/project/_/sql/new

-- Drop existing function if it exists (for updates)
drop function if exists match_sefaria_chunks(vector(1536), float, int);

-- Create the semantic search function
create or replace function match_sefaria_chunks (
    query_embedding vector(1536),
    match_threshold float default 0.5,
    match_count int default 10
)
returns table (
    id uuid,
    content text,
    sefaria_ref text,
    metadata jsonb,
    similarity float
)
language sql stable
as $$
    select
        sefaria_text_chunks.id,
        sefaria_text_chunks.content,
        sefaria_text_chunks.sefaria_ref,
        sefaria_text_chunks.metadata,
        1 - (sefaria_text_chunks.embedding <=> query_embedding) as similarity
    from sefaria_text_chunks
    where 1 - (sefaria_text_chunks.embedding <=> query_embedding) > match_threshold
    order by (sefaria_text_chunks.embedding <=> query_embedding) asc
    limit match_count;
$$;

-- Grant execute permission to authenticated and anon users
grant execute on function match_sefaria_chunks(vector(1536), float, int) to authenticated, anon;

-- Test the function (optional - requires an embedding vector)
-- select * from match_sefaria_chunks(
--     '[0.1, 0.2, ...]'::vector(1536),  -- Replace with actual embedding
--     0.5,
--     5
-- );
