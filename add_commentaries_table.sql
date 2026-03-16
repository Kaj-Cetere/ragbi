-- =============================================================================
-- Migration: Add table for cached meforshim/commentaries
-- Run this in Supabase SQL Editor
-- =============================================================================

CREATE TABLE IF NOT EXISTS sefaria_commentaries (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  base_ref text NOT NULL,
  commentary_ref text NOT NULL,
  commentator text NOT NULL,
  text_he text NOT NULL,
  base_book text,
  siman int,
  seif int,
  source text DEFAULT 'sefaria' NOT NULL,
  fetched_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
  created_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
  updated_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_commentaries_base_ref_commentary_ref
ON sefaria_commentaries(base_ref, commentary_ref);

CREATE INDEX IF NOT EXISTS idx_commentaries_base_ref
ON sefaria_commentaries(base_ref);

CREATE INDEX IF NOT EXISTS idx_commentaries_commentator
ON sefaria_commentaries(commentator);

CREATE INDEX IF NOT EXISTS idx_commentaries_base_book
ON sefaria_commentaries(base_book);

CREATE OR REPLACE FUNCTION set_commentaries_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = timezone('utc'::text, now());
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_set_commentaries_updated_at ON sefaria_commentaries;
CREATE TRIGGER trg_set_commentaries_updated_at
BEFORE UPDATE ON sefaria_commentaries
FOR EACH ROW
EXECUTE FUNCTION set_commentaries_updated_at();
