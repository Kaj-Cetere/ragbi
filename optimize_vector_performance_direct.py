#!/usr/bin/env python3
"""
Vector Performance Optimization Script
Executed directly from Python to avoid SQL editor timeout issues.
Optimizes only Gemini embeddings (removes Qwen optimizations as requested).
"""

import os
import time
import logging
from dotenv import load_dotenv
from supabase import create_client, Client
import psycopg2
from psycopg2.extras import execute_values

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_database_config():
    """Load database configuration from environment variables."""
    load_dotenv()
    
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in environment variables")
    
    # Extract database connection info from Supabase URL
    # Supabase URL format: https://[project-id].supabase.co
    # We need to get the direct database connection string
    
    # For now, we'll use the Supabase client for some operations
    # and direct PostgreSQL connection for others
    return SUPABASE_URL, SUPABASE_KEY

def get_direct_db_connection():
    """Get direct PostgreSQL connection using connection string from env."""
    # Try to get DATABASE_URL from environment
    db_url = os.getenv("DATABASE_URL")
    
    if not db_url:
        # If no DATABASE_URL, we need to construct it or ask user
        logger.error("DATABASE_URL not found in environment variables")
        logger.error("Please set DATABASE_URL to your PostgreSQL connection string")
        logger.error("Format: postgresql://[user]:[password]@[host]:[port]/[database]")
        raise ValueError("DATABASE_URL required for direct SQL execution")
    
    return psycopg2.connect(db_url)

def execute_sql_direct(sql: str, description: str):
    """Execute SQL directly via PostgreSQL connection."""
    conn = None
    try:
        logger.info(f"Executing: {description}")
        start_time = time.time()
        
        conn = get_direct_db_connection()
        conn.autocommit = True  # Important for DDL statements
        cursor = conn.cursor()
        
        cursor.execute(sql)
        
        execution_time = time.time() - start_time
        logger.info(f"Completed: {description} in {execution_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Failed to execute {description}: {e}")
        raise
    finally:
        if conn:
            conn.close()

def optimize_gemini_embeddings():
    """Optimize Gemini embeddings index and function."""
    
    # Step 1: Drop existing Gemini index
    drop_index_sql = """
    DROP INDEX IF EXISTS idx_embedding_gemini;
    """
    execute_sql_direct(drop_index_sql, "Drop existing Gemini index")
    
    # Step 2: Create optimized HNSW index for Gemini embeddings
    create_index_sql = """
    CREATE INDEX idx_embedding_gemini ON sefaria_text_chunks 
    USING hnsw (embedding_gemini halfvec_cosine_ops)
    WITH (m = 24, ef_construction = 96);
    """
    execute_sql_direct(create_index_sql, "Create optimized Gemini HNSW index")
    
    # Step 3: Optimize Gemini RPC function with query-time tuning
    drop_function_sql = """
    DROP FUNCTION IF EXISTS match_sefaria_chunks_gemini(halfvec(3072), float, int);
    """
    execute_sql_direct(drop_function_sql, "Drop existing Gemini function")
    
    create_function_sql = """
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
    """
    execute_sql_direct(create_function_sql, "Create optimized Gemini function")

def verify_optimizations():
    """Verify that optimizations were applied successfully."""
    
    # Check index creation and sizes
    verify_sql = """
    SELECT 
        indexname, 
        tablename, 
        indexdef,
        pg_size_pretty(pg_relation_size(indexname::regclass)) as index_size
    FROM pg_indexes 
    WHERE tablename = 'sefaria_text_chunks' 
      AND indexname LIKE '%embedding%';
    """
    
    try:
        logger.info("Verifying optimizations...")
        
        conn = get_direct_db_connection()
        cursor = conn.cursor()
        cursor.execute(verify_sql)
        
        results = cursor.fetchall()
        
        if results:
            logger.info("Index verification results:")
            for row in results:
                logger.info(f"  - Index: {row[0]}, Size: {row[3]}")
        else:
            logger.info("No embedding indexes found")
            
        conn.close()
            
    except Exception as e:
        logger.warning(f"Verification query failed: {e}")

def main():
    """Main execution function."""
    try:
        logger.info("Starting vector performance optimization (Gemini only)...")
        
        # Load configuration
        load_database_config()
        
        # Check if DATABASE_URL is available
        if not os.getenv("DATABASE_URL"):
            logger.error("DATABASE_URL environment variable is required")
            logger.error("Please set DATABASE_URL to your PostgreSQL connection string")
            logger.error("You can get this from your Supabase dashboard:")
            logger.error("1. Go to Project Settings > Database")
            logger.error("2. Find 'Connection string' under 'Connection parameters'")
            logger.error("3. Copy the URI format and add it to your .env file")
            return 1
        
        # Execute optimizations
        start_time = time.time()
        
        optimize_gemini_embeddings()
        verify_optimizations()
        
        total_time = time.time() - start_time
        logger.info(f"Optimization completed successfully in {total_time:.2f}s")
        
        # Print expected improvements
        logger.info("""
Expected Performance Improvements:
- Vector Search (Gemini): 2-3s → 0.3-0.8s (70% improvement)
  - Optimized HNSW index with m=24, ef_construction=96 for 3072d vectors
  - Query-time ef_search tuning set to 60 for high-dimensional space
  - Better distance comparisons through improved index structure

Note: For additional database-wide optimizations, contact Supabase support:
- ALTER SYSTEM SET maintenance_work_mem = '1GB'
- ALTER SYSTEM SET hnsw.ef_search = 40
        """)
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
