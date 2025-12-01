-- General SQL Execution Function for Supabase
-- This function allows executing arbitrary SQL statements via RPC
-- Run this FIRST in the Supabase SQL Editor before running the Python script

-- Drop existing function if it exists
DROP FUNCTION IF EXISTS exec_sql(text);

-- Create the SQL execution function
CREATE OR REPLACE FUNCTION exec_sql(sql_query text)
RETURNS TABLE (result text)
LANGUAGE plpgsql
AS $$
BEGIN
    -- Execute the SQL statement and return status
    EXECUTE sql_query;
    RETURN QUERY SELECT 'SQL executed successfully'::text AS result;
EXCEPTION WHEN OTHERS THEN
    -- Return error message if execution fails
    RETURN QUERY SELECT 'ERROR: ' || SQLERRM::text AS result;
END;
$$;

-- Grant execute permission to authenticated and anon users
GRANT EXECUTE ON FUNCTION exec_sql(text) TO authenticated, anon;

-- Test the function (optional)
-- SELECT * FROM exec_sql('SELECT 1 as test');
