-- General SQL Execution Function for Supabase
-- This function allows executing arbitrary SQL statements via RPC
-- Run this first in the Supabase SQL Editor

-- Drop existing function if it exists
drop function if exists exec_sql(text);

-- Create the SQL execution function
create or replace function exec_sql(sql_query text)
returns table (result text)
language plpgsql
as $$
begin
    -- Execute the SQL statement and return status
    execute sql_query;
    return query select 'SQL executed successfully'::text as result;
exception when others then
    -- Return error message if execution fails
    return query select 'ERROR: ' || sqlerrm::text as result;
end;
$$;

-- Grant execute permission to authenticated users
grant execute on function exec_sql(text) to authenticated, anon;
