



def find_embedding_jobs(
    session: Session,
    duckdb_cache_session: DuckDBPyConnection,
    source_table: str,
    cache_table: str,
):
    """
    Look in the source table for any rows that do not exist in the cache table."
    Place them in the jobs table.
    """
    ...