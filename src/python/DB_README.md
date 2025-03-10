# Document Store Integration

This module provides an integration to extract document embeddings and text from the PostgreSQL database and store them in a local DuckDB database for faster access and offline use.

## Features

- Extracts document texts and embeddings from the PostgreSQL database
- Truncates document content to appropriate token length using tiktoken
- Stores data in a portable DuckDB database file
- Provides a simple API for document and embedding retrieval
- Supports corpus-based filtering and pagination
- Includes example similarity search functionality with both NumPy and DuckDB
- Enables advanced vector operations directly in SQL

## Usage

### Creating a Document Store

To create a document store from the PostgreSQL database:

```python
from db_document_store import create_document_store

# Create a document store (default embedder_name is "openai_small")
create_document_store("data/document_store.db")

# Or with a specific embedder
create_document_store("data/document_store.db", embedder_name="other_embedder")
```

### Using the Document Store

```python
from db_document_store import DocStore

# Open the document store
with DocStore("data/document_store.db") as doc_store:
    # Get available corpora
    corpora = doc_store.get_corpora()
    print(f"Available corpora: {[c['name'] for c in corpora]}")
    
    # Get document count
    doc_count = doc_store.get_document_count("corpus_name")
    
    # Get documents with pagination
    docs = doc_store.get_documents(
        corpus_name="corpus_name",
        limit=10,
        offset=0
    )
    
    # Get document embeddings
    doc_ids, embeddings = doc_store.get_document_embeddings(
        corpus_name="corpus_name",
        limit=100
    )
    
    # Get a specific document
    doc = doc_store.get_document_by_id(doc_id=123)
    
    # Run custom SQL queries
    results = doc_store.run_query(
        "SELECT c.name, COUNT(*) FROM document d JOIN corpus c ON d.corpus_id = c.id GROUP BY c.name"
    )
```

### Similarity Search Example

The module includes an example script (`doc_store_example.py`) that demonstrates how to use the document store for similarity search:

```bash
python src/python/doc_store_example.py
```

The example shows how to:
1. Create a document store if it doesn't exist
2. Retrieve documents from a corpus
3. Perform similarity search using document embeddings
   - Using NumPy for vector operations
   - Using DuckDB's native vector functions
4. Display the most similar documents

## Implementation Details

### Document Truncation

Documents are truncated to fit within a specified token limit (default: 8191 tokens) using tiktoken, which is the same method used in the batch embeddings process. This ensures consistency between the stored text and the text used to generate embeddings.

### Data Storage

The DuckDB database contains two main tables:
- `corpus`: Stores information about each corpus
- `document`: Stores document content, corpus information, and embedding vectors as JSON

### Why DuckDB?

DuckDB offers several advantages for this use case:
- Fast analytical queries, optimized for OLAP workloads
- Native vector operations that enable efficient similarity search
- Self-contained database in a single file (like SQLite)
- Pandas integration for seamless data exchange
- SQL interface for flexible querying

### Performance Considerations

- Batched inserts using pandas DataFrames for better performance
- Document embeddings are stored as JSON strings and can be parsed to arrays in SQL
- Indices are created on frequently queried columns for faster lookup
- Vector operations can be performed directly in SQL, eliminating the need to load all vectors into memory

## Requirements

- SQLAlchemy (for PostgreSQL connection)
- duckdb
- tiktoken
- numpy
- pandas 
- tqdm