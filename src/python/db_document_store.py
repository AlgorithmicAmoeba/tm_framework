import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import duckdb
import numpy as np
import pandas as pd
import tiktoken
from sqlalchemy import select
from tqdm import tqdm

from database import get_session
from models import Corpus, Document, DocumentType, Embedder, Embedding
import configuration as cfg


class DocStore:
    """
    A document store that extracts embeddings and text from the PostgreSQL database
    and stores them in a local DuckDB database for faster access.
    """
    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize the document store.
        
        Args:
            db_path: Path to the DuckDB database file
        """
        self.db_path = str(Path(db_path))
        self._conn = None
        
    def __enter__(self):
        self._conn = duckdb.connect(self.db_path)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._conn:
            self._conn.close()
            self._conn = None
            
    @property
    def conn(self):
        """Get the DuckDB connection."""
        if self._conn is None:
            self._conn = duckdb.connect(self.db_path)
        return self._conn
    
    def _create_tables(self):
        """Create the necessary tables in the DuckDB database."""
        # Create corpus table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS corpus (
            id INTEGER PRIMARY KEY,
            name VARCHAR UNIQUE,
            description VARCHAR
        )
        """)
        
        # Create document table with embedding JSON column
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS document (
            id INTEGER PRIMARY KEY,
            corpus_id INTEGER,
            content VARCHAR,
            embedding_json VARCHAR,
            FOREIGN KEY (corpus_id) REFERENCES corpus (id)
        )
        """)
        
        # Create index on corpus_id for faster queries
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_document_corpus_id ON document (corpus_id)")
    
    def extract_and_store(self, pg_session, embedder_name: str = "openai_small", chunk_size: int = 8191):
        """
        Extract documents and embeddings from PostgreSQL and store them in DuckDB.
        
        Args:
            pg_session: SQLAlchemy session for PostgreSQL
            embedder_name: Name of the embedder to use for embeddings
            chunk_size: Maximum token count for document content
        """
        # Create tables
        self._create_tables()
        
        # Get the tokenizer for limiting document content length
        enc = tiktoken.encoding_for_model("gpt-4o")
        
        # Get embedder ID
        embedder = pg_session.query(Embedder).filter_by(name=embedder_name).first()
        if not embedder:
            raise ValueError(f"Embedder '{embedder_name}' not found in the database")
        
        # Get document type ID for raw documents
        raw_document_type = pg_session.query(DocumentType).filter_by(name='raw').first()
        if not raw_document_type:
            raise ValueError("Document type 'raw' not found in the database")
        
        # Get all corpora
        corpora = pg_session.query(Corpus).all()
        
        # Insert corpora into DuckDB
        for corpus in corpora:
            self.conn.execute(
                """
                INSERT INTO corpus (id, name, description) 
                VALUES (?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                name = excluded.name,
                description = excluded.description
                """,
                [corpus.id, corpus.name, corpus.description]
            )
        
        # Process each corpus
        for corpus in tqdm(corpora, desc="Processing corpora"):
            print(f"Processing corpus: {corpus.name}")
            
            # Get documents with embeddings for this corpus
            stmt = (
                select(Document, Embedding.vector)
                .join(Embedding, Document.id == Embedding.document_id)
                .filter(
                    Document.corpus_id == corpus.id,
                    Document.type_id == raw_document_type.id,
                    Embedding.embedder_id == embedder.id
                )
            )
            
            result = pg_session.execute(stmt).all()
            
            # Prepare for batch insertion
            batch_size = 100
            documents_batch = []
            
            for i, (document, vector) in enumerate(tqdm(result, desc=f"Processing documents in {corpus.name}")):
                # Shorten document text using tiktoken
                tokens = enc.encode(document.content)
                if len(tokens) > chunk_size:
                    tokens = tokens[:chunk_size]
                    shortened_content = enc.decode(tokens)
                else:
                    shortened_content = document.content
                
                # Convert vector to JSON string
                vector_json = json.dumps(vector)
                
                documents_batch.append({
                    'id': document.id,
                    'corpus_id': corpus.id,
                    'content': shortened_content,
                    'embedding_json': vector_json
                })
                
                # Insert in batches
                if len(documents_batch) >= batch_size:
                    # Convert to DataFrame and insert
                    df = pd.DataFrame(documents_batch)
                    self.conn.execute("INSERT INTO document SELECT * FROM df")
                    documents_batch = []
            
            # Insert any remaining documents
            if documents_batch:
                df = pd.DataFrame(documents_batch)
                self.conn.execute("INSERT INTO document SELECT * FROM df")
    
    def get_document_count(self, corpus_name: Optional[str] = None) -> int:
        """
        Get the number of documents in the database.
        
        Args:
            corpus_name: Optional corpus name to filter by
            
        Returns:
            Number of documents
        """
        if corpus_name:
            query = """
            SELECT COUNT(*) FROM document 
            JOIN corpus ON document.corpus_id = corpus.id
            WHERE corpus.name = ?
            """
            result = self.conn.execute(query, [corpus_name]).fetchone()
        else:
            result = self.conn.execute("SELECT COUNT(*) FROM document").fetchone()
            
        return result[0]
    
    def get_documents(self, 
                     corpus_name: Optional[str] = None, 
                     limit: Optional[int] = None, 
                     offset: int = 0) -> List[Dict]:
        """
        Get documents from the database.
        
        Args:
            corpus_name: Optional corpus name to filter by
            limit: Maximum number of documents to retrieve
            offset: Number of documents to skip
            
        Returns:
            List of document dictionaries with id, content, and embedding
        """
        if corpus_name:
            query = """
                SELECT d.id, d.content, d.embedding_json
                FROM document d
                JOIN corpus c ON d.corpus_id = c.id
                WHERE c.name = ?
            """
            params = [corpus_name]
        else:
            query = """
                SELECT id, content, embedding_json
                FROM document
            """
            params = []
            
        # Add limit and offset
        if limit:
            query += f" LIMIT {limit}"
        if offset:
            query += f" OFFSET {offset}"
            
        result = self.conn.execute(query, params).fetchall()
        
        documents = []
        for doc_id, content, embedding_json in result:
            embedding = json.loads(embedding_json)
            documents.append({
                "id": doc_id,
                "content": content,
                "embedding": embedding
            })
            
        return documents
    
    def get_document_embeddings(self, 
                              corpus_name: Optional[str] = None, 
                              limit: Optional[int] = None, 
                              offset: int = 0) -> Tuple[List[int], np.ndarray]:
        """
        Get document embeddings from the database.
        
        Args:
            corpus_name: Optional corpus name to filter by
            limit: Maximum number of documents to retrieve
            offset: Number of documents to skip
            
        Returns:
            Tuple of (document IDs, embeddings array)
        """
        if corpus_name:
            query = """
                SELECT d.id, d.embedding_json
                FROM document d
                JOIN corpus c ON d.corpus_id = c.id
                WHERE c.name = ?
            """
            params = [corpus_name]
        else:
            query = """
                SELECT id, embedding_json
                FROM document
            """
            params = []
            
        # Add limit and offset
        if limit:
            query += f" LIMIT {limit}"
        if offset:
            query += f" OFFSET {offset}"
            
        result = self.conn.execute(query, params).fetchall()
        
        doc_ids = []
        embeddings = []
        
        for doc_id, embedding_json in result:
            embedding = json.loads(embedding_json)
            doc_ids.append(doc_id)
            embeddings.append(embedding)
            
        return doc_ids, np.array(embeddings)
    
    def get_corpora(self) -> List[Dict]:
        """
        Get all corpora in the database.
        
        Returns:
            List of corpus dictionaries with id and name
        """
        result = self.conn.execute("SELECT id, name, description FROM corpus").fetchall()
        
        corpora = []
        for corpus_id, name, description in result:
            corpora.append({
                "id": corpus_id,
                "name": name,
                "description": description
            })
            
        return corpora
    
    def get_document_by_id(self, doc_id: int) -> Optional[Dict]:
        """
        Get a document by its ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document dictionary or None if not found
        """
        query = """
        SELECT id, content, embedding_json
        FROM document
        WHERE id = ?
        """
        
        result = self.conn.execute(query, [doc_id]).fetchone()
        if not result:
            return None
            
        doc_id, content, embedding_json = result
        embedding = json.loads(embedding_json)
        
        return {
            "id": doc_id,
            "content": content,
            "embedding": embedding
        }
    
    def run_query(self, query: str, params: Optional[List] = None) -> List:
        """
        Run a custom SQL query against the DuckDB database.
        
        Args:
            query: SQL query to execute
            params: Optional list of parameters for the query
            
        Returns:
            List of rows as tuples
        """
        if params:
            return self.conn.execute(query, params).fetchall()
        return self.conn.execute(query).fetchall()


def create_document_store(output_path: Union[str, Path], embedder_name: str = "openai_small"):
    """
    Create a document store by extracting data from PostgreSQL.
    
    Args:
        output_path: Path to the output DuckDB database file
        embedder_name: Name of the embedder to use for embeddings
    """
    # Load configuration
    config = cfg.load_config_from_env()
    
    # Convert to Path object and ensure output directory exists
    path = Path(output_path)
    if path.parent != Path('.'):
        path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract data and create DuckDB database
    with get_session(config.database) as pg_session:
        with DocStore(str(path)) as doc_store:
            doc_store.extract_and_store(pg_session, embedder_name)
    
    print(f"Document store created at {path}")


if __name__ == "__main__":
    # Example usage
    create_document_store("ignore/embedding_store.db")