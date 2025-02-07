import os
import json
import tiktoken
from sqlalchemy.orm import Session
from typing import Dict, List
import openai
from models import Corpus, Document
from database import get_session
from configuration import load_config_from_env

def create_batch_files(session: Session, output_dir: str) -> Dict[str, str]:
    """Create batch files for each corpus and return mapping of corpus name to file path."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    corpus_files = {}
    corpora = session.query(Corpus).all()
    
    for corpus in corpora:
        documents = session.query(Document).filter_by(corpus_id=corpus.id).all()
        batch_data = [{"id": doc.id, "content": doc.content} for doc in documents]
        
        file_path = os.path.join(output_dir, f"{corpus.name}_batch.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, ensure_ascii=False, indent=2)
        
        corpus_files[corpus.name] = file_path
    
    return corpus_files

def count_tokens_and_cost(file_paths: Dict[str, str]) -> Dict[str, Dict]:
    """Count tokens and estimate cost for each corpus using tiktoken."""
    encoding = tiktoken.get_encoding("cl100k_base")  # ada-002 uses this encoding
    results = {}
    
    # Current OpenAI API pricing for ada-002 embeddings
    PRICE_PER_1K_TOKENS = 0.0001
    
    for corpus_name, file_path in file_paths.items():
        with open(file_path, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
        
        total_tokens = 0
        for item in batch_data:
            tokens = len(encoding.encode(item['content']))
            total_tokens += tokens
        
        estimated_cost = (total_tokens / 1000) * PRICE_PER_1K_TOKENS
        
        results[corpus_name] = {
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost,
            "document_count": len(batch_data),
            "file_path": file_path
        }
    
    return results

def embed_documents(client: openai.Client, file_path: str, output_dir: str) -> str:
    """Generate embeddings for documents in a batch file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        batch_data = json.load(f)
    
    results = []
    for item in batch_data:
        response = client.embeddings.create(
            input=item['content'],
            model="text-embedding-ada-002"
        )
        results.append({
            "id": item['id'],
            "embedding": response.data[0].embedding
        })
    
    output_path = os.path.join(output_dir, f"embeddings_{os.path.basename(file_path)}")
    with open(output_path, 'w') as f:
        json.dump(results, f)
    
    return output_path

def main():
    # Load configuration and create OpenAI client
    config = load_config_from_env()
    client = openai.Client(api_key=config.openai.api_key)
    session = get_session(config.database)
    
    # Create directories
    batch_dir = "batches"
    embeddings_dir = "embeddings"
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Create batch files
    print("\nCreating batch files...")
    corpus_files = create_batch_files(session, batch_dir)
    
    # Calculate tokens and costs
    print("\nCalculating token counts and costs...")
    token_results = count_tokens_and_cost(corpus_files)
    
    # Print token counts and costs
    total_cost = 0
    print("\nToken count and cost estimation per corpus:")
    print("-" * 50)
    for corpus_name, stats in token_results.items():
        print(f"\nCorpus: {corpus_name}")
        print(f"Documents: {stats['document_count']}")
        print(f"Total tokens: {stats['total_tokens']:,}")
        print(f"Estimated cost: ${stats['estimated_cost_usd']:.4f}")
        total_cost += stats['estimated_cost_usd']
    
    print(f"\nTotal estimated cost for all corpora: ${total_cost:.4f}")
    
    # Confirm before proceeding
    proceed = input("\nDo you want to proceed with generating embeddings? (yes/no): ")
    if proceed.lower() != 'yes':
        print("Operation cancelled.")
        return
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    embedding_results = {}
    for corpus_name, stats in token_results.items():
        print(f"\nProcessing corpus: {corpus_name}")
        embedding_file = embed_documents(client, stats['file_path'], embeddings_dir)
        embedding_results[corpus_name] = embedding_file
        print(f"Embeddings saved to: {embedding_file}")
    
    print("\nAll embeddings generated successfully!")

if __name__ == '__main__':
    main()
