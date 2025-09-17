"""
Tokenizer testing script for comparing tokenization efficiency across different tokenizers.
This script calculates the average number of characters per token for different tokenizers
on a set of texts from the database.
"""
import logging
import statistics
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

import tiktoken
from sentence_transformers import SentenceTransformer
from sentence_transformers import SparseEncoder
from sqlalchemy import text
from sqlalchemy.orm import Session

from database import get_session
import configuration as cfg


@dataclass
class TokenizerResult:
    """Results for a single tokenizer on a set of texts."""
    tokenizer_name: str
    total_texts: int
    total_characters: int
    total_tokens: int
    avg_chars_per_token: float
    median_chars_per_token: float
    min_chars_per_token: float
    max_chars_per_token: float


class TokenizerTester:
    """Main class for testing tokenizers on text data."""
    
    def __init__(self, tokenizer_configs: List[Dict[str, Any]]):
        """
        Initialize the tokenizer tester.
        
        Args:
            tokenizer_configs: List of tokenizer configurations with 'name' and 'type' keys
        """
        self.tokenizer_configs = tokenizer_configs
        self.tokenizers = self._create_tokenizers()
    
    def _create_tokenizers(self) -> Dict[str, Any]:
        """
        Create tokenizer instances from the provided configurations.
        
        Returns:
            Dictionary mapping tokenizer names to tokenizer instances
        """
        tokenizers = {}
        for config in self.tokenizer_configs:
            name = config['name']
            tokenizer_type = config['type']
            
            try:
                if tokenizer_type == 'tiktoken':
                    tokenizers[name] = tiktoken.get_encoding(name)
                elif tokenizer_type == 'sentence_transformer':
                    tokenizers[name] = SentenceTransformer(name)
                elif tokenizer_type == 'sparse_encoder':
                    tokenizers[name] = SparseEncoder(name)
                    tokenizers[name].max_seq_length = 256
                else:
                    raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
                
                logging.info(f"Successfully created tokenizer: {name} ({tokenizer_type})")
            except Exception as e:
                logging.error(f"Failed to create tokenizer '{name}' ({tokenizer_type}): {e}")
                raise
        
        return tokenizers
    
    def fetch_texts_from_database(
        self, 
        session: Session, 
        corpus_name: str, 
        limit: int = 1000
    ) -> List[str]:
        """
        Fetch texts from the database for a specified corpus.
        
        Args:
            session: Database session
            corpus_name: Name of the corpus to fetch texts from
            limit: Maximum number of texts to fetch
            
        Returns:
            List of text contents
        """
        logging.info(f"Fetching up to {limit} texts from corpus: {corpus_name}")
        
        fetch_query = text("""
            SELECT content 
            FROM pipeline.used_raw_document
            WHERE corpus_name = :corpus_name
            LIMIT :limit
        """)
        
        result = session.execute(
            fetch_query, 
            {"corpus_name": corpus_name, "limit": limit}
        )
        
        texts = [row[0] for row in result]
        logging.info(f"Fetched {len(texts)} texts from corpus: {corpus_name}")
        
        return texts
    
    def _tokenize_text(self, tokenizer: Any, text: str, tokenizer_type: str) -> List[int]:
        """
        Tokenize text using the appropriate method for the tokenizer type.
        
        Args:
            tokenizer: The tokenizer instance
            text: Text to tokenize
            tokenizer_type: Type of tokenizer ('tiktoken', 'sentence_transformer', 'sparse_encoder')
            
        Returns:
            List of token IDs
        """
        if tokenizer_type == 'tiktoken':
            return tokenizer.encode(text)
        elif tokenizer_type == 'sentence_transformer':
            # For SentenceTransformer, we need to access the tokenizer
            tokens = tokenizer.tokenize([text])["input_ids"][0]
            tokens = tokens[1:-1] # remove the [CLS] and [SEP] tokens
            decoded_text = tokenizer.tokenizer.decode(tokens)
            char_count = len(decoded_text)
            token_count = len(tokens)
            return char_count, token_count
        elif tokenizer_type == 'sparse_encoder':
            # For SparseEncoder, we need to access the tokenizer
            tokens = tokenizer.tokenize([text])["input_ids"][0]
            tokens = tokens[1:-1] # remove the [CLS] and [SEP] tokens
            decoded_text = tokenizer.tokenizer.decode(tokens)
            char_count = len(decoded_text)
            token_count = len(tokens)
            return char_count, token_count
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
    
    def calculate_chars_per_token(self, texts: List[str]) -> Dict[str, TokenizerResult]:
        """
        Calculate characters per token statistics for each tokenizer.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            Dictionary mapping tokenizer names to TokenizerResult objects
        """
        logging.info(f"Calculating chars per token for {len(texts)} texts using {len(self.tokenizers)} tokenizers")
        
        results = {}
        
        for config in self.tokenizer_configs:
            tokenizer_name = config['name']
            tokenizer_type = config['type']
            tokenizer = self.tokenizers[tokenizer_name]
            
            logging.info(f"Processing tokenizer: {tokenizer_name} ({tokenizer_type})")
            
            chars_per_token_list = []
            total_chars = 0
            total_tokens = 0
            
            for text in texts:
                if not text or not text.strip():
                    continue
                
                try:
                    # Tokenize the text using the appropriate method
                    char_count, token_count = self._tokenize_text(tokenizer, text, tokenizer_type)
                    
                    if token_count > 0:
                        chars_per_token = char_count / token_count
                        chars_per_token_list.append(chars_per_token)
                        total_chars += char_count
                        total_tokens += token_count
                        
                except Exception as e:
                    logging.warning(f"Failed to tokenize text with {tokenizer_name}: {e}")
                    continue
            
            # Calculate statistics
            if chars_per_token_list:
                avg_chars_per_token = total_chars / total_tokens
                median_chars_per_token = statistics.median(chars_per_token_list)
                min_chars_per_token = min(chars_per_token_list)
                max_chars_per_token = max(chars_per_token_list)
            else:
                avg_chars_per_token = 0
                median_chars_per_token = 0
                min_chars_per_token = 0
                max_chars_per_token = 0
            
            results[tokenizer_name] = TokenizerResult(
                tokenizer_name=tokenizer_name,
                total_texts=len(chars_per_token_list),
                total_characters=total_chars,
                total_tokens=total_tokens,
                avg_chars_per_token=avg_chars_per_token,
                median_chars_per_token=median_chars_per_token,
                min_chars_per_token=min_chars_per_token,
                max_chars_per_token=max_chars_per_token
            )
            
            logging.info(f"Completed tokenizer: {tokenizer_name}")
        
        return results
    
    def print_results(self, results: Dict[str, TokenizerResult]) -> None:
        """
        Print the tokenizer comparison results in a formatted table.
        
        Args:
            results: Dictionary of tokenizer results
        """
        print("\n" + "="*100)
        print("TOKENIZER COMPARISON RESULTS")
        print("="*100)
        
        # Header
        print(f"{'Tokenizer':<20} {'Texts':<8} {'Avg Chars/Token':<15} {'Median':<10} {'Min':<8} {'Max':<8} {'Total Chars':<12} {'Total Tokens':<12}")
        print("-"*100)
        
        # Results
        for tokenizer_name, result in results.items():
            print(f"{tokenizer_name:<20} "
                  f"{result.total_texts:<8} "
                  f"{result.avg_chars_per_token:<15.2f} "
                  f"{result.median_chars_per_token:<10.2f} "
                  f"{result.min_chars_per_token:<8.2f} "
                  f"{result.max_chars_per_token:<8.2f} "
                  f"{result.total_characters:<12,} "
                  f"{result.total_tokens:<12,}")
        
        print("="*100)
        
        # Summary
        print("\nSUMMARY:")
        print(f"Best average chars per token: {max(results.values(), key=lambda x: x.avg_chars_per_token).tokenizer_name}")
        print(f"Worst average chars per token: {min(results.values(), key=lambda x: x.avg_chars_per_token).tokenizer_name}")
        
        # Token efficiency comparison
        print("\nTOKEN EFFICIENCY (higher chars per token = more efficient):")
        sorted_results = sorted(results.values(), key=lambda x: x.avg_chars_per_token, reverse=True)
        for i, result in enumerate(sorted_results, 1):
            print(f"{i}. {result.tokenizer_name}: {result.avg_chars_per_token:.2f} chars/token")


def get_available_corpora() -> List[str]:
    """
    Get list of available corpora from the chunking main.py file.
    
    Returns:
        List of corpus names
    """
    return [
        "newsgroups",
        "wikipedia_sample", 
        "imdb_reviews",
        "trec_questions",
        "twitter-financial-news",
        "pubmed-multilabel",
        "patent-classification",
        "goodreads-bookgenres",
        "battery-abstracts",
        "t2-ragbench-convfinqa"
    ]


def main():
    """Main execution function."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration - tokenizer configurations with name and type
    tokenizer_configs = [
        {
            'name': 'naver/splade-v3',
            'type': 'sparse_encoder'
        },
        {
            'name': 'all-MiniLM-L6-v2',
            'type': 'sentence_transformer'
        },
    ]
    
    corpus_name = "wikipedia_sample"  # Change this to test different corpora
    text_limit = 1000  # Number of texts to fetch for testing
    
    print(f"Testing tokenizers:")
    for config in tokenizer_configs:
        print(f"  - {config['name']} ({config['type']})")
    print(f"Using corpus: {corpus_name}")
    print(f"Text limit: {text_limit}")
    print(f"Available corpora: {get_available_corpora()}")
    
    # Load configuration
    config = cfg.load_config_from_env()
    db_config = config.database
    
    # Create database session
    with get_session(db_config) as session:
        # Initialize tokenizer tester
        tester = TokenizerTester(tokenizer_configs)
        
        # Fetch texts from database
        texts = tester.fetch_texts_from_database(session, corpus_name, text_limit)
        
        if not texts:
            logging.error(f"No texts found for corpus: {corpus_name}")
            return
        
        # Calculate results
        results = tester.calculate_chars_per_token(texts)
        
        # Print results
        tester.print_results(results)


if __name__ == '__main__':
    main()
