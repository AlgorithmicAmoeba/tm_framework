"""
Corpus features pipeline for calculating various statistical features of text corpora.
This pipeline calculates document-level distributions and their statistical metrics,
as well as dataset-level metrics.
"""
import logging
import math
import re
import zlib
from typing import Any

import tqdm
import numpy as np
import spacy
import torch
from scipy import stats
from sqlalchemy import text
from sqlalchemy.orm import Session
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from database import get_session
import configuration as cfg


class CorpusFeatureCalculator:
    """Calculate various corpus-level features from text documents."""
    
    def __init__(self):
        # Initialize spaCy for sentence tokenization
        self.nlp = spacy.load("en_core_web_sm")
        # Remove parser and add sentencizer for faster processing
        if self.nlp.has_pipe("parser"):
            self.nlp.remove_pipe("parser")
        if not self.nlp.has_pipe("sentencizer"):
            self.nlp.add_pipe("sentencizer")
        
        # Initialize GPT-2 for perplexity calculation
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
    
    def calculate_document_distributions(self, texts: list[str]) -> dict[str, list[float]]:
        """
        Calculate document-level distributions for all texts.
        
        Args:
            texts: List of document texts
            
        Returns:
            Dictionary mapping distribution names to lists of values per document
        """
        distributions = {
            'words_per_document': [],
            'sentences_per_document': [],
            'characters_per_word': [],
            'words_per_sentence': [],
            # 'gpt2_perplexity': [],
            'compression_ratio': [],
            'word_entropy': []
        }
        
        # Process texts in batches with spaCy for better performance
        batch_size = 1000  # Adjust based on memory constraints
        
        for i in tqdm.tqdm(range(0, len(texts), batch_size), desc="Processing text batches"):
            batch_texts = texts[i:i + batch_size]
            
            # Process batch with spaCy - much faster than individual processing
            docs = list(self.nlp.pipe(batch_texts, disable=['ner', 'lemmatizer']))
            
            for text, doc in zip(batch_texts, docs):
                # Extract tokens and sentences more efficiently
                tokens = [token for token in doc if not token.is_space]
                sentences = list(doc.sents)
                
                # Cache lengths to avoid recalculation
                num_words = len(tokens)
                num_sentences = len(sentences)
                
                distributions['words_per_document'].append(num_words)
                distributions['sentences_per_document'].append(num_sentences)
                
                # Characters per word - vectorized calculation
                if num_words > 0:
                    char_counts = [len(token.text) for token in tokens]
                    distributions['characters_per_word'].extend(char_counts)
                else:
                    distributions['characters_per_word'].append(0)
                
                # Words per sentence - optimized calculation
                if num_sentences > 0:
                    words_per_sent = [sum(1 for token in sent if not token.is_space) 
                                    for sent in sentences]
                    distributions['words_per_sentence'].extend(words_per_sent)
                else:
                    distributions['words_per_sentence'].append(0)
                
                # Compression ratio
                compression_ratio = self._calculate_compression_ratio(text)
                distributions['compression_ratio'].append(compression_ratio)
                
                # Word entropy - pass token texts directly
                word_texts = [token.text for token in tokens]
                word_entropy = self._calculate_word_entropy(word_texts)
                distributions['word_entropy'].append(word_entropy)
        
        return distributions
    
    def _calculate_gpt2_perplexity(self, text: str) -> float:
        """Calculate GPT-2 log perplexity for a document using sliding window with 10% overlap."""
        try:
            # Tokenize the full text
            tokens = self.gpt2_tokenizer.encode(text)
            
            if len(tokens) == 0:
                return float('inf')
            
            # Parameters for sliding window
            max_length = 1024
            overlap_ratio = 0.1
            stride = int(max_length * (1 - overlap_ratio))  # 90% of max_length
            
            # If text fits in one window, process directly
            if len(tokens) <= max_length:
                try:
                    encodings = self.gpt2_tokenizer.encode_plus(
                        text, 
                        return_tensors='pt', 
                        truncation=True, 
                        max_length=max_length
                    )
                    
                    with torch.no_grad():
                        outputs = self.gpt2_model(**encodings, labels=encodings.input_ids)
                        return outputs.loss.item()  # Return log perplexity (cross-entropy loss)
                        
                except Exception as e:
                    logging.warning(f"Error in single window perplexity calculation: {e}")
                    return float('inf')
            
            # Process text in sliding windows
            log_likelihoods = []
            total_tokens = 0
            
            for i in range(0, len(tokens), stride):
                # Extract window tokens
                window_tokens = tokens[i:i + max_length]
                
                if len(window_tokens) < 2:  # Need at least 2 tokens for loss calculation
                    continue
                
                try:
                    # Convert back to text and tokenize properly
                    window_text = self.gpt2_tokenizer.decode(window_tokens)
                    encodings = self.gpt2_tokenizer.encode_plus(
                        window_text,
                        return_tensors='pt',
                        truncation=True,
                        max_length=max_length
                    )
                    
                    with torch.no_grad():
                        outputs = self.gpt2_model(**encodings, labels=encodings.input_ids)
                        loss = outputs.loss.item()
                        
                        # Weight by number of tokens in this window
                        window_length = encodings.input_ids.shape[1]
                        log_likelihoods.append(loss * window_length)
                        total_tokens += window_length
                        
                except Exception as e:
                    logging.warning(f"Error in window perplexity calculation: {e}")
                    continue
            
            # Calculate weighted average log perplexity
            if total_tokens > 0 and log_likelihoods:
                avg_log_perplexity = sum(log_likelihoods) / total_tokens
                return avg_log_perplexity
            else:
                return float('inf')
                
        except Exception as e:
            logging.warning(f"Error calculating GPT-2 log perplexity: {e}")
            return float('inf')
    
    def _calculate_compression_ratio(self, text: str) -> float:
        """Calculate compression ratio using zlib."""
        if not text:
            return 0.0
        
        original_size = len(text.encode('utf-8'))
        compressed_size = len(zlib.compress(text.encode('utf-8')))
        
        return compressed_size / original_size if original_size > 0 else 0.0
    
    def _calculate_word_entropy(self, words: list[str]) -> float:
        """Calculate word entropy for a document."""
        if not words:
            return 0.0
        
        # Count word frequencies
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Calculate probabilities and entropy
        total_words = len(words)
        entropy = 0.0
        
        for count in word_counts.values():
            prob = count / total_words
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def calculate_distribution_metrics(self, values: list[float]) -> dict[str, float]:
        """
        Calculate statistical metrics for a distribution.
        
        Args:
            values: List of numerical values
            
        Returns:
            Dictionary of metric names to values
        """
        if not values or all(math.isnan(v) or math.isinf(v) for v in values):
            return {
                'average': 0.0,
                'iqr_range': 0.0,
                'kurtosis': 0.0,
                'median': 0.0,
                'std_dev': 0.0,
                'fano_factor': 0.0,
                'kl_div_uniform': 0.0
            }
        
        # Filter out invalid values
        valid_values = [v for v in values if not (math.isnan(v) or math.isinf(v))]
        
        if not valid_values:
            return {
                'average': 0.0,
                'iqr_range': 0.0,
                'kurtosis': 0.0,
                'median': 0.0,
                'std_dev': 0.0,
                'fano_factor': 0.0,
                'kl_div_uniform': 0.0
            }
        
        arr = np.array(valid_values)
        
        # Calculate basic statistics
        mean_val = np.mean(arr)
        median_val = np.median(arr)
        std_val = np.std(arr)
        
        # IQR range
        q75, q25 = np.percentile(arr, [75, 25])
        iqr_range = q75 - q25
        
        # Kurtosis
        kurtosis_val = stats.kurtosis(arr)
        
        # Fano factor (variance / mean)
        fano_factor = np.var(arr) / mean_val if mean_val > 0 else 0.0
        
        # KL divergence versus uniform distribution
        kl_div = self._calculate_kl_divergence_uniform(arr)
        
        return {
            'average': float(mean_val),
            'iqr_range': float(iqr_range),
            'kurtosis': float(kurtosis_val),
            'median': float(median_val),
            'std_dev': float(std_val),
            'fano_factor': float(fano_factor),
            'kl_div_uniform': float(kl_div)
        }
    
    def _calculate_kl_divergence_uniform(self, values: np.ndarray) -> float:
        """Calculate KL divergence versus uniform distribution."""
        if len(values) == 0:
            return 0.0
        
        # Create histogram
        hist, bin_edges = np.histogram(values, bins=min(50, len(values)), density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        
        # Normalize to get probabilities
        p = hist * bin_width
        p = p[p > 0]  # Remove zero probabilities
        
        if len(p) == 0:
            return 0.0
        
        # Uniform distribution probability
        q = 1.0 / len(p)
        
        # Calculate KL divergence
        kl_div = np.sum(p * np.log(p / q))
        
        return kl_div
    
    def calculate_dataset_metrics(self, texts: list[str]) -> dict[str, float]:
        """
        Calculate dataset-level metrics.
        
        Args:
            texts: List of document texts
            
        Returns:
            Dictionary of dataset metric names to values
        """
        # Number of documents
        num_documents = len(texts)
        
        # Document compression ratio (average across all documents)
        if texts:
            compression_ratios = [self._calculate_compression_ratio(text) for text in texts]
            avg_compression_ratio = np.mean(compression_ratios)
        else:
            avg_compression_ratio = 0.0
        
        return {
            'num_documents': float(num_documents),
            'document_compression_ratio': float(avg_compression_ratio)
        }


def store_corpus_features(session: Session, corpus_name: str, features: dict[str, float]):
    """
    Store corpus features in the database using upsert pattern.
    
    Args:
        session: Database session
        corpus_name: Name of the corpus
        features: Dictionary of feature names to values
    """
    for feature_name, feature_value in features.items():
        upsert_query = text("""
            INSERT INTO pipeline.corpus_features (corpus_name, feature_name, feature_value)
            VALUES (:corpus_name, :feature_name, :feature_value)
            ON CONFLICT (corpus_name, feature_name)
            DO UPDATE SET 
                feature_value = :feature_value,
                updated_at = CURRENT_TIMESTAMP
        """)
        
        session.execute(
            upsert_query,
            {
                "corpus_name": corpus_name,
                "feature_name": feature_name,
                "feature_value": feature_value
            }
        )


def store_corpus_distributions(session: Session, corpus_name: str, distributions: dict[str, list[float]]):
    """
    Store raw distribution data (feature vectors) in the database using upsert pattern.
    
    Args:
        session: Database session
        corpus_name: Name of the corpus
        distributions: Dictionary mapping distribution names to lists of data points
    """
    for distribution_name, data_points in distributions.items():
        if distribution_name == 'characters_per_word':
            continue
        # Filter out infinite and NaN values
        valid_points = [float(x) for x in data_points if not (math.isnan(x) or math.isinf(x))]
        
        if not valid_points:
            continue
            
        upsert_query = text("""
            INSERT INTO pipeline.corpus_distributions (corpus_name, distribution_name, data_points, num_points)
            VALUES (:corpus_name, :distribution_name, :data_points, :num_points)
            ON CONFLICT (corpus_name, distribution_name)
            DO UPDATE SET 
                data_points = :data_points,
                num_points = :num_points,
                updated_at = CURRENT_TIMESTAMP
        """)
        
        session.execute(
            upsert_query,
            {
                "corpus_name": corpus_name,
                "distribution_name": distribution_name,
                "data_points": valid_points,
                "num_points": len(valid_points)
            }
        )


def calculate_corpus_features(session: Session, corpus_name: str, force_recalculate: bool = False):
    """
    Calculate and store all corpus features for a given corpus.
    
    Args:
        session: Database session
        corpus_name: Name of the corpus to process
        force_recalculate: If True, recalculate even if distributions already exist
        
    Returns:
        Number of features calculated
    """
    # Fetch documents from the corpus
    fetch_query = text("""
        SELECT content FROM pipeline.document 
        WHERE corpus_name = :corpus_name
        ORDER BY id
    """)
    
    result = session.execute(fetch_query, {"corpus_name": corpus_name})
    texts = [row.content for row in result.fetchall()]
    
    if not texts:
        logging.warning(f"No documents found for corpus '{corpus_name}'")
        return 0
    
    num_documents = len(texts)
    logging.info(f"Processing {num_documents} documents for corpus '{corpus_name}'")
    
    # Check if distributions already exist with correct number of points
    if not force_recalculate:
        check_query = text("""
            SELECT distribution_name, num_points 
            FROM pipeline.corpus_distributions 
            WHERE corpus_name = :corpus_name
        """)
        
        existing_distributions = session.execute(check_query, {"corpus_name": corpus_name}).fetchall()
        
        if existing_distributions:
            # Check if all expected distributions exist with correct point counts
            existing_dist_dict = {row.distribution_name: row.num_points for row in existing_distributions}
            
            expected_distributions = [
                'words_per_document', 'sentences_per_document', 'characters_per_word',
                'words_per_sentence', 'compression_ratio', 'word_entropy'
            ]
            
            # Check if we have all expected distributions
            has_all_distributions = all(dist in existing_dist_dict for dist in expected_distributions)
            
            if has_all_distributions:
                # Check if document-level distributions have correct counts
                doc_level_distributions = ['words_per_document', 'sentences_per_document', 'compression_ratio', 'word_entropy']
                doc_counts_correct = all(
                    existing_dist_dict[dist] == num_documents 
                    for dist in doc_level_distributions 
                    if dist in existing_dist_dict
                )
                
                if doc_counts_correct:
                    logging.info(f"Corpus '{corpus_name}' already has complete distributions with correct counts. Skipping calculation.")
                    logging.info("Use force_recalculate=True to recalculate anyway.")
                    return 0
    
    # Initialize calculator
    calculator = CorpusFeatureCalculator()
    
    # Calculate document distributions
    logging.info("Calculating document distributions...")
    distributions = calculator.calculate_document_distributions(texts)
    
    # Calculate GPT-2 perplexity separately (expensive operation)
    # logging.info("Calculating GPT-2 perplexity...")
    # gpt2_perplexities = []
    # for text in tqdm.tqdm(texts, desc="GPT-2 perplexity"):
    #     perplexity = calculator._calculate_gpt2_perplexity(text)
    #     gpt2_perplexities.append(perplexity)
    # distributions['gpt2_perplexity'] = gpt2_perplexities
    
    # Store raw distributions in database
    logging.info("Storing raw distribution data...")
    store_corpus_distributions(session, corpus_name, distributions)
    
    # Calculate distribution metrics
    logging.info("Calculating distribution metrics...")
    all_features = {}
    
    # distribution_metrics = ['average', 'iqr_range', 'kurtosis', 'median', 'std_dev', 'fano_factor', 'kl_div_uniform']
    
    for dist_name, dist_values in distributions.items():
        metrics = calculator.calculate_distribution_metrics(dist_values)
        for metric_name, metric_value in metrics.items():
            feature_name = f"{dist_name}_{metric_name}"
            all_features[feature_name] = metric_value
    
    # Calculate dataset metrics
    logging.info("Calculating dataset metrics...")
    dataset_metrics = calculator.calculate_dataset_metrics(texts)
    all_features.update(dataset_metrics)
    
    # Store features in database
    logging.info(f"Storing {len(all_features)} features in database...")
    store_corpus_features(session, corpus_name, all_features)
    
    session.commit()
    
    logging.info(f"Successfully calculated and stored {len(all_features)} features for corpus '{corpus_name}'")
    
    return len(all_features)


def calculate_features_for_all_corpora(session: Session, force_recalculate: bool = False):
    """
    Calculate features for all corpora in the database.
    
    Args:
        session: Database session
        force_recalculate: If True, recalculate even if distributions already exist
        
    Returns:
        Dictionary mapping corpus names to number of features calculated
    """
    # Get all corpus names
    fetch_corpora_query = text("SELECT name FROM pipeline.corpus ORDER BY name")
    result = session.execute(fetch_corpora_query)
    corpus_names = [row.name for row in result.fetchall()]
    
    results = {}
    
    for corpus_name in corpus_names:
        logging.info(f"Processing corpus: {corpus_name}")
        try:
            feature_count = calculate_corpus_features(session, corpus_name, force_recalculate)
            results[corpus_name] = feature_count
        except Exception as e:
            logging.error(f"Error processing corpus '{corpus_name}': {e}")
            results[corpus_name] = 0
    
    return results


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = cfg.load_config_from_env()
    db_config = config.database
    
    # Create database session
    with get_session(db_config) as session:
        logging.info("Starting corpus features calculation for all corpora...")
        results = calculate_features_for_all_corpora(session)
        
        logging.info("Feature calculation results:")
        for corpus_name, feature_count in results.items():
            logging.info(f"  {corpus_name}: {feature_count} features")
        
        total_features = sum(results.values())
        logging.info(f"Total features calculated: {total_features}")
        
        logging.info("Corpus features calculation completed successfully.")