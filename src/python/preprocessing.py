import re
import string
from typing import Optional, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import tqdm

class TextPreprocessor:
    """Unified text preprocessing class that handles both tokenization and vocabulary building."""
    
    def __init__(self,
            top_n: Optional[int] = None,
            min_words_per_document: Optional[int] = None,
            min_df: float = 0.0,
            max_df: float = 1.0,
            min_chars: int = 3,
            remove_stopwords: bool = False,
            remove_urls: bool = True,
            remove_numbers: bool = True,
            language: str = 'en_core_web_sm'
        ):
        """
        Initialize the text preprocessor.
        
        Args:
            top_n: Maximum number of words to keep in vocabulary
            min_words_per_document: Minimum words required per document
            min_df: Minimum document frequency for words
            max_df: Maximum document frequency for words
            min_chars: Minimum characters per word
            remove_stopwords: Whether to remove stopwords
            remove_urls: Whether to remove URLs
            remove_numbers: Whether to remove numbers
            language: Spacy language model to use
        """
        self._top_n = top_n
        self._min_words_per_document = min_words_per_document
        self._min_df = min_df
        self._max_df = max_df
        self._min_chars = min_chars
        self._remove_stopwords = remove_stopwords
        self._remove_urls = remove_urls
        self._remove_numbers = remove_numbers

        # Initialize processing patterns
        self._patterns = {
            'url': re.compile(r'https?://\S+'),
            'apostrophe': re.compile(r"(?<=\w)'|'(?=\w)"),
            'punctuation': re.compile(rf"[{re.escape(string.punctuation)}]"),
            'whitespace': re.compile(r'\s+'),
            'numeric': re.compile(r'\d')
        }

        # Initialize vocabulary state
        self._vocabulary = set()
        self._vocabulary_scores = None
        self._tfidf_matrix = None

        # Load spacy model if needed
        if remove_stopwords:
            self._spacy_model = spacy.load(
                language,
                exclude=['tok2vec', 'attribute_ruler', 'ner']
            )
        else:
            self._spacy_model = None

    def clean_text(self, text: str) -> str:
        """Clean a single text string."""
        # Remove URLs if configured
        if self._remove_urls:
            text = self._patterns['url'].sub('', text)
        
        # Remove apostrophes between word characters
        text = self._patterns['apostrophe'].sub('', text)
        
        # Replace punctuation with space
        text = self._patterns['punctuation'].sub(' ', text)
        
        # Reduce whitespace to single spaces
        text = self._patterns['whitespace'].sub(' ', text)
        
        return text.strip()

    def tokenize(self, text: str) -> list[str]:
        """Tokenize a single text string."""
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Split into tokens
        tokens = cleaned_text.split(' ')
        
        # Apply filters
        tokens = [
            token.lower() for token in tokens 
            if token and  # Non-empty
            len(token) >= self._min_chars and  # Meets minimum length
            (not self._remove_numbers or not self._patterns['numeric'].search(token))  # Number filtering
        ]
        
        # Remove stopwords if configured
        if self._remove_stopwords:
            tokens = [token for token in tokens if not self._spacy_model.vocab[token].is_stop]
            
        return tokens

    def process_texts(self, texts: list[str]) -> list[list[str]]:
        """Process multiple texts into tokens."""
        return [self.tokenize(text) for text in tqdm.tqdm(texts, desc="Tokenizing")]

    def fit_transform(self, texts: list[str]) -> list[list[str]]:
        """Process texts and build vocabulary."""
        # First tokenize all texts
        tokenized_texts = self.process_texts(texts)

        # Join tokenized texts back into strings for TF-IDF processing
        documents = [' '.join(tokens) for tokens in tokenized_texts]

        # Calculate initial vocabulary size
        big_vocabulary = set(word for tokens in tokenized_texts for word in tokens if len(word) >= self._min_chars)

        # Calculate TF-IDF scores with document frequency filtering
        vectorizer = TfidfVectorizer(
            norm=None,
            lowercase=False,
            min_df=self._min_df,
            max_df=self._max_df,
            max_features=self._top_n,
            token_pattern=rf"(?u)\b[\w|\-]{{{self._min_chars},}}\b"
        )
        self._tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        self._vocabulary = {feature_names[i]: i for i in range(len(feature_names))}

        print(f"Vocabulary filtered from {len(big_vocabulary)} to {len(feature_names)} words")

        # Filter texts based on vocabulary and minimum words requirement
        filtered_texts = [self._filter_tokens(tokens) for tokens in tqdm.tqdm(tokenized_texts, desc="Filtering")]
        
        if self._min_words_per_document:
            text_lens = [len(tokens) for tokens in filtered_texts]
            min_words_threshold_mask = [text_len >= self._min_words_per_document for text_len in text_lens]
            filtered_texts = [filtered_texts[i] for i, mask in enumerate(min_words_threshold_mask) if mask]
            self._tfidf_matrix = self._tfidf_matrix[min_words_threshold_mask]

            print(f"Filtered documents from {len(tokenized_texts)} to {len(filtered_texts)}")

        return filtered_texts

    def transform(self, texts: list[str]) -> list[list[str]]:
        """Transform new texts using existing vocabulary."""
        if not self._vocabulary:
            raise ValueError("Vocabulary has not been determined. Call 'fit_transform' first.")
        
        tokenized_texts = self.process_texts(texts)
        return [self._filter_tokens(tokens) for tokens in tqdm.tqdm(tokenized_texts, desc="Filtering")]

    def _filter_tokens(self, tokens: list[str]) -> list[str]:
        """Filter tokens based on vocabulary."""
        return [token for token in tokens if token in self._vocabulary]

    @property
    def vocabulary(self) -> list[str]:
        """Get the sorted vocabulary list."""
        return [word for word, _ in sorted(self._vocabulary.items(), key=lambda x: x[1])]

    @property
    def vocabulary_scores(self) -> dict[str, float]:
        """Get vocabulary scores."""
        return self._vocabulary_scores

    @property
    def tfidf_matrix(self) -> np.ndarray:
        """Get the TF-IDF matrix."""
        return self._tfidf_matrix