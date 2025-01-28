import re
import string
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import tqdm

class UnicodeTokenizer:
    def __init__(self, remove_urls: bool = True):
        self.remove_urls = remove_urls

        # Regex to match apostrophes between word characters (unicode)
        self.apostrophe_pattern = re.compile(r"(?<=\w)'|'(?=\w)")
        # Regex to match any punctuation character
        self.punctuation_pattern = re.compile(rf"[{re.escape(string.punctuation)}]")
        # Regex to match consecutive whitespace characters
        self.whitespace_pattern = re.compile(r'\s+')
        # Regex to match any numeric character
        self.numeric_pattern = re.compile(r'\d')
        # Regex to match URLs
        self.url_pattern = re.compile(r'https?://\S+')

    def clean_text(self, text: str) -> str:
        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub('', text)
        # Replace apostrophes between word characters
        text = self.apostrophe_pattern.sub('', text)
        # Replace punctuation with space
        text = self.punctuation_pattern.sub(' ', text)
        # Reduce consecutive whitespace to a single space
        text = self.whitespace_pattern.sub(' ', text)
        return text.strip()

    def tokenize(self, text: str) -> list[str]:
        # Clean the text
        cleaned_text = self.clean_text(text)
        # Split into tokens
        tokens = cleaned_text.split(' ')
        # Remove tokens with numeric characters and empty tokens
        tokens = [token.lower() for token in tokens if token and not self.numeric_pattern.search(token)]
        return tokens

    def process_texts(self, texts: list[str]) -> list[list[str]]:
        return [self.tokenize(text) for text in tqdm.tqdm(texts, desc="Tokenizing")]


class Vocabulariser:
    def __init__(self,
            top_n: int,
            min_words_per_document: Optional[int] = None, 
            min_df: float = 0.0,
            max_df: float = 1.0,
            min_chars: int = 3,
            remove_stopwords: bool = False,
        ):
        self._top_n = top_n
        self._min_words_per_document = min_words_per_document
        self._min_df = min_df
        self._max_df = max_df
        self._min_chars = min_chars
        self._remove_stopwords = remove_stopwords

        self._vocabulary = set()
        self._vocalulary_scores = None
        self._tfidf_matrix = None

        self._spacy_model = spacy.load(
            'en_core_web_sm',
            exclude=['tok2vec', 'attribute_ruler', 'ner']
        )

    def fit_transform(self, tokenized_texts: list[list[str]]) -> list[list[str]]:
        if self._remove_stopwords:
            tokenized_texts = [
                [token for token in tokens if not self._spacy_model.vocab[token].is_stop]
                for tokens in tokenized_texts
            ]

        # Join tokenized texts back into strings for TF-IDF processing
        documents = [' '.join(tokens) for tokens in tokenized_texts]

        big_vocabulary = set(word for tokens in tokenized_texts for word in tokens if len(word) >= self._min_chars)

        # Calculate TF-IDF scores with document frequency filtering
        vectorizer = TfidfVectorizer(
            norm=None,
            lowercase=False,
            min_df=self._min_df,
            max_df=self._max_df,
            max_features=self._top_n,
            token_pattern=rf"(?u)\b[\w|\-]{{{self._min_chars},}}\b"  # Modified to use min_chars
        )
        tfidf_matrix = vectorizer.fit_transform(documents)
        self._tfidf_matrix = tfidf_matrix

        feature_names = vectorizer.get_feature_names_out()

        self._vocabulary = set(feature_names)

        print(f"Vocabulary filtered from {len(big_vocabulary)} to {len(feature_names)} words")

        filtered_texts = [self._filter_tokens(tokens) for tokens in tqdm.tqdm(tokenized_texts, desc="Filtering")]
        text_lens = [len(tokens) for tokens in filtered_texts]
        min_words_threshold_mask = [text_len >= self._min_words_per_document for text_len in text_lens]
        filtered_texts = [filtered_texts[i] for i, mask in enumerate(min_words_threshold_mask) if mask]

        print(f"Filtered documents from {len(tokenized_texts)} to {len(filtered_texts)}")

        self._tfidf_matrix = tfidf_matrix[min_words_threshold_mask]

        # Return the transformed texts
        return filtered_texts

    def transform(self, tokenized_texts: list[list[str]], filter_tfidf=False) -> list[list[str]]:
        if not self._vocabulary:
            raise ValueError("Vocabulary has not been determined. Call 'fit' before 'transform'.")
        
        filtered_texts = [self._filter_tokens(tokens) for tokens in tqdm.tqdm(tokenized_texts, desc="Filtering")]
        return filtered_texts

    def _filter_tokens(self, tokens: list[str]) -> list[str]:
        return [token for token in tokens if token in self._vocabulary]

    @property
    def vocabulary(self) -> set:
        return self._vocabulary

    @property
    def top_n(self) -> int:
        return self._top_n
    
    @property
    def vocabulary_scores(self) -> dict[str, float]:
        return self._vocalulary_scores

    @property
    def tfidf_matrix(self) -> np.ndarray:
        return self._tfidf_matrix