import re
import string
import concurrent.futures
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class UnicodeTokenizer:
    def __init__(self):
        # Regex to match apostrophes between word characters (unicode)
        self.apostrophe_pattern = re.compile(r"(?<=\w)'|'(?=\w)")
        # Regex to match any punctuation character
        self.punctuation_pattern = re.compile(rf"[{re.escape(string.punctuation)}]")
        # Regex to match consecutive whitespace characters
        self.whitespace_pattern = re.compile(r'\s+')
        # Regex to match any numeric character
        self.numeric_pattern = re.compile(r'\d')

    def clean_text(self, text: str) -> str:
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

    def process_texts(self, texts: list[str], executor: Optional[concurrent.futures.Executor] = None) -> list[list[str]]:
        if executor is None:
            # Process sequentially if no executor is provided
            return [self.tokenize(text) for text in texts]
        else:
            # Use the executor for parallel processing
            futures = [executor.submit(self.tokenize, text) for text in texts]
            return [future.result() for future in concurrent.futures.as_completed(futures)]


class Vocabulariser:
    def __init__(self, top_n: int):
        self._top_n = top_n
        self._vocabulary = set()

    def fit(self, tokenized_texts: list[list[str]], executor: concurrent.futures.Executor = None) -> list[list[str]]:
        # Join tokenized texts back into strings for TF-IDF processing
        documents = [' '.join(tokens) for tokens in tokenized_texts]

        # Calculate TF-IDF scores
        vectorizer = TfidfVectorizer(norm=None, use_idf=True, lowercase=False)
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()

        # Calculate average TF-IDF scores for each word
        tfidf_array = tfidf_matrix.toarray()
        non_zero_counts = (tfidf_array > 0).sum(axis=0)
        total_scores = tfidf_array.sum(axis=0)
        average_scores = total_scores / np.maximum(non_zero_counts, 1)

        # Select top N words with highest average scores
        top_indices = np.argsort(average_scores)[-self._top_n:]
        self._vocabulary = {feature_names[i] for i in top_indices}

        # Return the transformed texts
        return self.transform(tokenized_texts, executor)

    def transform(self, tokenized_texts: list[list[str]], executor: concurrent.futures.Executor = None) -> list[list[str]]:
        if not self._vocabulary:
            raise ValueError("Vocabulary has not been determined. Call 'fit' before 'transform'.")
        
        if executor is None:
            return [self._filter_tokens(tokens) for tokens in tokenized_texts]
        else:
            futures = [executor.submit(self._filter_tokens, tokens) for tokens in tokenized_texts]
            return [future.result() for future in concurrent.futures.as_completed(futures)]

    def _filter_tokens(self, tokens: list[str]) -> list[str]:
        return [token for token in tokens if token in self._vocabulary]

    @property
    def vocabulary(self) -> set:
        return self._vocabulary

    @property
    def top_n(self) -> int:
        return self._top_n
