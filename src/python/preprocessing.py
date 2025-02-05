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
            min_chars: int = None,
            remove_stopwords: bool = True,
            lemmatize: bool = True,
            remove_numbers: bool = True,
            remove_urls: bool = False,
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
        self._lemmatize = lemmatize

        # Initialize processing patterns
        self._patterns = {
            'url': re.compile(r'https?://\S+'),
            'whitespace': re.compile(r'\s+'),
            'apostrophe': re.compile(r"(?<=\w)'|'(?=\w)"),
            'punctuation': re.compile(rf"[{re.escape(string.punctuation)}]"),
            'numeric': re.compile(r'\d')
        }

        # Initialize vocabulary state
        self._vocabulary = set()
        self._vocabulary_scores = None
        self._tfidf_matrix = None

        self._cleaned_texts = None

        # Load spacy model if needed
        if remove_stopwords or lemmatize:
            self._spacy_model = spacy.load(
                language,
                exclude=[
                    # 'tok2vec',
                    # 'tagger',
                    # 'parser',
                    # 'attribute_ruler',
                    # 'lemmatizer',
                    # 'ner',
                ]
            )
        else:
            self._spacy_model = None
        
    def fit_transform(self, texts: list[str]) -> list[list[str]]:
        """Process texts and build vocabulary."""

        cleaned_texts = self.clean_texts(texts)
        self._cleaned_texts = cleaned_texts

        tokenized_texts = [text.split(' ') for text in cleaned_texts]
        # Calculate initial vocabulary size
        big_vocabulary = set(word for tokens in tokenized_texts for word in tokens)

        # Calculate TF-IDF scores with document frequency filtering
        vectorizer = TfidfVectorizer(
            norm=None,
            lowercase=False,
            min_df=self._min_df,
            max_df=self._max_df,
            max_features=self._top_n,
            # token_pattern=rf"(?u)\b[\w|\-]{{{self._min_chars},}}\b"
        )
        self._tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
        feature_names = vectorizer.get_feature_names_out()
        self._vocabulary = {feature_names[i]: i for i in range(len(feature_names))}

        print(f"Vocabulary filtered from {len(big_vocabulary)} to {len(feature_names)} words")

        # Filter texts based on vocabulary
        filtered_texts = [
            [token for token in tokens if token in self._vocabulary]
            for tokens in tqdm.tqdm(tokenized_texts, desc="Filtering")
        ]
        
        if self._min_words_per_document:
            text_lens = [len(tokens) for tokens in filtered_texts]
            min_words_threshold_mask = [text_len >= self._min_words_per_document for text_len in text_lens]
            filtered_texts = [filtered_texts[i] for i, mask in enumerate(min_words_threshold_mask) if mask]
            self._tfidf_matrix = self._tfidf_matrix[min_words_threshold_mask]

            print(f"Filtered documents from {len(tokenized_texts)} to {len(filtered_texts)}")

        return filtered_texts
    
    def clean_texts(self, texts: List[str]) -> List[str]:

        pre_spacy_texts = self.pre_spacy_clean_texts(texts)
        spacy_cleaned_texts = self.spacy_clean_texts(pre_spacy_texts)
        cleaned_texts = self.post_spacy_clean_texts(spacy_cleaned_texts)

        return cleaned_texts
    
    def spacy_clean_texts(self, texts: List[str]) -> List[str]:
        """Clean multiple texts using Spacy. Do stopword removal and lematization."""
        if not self._lemmatize and not self._remove_stopwords:
            return texts
        
        tokenized_texts = list(self._spacy_model.pipe(texts, batch_size=100, n_process=16, disable=['ner']))
        # tokenized_texts = [
        #     self._spacy_model(text) for text in tqdm.tqdm(texts, desc="Spacy tokenizing")
        # ]

        if self._remove_stopwords:
            tokenized_texts = [
                [token for token in tokens if not token.is_stop]
                for tokens in tokenized_texts
            ]

        if self._lemmatize:
            tokenized_texts = [
                [token.lemma_ for token in tokens]
                for tokens in tokenized_texts
            ]

            # flag = False
            # for i, tokens in enumerate(tokenized_texts):
            #     for token in tokens:
            #         if token == "christians":
            #             print(f"{i}, {token}")
            #             flag = True
            #             break
            #     if flag:
            #         break

        cleaned_texts = [
            ' '.join(tokens) for tokens in tokenized_texts
        ]

        return cleaned_texts
    
    def pre_spacy_clean_texts(self, texts: List[str]) -> List[str]:
        if self._remove_urls:
            cleaned_texts = [
                self._patterns['url'].sub('', text)
                for text in texts
            ]

        return cleaned_texts
    
    def post_spacy_clean_texts(self, texts: List[str]) -> List[str]:

        processes = [
            lambda text: text.lower(),
            # lambda text: self._patterns['apostrophe'].sub('', text),
            lambda text: self._patterns['punctuation'].sub(' ', text),
            lambda text: self._patterns['whitespace'].sub(' ', text),
            lambda text: self._patterns['numeric'].sub('', text),
            lambda text: text.strip(),

            # remove words with less than min_chars
            lambda text: ' '.join(
                [token for token in text.split(' ') if len(token) >= self._min_chars]
                ),
        ]

        cleaned_texts = texts
        for process in tqdm.tqdm(processes, desc="Post-processing"):
            cleaned_texts = [process(text) for text in cleaned_texts]

        return cleaned_texts

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
    
    @property
    def tokenized_texts(self) -> list[list[str]]:
        """Get the tokenized texts."""
        return [text.split(' ') for text in self._cleaned_texts]