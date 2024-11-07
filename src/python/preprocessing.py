import re
import string
import concurrent.futures
from typing import Optional

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
        tokens = [token for token in tokens if token and not self.numeric_pattern.search(token)]
        return tokens

    def process_texts(self, texts: list[str], executor: Optional[concurrent.futures.Executor] = None) -> list[list[str]]:
        if executor is None:
            # Process sequentially if no executor is provided
            return [self.tokenize(text) for text in texts]
        else:
            # Use the executor for parallel processing
            futures = [executor.submit(self.tokenize, text) for text in texts]
            return [future.result() for future in concurrent.futures.as_completed(futures)]
