from typing import List
from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    """Tokenizer for text."""
    def __init__(self, pre_tokenizer=None):
        if pre_tokenizer is None:
            self.pre_tokenizer = lambda text: text.lower().split()
        else:
            self.pre_tokenizer = pre_tokenizer

    def pre_tokenize(self, text: str) -> List[str]:
        """Initial tokenization of text"""
        return self.pre_tokenizer(text)

    @abstractmethod
    def vocab_size(self):
        """Gets the size of vocabulary."""
        pass
    
    @abstractmethod
    def encode(self, text: str, max_length: int) -> List[int]:
        """Encodes input text with vocabulary into list of integers (tokens)"""
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Decodes list of integers (tokens) into text.

        Text representation of tokens simply concatenated with spaces.
        """
        pass
