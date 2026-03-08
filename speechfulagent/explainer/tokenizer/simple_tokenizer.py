from typing import List, Dict

from .base_tokenizer import BaseTokenizer


class SimpleTokenizer(BaseTokenizer):
    """Simplest tokenizer for text."""
    def __init__(self, pre_tokenizer=None):
        super().__init__(pre_tokenizer)
        self.vocab = {}
        self.special_tokens = {
            "<UNK>": 0, 
            "<PAD>": 1,
            "<BOS>": 2, 
            "<EOS>": 3
        }
        self.inverse_vocab = {}

    def build_vocab(self, corpus: List[str]) -> None:
        """Builds vocabulary and inverse vocabulary from corpus"""
        self.vocab = {
            "<UNK>": 0, 
            "<PAD>": 1,
            "<BOS>": 2, 
            "<EOS>": 3
        }
        all_tokens = set()
        for text in corpus:
            tokens = self.pre_tokenize(text)
            all_tokens.update(tokens)
        all_tokens = list(all_tokens)
        tokens_vocab = {token: all_tokens.index(token) + 4 for token in all_tokens}
        self.vocab = {**self.vocab, **tokens_vocab}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def set_vocab(self, vocab: Dict[str, int]):
        """Loads external vocab into tokenizer."""
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def vocab_size(self):
        """Gets the size of vocabulary."""
        return len(self.vocab.keys())
    
    def encode(self, text: str, max_length: int) -> List[int]:
        """Encodes input text with vocabulary into list of integers (tokens)"""
        tokens = [self.special_tokens["<BOS>"]]
        pre = self.pre_tokenize(text)
        for tok in pre:
            if tok in self.vocab:
                tokens.append(self.vocab[tok])
            else:
                tokens.append(self.special_tokens["<UNK>"])
        if len(tokens) >= max_length:
            tokens = tokens[:max_length]
            tokens[-1] = self.special_tokens["<EOS>"]
        else:
            tokens.append(self.special_tokens["<EOS>"])
            for _ in range(max_length - len(tokens)):
                tokens.append(self.special_tokens["<PAD>"])
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decodes list of integers (tokens) into text.

        Text representation of tokens simply concatenated with spaces.
        """
        text = [self.inverse_vocab[tok] for tok in tokens]
        if self.special_tokens["<PAD>"] in tokens:
            pad_index = text.index("<PAD>")
            text = text[:pad_index]
        return " ".join(text)
