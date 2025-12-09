from typing import List, Dict


def pre_tokenize(text: str) -> List[str]:
    return text.lower().split()

def build_vocab(corpus: List[str]) -> Dict[str, int]:
    vocab = {
        "<UNK>": 0, 
        "<PAD>": 1,
        "<BOS>": 2, 
        "<EOS>": 3
    }
    all_tokens = set()
    for text in corpus:
        tokens = pre_tokenize(text)
        all_tokens.update(tokens)
    all_tokens = list(all_tokens)
    tokens_vocab = {token: all_tokens.index(token) + 4 for token in all_tokens}
    return {**vocab, **tokens_vocab}

def tokenize(text: str, max_length: int, vocab: Dict[str, int]) -> List[int]:
    tokens = [2] # <BOS>
    pre = pre_tokenize(text)
    for tok in pre:
        if tok in vocab:
            tokens.append(vocab[tok])
        else:
            tokens.append(0) # <UNK>
    if len(tokens) >= max_length:
        tokens = tokens[:max_length]
        tokens[-1] = 3 # <EOS>
    else:
        tokens.append(3) # <EOS>
        for _ in range(max_length - len(tokens)):
            tokens.append(1) # <PAD>
    return tokens
    