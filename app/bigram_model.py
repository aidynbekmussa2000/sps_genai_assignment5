# app/bigram_model.py
import re
import random
from typing import List, Dict

class BigramModel:
    def __init__(self, corpus: List[str]):
        """
        Build bigram counts from a list of sentences.
        """
        self.bigram_counts: Dict[str, Dict[str, int]] = {}
        self._build_counts(corpus)

    def _tokenize(self, text: str) -> List[str]:
        # Super simple tokenizer: lowercase + split on spaces/punctuation
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def _build_counts(self, corpus: List[str]) -> None:
        for sentence in corpus:
            tokens = self._tokenize(sentence)
            for w1, w2 in zip(tokens, tokens[1:]):
                if w1 not in self.bigram_counts:
                    self.bigram_counts[w1] = {}
                self.bigram_counts[w1][w2] = self.bigram_counts[w1].get(w2, 0) + 1

    def _next_word(self, current_word: str) -> str:
        """
        Sample a next word based on bigram frequencies.
        """
        followers = self.bigram_counts.get(current_word)
        if not followers:
            # if we have no information, just stop
            return ""

        # Make a weighted random choice based on counts
        words = list(followers.keys())
        weights = list(followers.values())
        return random.choices(words, weights=weights, k=1)[0]

    def generate_text(self, start_word: str, length: int) -> str:
        """
        Generate text of `length` words starting from `start_word`.
        """
        current = start_word.lower()
        tokens = [current]

        for _ in range(length - 1):
            nxt = self._next_word(current)
            if not nxt:
                break
            tokens.append(nxt)
            current = nxt

        return " ".join(tokens)