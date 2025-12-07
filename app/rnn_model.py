# app/rnn_model.py

import re
from collections import Counter

import requests
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 100, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden


class RNNTextGenerator:
    def __init__(self):
        """
        - Download & preprocess Count of Monte Cristo
        - Build vocab and inverse vocab as in the notebook
        - Create LSTMModel and load weights from model.pt
        """
        # 1. Load and clean text (simplified)
        url = "https://www.gutenberg.org/cache/epub/1184/pg1184.txt"
        text = requests.get(url).text

        start_idx = text.find("Chapter 1.")
        if start_idx == -1:
            start_idx = 0
        text = text[start_idx:]

        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        tokens = text.split()

        # 2. Build vocab exactly as during training
        counter = Counter(tokens)
        vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common(9998))}
        vocab["<PAD>"] = 0
        vocab["<UNK>"] = 1
        inv_vocab = {idx: word for word, idx in vocab.items()}

        self.vocab = vocab
        self.inv_vocab = inv_vocab

        # 3. Load trained LSTM weights
        self.model = LSTMModel()
        state_dict = torch.load("model.pt", map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def generate_text(self, start_word: str, length: int = 50, temperature: float = 1.0) -> str:
        """
        Generate `length` more tokens starting from `start_word`.
        """
        self.model.eval()
        words = start_word.lower().split()
        input_ids = [self.vocab.get(w, self.vocab["<UNK>"]) for w in words]
        input_tensor = torch.tensor(input_ids).unsqueeze(0)
        hidden = None

        with torch.no_grad():
            for _ in range(length):
                output, hidden = self.model(input_tensor, hidden)
                logits = output[0, -1] / temperature
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()

                words.append(self.inv_vocab.get(next_id, "<UNK>"))

                input_ids.append(next_id)
                input_tensor = torch.tensor(input_ids).unsqueeze(0)

        return " ".join(words)