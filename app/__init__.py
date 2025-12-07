import torch
import json

class RNNTextGenerator:
    def __init__(self):
        # Load vocabulary (adjust filename if needed)
        with open("app/vocab.json", "r") as f:
            self.vocab = json.load(f)

        self.itos = self.vocab["itos"]       # index → string
        self.stoi = {w: i for i, w in enumerate(self.itos)}  # string → index
        vocab_size = len(self.itos)

        # Import your model class defined in the notebook
        from app.my_rnn_architecture import MyRNNModel

        # Initialize model architecture (same as training)
        self.model = MyRNNModel(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=256
        )

        # Load weights
        self.model.load_state_dict(torch.load("app/model.pt", map_location="cpu"))
        self.model.eval()