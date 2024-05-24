import torch
from typing import Tuple

from src.data.tokenizer import Tokenizer

DATA_FILE = "data/shakespeare.txt"


class Loader:
    def __init__(self, tokenizer: Tokenizer, train_split_size: float = 0.9) -> None:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            text = f.read()
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        train_data_size = int(train_split_size * len(data))

        self.train_data = data[:train_data_size]
        self.val_data = data[train_data_size:]

    def get_batch(
        self, split: str, batch_size: int, block_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x, y
