import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from src.model.moe import SparseMoE, Router


class Head(nn.Module):
    def __init__(
        self, head_size: int, n_embed: int, block_size: int, dropout: float = 0.1
    ):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        n_embed: int,
        block_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(
                    head_size=head_size,
                    n_embed=n_embed,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class Block(nn.Module):
    """
    Mixture of Experts Transformer block
    """

    def __init__(
        self,
        n_embed: int,
        n_head: int,
        num_experts: int,
        top_k: int,
        block_size: int,
        router_class: Router,
    ) -> None:
        super().__init__()
        head_size = n_embed // n_head

        self.sa = MultiHeadAttention(
            num_heads=n_head,
            head_size=head_size,
            n_embed=n_embed,
            block_size=block_size,
        )
        self.smoe = SparseMoE(
            n_embed=n_embed,
            num_experts=num_experts,
            top_k=top_k,
            router_class=router_class,
        )
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        output = self.smoe(self.ln2(x))
        x = x + output
        return x


class SparseMoELanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        n_layer: int,
        n_head: int,
        block_size: int,
        num_experts: int,
        top_k: int,
        router_class: Router,
    ) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embed=n_embed,
                    n_head=n_head,
                    num_experts=num_experts,
                    top_k=top_k,
                    router_class=router_class,
                    block_size=block_size,
                )
                for _ in range(n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(n_embed)  # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, block_size: int
    ) -> torch.Tensor:
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            # get the predictions
            logits, _ = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
