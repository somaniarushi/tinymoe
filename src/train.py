import torch
import torch.nn as nn
from torch.nn import init
from typing import Callable, List, TypedDict, NamedTuple

from src.data.loader import Loader
from src.data.tokenizer import Tokenizer
from src.model.transformer import SparseMoELanguageModel
from src.model.moe import Router
from src.model.routers import NoisyTopKRouter

torch.manual_seed(42)


class CheckpointLoss(TypedDict):
    train: float
    val: float


@torch.no_grad()
def estimate_loss(
    model: SparseMoELanguageModel, get_batch: Callable, eval_iters: int = 100
) -> CheckpointLoss:
    device = next(model.parameters()).device
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            X, Y = X.to(device), Y.to(device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return CheckpointLoss(train=out["train"], val=out["val"])


def kaiming_init_weights(m):
    if isinstance(m, (nn.Linear)):
        init.kaiming_normal_(m.weight)


def get_model(
    vocab_size: int,
    block_size: int,
    router_class: Router,
    n_embed: int = 128,
    n_layer: int = 8,
    n_head: int = 8,
    num_experts: int = 8,
    top_k: int = 2,
) -> SparseMoELanguageModel:
    model = SparseMoELanguageModel(
        vocab_size=vocab_size,
        n_embed=n_embed,
        n_layer=n_layer,
        n_head=n_head,
        block_size=block_size,
        num_experts=num_experts,
        top_k=top_k,
        router_class=router_class,
    )
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")
    model.apply(kaiming_init_weights)
    return model


class TrainerResults(NamedTuple):
    model: SparseMoELanguageModel
    train_losses: List[float]
    val_losses: List[float]


def train_loop(
    lr: float = 1e-3,
    max_iters: int = 100,
    eval_interval: int = 10,
    batch_size: int = 16,
    block_size: int = 32,
    top_k: int = 2,
) -> TrainerResults:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer()
    loader = Loader(tokenizer)

    model = get_model(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        top_k=top_k,
        router_class=NoisyTopKRouter,  # type: ignore
    ).to(device)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(
                model=model,
                get_batch=lambda split: loader.get_batch(split, batch_size, block_size),
            )
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            val_losses.append(losses["val"])

        # sample a batch of data
        xb, yb = loader.get_batch("train", batch_size=batch_size, block_size=block_size)
        xb, yb = xb.to(device), yb.to(device)

        # evaluate the loss
        _, loss = model(xb, yb)
        train_losses.append(loss.item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return TrainerResults(model=model, train_losses=train_losses, val_losses=val_losses)


if __name__ == "__main__":
    train_loop()
