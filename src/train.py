import torch
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
    return model


class TrainerResults(NamedTuple):
    model: SparseMoELanguageModel
    train_losses: List[float]
    val_losses: List[float]


def train_loop(
    lr: float = 1e-4,
    max_iters: int = 1000,
    eval_interval: int = 100,
    batch_size: int = 16,
    block_size: int = 32,
) -> TrainerResults:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer()
    loader = Loader(tokenizer)
    model = get_model(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        router_class=NoisyTopKRouter,  # type: ignore
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    for iteration in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iteration % eval_interval == 0 or iteration == max_iters - 1:
            losses: CheckpointLoss = estimate_loss(
                model=model,
                get_batch=lambda split: loader.get_batch(split, batch_size, block_size),
            )
            val_losses.append(losses["val"])
            print(
                f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        # sample a batch of data
        xb, yb = loader.get_batch("train", batch_size, block_size)
        xb, yb = xb.to(device), yb.to(device)

        # evaluate the loss
        _, loss = model(xb, yb)
        loss.backward()
        train_losses.append(loss.item())

        optimizer.zero_grad(set_to_none=True)
        optimizer.step()

    return TrainerResults(model, train_losses, val_losses)
