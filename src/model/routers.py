import torch
import torch.nn as nn
from typing import Tuple
from torch.nn import functional as F

from src.model.moe import Router


class NoisyTopKRouter(Router):
    def __init__(self, n_embed: int, num_experts: int, top_k: int) -> None:
        super().__init__(n_embed, num_experts, top_k)
        self.top_k_route_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x is the output tensor from multi-head self attention block
        logits = self.top_k_route_linear(x)

        # Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits) * F.softplus(self.noise_linear(x))

        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float("-inf"))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)

        return router_output, indices
