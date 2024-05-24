import torch
import torch.nn as nn
from typing import Tuple
from abc import ABC, abstractmethod


class Router(nn.Module, ABC):
    def __init__(self, n_embed: int, num_experts: int, top_k: int) -> None:
        super(Router, self).__init__()
        self.top_k = top_k
        self.num_experts = num_experts

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class SparseMoE(nn.Module):
    def __init__(
        self, n_embed: int, num_experts: int, top_k: int, router_class: Router
    ) -> None:
        super(SparseMoE, self).__init__()
        self.router = router_class(
            n_embed=n_embed, num_experts=num_experts, top_k=top_k
        )
        self.experts = nn.ModuleList(
            [Expert(n_embed=n_embed) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # Process each expert
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output


class Expert(nn.Module):
    """
    An MLP is a simple linear layer followed by a non-linearity
    i.e. each Expert
    """

    def __init__(self, n_embed: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
