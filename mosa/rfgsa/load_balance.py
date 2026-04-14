"""Adaptive load-balancing loss (Routing-Free MoE, Section 3.2)."""

import torch
from torch import nn


class RoutingFreeLoadBalance(nn.Module):
    """Adaptive load-balancing without any router, Softmax, or TopK.

    Jointly optimises two complementary objectives via a single auxiliary loss:
      L_EB  — expert-balancing: tokens distributed evenly across heads
      L_TB  — token-balancing:  each token activates a similar number of heads
      L_LB  = mu * L_EB + (1-mu) * L_TB

    An adaptive coefficient lambda_t is updated every training step to drive
    the empirical activation density toward a configurable target rho_inf.
    """

    def __init__(
        self,
        target_density: float = 0.25,
        mu: float = 0.5,
        lambda_init: float = 1e-10,
        eta: float = 0.02,
    ):
        super().__init__()
        self.target_density = target_density
        self.mu = mu
        self.eta = eta
        self.register_buffer("lambda_coeff", torch.tensor(lambda_init))

    def forward(
        self,
        scores: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            scores:      G_h(x) continuous activations  — (B, T, E)
            active_mask: binary 1{G_h(x) > 0}           — (B, T, E)

        Returns:
            Scalar weighted load-balance loss  (lambda_t * L_LB).
        """
        f_expert = active_mask.float().mean(dim=1)   # (B, E)
        g_expert = scores.mean(dim=1)                # (B, E)
        L_EB = (f_expert * g_expert).mean()

        f_token = active_mask.float().mean(dim=2)    # (B, T)
        g_token = scores.mean(dim=2)                 # (B, T)
        L_TB = (f_token * g_token).mean()

        L_LB = self.mu * L_EB + (1.0 - self.mu) * L_TB

        if self.training:
            rho_t = active_mask.float().mean().item()
            sign = 1.0 if rho_t > self.target_density else -1.0
            self.lambda_coeff.fill_(
                (self.lambda_coeff * (1.0 + self.eta) ** sign).item()
            )

        return self.lambda_coeff * L_LB
