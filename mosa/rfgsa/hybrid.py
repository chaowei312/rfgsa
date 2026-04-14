"""RFGSA hybrid wrapper — sparse RFGSA heads + dense/local heads."""

from typing import Optional, Tuple

import torch
from torch import nn

from .core import PureRFGSA
from ..hybrid import Dense, MHLocalAttention


class RFGSA(nn.Module):
    """Hybrid of Router-Free Gated Sparse Attention + dense/local heads.

    Drop-in replacement for ``mosa.MoSA`` except that ``forward`` returns
    ``(output, aux_loss)`` instead of just ``output``.
    """

    def __init__(
        self,
        h: int,
        h_prim: int,
        num_rfgsa_heads: int,
        num_other_heads: int,
        max_seq_len: int,
        sparsity: int,
        gate_rank: int = 32,
        kernel_size: Optional[int] = None,
        hybrid_type: str = "dense",
        include_first: int = 0,
        rotate_fraction: float = 0.5,
        rope_base: float = 10000.0,
        target_density: float = 0.25,
        mu: float = 0.5,
    ):
        super().__init__()

        if num_rfgsa_heads > 0:
            self.sparse_heads = PureRFGSA(
                n_heads=num_rfgsa_heads,
                sparsity=sparsity,
                h=h,
                h_prim=h_prim,
                gate_rank=gate_rank,
                kernel_size=kernel_size,
                include_first=include_first,
                rotate_fraction=rotate_fraction,
                rope_base=rope_base,
                target_density=target_density,
                mu=mu,
            )
        else:
            self.sparse_heads = None

        if num_other_heads > 0:
            if hybrid_type == "dense":
                self.other_heads = Dense(
                    h, h_prim, num_other_heads, rotate_fraction, rope_base,
                )
            elif hybrid_type == "local":
                k = max_seq_len // sparsity
                self.other_heads = MHLocalAttention(h, h_prim, num_other_heads, k)
            else:
                raise ValueError(f"hybrid_type '{hybrid_type}' not recognized")
        else:
            self.other_heads = None

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output:   (B, T, h)
            aux_loss: scalar (0.0 when there are no sparse heads)
        """
        output = torch.zeros_like(X)
        aux_loss = torch.tensor(0.0, device=X.device, dtype=X.dtype)

        if self.sparse_heads is not None:
            sparse_out, aux_loss = self.sparse_heads(X)
            output = output + sparse_out

        if self.other_heads is not None:
            output = output + self.other_heads(X)

        return output, aux_loss
