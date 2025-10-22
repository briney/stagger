import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU feedforward block.

    Implements x -> W_down( SiLU(W_gate x) ⊙ (W_up x) ).
    """

    def __init__(self, d_model: int, expansion: float, dropout: float):
        super().__init__()
        hidden = int(d_model * expansion)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.w_up(x)
        gate = F.silu(self.w_gate(x))
        y = self.w_down(up * gate)
        return self.dropout(y)
