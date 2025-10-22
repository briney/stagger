import torch
import torch.nn as nn

from .attention import MHAWithRoPE
from .mlp import SwiGLU


class PreLNEncoderBlock(nn.Module):
    """Pre-LN Transformer encoder block (MHA + SwiGLU)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_dropout: float,
        resid_dropout: float,
        rope: object,
        norm_type: str = "layernorm",
        ffn_mult: float = 4.0,
    ):
        super().__init__()
        Norm = nn.LayerNorm  # RMSNorm optional later
        if norm_type != "layernorm":
            # Placeholder to keep interface stable; can add RMSNorm later.
            Norm = nn.LayerNorm

        self.norm1 = Norm(d_model)
        self.attn = MHAWithRoPE(
            d_model=d_model, n_heads=n_heads, dropout=attn_dropout, rope=rope
        )
        self.drop1 = nn.Dropout(resid_dropout)

        self.norm2 = Norm(d_model)
        self.mlp = SwiGLU(d_model=d_model, expansion=ffn_mult, dropout=resid_dropout)
        self.drop2 = nn.Dropout(resid_dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.attn(h, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + self.drop1(h)

        h = self.norm2(x)
        h = self.mlp(h)
        x = x + self.drop2(h)
        return x
