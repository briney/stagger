import torch
import torch.nn as nn

from .blocks import PreLNEncoderBlock
from .rope import RotaryEmbedding


class Encoder(nn.Module):
    """Stack of Pre-LN encoder blocks with shared RoPE."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        attn_dropout: float,
        rope_base: float,
        rope_dim: int | None = None,
        ffn_mult: float = 4.0,
        norm_type: str = "layernorm",
    ):
        super().__init__()
        self.rope = RotaryEmbedding(base=rope_base, rope_dim=rope_dim)
        self.layers = nn.ModuleList(
            [
                PreLNEncoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    attn_dropout=attn_dropout,
                    resid_dropout=dropout,
                    rope=self.rope,
                    norm_type=norm_type,
                    ffn_mult=ffn_mult,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        h: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return self.final_norm(h)
