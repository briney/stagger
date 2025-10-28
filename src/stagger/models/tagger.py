from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .head import CodebookClassifier


class TaggerModel(nn.Module):
    """Encoder-only tagger that predicts a structure token per residue."""

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_mult: float,
        dropout: float,
        attn_dropout: float,
        rope_base: float,
        codebook: torch.Tensor,
        classifier_kwargs: dict[str, Any] | None = None,
        norm_type: str = "layernorm",
    ):
        super().__init__()
        self.pad_id = pad_id
        self.codebook_size = codebook.shape[0]
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

        self.encoder = Encoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            attn_dropout=attn_dropout,
            rope_base=rope_base,
            rope_dim=None,
            ffn_mult=ffn_mult,
            norm_type=norm_type,
        )
        self.classifier = CodebookClassifier(
            d_in=d_model,
            codebook=codebook,
            **(classifier_kwargs or {}),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        ignore_index: int = -100,
    ):
        # Embedding
        h = self.embed(tokens)  # [B, L, d_model]

        # If mask not provided, build from pad_id
        if key_padding_mask is None:
            key_padding_mask = tokens == self.pad_id  # [B, L], True = pad

        # Encode
        h = self.encoder(
            h, key_padding_mask=key_padding_mask, attn_mask=None
        )  # [B, L, d_model]

        # Classify
        logits = self.classifier(h)  # [B, L, C]

        loss = None
        if labels is not None:
            # Flatten for CE; ignore_index for missing/unmodeled residues
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=ignore_index,
            )
        return {"logits": logits, "loss": loss}
