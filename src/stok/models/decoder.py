import torch
import torch.nn as nn
from x_transformers import ContinuousTransformerWrapper, Encoder

from .head import Dim6RotStructureHead

__all__ = [
    "GeometricDecoder",
]


class GeometricDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_mult: float,
        max_length: int,
        d_code: int,
        num_memory_tokens: int = 0,
        attn_kv_heads: int = 2,
    ):
        super(GeometricDecoder, self).__init__()

        self.decoder_output_scaling_factor = 1.0

        # input projection (from codebook space to decoder space)
        self.projector_in = nn.Linear(d_code, d_model, bias=False)

        # decoder stack
        self.decoder_stack = ContinuousTransformerWrapper(
            dim_in=d_model,
            dim_out=d_model,
            max_seq_len=max_length,
            num_memory_tokens=num_memory_tokens,
            attn_layers=Encoder(
                dim=d_model,
                ff_mult=ffn_mult,
                ff_glu=True,  # gate-based feed-forward (GLU family)
                ff_swish=True,  # use Swish instead of GELU → SwiGLU
                ff_no_bias=True,  # removes the two Linear biases in SwiGLU / MLP
                depth=n_layers,
                heads=n_heads,
                rotary_pos_emb=True,
                attn_flash=True,
                attn_kv_heads=attn_kv_heads,
                attn_qk_norm=True,
                pre_norm=True,
                residual_attn=False,
            ),
        )

        # output projection
        self.affine_output_projection = Dim6RotStructureHead(
            d_model,
            trans_scale_factor=1.0,
            predict_torsion_angles=False,
        )

    def forward(
        self,
        structure_tokens: torch.Tensor,
        mask: torch.Tensor,
        *,
        true_lengths: torch.Tensor | None = None,
    ):
        x = self.projector_in(structure_tokens)

        decoder_mask_bool = mask.to(torch.bool)
        x = self.decoder_stack(x, mask=decoder_mask_bool)

        bb_pred = self.affine_output_projection(
            x, affine=None, affine_mask=torch.zeros_like(mask), preds_only=True
        )

        return bb_pred.flatten(-2) * self.decoder_output_scaling_factor
