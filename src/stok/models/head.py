import torch
import torch.nn as nn
import torch.nn.functional as F


class CodebookClassifier(nn.Module):
    """Per-residue classifier tied to a (frozen) VQ codebook.

    Supports distance-based logits (mirrors Euclidean VQ) or cosine logits.
    A small linear projector maps encoder features to code space.
    """

    def __init__(
        self,
        d_in: int,
        codebook: torch.Tensor,
        use_cosine: bool = False,
        learnable_temperature: bool = True,
        bias_from_code_norm: bool = True,
        projector_dim: int | None = None,
    ):
        """Initialize codebook classifier.

        Args:
            d_in: Input feature dimension.
            codebook: Codebook tensor of shape [C, d_code].
            use_cosine: If True, use cosine similarity; else use distance-based logits.
            learnable_temperature: If True, use learnable temperature scaling.
            bias_from_code_norm: If True, precompute -||e||^2 bias term for
                distance-based logits.
            projector_dim: Optional projector dimension. If None, uses d_code.
        """
        super().__init__()
        assert codebook.dim() == 2, "codebook must be [C, d_code]"
        C, d_code = codebook.shape
        self.C = C
        self.d_code = d_code
        self.use_cosine = use_cosine
        self.bias_from_code_norm = bias_from_code_norm and (not use_cosine)

        # register the codebook as a non-trainable buffer
        self.register_buffer("E", codebook.detach(), persistent=True)

        # project to codebook space
        p_out = projector_dim or d_code
        self.project = nn.Linear(d_in, p_out, bias=False)
        self.ln = nn.LayerNorm(p_out)

        # when projector_dim != d_code, add a final projection to d_code
        self.to_code = (
            nn.Identity() if p_out == d_code else nn.Linear(p_out, d_code, bias=False)
        )

        # temperature/scale
        self.inv_tau = (
            nn.Parameter(torch.tensor(1.0)) if learnable_temperature else None
        )

        # precompute -||e||^2 bias if using distance head (if bias_from_code_norm is True)
        if self.bias_from_code_norm:
            with torch.no_grad():
                bias = -(self.E**2).sum(dim=1)  # [C]
            self.register_buffer("code_bias", bias, persistent=True)
        else:
            self.register_buffer("code_bias", torch.zeros(C), persistent=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward pass through classifier.

        Args:
            h: Input tensor of shape [B, L, d_in].

        Returns:
            Logits tensor of shape [B, L, C].
        """
        # B, L, _ = h.shape
        h = self.ln(self.project(h))
        h = self.to_code(h)  # [B, L, d_code]

        if self.use_cosine:
            # normalize, cosine similarity scaled by inv_tau (if provided)
            h_norm = F.normalize(h, dim=-1)
            e_norm = F.normalize(self.E, dim=-1)
            logits = torch.einsum("bld,cd->blc", h_norm, e_norm)  # [B, L, C]
            scale = self.inv_tau if self.inv_tau is not None else 1.0
            logits = scale * logits
        else:
            # distance head: 2 h·e - ||e||^2
            logits = 2.0 * torch.einsum("bld,cd->blc", h, self.E)  # [B, L, C]
            if self.bias_from_code_norm:
                logits = logits + self.code_bias  # broadcast [C] -> [B,L,C]
            if self.inv_tau is not None:
                logits = self.inv_tau * logits
        return logits
