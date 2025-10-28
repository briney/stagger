import torch
import torch.nn.functional as F


def token_ce_loss(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Compute cross-entropy loss over structure tokens.

    Args:
        logits: Logits tensor of shape [B, L, C].
        labels: Target labels of shape [B, L].
        ignore_index: Index to ignore in loss computation. Defaults to -100.

    Returns:
        Scalar loss tensor.
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index,
    )
