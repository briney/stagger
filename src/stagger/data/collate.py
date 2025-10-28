import torch


def simple_pad_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate batch of token/label pairs.

    Currently assumes all sequences are equal length and stacks them.
    For variable-length sequences, this function would need to pad to max length.

    Args:
        batch: List of (tokens, labels) tuples where each tensor has shape [L].
        pad_id: Padding token ID (currently unused).

    Returns:
        Tuple of (tokens, labels) with shapes [B, L] and [B, L].
    """
    tokens, labels = zip(*batch)
    return torch.stack(tokens, dim=0), torch.stack(labels, dim=0)
