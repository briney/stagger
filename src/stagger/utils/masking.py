import torch


def key_padding_mask_from_tokens(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Build key padding mask [B, L] where True marks padding positions."""
    return tokens == pad_id
