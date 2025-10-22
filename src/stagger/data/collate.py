import torch


def simple_pad_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences to the max length in the batch.

    Placeholder; assumes all inputs are already equal length by default.
    """
    tokens, labels = zip(*batch)
    return torch.stack(tokens, dim=0), torch.stack(labels, dim=0)
