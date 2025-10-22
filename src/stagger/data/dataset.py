import torch
from torch.utils.data import Dataset


class DummySequenceDataset(Dataset):
    """Placeholder dataset producing random token/label pairs for smoke tests."""

    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        vocab_size: int,
        num_classes: int,
        pad_id: int = 0,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.pad_id = pad_id

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.randint(low=1, high=self.vocab_size, size=(self.seq_len,))
        labels = torch.randint(low=0, high=self.num_classes, size=(self.seq_len,))
        # randomly pad a couple at end
        tokens[-2:] = self.pad_id
        labels[-2:] = -100
        return tokens.long(), labels.long()
