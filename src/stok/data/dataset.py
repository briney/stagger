import pandas as pd
import torch
from torch.utils.data import Dataset


class VQIndicesDataset(Dataset):
    """Dataset for loading VQ indices from a CSV file."""

    def __init__(self, csv_path: str, max_length: int):
        self.data = pd.read_csv(csv_path)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        pid = row["pid"]
        seq = row["protein_sequence"]
        indices = [int(i) for i in row["indices"].split()]  #  space-separate string

        idx_length = len(indices)
        pad_length = self.max_length - idx_length

        # pad indices with -1 and create a mask
        padded_indices = indices + [-1] * pad_length
        mask = [True] * idx_length + [False] * pad_length

        # make tensors
        indices_tensor = torch.tensor(padded_indices, dtype=torch.long)
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        nan_mask = indices_tensor != -1

        return {
            "pid": pid,
            "indices": indices_tensor,
            "seq": seq,
            "masks": mask_tensor,
            "nan_masks": nan_mask,
        }


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
        """Initialize dummy dataset.

        Args:
            num_samples: Number of samples in dataset.
            seq_len: Sequence length for each sample.
            vocab_size: Vocabulary size for token generation.
            num_classes: Number of classes for label generation.
            pad_id: Padding token ID.
        """
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.pad_id = pad_id

    def __len__(self) -> int:
        """Return dataset size.

        Returns:
            Number of samples in dataset.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (tokens, labels) with shapes [seq_len] and [seq_len].
        """
        tokens = torch.randint(low=1, high=self.vocab_size, size=(self.seq_len,))
        labels = torch.randint(low=0, high=self.num_classes, size=(self.seq_len,))
        # randomly pad a couple at end
        tokens[-2:] = self.pad_id
        labels[-2:] = -100
        return tokens.long(), labels.long()
