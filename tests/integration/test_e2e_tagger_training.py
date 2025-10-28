import torch
from torch.utils.data import DataLoader, Dataset

from stagger.models.tagger import TaggerModel
from tests.utils.synthetic import make_collate_fn, random_protein_sequence


class SeqDataset(Dataset):
    """Simple dataset of random protein sequences."""

    def __init__(self, n: int, min_len: int = 96, max_len: int = 128):
        """Initialize dataset with random sequences.

        Args:
            n: Number of sequences.
            min_len: Minimum sequence length. Defaults to 96.
            max_len: Maximum sequence length. Defaults to 128.
        """
        self.data = [random_protein_sequence(min_len, max_len) for _ in range(n)]

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> str:
        """Get a sequence from the dataset.

        Args:
            idx: Sequence index.

        Returns:
            Protein sequence string.
        """
        return self.data[idx]


def build_model(
    tiny_model_hparams, codebook_tensor, device: torch.device
) -> TaggerModel:
    """Build a tagger model with test hyperparameters.

    Args:
        tiny_model_hparams: Model hyperparameters dictionary.
        codebook_tensor: Codebook tensor.
        device: Device to place model on.

    Returns:
        TaggerModel instance on specified device.
    """
    model = TaggerModel(
        vocab_size=tiny_model_hparams["vocab_size"],
        pad_id=tiny_model_hparams["pad_id"],
        d_model=tiny_model_hparams["d_model"],
        n_heads=tiny_model_hparams["n_heads"],
        n_layers=tiny_model_hparams["n_layers"],
        ffn_mult=tiny_model_hparams["ffn_mult"],
        dropout=tiny_model_hparams["dropout"],
        attn_dropout=tiny_model_hparams["attn_dropout"],
        rope_base=tiny_model_hparams["rope_base"],
        codebook=codebook_tensor,
        classifier_kwargs=dict(
            use_cosine=False,
            learnable_temperature=True,
            bias_from_code_norm=True,
            projector_dim=None,
        ),
    ).to(device)
    return model


def test_model_instantiation_and_forward(
    tokenizer, tiny_model_hparams, codebook_tensor, ignore_index, device
):
    """Test model instantiation and forward pass."""
    model = build_model(tiny_model_hparams, codebook_tensor, device)

    ds = SeqDataset(n=2)
    collate = make_collate_fn(
        tokenizer, tiny_model_hparams["codebook_size"], ignore_index
    )
    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate)

    tokens, labels = next(iter(dl))
    tokens, labels = tokens.to(device), labels.to(device)

    out = model(tokens=tokens, labels=labels, ignore_index=ignore_index)
    assert out["logits"].shape[:2] == tokens.shape
    assert torch.isfinite(out["loss"]).item()


def test_end_to_end_single_optimizer_step(
    tokenizer, tiny_model_hparams, codebook_tensor, ignore_index, device
):
    """Test end-to-end training with a single optimizer step."""
    model = build_model(tiny_model_hparams, codebook_tensor, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ds = SeqDataset(n=2)
    collate = make_collate_fn(
        tokenizer, tiny_model_hparams["codebook_size"], ignore_index
    )
    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate)

    tokens, labels = next(iter(dl))
    tokens, labels = tokens.to(device), labels.to(device)

    model.train()
    with torch.no_grad():
        ref = next(p for p in model.parameters() if p.requires_grad).clone()

    for _ in range(2):
        out = model(tokens=tokens, labels=labels, ignore_index=ignore_index)
        loss = out["loss"]
        loss.backward()
        opt.step()
        opt.zero_grad()

    with torch.no_grad():
        changed = (
            (next(p for p in model.parameters() if p.requires_grad) - ref)
            .abs()
            .sum()
            .item()
        )
    assert changed > 0.0
