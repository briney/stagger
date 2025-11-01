import torch

from stok.cli.train import _tokenize_and_align
from stok.utils.tokenizer import Tokenizer


def test_tokenize_and_align_alignment_and_ignore_index():
    tokenizer = Tokenizer()
    max_len = 8
    ignore_index = -100
    pad_id = 1

    seq = "LAGVSEK"  # short sequence
    indices = torch.tensor([3, 1, 4, 1], dtype=torch.long)

    tokens, labels = _tokenize_and_align(
        [{"seq": seq, "indices": indices}],
        tokenizer,
        max_len=max_len,
        ignore_index=ignore_index,
        pad_id=pad_id,
    )

    assert tokens.shape == (1, max_len)
    assert labels.shape == (1, max_len)

    L = tokens.shape[1]
    copy_len = min(len(indices), max(0, L - 2))

    # BOS position (0) must be ignored
    assert labels[0, 0].item() == ignore_index
    # The supervised span must start at position 1 and match indices (truncated if needed)
    if copy_len > 0:
        assert torch.equal(labels[0, 1 : 1 + copy_len], indices[:copy_len])
    # Any positions beyond the supervised span at least include EOS/PAD which should be ignored
    assert (labels[0, 1 + copy_len :] == ignore_index).any().item()


