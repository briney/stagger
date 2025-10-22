import random
from typing import Callable, List, Tuple

import torch

# Amino-acid-like alphabet consistent with Tokenizer's DEFAULT_VOCAB letters
_AMINO_ALPHABET = list("LAGV SERTIDPKQNFYMHW CXBUOZ.-".replace(" ", ""))


def random_protein_sequence(min_len: int = 50, max_len: int = 200) -> str:
    length = random.randint(min_len, max_len)
    return "".join(random.choice(_AMINO_ALPHABET) for _ in range(length))


def build_batch(
    tokenizer, seqs: List[str], codebook_size: int, ignore_index: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(seqs, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"]  # [B, L]
    attn = enc["attention_mask"]  # [B, L]
    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id

    B, L = input_ids.shape
    labels = torch.full((B, L), fill_value=ignore_index, dtype=torch.long)

    # valid = token present in attention AND not BOS/EOS
    valid_mask = (attn == 1) & (input_ids != bos_id) & (input_ids != eos_id)
    # Ensure at least one valid supervised token to avoid NaN loss
    if valid_mask.any():
        rand_lbls = torch.randint(low=0, high=codebook_size, size=(B, L))
        labels[valid_mask] = rand_lbls[valid_mask]
    else:
        # Fallback: mark the last non-pad position of each sequence as valid
        for b in range(B):
            idxs = torch.nonzero(attn[b] == 1, as_tuple=False).flatten()
            if len(idxs) > 2:  # skip BOS/EOS
                j = idxs[-2].item()
                if input_ids[b, j] != bos_id and input_ids[b, j] != eos_id:
                    labels[b, j] = int(torch.randint(0, codebook_size, ()).item())
    return input_ids.long(), labels.long()


def make_collate_fn(
    tokenizer, codebook_size: int, ignore_index: int
) -> Callable[[List[str]], Tuple[torch.Tensor, torch.Tensor]]:
    def _collate(batch: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_batch(tokenizer, batch, codebook_size, ignore_index)

    return _collate
