import math

import torch

from stok.cli.train import _build_scheduler, _compute_accuracy


def test_build_scheduler_warmup_then_cosine_decay():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=1.0)
    warmup_steps, total_steps = 3, 10
    sched = _build_scheduler(opt, warmup_steps=warmup_steps, total_steps=total_steps)

    lrs: list[float] = []
    for _ in range(total_steps):
        sched.step()
        lrs.append(sched.get_last_lr()[0])

    # warmup monotonic increasing
    assert lrs[0] < lrs[1] <= lrs[2] <= 1.0 + 1e-6
    # post-warmup non-increasing
    for i in range(warmup_steps + 1, total_steps):
        assert lrs[i] <= lrs[i - 1] + 1e-6
    # bounds
    assert all(0.0 - 1e-6 <= lr <= 1.0 + 1e-6 for lr in lrs)


def test_compute_accuracy_with_ignore_index():
    ignore_index = -100
    logits = torch.tensor(
        [
            [2.0, 1.0],  # pred 0, ignored
            [0.1, 0.9],  # pred 1, correct
            [0.9, 0.1],  # pred 0, incorrect
        ]
    )
    labels = torch.tensor([ignore_index, 1, 1])
    acc = _compute_accuracy(logits, labels, ignore_index)
    assert math.isclose(acc, 0.5, rel_tol=1e-6, abs_tol=1e-6)


