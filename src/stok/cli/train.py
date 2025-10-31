import math
import sys
from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader

from stok.data.dataset import DummySequenceDataset, VQIndicesDataset
from stok.models.stok import STokModel
from stok.utils.codebook import load_codebook
from stok.utils.tokenizer import Tokenizer


def _maybe_get_accelerator():
    try:
        from accelerate import Accelerator  # type: ignore

        return Accelerator()
    except Exception:
        return None


def _build_scheduler(
    optimizer: torch.optim.Optimizer, *, warmup_steps: int, total_steps: int
):
    def lr_lambda(current_step: int):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # cosine decay from 1 -> 0
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _compute_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int
) -> float:
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        mask = labels != ignore_index
        if mask.sum().item() == 0:
            return 0.0
        correct = (preds[mask] == labels[mask]).sum().item()
        total = mask.sum().item()
        return float(correct) / float(total)


def _tokenize_and_align(
    batch: list[dict[str, Any]] | list[tuple[torch.Tensor, torch.Tensor]],
    tokenizer: Optional[Tokenizer],
    *,
    max_len: int,
    ignore_index: int,
    pad_id: int,
):
    # If using DummySequenceDataset, batch is tuples(tokens, labels)
    if tokenizer is None:
        tokens, labels = zip(*batch)  # type: ignore[arg-type]
        return torch.stack(tokens, dim=0), torch.stack(labels, dim=0)

    # Else VQIndicesDataset dicts with 'seq' and 'indices'
    input_ids = []
    label_ids = []
    for item in batch:  # type: ignore[assignment]
        seq: str = item["seq"]
        indices: torch.Tensor = item["indices"].long()

        enc = tokenizer(
            seq,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        ids = enc["input_ids"][0]

        # Build labels aligned to tokens: CLS/EOS/PAD -> ignore_index
        L = ids.size(0)
        labels = torch.full((L,), ignore_index, dtype=torch.long)

        # positions 1..(1+len(indices)) receive labels, respecting truncation before EOS
        copy_len = min(len(indices), max(0, L - 2))
        if copy_len > 0:
            labels[1 : 1 + copy_len] = indices[:copy_len]

        input_ids.append(ids)
        label_ids.append(labels)

    tokens = torch.stack(input_ids, dim=0)
    labels = torch.stack(label_ids, dim=0)
    return tokens, labels


def _build_dataloaders(cfg: DictConfig, *, codebook_size: int, pad_id: int):
    batch_size: int = cfg.data.batch_size
    max_len: int = cfg.data.max_len
    num_workers: int = cfg.data.num_workers
    pin_memory: bool = cfg.data.pin_memory
    ignore_index: int = cfg.model.classifier.ignore_index

    tokenizer: Optional[Tokenizer] = None
    collate_fn = None

    if cfg.data.get("train"):
        # CSV-backed dataset; tokenize in collate
        train_ds = VQIndicesDataset(csv_path=str(cfg.data.train), max_length=max_len)
        eval_ds = (
            VQIndicesDataset(csv_path=str(cfg.data.eval), max_length=max_len)
            if cfg.data.get("eval")
            else None
        )
        tokenizer = Tokenizer()

        def collate(batch):
            return _tokenize_and_align(
                batch,
                tokenizer,
                max_len=max_len,
                ignore_index=ignore_index,
                pad_id=pad_id,
            )

        collate_fn = collate
    else:
        # Fallback dummy data for quick smoke training
        train_ds = DummySequenceDataset(
            num_samples=512,
            seq_len=min(max_len, 256),
            vocab_size=cfg.model.encoder.vocab_size,
            num_classes=codebook_size,
            pad_id=pad_id,
        )
        eval_ds = DummySequenceDataset(
            num_samples=128,
            seq_len=min(max_len, 256),
            vocab_size=cfg.model.encoder.vocab_size,
            num_classes=codebook_size,
            pad_id=pad_id,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    eval_loader = (
        DataLoader(
            eval_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            drop_last=False,
        )
        if eval_ds is not None
        else None
    )
    return train_loader, eval_loader


def _maybe_init_wandb(cfg: DictConfig, *, is_main_process: bool):
    wb = None
    if cfg.train.get("wandb") and cfg.train.wandb.get("enabled", True):
        if is_main_process:
            try:
                import wandb  # type: ignore

                wandb.init(
                    project=cfg.train.wandb.get("project", "stok"),
                    entity=cfg.train.wandb.get("entity"),
                    group=cfg.train.wandb.get("group"),
                    name=cfg.train.wandb.get("name"),
                    tags=list(cfg.train.wandb.get("tags", [])),
                    config=OmegaConf.to_container(cfg, resolve=True),
                )
                wb = wandb
            except Exception:
                # proceed without W&B
                wb = None
    return wb


def run_training(cfg: DictConfig):
    accelerator = _maybe_get_accelerator()
    is_main = accelerator.is_main_process if accelerator else True
    printer = accelerator.print if accelerator else print

    # Load codebook and build model
    codebook = load_codebook(
        preset=cfg.model.codebook.get("preset"),
        path=cfg.model.codebook.get("path"),
    )
    codebook_size = codebook.shape[0]
    model = STokModel(
        vocab_size=cfg.model.encoder.vocab_size,
        pad_id=cfg.model.encoder.pad_id,
        d_model=cfg.model.encoder.d_model,
        n_heads=cfg.model.encoder.n_heads,
        n_layers=cfg.model.encoder.n_layers,
        ffn_mult=cfg.model.encoder.ffn_mult,
        dropout=cfg.model.encoder.dropout,
        attn_dropout=cfg.model.encoder.attn_dropout,
        codebook=codebook,
        classifier_kwargs=dict(
            use_cosine=cfg.model.classifier.use_cosine,
            learnable_temperature=cfg.model.classifier.learnable_temperature,
            bias_from_code_norm=cfg.model.classifier.bias_from_code_norm,
            projector_dim=cfg.model.classifier.projector_dim,
        ),
        norm_type=cfg.model.encoder.norm,
    )

    # Data
    train_loader, eval_loader = _build_dataloaders(
        cfg, codebook_size=codebook_size, pad_id=cfg.model.encoder.pad_id
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.optimizer.lr,
        betas=tuple(cfg.train.optimizer.betas),
        weight_decay=cfg.train.optimizer.weight_decay,
    )

    # Determine training steps
    grad_accum_steps: int = cfg.train.get("grad_accum_steps", 1)
    if cfg.train.get("epochs") is not None:
        steps_per_epoch = math.ceil(len(train_loader))
        max_steps = int(cfg.train.epochs) * steps_per_epoch
    else:
        max_steps = int(cfg.train.get("num_steps", 10000))

    scheduler = _build_scheduler(
        optimizer,
        warmup_steps=int(cfg.train.scheduler.get("warmup_steps", 0)),
        total_steps=max_steps,
    )

    # Prepare with Accelerate (if available)
    if accelerator:
        to_prepare = [model, optimizer, train_loader]
        if eval_loader is not None:
            to_prepare.append(eval_loader)
        prepared = accelerator.prepare(*to_prepare)
        # Unpack prepared components in order
        model = prepared[0]
        optimizer = prepared[1]
        train_loader = prepared[2]
        if eval_loader is not None:
            eval_loader = prepared[3]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    # W&B
    wb = _maybe_init_wandb(cfg, is_main_process=is_main)

    # Training loop
    model.train()
    global_step = 0
    running_loss = 0.0
    log_interval = int(cfg.train.get("log_steps", 50))
    eval_interval = int(cfg.train.get("eval_steps", 1000))
    ignore_index = int(cfg.model.classifier.ignore_index)
    grad_clip = float(cfg.train.get("grad_clip_norm", 1.0))

    while global_step < max_steps:
        for batch in train_loader:
            if accelerator is None:
                tokens, labels = batch
                tokens = tokens.to(model.embed.weight.device)
                labels = labels.to(model.embed.weight.device)
            else:
                tokens, labels = batch

            outputs = model(tokens=tokens, labels=labels, ignore_index=ignore_index)
            loss: torch.Tensor = outputs["loss"]

            # Normalize by grad accumulation
            loss_to_backprop = loss / grad_accum_steps
            if accelerator:
                accelerator.backward(loss_to_backprop)
            else:
                loss_to_backprop.backward()

            if (global_step + 1) % grad_accum_steps == 0:
                if grad_clip is not None and grad_clip > 0:
                    if accelerator:
                        accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                    else:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.detach().item())

            # Logging
            if (global_step + 1) % log_interval == 0 and is_main:
                with torch.no_grad():
                    acc = _compute_accuracy(outputs["logits"], labels, ignore_index)
                lr = scheduler.get_last_lr()[0]
                msg = f"step {global_step+1}/{max_steps} | loss {running_loss/log_interval:.4f} | acc {acc:.4f} | lr {lr:.2e}"
                printer(msg)
                if wb is not None:
                    wb.log(
                        {
                            "train/loss": running_loss / log_interval,
                            "train/acc": acc,
                            "lr": lr,
                        },
                        step=global_step + 1,
                    )
                running_loss = 0.0

            # Eval
            if (global_step + 1) % eval_interval == 0 and eval_loader is not None:
                eval_loss_sum = 0.0
                eval_acc_sum = 0.0
                eval_batches = 0.0
                model.eval()
                with torch.no_grad():
                    for ev in eval_loader:
                        etok, elab = ev
                        if accelerator is None:
                            etok = etok.to(model.embed.weight.device)
                            elab = elab.to(model.embed.weight.device)
                        out = model(tokens=etok, labels=elab, ignore_index=ignore_index)
                        eval_loss_sum += float(out["loss"].item())
                        eval_acc_sum += _compute_accuracy(
                            out["logits"], elab, ignore_index
                        )
                        eval_batches += 1.0
                model.train()

                # Aggregate across processes if using Accelerate
                if accelerator:
                    metrics_local = torch.tensor(
                        [eval_loss_sum, eval_acc_sum, eval_batches],
                        dtype=torch.float64,
                        device=model.embed.weight.device,
                    )
                    gathered = accelerator.gather_for_metrics(metrics_local)
                    # gathered is [N, 3]; reduce by sum
                    eval_loss_sum = float(gathered[:, 0].sum().item())
                    eval_acc_sum = float(gathered[:, 1].sum().item())
                    eval_batches = float(gathered[:, 2].sum().item())

                eval_loss = eval_loss_sum / max(1.0, eval_batches)
                eval_acc = eval_acc_sum / max(1.0, eval_batches)

                if is_main:
                    printer(f"eval | loss {eval_loss:.4f} | acc {eval_acc:.4f}")
                    if wb is not None:
                        wb.log(
                            {"eval/loss": eval_loss, "eval/acc": eval_acc},
                            step=global_step + 1,
                        )

            global_step += 1
            if global_step >= max_steps:
                break

    if is_main:
        printer("Training complete.")


if __name__ == "__main__":
    print(
        "This module is intended to be invoked via the CLI: `stok train ...`",
        file=sys.stderr,
    )
