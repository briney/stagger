from click.testing import CliRunner

from stok.cli.cli import cli


def test_cli_train_smoke_runs_end_to_end_cpu():
    runner = CliRunner()
    overrides = [
        # tiny model for speed
        "model.encoder.d_model=64",
        "model.encoder.n_layers=2",
        "model.encoder.n_heads=4",
        "model.encoder.ffn_mult=1.0",
        "model.encoder.dropout=0.0",
        "model.encoder.attn_dropout=0.0",
        # small codebook preset
        "model.codebook.preset=lite",
        # small data loader
        "data.batch_size=2",
        "data.max_len=64",
        "data.num_workers=0",
        "data.pin_memory=false",
        # fast training
        "train.num_steps=3",
        "train.log_steps=1",
        "train.eval_steps=100000",
        "train.grad_accum_steps=1",
        # disable external logging
        "train.wandb.enabled=false",
    ]

    result = runner.invoke(cli, ["train", *overrides])  # type: ignore[arg-type]
    assert result.exit_code == 0, result.output
    assert "Training complete." in result.output


