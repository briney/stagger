from importlib.resources import as_file, files

import click
from hydra import compose, initialize_config_dir

from stok.cli.smoke_test import run_smoke_test


@click.group()
def cli():
    """STok command line."""


@cli.command(
    name="smoke-test",
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
)
@click.pass_context
def smoke_test(ctx: click.Context):
    """Run the STok smoke test.

    Forwards any unknown options/arguments as Hydra overrides.
    Example: stok smoke-test model.encoder.n_layers=6
    """
    overrides = list(ctx.args)
    with as_file(files("stok").joinpath("configs")) as cfg_dir:
        with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
            cfg = compose(config_name="config", overrides=overrides)
    run_smoke_test(cfg)


@cli.command(
    name="train",
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
)
@click.pass_context
def train_cmd(ctx: click.Context):
    """Run encoder training.

    Forwards any unknown options/arguments as Hydra overrides.
    Example: stok train train.num_steps=5000 data.train=/path/train.csv
    """
    overrides = list(ctx.args)
    with as_file(files("stok").joinpath("configs")) as cfg_dir:
        with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
            cfg = compose(config_name="config", overrides=overrides)
    from .train import run_training

    run_training(cfg)


if __name__ == "__main__":
    cli()
