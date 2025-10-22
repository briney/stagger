# stagger

Encoder-only protein tagger using SDPA attention with RoPE and a SwiGLU MLP, managed via Hydra. The classifier can be tied to a frozen VQ codebook for per-residue structure tokens.

## Install (optional editable)

```bash
python -m pip install -e .
```

## Dev smoke test

```bash
python -m stagger.cli.build_model +model.d_model=512 +model.n_layers=6
```

This prints the config, model parameter count, and runs a tiny forward pass.
