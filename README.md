# stagger

Encoder-only protein tagger using SDPA attention with RoPE and a SwiGLU MLP, managed via Hydra. The classifier can be tied to a frozen VQ codebook for per-residue structure tokens.

## Install (optional editable)

```bash
python -m pip install -e .
```

## Dev smoke test

```bash
python src/stagger/cli/smoke_test.py model.d_model=512 model.n_heads=8 model.n_layers=6
```

This prints the config, model parameter count, and runs a tiny forward pass.

## Codebook presets and custom paths

By default the model uses the built-in codebook preset `base`.

- Use a different built-in preset:

  ```bash
  python -m stagger.cli.smoke_test model.codebook.preset=lite
  ```

- Use a custom codebook file (overrides preset):

  ```bash
  python -m stagger.cli.smoke_test model.codebook.path=/abs/path/to/codebook.pt
  ```

Configuration fields:

```yaml
model:
  codebook:
    preset: "base"   # one of: "base", "lite" (default: base)
    path: null       # custom file path; when set, overrides preset
```
