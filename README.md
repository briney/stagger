# STōk: structure tokenizer

Encoder-only protein structure tokenizer using SDPA attention with RoPE and a SwiGLU MLP, managed via Hydra. The classifier can be tied to a frozen VQ codebook for per-residue structure tokens.

## install

```bash
pip install stok
```

## smoke test

The following `smoke test`command will print the config, model parameter count, and run a tiny forward pass:

```bash
python -m stok.cli.smoke_test
```

Config overrides can be used to run a smoke test using a different model architecture. This is useful for testing different architectures to ensure that the selected hyperparameters are compatible.

```bash
python -m stok.cli.smoke_test model.encoder.d_model=512 model.encoder.n_heads=8 model.encoder.n_layers=6
```

## codebook presets and custom files

By default the model uses the built-in codebook preset `base`, which corresponds to the codebook used in the [Large](https://github.com/mahdip72/vq_encoder_decoder?tab=readme-ov-file#pretrained-models) GCP-VQVAE model. Config overrides can be used to change the codebook.

- Use a different built-in preset (for example, the codebook use in the [Lite](https://github.com/mahdip72/vq_encoder_decoder?tab=readme-ov-file#pretrained-models) GCP-VQVAE model variant:

  ```bash
  python -m stok.cli.smoke_test model.codebook.preset=lite
  ```

- Use a custom codebook file (overrides preset):

  ```bash
  python -m stok.cli.smoke_test model.codebook.path=/abs/path/to/codebook.pt
  ```
If using a custom codebook file, it must be a PyTorch tensor saved in `.pt` format and of shape `[C, d_code]`, where `C` is the codebook size and `d_code` is the codebook dimension. If `d_code` does not match the encoder model dimension, a linear projection will be automatically added to the classifier head.


Configuration fields:

```yaml
model:
  codebook:
    preset: "base"   # one of: "base", "lite" (default: base)
    path: null       # custom file path; when set, overrides preset
```
