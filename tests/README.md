# STōk Test Suite

This directory contains tests for the Stagger project, organized for fast, CPU-only execution in CI. Tests aim to exercise real components (tokenizer, models, and data flow) end-to-end with tiny configurations suitable for continuous runs.

## Integration Tests

- e2e Tagger training (`integration/test_e2e_tagger_training.py`)
  - Purpose: Verify the complete pipeline from tokenization to model forward/backward and optimizer steps runs without error.
  - Scope: Uses the real `stok.utils.tokenizer.Tokenizer`, actual `StokModel`, and a tiny synthetic codebook. Synthetic protein-like sequences (length ~96–128) are tokenized; labels are generated per-token except BOS/EOS/PAD positions which are ignored.
  - Pass criteria: A forward pass returns logits of shape `[B, L, C]` and a finite loss; two short optimizer steps complete and change at least one trainable parameter value.

- Package data (`integration/test_package_data.py`)
  - Purpose: Verify that configs and checkpoint files are properly packaged and accessible after installation.
  - Scope: Checks that config files and built-in codebook checkpoint files (`base.pt`, `lite.pt`) are included in the installed package and can be accessed via `importlib.resources`.
  - Pass criteria: All expected config and checkpoint files exist in the installed package.

- Codebook loading (`integration/test_codebook_loading.py`)
  - Purpose: Ensure the codebook loader supports explicit path overrides and preset-based loading.
  - Scope: Calls `stok.utils.codebook.load_codebook` with a custom `.pt` path (which must override any preset) and verifies the `lite` preset loads from package resources.
  - Pass criteria: When both `preset` and `path` are provided, the loaded tensor matches the saved custom tensor shape and values; the `lite` preset returns a 2D tensor with positive dimensions.

- Decoder loader (`integration/test_decoder_loader.py`)
  - Purpose: Validate pretrained decoder loading via explicit path or preset download with caching and freezing behavior.
  - Scope: Uses `stok.models.decoder.load_pretrained_decoder` with a temp checkpoint path to check path override and `freeze=True`; simulates download for presets and verifies cache reuse via `STOK_DECODER_CACHE`.
  - Pass criteria: With `path`, the model loads on CPU, is in eval mode when frozen, all params have `requires_grad=False`, and input/output projector shapes are as expected; with preset download, the first call downloads and caches once and the second call reuses the cache without re-downloading.

- CLI training smoke (`integration/test_cli_train_smoke.py`)
  - Purpose: Exercise the `stok train` CLI end-to-end on dummy data.
  - Scope: Invokes Click CLI with Hydra overrides for a tiny model (e.g., `d_model=64`, `n_layers=2`, `n_heads=4`, `ffn_mult=1.0`), small data loader (`batch_size=2`, `max_len=64`, `num_workers=0`), `model.codebook.preset=lite`, a few steps (`train.num_steps=3`), and `train.wandb.enabled=false`.
  - Pass criteria: CLI exits with code 0 and prints `Training complete.`.

- CLI training with CSV (`integration/test_cli_train_with_csv.py`)
  - Purpose: End-to-end training on a tiny real CSV-backed dataset to validate tokenizer alignment and `VQIndicesDataset` integration.
  - Scope: Generates small `train.csv` and `eval.csv` with columns `pid,protein_sequence,indices`, sets a small `data.max_len` and indices length (`max_len-2`), uses the same tiny model overrides, and triggers evaluation (`train.eval_steps=2`).
  - Pass criteria: CLI exits with code 0, prints an eval line (`eval | loss ...`), and ends with `Training complete.`.

- Programmatic training (`integration/test_run_training_programmatic.py`)
  - Purpose: Run `run_training` directly (non-CLI) to ensure programmatic usage works with Hydra-composed configs.
  - Scope: Composes config from packaged `stok/configs` via `initialize_config_dir`/`compose` and uses the same tiny overrides as the CLI smoke test.
  - Pass criteria: No exceptions during the run; captured stdout contains `Training complete.`.

## Unit Tests

- FAPE loss (`unit/test_fape_loss.py`)
  - Purpose: Verify Frame-Aligned Point Error (FAPE) correctness and behavior.
  - Scope: Uses synthetic, stable N–CA–C coordinates to test:
    - Identity: loss ≈ 0 when predictions equal ground truth.
    - Rigid invariance: loss unchanged under same global rotation/translation.
    - Masking/NaNs: inferred masking from NaNs matches explicit `residue_mask`.
  - Pass criteria: All assertions pass; loss values are finite and consistent across invariance and masking scenarios.

- Train helpers (`unit/test_train_helpers.py`)
  - Purpose: Validate scheduler shape and accuracy computation.
  - Scope: `_build_scheduler` warmup increase then cosine decay (bounds in `[0, 1]`); `_compute_accuracy` respects `ignore_index` and matches expected ratio on a small constructed example.
  - Pass criteria: Learning rates follow expected monotonic segments within bounds; accuracy equals the expected value.

- Tokenize and align (`unit/test_tokenize_and_align.py`)
  - Purpose: Verify token-label alignment for CSV inputs.
  - Scope: Uses real `Tokenizer` and `_tokenize_and_align` with a short sequence and indices; asserts BOS/EOS/PAD positions are ignored, supervised span starts at position 1, and indices are truncated to `max_len-2`.
  - Pass criteria: Output shapes match `max_len`; labels at [0] are `ignore_index`; labels[1:1+copy_len] equal provided indices slice; remaining positions include `ignore_index`.

## Conventions

- Tests are CPU-only to ensure CI reliability and speed.
- Tiny model sizes and small codebooks keep runtime to a few seconds.
- Synthetic utilities live in `tests/utils` and are shared across tests.
As new tests are added, update this README with a concise description of each test and its purpose.
