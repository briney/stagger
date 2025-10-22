# Stagger Test Suite

This directory contains tests for the Stagger project, organized for fast, CPU-only execution in CI. Tests aim to exercise real components (tokenizer, models, and data flow) end-to-end with tiny configurations suitable for continuous runs.

## Integration Tests

- e2e Tagger training (`integration/test_e2e_tagger_training.py`)
  - Purpose: Verify the complete pipeline from tokenization to model forward/backward and optimizer steps runs without error.
  - Scope: Uses the real `stagger.utils.tokenizer.Tokenizer`, actual `TaggerModel`, and a tiny synthetic codebook. Synthetic protein-like sequences (length ~96–128) are tokenized; labels are generated per-token except BOS/EOS/PAD positions which are ignored.
  - Pass criteria: A forward pass returns logits of shape `[B, L, C]` and a finite loss; two short optimizer steps complete and change at least one trainable parameter value.

## Conventions

- Tests are CPU-only to ensure CI reliability and speed.
- Tiny model sizes and small codebooks keep runtime to a few seconds.
- Synthetic utilities live in `tests/utils` and are shared across tests.
As new tests are added, update this README with a concise description of each test and its purpose.
