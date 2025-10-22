import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from stagger.models.tagger import TaggerModel


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Hydra entry that builds the model and runs a tiny forward pass."""
    print(OmegaConf.to_yaml(cfg))

    # Load or synthesize a codebook for a quick smoke test
    C = cfg.model.codebook_size
    d_model = cfg.model.d_model
    # For a real run, load the actual VQ codebook tensor [C, d_code]
    codebook = torch.randn(C, d_model) * 0.02

    model = TaggerModel(
        vocab_size=cfg.model.vocab_size,
        pad_id=cfg.model.pad_id,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        ffn_mult=cfg.model.ffn_mult,
        dropout=cfg.model.dropout,
        attn_dropout=cfg.model.attn_dropout,
        rope_base=cfg.model.rope.base,
        codebook=codebook,
        classifier_kwargs=dict(
            use_cosine=cfg.model.classifier.use_cosine,
            learnable_temperature=cfg.model.classifier.learnable_temperature,
            bias_from_code_norm=cfg.model.classifier.bias_from_code_norm,
            projector_dim=cfg.model.classifier.projector_dim,
        ),
        norm_type=cfg.model.norm,
    )

    if cfg.print_model_summary:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model params: {n_params/1e6:.2f}M")

    # Tiny forward sanity check
    B, L = 2, 16
    tokens = torch.randint(low=1, high=cfg.model.vocab_size, size=(B, L))
    tokens[:, -2:] = cfg.model.pad_id
    labels = torch.randint(low=0, high=cfg.model.codebook_size, size=(B, L))
    labels[:, -2:] = (
        cfg.model.classifier.ignore_index
        if "ignore_index" in cfg.model.classifier
        else -100
    )

    out = model(
        tokens=tokens, labels=labels, ignore_index=cfg.model.classifier.ignore_index
    )
    print("logits:", out["logits"].shape, "loss:", float(out["loss"]))
    print("OK")


if __name__ == "__main__":
    main()
