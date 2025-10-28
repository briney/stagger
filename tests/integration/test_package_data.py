import importlib.resources as r


def test_packaged_configs_and_checkpoints_exist():
    root = r.files("stagger")
    assert (root / "configs" / "config.yaml").is_file()
    assert (root / "checkpoints" / "codebook" / "base.pt").is_file()
    assert (root / "checkpoints" / "codebook" / "lite.pt").is_file()
