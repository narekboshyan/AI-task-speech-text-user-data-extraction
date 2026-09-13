"""Smoke test: the package imports and config.yaml is well-formed."""
from pathlib import Path

import yaml

import phonebot


def test_package_imports():
    assert phonebot is not None


def test_config_has_required_sections():
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    assert set(cfg) >= {"paths", "asr", "llm"}
    assert Path(cfg["paths"]["recordings"]).is_dir()
    assert Path(cfg["paths"]["ground_truth"]).is_file()
