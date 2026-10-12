"""Unit tests for utilities module (src/utils/)."""

import logging
from pathlib import Path
import pytest

from src.utils.config import load_yaml_config
from src.utils.logging import get_logger


def test_load_yaml_config(tmp_path):
    """Verify load_yaml_config parses YAML files correctly."""
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("random_seed: 42\nsplits:\n  train_end: '2015-12-31'\n", encoding="utf-8")

    cfg = load_yaml_config(yaml_file)
    assert cfg["random_seed"] == 42
    assert cfg["splits"]["train_end"] == "2015-12-31"

    with pytest.raises(FileNotFoundError):
        load_yaml_config(tmp_path / "nonexistent.yaml")


def test_get_logger():
    """Verify get_logger creates a configured logger."""
    logger = get_logger("test_logger", level=logging.DEBUG)
    assert logger.name == "test_logger"
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) >= 1

