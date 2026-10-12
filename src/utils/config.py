"""Configuration utility module per docs/architecture.md."""

from pathlib import Path
from typing import Any, Dict, Union
import yaml


def load_yaml_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file into a dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

