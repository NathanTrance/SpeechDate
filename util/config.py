"""Configuration loading helpers."""

from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def load_config(config_path) -> Dict[str, Any]:
    """Load a YAML config file into a dict. Accepts a str or Path."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config


def default_config_path(step_name: str) -> Path:
    """Return the default config path for a step, e.g. config/<step_name>.yaml."""
    return DEFAULT_CONFIG_DIR / f"{step_name}.yaml"
