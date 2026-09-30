"""Shared helpers for config management."""

import yaml
from box import Box


def load_config(file_path):
    """Load a YAML configuration file into a Box for attribute access."""
    with open(file_path, "r") as file:
        return Box(yaml.safe_load(file))
