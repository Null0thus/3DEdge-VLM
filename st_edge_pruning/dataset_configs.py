from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _parse_scalar(raw: str) -> Any:
    """Parse a small YAML scalar without depending on PyYAML."""

    value = raw.strip()
    if value in {"null", "None", "~"}:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_config_file(path: str | Path) -> Dict[str, Any]:
    """Load a JSON/YAML-like config file.

    PyYAML is used when available. A tiny fallback parser is kept so the
    experiment entrypoint can still read our simple flat config files.
    """

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return data or {}
    except Exception:
        data: Dict[str, Any] = {}
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            data[key.strip()] = _parse_scalar(value)
        return data


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge config dictionaries from low to high priority."""

    merged: Dict[str, Any] = {}
    for config in configs:
        for key, value in config.items():
            if value is not None:
                merged[key] = value
    return merged


def load_dataset_config(dataset: str, explicit_path: str | None = None) -> Dict[str, Any]:
    """Load the dataset config selected by --dataset."""

    path = Path(explicit_path) if explicit_path else Path("configs") / "datasets" / f"{dataset.lower()}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    return load_config_file(path)
