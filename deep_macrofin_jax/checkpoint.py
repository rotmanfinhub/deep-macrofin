"""Portable, pickle-free parameter checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _encode(value: Any, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    if isinstance(value, (jax.Array, np.ndarray)):
        key = f"array_{len(arrays)}"
        arrays[key] = np.asarray(value)
        return {"type": "array", "key": key}
    if isinstance(value, dict):
        return {
            "type": "dict",
            "items": [[str(key), _encode(item, arrays)] for key, item in value.items()],
        }
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_encode(item, arrays) for item in value]}
    if isinstance(value, list):
        return {"type": "list", "items": [_encode(item, arrays) for item in value]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"type": "scalar", "value": value}
    raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")


def _decode(spec: dict[str, Any], arrays: Any) -> Any:
    kind = spec["type"]
    if kind == "array":
        return jnp.asarray(arrays[spec["key"]])
    if kind == "dict":
        return {key: _decode(value, arrays) for key, value in spec["items"]}
    if kind == "tuple":
        return tuple(_decode(value, arrays) for value in spec["items"])
    if kind == "list":
        return [_decode(value, arrays) for value in spec["items"]]
    if kind == "scalar":
        return spec["value"]
    raise ValueError(f"Unknown checkpoint node type: {kind}")


def save_checkpoint(
    path: str | Path, params: Any, metadata: dict[str, Any] | None = None
) -> Path:
    """Save model parameters and JSON metadata to one compressed ``.npz`` file."""

    target = Path(path)
    if target.suffix != ".npz":
        target = target.with_suffix(".npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    structure = _encode(jax.device_get(params), arrays)
    manifest = json.dumps(
        {"format_version": 1, "structure": structure, "metadata": metadata or {}},
        sort_keys=True,
    )
    with target.open("wb") as handle:
        np.savez_compressed(handle, manifest=np.asarray(manifest), **arrays)
    return target


def load_checkpoint(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load parameters and metadata from :func:`save_checkpoint`."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest"]))
        if manifest.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        params = _decode(manifest["structure"], archive)
    return params, manifest.get("metadata", {})
