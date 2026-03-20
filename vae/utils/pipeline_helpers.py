from __future__ import annotations

from copy import deepcopy
from typing import Optional


def deep_update_dict(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base` without mutating inputs."""
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update_dict(out[key], value)
        else:
            out[key] = value
    return out


def resolve_target_sampling_mode(cfg: dict) -> str:
    """Resolve target sampling mode with backward compatibility.

    Priority:
      1) target_sampling_mode
      2) legacy run_training_dist boolean
    """
    explicit = cfg.get('target_sampling_mode', None)
    if explicit is not None:
        mode = str(explicit).strip().lower()
    else:
        mode = 'training_dist' if bool(cfg.get('run_training_dist', False)) else 'single_target'

    aliases = {
        'single': 'single_target',
        'single_target': 'single_target',
        'training': 'training_dist',
        'training_dist': 'training_dist',
        'uniform': 'uniform_range',
        'uniform_range': 'uniform_range',
        'uniform_strict': 'uniform_range_strict',
        'uniform_range_strict': 'uniform_range_strict',
    }
    resolved = aliases.get(mode)
    if resolved is None:
        raise ValueError(
            'target_sampling_mode must be one of: '
            'single_target, training_dist, uniform_range, uniform_range_strict'
        )
    return resolved


def coerce_int(value, *, default: int = 0) -> int:
    if value is None:
        return int(default)
    raw = str(value).strip()
    if raw == '':
        return int(default)
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def coerce_int_or_none(value) -> Optional[int]:
    if value is None:
        return None
    raw = str(value).strip()
    if raw == '':
        return None
    try:
        return int(float(raw))
    except Exception:
        return None


def coerce_float(value, *, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    raw = str(value).strip()
    if raw == '':
        return default
    try:
        return float(raw)
    except Exception:
        return default
