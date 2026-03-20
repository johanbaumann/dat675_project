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


def create_sampling_stats() -> dict:
    """Create a fresh sampling quality counter dict.

    Keep this as the single source of truth so all sampling/tracking paths use
    the same schema and new counters only need to be added in one place.
    """
    return {
        'total_generated': 0,
        'accepted': 0,
        'invalid_or_empty': 0,
        'discarded_cleanup': 0,
        'in_training': 0,
        'duplicate': 0,
        'rejected_by_filter': 0,
        'rejected_by_validation_scaffold': 0,
        'rejected_by_heldout_scaffold': 0,
        'rejected_by_strict_bin_quota': 0,
        'salt_stripped': 0,
        'tautomer_canonicalized': 0,
    }


def compute_vun_from_counts(
    *,
    total_generated: int,
    invalid_or_empty: int,
    discarded_cleanup: int,
    in_training: int,
    duplicate: int,
    accepted: int,
) -> dict:
    """Compute validity/uniqueness/novelty/acceptance from aggregate counters."""
    total = int(total_generated)
    if total <= 0:
        return {
            'validity': 0.0,
            'uniqueness': 0.0,
            'novelty': 0.0,
            'acceptance_rate': 0.0,
            'valid_count': 0,
            'unique_count': 0,
            'novel_count': 0,
        }

    valid_count = int(total - int(invalid_or_empty) - int(discarded_cleanup))
    unique_count = int(total - int(duplicate))
    novel_count = int(total - int(in_training))
    accepted_count = int(accepted)

    return {
        'validity': float(valid_count) / float(total),
        'uniqueness': float(unique_count) / float(total),
        'novelty': float(novel_count) / float(total),
        'acceptance_rate': float(accepted_count) / float(total),
        'valid_count': int(valid_count),
        'unique_count': int(unique_count),
        'novel_count': int(novel_count),
    }
