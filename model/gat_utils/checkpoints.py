from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

import torch

from .target_scaling import (
	build_target_standardizer_from_stats,
	get_target_standardization_config,
)
from .training_helpers import build_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== checkpoint helpers ====================
def get_fold_checkpoint_path(target_folder: str, fold_idx: int) -> Path:
	"""Canonical path for a per-fold best-model checkpoint.

	Checkpoints live inside the dataset folder so each dataset mix keeps
	its own set of saved weights and they never overwrite each other.
	"""
	return Path(target_folder) / "checkpoints" / f"best_model_fold_{fold_idx}.pth"


def save_fold_checkpoint(
	checkpoint: dict[str, Any],
	checkpoint_path: Path,
	*,
	retries: int = 5,
	retry_delay_sec: float = 0.25,
) -> None:
	"""Persist a checkpoint robustly on Windows.

	Uses write-then-replace for atomic updates and retries to tolerate
	transient file-lock/IO hiccups from sync/indexing processes.
	"""
	checkpoint_path = Path(checkpoint_path)
	checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")

	last_error = None
	for attempt in range(1, max(int(retries), 1) + 1):
		try:
			torch.save(checkpoint, tmp_path)
			os.replace(tmp_path, checkpoint_path)
			return
		except Exception as exc:  # noqa: BLE001 - preserve original exception context.
			last_error = exc
			try:
				if tmp_path.exists():
					tmp_path.unlink()
			except OSError:
				pass
			if attempt < retries:
				time.sleep(retry_delay_sec * attempt)

	raise RuntimeError(
		"Failed to save checkpoint after retries. "
		f"path='{checkpoint_path.resolve()}' cwd='{Path.cwd()}'"
	) from last_error


def load_fold_checkpoint(
	checkpoint_path: Path,
	config: dict[str, Any],
	model_class,
	feature_context: dict[str, Any],
) -> tuple:
	"""Load a fold checkpoint and return (model, target_standardizer | None).

	Handles both the current rich format::

		{"state_dict": ..., "fold": int, "target_mean": float,
		 "target_std": float, "val_rmse": float}

	and the legacy bare state_dict format for backward compatibility.
	"""
	ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

	if isinstance(ckpt, dict) and "state_dict" in ckpt:
		state_dict = ckpt["state_dict"]
		target_mean = float(ckpt.get("target_mean", 0.0))
		target_std = float(ckpt.get("target_std", 1.0))
	else:
		# Legacy format: checkpoint is a plain state_dict (OrderedDict).
		state_dict = ckpt
		target_mean = 0.0
		target_std = 1.0

	model = build_model(model_class, config, feature_context)
	model.load_state_dict(state_dict)
	model.eval()

	use_std = get_target_standardization_config(config)["enabled"]
	target_standardizer = None
	if use_std:
		target_standardizer = build_target_standardizer_from_stats(target_mean, target_std)

	return model, target_standardizer
