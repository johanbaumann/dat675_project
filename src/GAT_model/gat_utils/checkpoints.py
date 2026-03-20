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
def _sanitize_dataset_label(label: str) -> str:
	value = str(label).strip()
	value = value.replace("%", "_percent")
	value = value.replace(" ", "_")
	return value


def get_checkpoints_dir(target_folder: str) -> Path:
	return Path(target_folder) / "checkpoints"


def get_fold_checkpoint_path(target_folder: str, fold_idx: int) -> Path:
	"""Canonical path for a per-fold best-model checkpoint.

	Checkpoints live inside the dataset folder so each dataset mix keeps
	its own set of saved weights and they never overwrite each other.
	"""
	return get_checkpoints_dir(target_folder) / f"best_model_fold_{fold_idx}.pth"


def get_epoch_checkpoint_path(target_folder: str, fold_idx: int, epoch: int) -> Path:
	"""Canonical path for a periodic per-epoch checkpoint."""
	dataset_label = _sanitize_dataset_label(Path(target_folder).name)
	filename = f"model_{dataset_label}_cv_iteration_{fold_idx}_epoch_{epoch}.pth"
	return get_checkpoints_dir(target_folder) / filename


def infer_checkpoint_path_from_selection(
	target_folder: str,
	fold_idx: int,
	selection: int | str | Path | None,
) -> Path | None:
	"""Infer a checkpoint path from a shorthand selection value.

	Supported values:
	- None: no override (caller should use default fold checkpoint)
	- int: interpreted as epoch number for canonical epoch checkpoint naming
	- numeric str (e.g. "22"): interpreted as epoch number
	- non-numeric str / Path: treated as explicit path or filename
	"""
	if selection is None:
		return None

	if isinstance(selection, int):
		if selection < 1:
			raise ValueError("Epoch checkpoint shorthand must be >= 1.")
		return get_epoch_checkpoint_path(target_folder, fold_idx, int(selection))

	if isinstance(selection, Path):
		return selection

	selection_text = str(selection).strip()
	if not selection_text:
		return None

	if selection_text.isdigit():
		epoch = int(selection_text)
		if epoch < 1:
			raise ValueError("Epoch checkpoint shorthand must be >= 1.")
		return get_epoch_checkpoint_path(target_folder, fold_idx, epoch)

	return Path(selection_text)


def resolve_checkpoint_path(
	target_folder: str,
	fold_idx: int,
	selected_checkpoint: int | str | Path | None = None,
	*,
	workspace_root: str | Path | None = None,
) -> Path:
	"""Resolve a user-selected checkpoint path or fall back to the best checkpoint.

	Relative filenames are first resolved inside <target_folder>/checkpoints.
	If a workspace_root is provided and the relative path includes directories,
	it is resolved from the workspace root instead.
	"""
	inferred = infer_checkpoint_path_from_selection(
		target_folder,
		fold_idx,
		selected_checkpoint,
	)
	if inferred is None:
		return get_fold_checkpoint_path(target_folder, fold_idx)

	path_obj = inferred
	if path_obj.is_absolute():
		return path_obj

	if len(path_obj.parts) == 1:
		return get_checkpoints_dir(target_folder) / path_obj

	if workspace_root is not None:
		return Path(workspace_root) / path_obj

	return Path(target_folder) / path_obj


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
		stored_use_std = ckpt.get("target_standardization_enabled", None)
		target_mean = float(ckpt.get("target_mean", 0.0))
		target_std = float(ckpt.get("target_std", 1.0))
	else:
		# Legacy format: checkpoint is a plain state_dict (OrderedDict).
		state_dict = ckpt
		stored_use_std = None
		target_mean = 0.0
		target_std = 1.0

	model = build_model(model_class, config, feature_context)
	model.load_state_dict(state_dict)
	model.eval()

	if stored_use_std is None:
		use_std = get_target_standardization_config(config)["enabled"]
	else:
		# Prefer persisted training-time behavior for consistency at inference.
		use_std = bool(stored_use_std)
	target_standardizer = None
	if use_std:
		target_standardizer = build_target_standardizer_from_stats(target_mean, target_std)

	return model, target_standardizer
