from __future__ import annotations

from typing import Any

import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== target scaling ====================
# Optional per-fold target standardization. If enabled, we fit mean/std on the
# training fold only, train the model to predict standardized targets, and invert
# predictions back to the original target scale for metrics and reporting.


def get_target_standardization_config(config: dict[str, Any]) -> dict[str, Any]:
	target_cfg = config.get("training", {}).get("target_standardization", {})
	if not isinstance(target_cfg, dict):
		target_cfg = {}
	return {
		"enabled": bool(target_cfg.get("enabled", False)),
		"epsilon": float(target_cfg.get("epsilon", 1e-8)),
	}


def fit_target_standardizer(
	train_targets: np.ndarray,
	*,
	epsilon: float = 1e-8,
) -> tuple[float, float]:
	"""Fit (mean, std) on the training fold targets.

	Uses population std (ddof=0) and guards against near-zero std.
	"""
	mean = float(np.mean(train_targets))
	std = float(np.std(train_targets))
	if not np.isfinite(std) or std < epsilon:
		std = 1.0
	return mean, std


def standardize_targets_tensor(
	y: torch.Tensor,
	*,
	mean: torch.Tensor,
	std: torch.Tensor,
) -> torch.Tensor:
	return (y - mean) / std


def invert_standardization_tensor(
	y_scaled: torch.Tensor,
	*,
	mean: torch.Tensor,
	std: torch.Tensor,
) -> torch.Tensor:
	return y_scaled * std + mean


def build_target_standardizer(
	train_targets: np.ndarray,
	*,
	epsilon: float = 1e-8,
) -> dict[str, Any]:
	mean, std = fit_target_standardizer(train_targets, epsilon=epsilon)
	return build_target_standardizer_from_stats(mean, std)


def build_target_standardizer_from_stats(
	mean: float,
	std: float,
) -> dict[str, Any]:
	mean_tensor = torch.tensor(mean, dtype=torch.float32, device=device)
	std_tensor = torch.tensor(std, dtype=torch.float32, device=device)
	return {
		"mean": mean,
		"std": std,
		"mean_tensor": mean_tensor,
		"std_tensor": std_tensor,
	}


def standardize_batch_targets(
	y_true: torch.Tensor,
	*,
	target_standardizer: dict[str, Any] | None,
) -> torch.Tensor:
	if target_standardizer is None:
		return y_true
	return standardize_targets_tensor(
		y_true,
		mean=target_standardizer["mean_tensor"],
		std=target_standardizer["std_tensor"],
	)


def invert_standardized_predictions(
	y_pred: torch.Tensor,
	*,
	target_standardizer: dict[str, Any] | None,
) -> torch.Tensor:
	if target_standardizer is None:
		return y_pred
	return invert_standardization_tensor(
		y_pred,
		mean=target_standardizer["mean_tensor"],
		std=target_standardizer["std_tensor"],
	)
