from __future__ import annotations

from typing import Any


MINIMIZE_METRICS = {"mse", "rmse", "mae"}
MAXIMIZE_METRICS = {"r2", "rho", "pearson"}
ALL_METRICS = MINIMIZE_METRICS | MAXIMIZE_METRICS


def as_dict(value: Any) -> dict[str, Any]:
	if isinstance(value, dict):
		return value
	return {}


def parse_float_in_range(
	value: Any,
	*,
	name: str,
	min_exclusive: float,
	max_inclusive: float,
) -> float:
	parsed = float(value)
	if parsed <= min_exclusive or parsed > max_inclusive:
		raise ValueError(f"{name} must be in ({min_exclusive}, {max_inclusive}].")
	return parsed


def parse_optional_positive_float(value: Any, *, name: str) -> float | None:
	if value is None:
		return None
	parsed = float(value)
	if parsed <= 0.0:
		raise ValueError(f"{name} must be > 0 when provided.")
	return parsed


def parse_non_negative_float(value: Any, *, name: str) -> float:
	parsed = float(value)
	if parsed < 0.0:
		raise ValueError(f"{name} must be >= 0.")
	return parsed


def parse_bool(value: Any, *, default: bool) -> bool:
	if value is None:
		return bool(default)
	return bool(value)


def parse_metric_name(metric_name: Any, *, field_name: str) -> str:
	name = str(metric_name).lower().strip()
	if name not in ALL_METRICS:
		raise ValueError(
			f"{field_name} must be one of {sorted(ALL_METRICS)}."
		)
	return name


def parse_target_folders(target_folder_value: Any) -> list[str]:
	field_name = "CONFIG['experiment']['target_folder']"
	if isinstance(target_folder_value, str):
		folder = target_folder_value.strip()
		if not folder:
			raise ValueError(f"{field_name} must not be empty.")
		return [folder]

	if isinstance(target_folder_value, (list, tuple)):
		folders: list[str] = []
		for idx, item in enumerate(target_folder_value):
			if not isinstance(item, str):
				raise ValueError(f"{field_name}[{idx}] must be a string path.")
			folder = item.strip()
			if not folder:
				raise ValueError(f"{field_name}[{idx}] must not be empty.")
			folders.append(folder)

		if not folders:
			raise ValueError(f"{field_name} list must not be empty.")

		# Preserve user order but remove duplicates to avoid accidental re-runs.
		unique_folders: list[str] = []
		seen: set[str] = set()
		for folder in folders:
			if folder in seen:
				continue
			seen.add(folder)
			unique_folders.append(folder)
		return unique_folders

	raise ValueError(f"{field_name} must be a string path or a list of string paths.")


def is_minimize_metric(metric_name: str) -> bool:
	return str(metric_name).lower().strip() in MINIMIZE_METRICS


def get_scheduler_config(config: dict[str, Any]) -> dict[str, Any]:
	scheduler_cfg = as_dict(config.get("scheduler", {}))
	enabled = bool(scheduler_cfg.get("enabled", False))
	monitor_metric = parse_metric_name(
		scheduler_cfg.get("monitor_metric", "rmse"),
		field_name="CONFIG['scheduler']['monitor_metric']",
	)
	mode = str(scheduler_cfg.get("mode", "min")).lower().strip()
	if mode not in {"min", "max"}:
		raise ValueError("CONFIG['scheduler']['mode'] must be 'min' or 'max'.")

	return {
		"enabled": enabled,
		"mode": mode,
		"factor": float(scheduler_cfg.get("factor", 0.75)),
		"patience": int(scheduler_cfg.get("patience", 3)),
		"monitor_metric": monitor_metric,
	}


def get_early_stopping_config(config: dict[str, Any]) -> dict[str, Any]:
	training_cfg = as_dict(config.get("training", {}))
	early_stop_cfg = as_dict(training_cfg.get("early_stopping", {}))
	min_delta = parse_non_negative_float(
		early_stop_cfg.get("min_delta", 0.0),
		name="CONFIG['training']['early_stopping']['min_delta']",
	)
	minimum_improvement = parse_non_negative_float(
		early_stop_cfg.get("minimum_improvement", 0.0),
		name="CONFIG['training']['early_stopping']['minimum_improvement']",
	)
	return {
		"enabled": bool(early_stop_cfg.get("enabled", False)),
		"monitor_metric": parse_metric_name(
			early_stop_cfg.get("monitor_metric", "rmse"),
			field_name="CONFIG['training']['early_stopping']['monitor_metric']",
		),
		"patience": int(early_stop_cfg.get("patience", 15)),
		"min_delta": min_delta,
		"minimum_improvement": minimum_improvement,
	}


def get_checkpointing_config(config: dict[str, Any]) -> dict[str, Any]:
	training_cfg = as_dict(config.get("training", {}))
	checkpoint_cfg = as_dict(training_cfg.get("checkpointing", {}))
	save_every_n_epochs = int(checkpoint_cfg.get("save_every_n_epochs", 0) or 0)
	if save_every_n_epochs < 0:
		raise ValueError(
			"CONFIG['training']['checkpointing']['save_every_n_epochs'] must be >= 0."
		)
	return {"save_every_n_epochs": save_every_n_epochs}


def get_optimization_config(config: dict[str, Any]) -> dict[str, Any]:
	opt_cfg = as_dict(config.get("optimization", {}))
	return {
		"learning_rate": float(opt_cfg.get("learning_rate", 1e-3)),
		"weight_decay": float(opt_cfg.get("weight_decay", 0.0)),
		"grad_clip_norm": float(opt_cfg.get("grad_clip_norm", 0.0)),
	}


def get_experiment_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
	exp_cfg = as_dict(config.get("experiment", {}))
	data_cfg = as_dict(config.get("data", {}))
	training_cfg = as_dict(config.get("training", {}))
	target_folders = parse_target_folders(exp_cfg.get("target_folder", "."))
	return {
		"target_folder": target_folders[0],
		"target_folders": target_folders,
		"actual_test_file": str(exp_cfg.get("actual_test_file", "heldout_testset.csv")),
		"total_folds": int(exp_cfg.get("total_folds", 5)),
		"batch_size": int(data_cfg.get("batch_size", 64)),
		"num_epochs": int(training_cfg.get("num_epochs", 200)),
		"print_every": int(training_cfg.get("print_every", 1)),
	}
