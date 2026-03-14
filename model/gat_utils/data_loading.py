from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from torch_geometric.data import Data

from .config_helpers import as_dict, parse_float_in_range, parse_optional_positive_float
from .features import smiles_to_graph


# ==================== data loading ====================
def get_synthetic_cv_config(config: dict[str, Any]) -> dict[str, Any]:
	synthetic_cfg = as_dict(as_dict(config.get("data", {})).get("synthetic_cv", {}))
	keep_percentile = parse_float_in_range(
		synthetic_cfg.get("keep_percentile", 1.0),
		name="CONFIG['data']['synthetic_cv']['keep_percentile']",
		min_exclusive=0.0,
		max_inclusive=1.0,
	)
	row_keep_raw = synthetic_cfg.get("row_keep_fraction", 1.0)
	row_keep_fraction = (
		None
		if row_keep_raw is None
		else parse_float_in_range(
			row_keep_raw,
			name="CONFIG['data']['synthetic_cv']['row_keep_fraction']",
			min_exclusive=0.0,
			max_inclusive=1.0,
		)
	)

	max_train_synth_to_real_ratio = parse_optional_positive_float(
		synthetic_cfg.get("max_train_synth_to_real_ratio", None),
		name="CONFIG['data']['synthetic_cv']['max_train_synth_to_real_ratio']",
	)

	label_source = str(synthetic_cfg.get("label_source", "pred")).lower().strip()
	if label_source not in {"pred", "target"}:
		raise ValueError(
			"CONFIG['data']['synthetic_cv']['label_source'] must be 'pred' or 'target'."
		)
	train_selection = str(
		synthetic_cfg.get("train_selection", "matching_fold")
	).lower().strip()
	if train_selection not in {"matching_fold", "all", "none"}:
		raise ValueError(
			"CONFIG['data']['synthetic_cv']['train_selection'] must be one of "
			"{'matching_fold', 'all', 'none'}."
		)
	validation_selection = str(
		synthetic_cfg.get("validation_selection", "all_except_train_and_fold")
	).lower().strip()
	if validation_selection not in {
		"all_except_train_and_fold",
		"single_next_non_train",
		"none",
	}:
		raise ValueError(
			"CONFIG['data']['synthetic_cv']['validation_selection'] must be one of "
			"{'all_except_train_and_fold', 'single_next_non_train', 'none'}."
		)
	return {
		"include_in_training": bool(synthetic_cfg.get("include_in_training", True)),
		"include_in_validation": bool(synthetic_cfg.get("include_in_validation", False)),
		"keep_percentile": keep_percentile,
		"row_keep_fraction": row_keep_fraction,
		"max_train_synth_to_real_ratio": max_train_synth_to_real_ratio,
		"label_source": label_source,
		"train_selection": train_selection,
		"validation_selection": validation_selection,
	}


def get_synthetic_pretraining_config(config: dict[str, Any]) -> dict[str, Any]:
	pre_cfg = as_dict(as_dict(config.get("training", {})).get("synthetic_pretraining", {}))
	epochs = int(pre_cfg.get("epochs", 0))
	if epochs < 0:
		raise ValueError("CONFIG['training']['synthetic_pretraining']['epochs'] must be >= 0.")

	min_synthetic_graphs = int(pre_cfg.get("min_synthetic_graphs", 32))
	if min_synthetic_graphs < 1:
		raise ValueError(
			"CONFIG['training']['synthetic_pretraining']['min_synthetic_graphs'] must be >= 1."
		)

	return {
		"enabled": bool(pre_cfg.get("enabled", False)),
		"epochs": epochs,
		"learning_rate": pre_cfg.get("learning_rate", None),
		"weight_decay": pre_cfg.get("weight_decay", None),
		"grad_clip_norm": pre_cfg.get("grad_clip_norm", None),
		"batch_size": int(pre_cfg.get("batch_size", config["data"]["batch_size"])),
		"shuffle": bool(pre_cfg.get("shuffle", True)),
		"min_synthetic_graphs": min_synthetic_graphs,
		"use_target_standardization": bool(pre_cfg.get("use_target_standardization", False)),
	}


def _extract_synthetic_iteration(path_obj: Path) -> int | None:
	match = re.match(r"^synthetic_data_iteration_(\d+)\.csv$", path_obj.name)
	if not match:
		return None
	return int(match.group(1))


def _get_available_synthetic_files(dir_path: Path) -> dict[int, Path]:
	files: dict[int, Path] = {}
	for path_obj in sorted(dir_path.glob("synthetic_data_iteration_*.csv")):
		iter_idx = _extract_synthetic_iteration(path_obj)
		if iter_idx is None:
			continue
		files[iter_idx] = path_obj
	return files


def _select_synthetic_indices(
	*,
	selection: str,
	val_idx: int,
	available_synth: dict[int, Path],
	context_label: str,
) -> set[int]:
	if selection == "matching_fold":
		if val_idx in available_synth:
			return {val_idx}
		print(f"  No matching synthetic file found for {context_label} this fold.")
		return set()
	if selection == "all":
		return set(available_synth.keys())
	return set()


def get_fold_files(
	target_dir: str,
	val_idx: int,
	config: dict[str, Any],
	total_folds: int = 5,
):
	dir_path = Path(target_dir)
	val_file = dir_path / f"fold_{val_idx}.csv"

	policy = get_synthetic_cv_config(config)
	available_synth = _get_available_synthetic_files(dir_path)
	train_fold_indices = [i for i in range(total_folds) if i != val_idx]

	train_files = [dir_path / f"fold_{i}.csv" for i in train_fold_indices]
	val_files = [val_file]
	train_synth_indices: set[int] = set()

	if policy["include_in_training"]:
		train_synth_indices = _select_synthetic_indices(
			selection=policy["train_selection"],
			val_idx=val_idx,
			available_synth=available_synth,
			context_label="training",
		)
		if train_synth_indices:
			train_synth_files = [available_synth[i] for i in sorted(train_synth_indices)]
			train_files.extend(train_synth_files)
			print(
				"  Added training synthetic file(s): "
				+ ", ".join(path.name for path in train_synth_files)
			)
		else:
			print("  No synthetic files selected for training.")
	else:
		print("  Synthetic data excluded from training by config.")

	if policy["include_in_validation"]:
		validation_selection = policy["validation_selection"]
		val_synth_candidates = [
			i
			for i in sorted(available_synth.keys())
			if i != val_idx and i not in train_synth_indices
		]

		if validation_selection == "single_next_non_train" and val_synth_candidates:
			# Deterministic rotation for reproducibility.
			chosen_idx = val_synth_candidates[val_idx % len(val_synth_candidates)]
			val_synth_candidates = [chosen_idx]
		elif validation_selection == "none":
			val_synth_candidates = []

		if val_synth_candidates:
			val_synth_files = [available_synth[i] for i in val_synth_candidates]
			val_files.extend(val_synth_files)
			print(
				"  Added validation synthetic file(s) (no overlap with train/current fold): "
				+ ", ".join(path.name for path in val_synth_files)
			)
		else:
			print(
				"  No validation synthetic files selected "
				"(disjoint-set rule may have excluded all candidates)."
			)
	else:
		print("  Synthetic data excluded from validation by config.")

	return train_files, val_files


def get_synthetic_pretraining_files(
	target_dir: str,
	val_idx: int,
	config: dict[str, Any],
) -> list[Path]:
	"""Select synthetic CSV files for pretraining, independent of finetune train inclusion.

	This lets users pretrain on synthetic data even when
	CONFIG['data']['synthetic_cv']['include_in_training'] is False.
	"""
	pre_cfg = get_synthetic_pretraining_config(config)
	if not pre_cfg["enabled"]:
		return []

	policy = get_synthetic_cv_config(config)
	available_synth = _get_available_synthetic_files(Path(target_dir))
	if not available_synth:
		print("  No synthetic files found for pretraining.")
		return []

	selected_indices = _select_synthetic_indices(
		selection=policy["train_selection"],
		val_idx=val_idx,
		available_synth=available_synth,
		context_label="pretraining",
	)
	selected_files = [available_synth[i] for i in sorted(selected_indices)]
	if selected_files:
		print(
			"  Selected synthetic pretraining file(s): "
			+ ", ".join(path.name for path in selected_files)
		)
	else:
		print("  No synthetic files selected for pretraining.")

	return selected_files


def choose_target_column(df: pd.DataFrame, file_path: Path, config: dict[str, Any]) -> str:
	is_synthetic = file_path.name.startswith("synthetic_data_iteration_")

	if is_synthetic:
		synth_policy = get_synthetic_cv_config(config)
		label_source = synth_policy["label_source"]
		if label_source == "pred":
			preferred = [
				config["data"]["synthetic_target_column"],
				*config["data"]["fallback_target_columns"],
				config["data"]["real_target_column"],
			]
		else:
			preferred = [
				"target_pIC50",
				config["data"]["real_target_column"],
				*config["data"]["fallback_target_columns"],
				config["data"]["synthetic_target_column"],
			]
	else:
		preferred = [
			config["data"]["real_target_column"],
			*config["data"]["fallback_target_columns"],
		]

	for col in preferred:
		if col in df.columns:
			return col

	raise ValueError(
		f"No valid target column found in {file_path}. Tried: {preferred}"
	)


def _resolve_synthetic_filter_column(config: dict[str, Any]) -> str:
	synth_policy = get_synthetic_cv_config(config)
	if synth_policy["label_source"] == "pred":
		return config["data"]["synthetic_target_column"]
	return "target_pIC50"


def _filter_synthetic_rows(df: pd.DataFrame, path_obj: Path, config: dict[str, Any]) -> pd.DataFrame:
	synth_policy = get_synthetic_cv_config(config)
	keep_percentile = float(synth_policy["keep_percentile"])
	if keep_percentile >= 1.0:
		filtered = df
	else:
		filter_col = _resolve_synthetic_filter_column(config)
		if filter_col not in df.columns:
			raise ValueError(
				f"Synthetic filtering column '{filter_col}' is missing in {path_obj}."
			)

		series = pd.to_numeric(df[filter_col], errors="coerce")
		valid_mask = series.notna()
		valid_series = series[valid_mask]
		if valid_series.empty:
			raise ValueError(
				f"No numeric values available in synthetic filter column '{filter_col}' for {path_obj}."
			)

		tail = (1.0 - keep_percentile) / 2.0
		lower = float(valid_series.quantile(tail))
		upper = float(valid_series.quantile(1.0 - tail))

		keep_mask = valid_mask & series.ge(lower) & series.le(upper)
		filtered = df.loc[keep_mask].copy()

		kept_count = int(len(filtered))
		total_count = int(len(df))
		if kept_count == 0:
			raise ValueError(
				f"Synthetic percentile filtering removed all rows in {path_obj}. "
				f"keep_percentile={keep_percentile}"
			)

		kept_values = pd.to_numeric(filtered[filter_col], errors="coerce").dropna()
		kept_mean = float(kept_values.mean()) if not kept_values.empty else float("nan")
		kept_std = float(kept_values.std(ddof=0)) if not kept_values.empty else float("nan")

		print(
			f"  Synthetic filter [{path_obj.name}] by '{filter_col}': "
			f"kept {kept_count}/{total_count} rows "
			f"({kept_count / max(total_count, 1):.1%}) | "
			f"mean={kept_mean:.4f} std={kept_std:.4f}"
		)

	row_keep_fraction = synth_policy["row_keep_fraction"]
	if row_keep_fraction is not None and row_keep_fraction < 1.0:
		n_rows = len(filtered)
		keep_n = max(1, int(np.floor(n_rows * row_keep_fraction)))
		if keep_n < n_rows:
			# Deterministic row sampling to keep runs reproducible.
			filtered = filtered.sample(n=keep_n, random_state=42).copy()
			print(
				f"  Synthetic row subsample [{path_obj.name}]: "
				f"kept {keep_n}/{n_rows} rows ({row_keep_fraction:.1%})."
			)

	return filtered


def load_dataset(file_paths, config: dict[str, Any], feature_context: dict[str, Any]):
	if not isinstance(file_paths, list):
		file_paths = [file_paths]

	data_list = []
	for file_path in file_paths:
		path_obj = Path(file_path)
		df = pd.read_csv(path_obj)
		if "smiles" not in df.columns:
			raise ValueError(f"Missing required 'smiles' column in {path_obj}")

		is_synthetic = path_obj.name.startswith("synthetic_data_iteration_")
		if is_synthetic:
			df = _filter_synthetic_rows(df, path_obj, config)

		target_col = choose_target_column(df, path_obj, config)

		for _, row in df.iterrows():
			smiles = row["smiles"]
			target = row[target_col]

			if pd.isna(smiles) or pd.isna(target):
				continue

			graph = smiles_to_graph(str(smiles), float(target), config, feature_context)
			if graph is not None:
				graph.is_synthetic = bool(is_synthetic)
				data_list.append(graph)

	return data_list


def get_targets_from_graphs(data_list):
	return np.array([float(data.y.item()) for data in data_list], dtype=np.float32)


def baseline_rmse(train_targets: np.ndarray, val_targets: np.ndarray) -> float:
	train_mean = float(np.mean(train_targets))
	preds = np.full_like(val_targets, train_mean, dtype=np.float32)
	return float(np.sqrt(mean_squared_error(val_targets, preds)))


def print_fold_data_summary(val_idx: int, train_data, val_data) -> None:
	train_targets = get_targets_from_graphs(train_data)
	val_targets = get_targets_from_graphs(val_data)
	fold_baseline_rmse = baseline_rmse(train_targets, val_targets)
	real_train, synthetic_train = split_real_and_synthetic(train_data)

	summary = (
		f"[Fold {val_idx}] train_n={len(train_data)} val_n={len(val_data)} "
		f"train_mean={np.mean(train_targets):.4f} val_mean={np.mean(val_targets):.4f} "
		f"baseline_rmse={fold_baseline_rmse:.4f}"
	)
	if len(synthetic_train) > 0:
		summary += (
			f" | train_original_n={len(real_train)} "
			f"train_synthetic_n={len(synthetic_train)}"
		)
	print(summary)


def split_real_and_synthetic(data_list: list[Data]) -> tuple[list[Data], list[Data]]:
	real_data = []
	synthetic_data = []
	for graph in data_list:
		if bool(getattr(graph, "is_synthetic", False)):
			synthetic_data.append(graph)
		else:
			real_data.append(graph)
	return real_data, synthetic_data


def cap_synthetic_train_ratio(
	train_data: list[Data],
	*,
	max_ratio: float | None,
	seed: int,
	fold_idx: int,
) -> list[Data]:
	if max_ratio is None:
		return train_data

	real_data, synthetic_data = split_real_and_synthetic(train_data)
	if not real_data or not synthetic_data:
		return train_data

	max_synth = int(np.floor(len(real_data) * max_ratio))
	if len(synthetic_data) <= max_synth:
		print(
			f"[Fold {fold_idx}] synthetic:real cap inactive "
			f"(ratio={len(synthetic_data) / max(len(real_data), 1):.3f} <= {max_ratio:.3f})."
		)
		return train_data

	rng = np.random.default_rng(seed + fold_idx)
	selected_indices = rng.choice(len(synthetic_data), size=max_synth, replace=False)
	selected_synth = [synthetic_data[int(i)] for i in selected_indices]

	rebalanced = real_data + selected_synth
	rng.shuffle(rebalanced)
	print(
		f"[Fold {fold_idx}] applied synthetic:real cap={max_ratio:.3f} | "
		f"real={len(real_data)} synthetic={len(selected_synth)} "
		f"(was {len(synthetic_data)} synthetic)."
	)
	return rebalanced
