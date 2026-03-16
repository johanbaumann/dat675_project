from __future__ import annotations

import copy
import gc
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from gat_predictor import CONFIG, GATModel
from gat_utils.config_helpers import (
	get_early_stopping_config,
	get_experiment_runtime_config,
	get_optimization_config,
	get_scheduler_config,
	is_minimize_metric,
	parse_target_folders,
)
from gat_utils.data_loading import load_dataset
from gat_utils.features import apply_feature_scalers, prepare_feature_config, set_seed
from gat_utils.fold_orchestration import run_single_fold
from gat_utils.pipeline import run_synthetic_pretraining_stage
from gat_utils.target_scaling import (
	build_target_standardizer_from_stats,
	get_target_standardization_config,
)
from gat_utils.training_helpers import build_model, predict


WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = WORKSPACE_ROOT / "gat_lr_sweep_holdout_results.csv"


SWEEP_CONFIG = {
	"datasets": {
		# One folder string or multiple folder strings are both supported.
		# Examples: ["67%"] or ["0%", "33%", "67%"]
		"target_folders": ["0%", "67%"],
		# Set to None to use CONFIG["experiment"]["actual_test_file"].
		"holdout_test_file": None,
	},
	"lrs": {
		"main": [3e-4, 5e-4, 7e-4, 1e-3],
		"pretrain": [2e-4, 4e-4, 1e-4],
	},
	"output": {
		# Relative paths are resolved from WORKSPACE_ROOT.
		"csv": str(DEFAULT_OUTPUT.name),
	},
}


def _resolve_output_path(output_value: str) -> Path:
	path = Path(output_value)
	if path.is_absolute():
		return path
	return WORKSPACE_ROOT / path


def _load_holdout_group_keys(test_path: Path, config: dict[str, Any]) -> pd.DataFrame | None:
	real_target_col = config["data"]["real_target_column"]
	try:
		df = pd.read_csv(test_path)
	except Exception:
		return None

	needed = ["smiles", real_target_col]
	if not all(col in df.columns for col in needed):
		return None

	df = df[df["smiles"].notna() & df[real_target_col].notna()].reset_index(drop=True)
	return df[["smiles"]].copy()


def _molecule_mean_rmse(
	*,
	y_true: list[float],
	y_pred: list[float],
	holdout_keys: pd.DataFrame | None,
) -> float:
	pred_df = pd.DataFrame({
		"True_pIC50": y_true,
		"Pred_pIC50": y_pred,
	})
	if holdout_keys is not None and len(holdout_keys) == len(pred_df):
		pred_df.insert(0, "smiles", holdout_keys["smiles"].to_list())
		molecule_df = (
			pred_df.groupby("smiles", as_index=False)
			.agg(True_pIC50=("True_pIC50", "mean"), Pred_pIC50=("Pred_pIC50", "mean"))
		)
	else:
		molecule_df = pred_df

	error = molecule_df["True_pIC50"].to_numpy(dtype=np.float64) - molecule_df["Pred_pIC50"].to_numpy(dtype=np.float64)
	return float(np.sqrt(np.mean(np.square(error))))


def _evaluate_best_fold_on_holdout(
	*,
	config: dict[str, Any],
	feature_context: dict[str, Any],
	raw_holdout_data: list,
	holdout_keys: pd.DataFrame | None,
	fold_output: dict[str, Any],
	use_target_standardization: bool,
) -> float:
	best_model = build_model(GATModel, config, feature_context)
	best_model.load_state_dict(fold_output["best_model_weights"])

	holdout_data = copy.deepcopy(raw_holdout_data)
	apply_feature_scalers(holdout_data, feature_scalers=fold_output["feature_scalers"])
	holdout_loader = DataLoader(
		holdout_data,
		batch_size=config["data"]["batch_size"],
		shuffle=False,
	)

	target_standardizer = None
	if use_target_standardization:
		target_standardizer = build_target_standardizer_from_stats(
			float(fold_output["train_target_mean"]),
			float(fold_output["train_target_std"]),
		)

	y_true, y_pred = predict(
		best_model,
		holdout_loader,
		target_standardizer=target_standardizer,
	)
	rmse = _molecule_mean_rmse(
		y_true=y_true,
		y_pred=y_pred,
		holdout_keys=holdout_keys,
	)

	del best_model
	del holdout_data
	del holdout_loader
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	gc.collect()
	return rmse


def _run_single_dataset_sweep(
	*,
	base_config: dict[str, Any],
	target_folder: str,
	test_path: Path,
) -> dict[str, Any]:
	dataset_config = copy.deepcopy(base_config)
	dataset_config["experiment"]["target_folder"] = target_folder

	set_seed(dataset_config["experiment"]["seed"])
	feature_context = prepare_feature_config(dataset_config)
	target_std_cfg = get_target_standardization_config(dataset_config)
	use_target_standardization = target_std_cfg["enabled"]

	runtime_cfg = get_experiment_runtime_config(dataset_config)
	optimization_cfg = get_optimization_config(dataset_config)
	early_stop_cfg = get_early_stopping_config(dataset_config)
	scheduler_cfg = get_scheduler_config(dataset_config)
	if scheduler_cfg["enabled"]:
		expected_mode = "min" if is_minimize_metric(scheduler_cfg["monitor_metric"]) else "max"
		if scheduler_cfg["mode"] != expected_mode:
			scheduler_cfg = {**scheduler_cfg, "mode": expected_mode}

	raw_holdout_data = load_dataset(str(test_path), dataset_config, feature_context)
	if len(raw_holdout_data) == 0:
		raise RuntimeError(
			f"Holdout test set produced 0 valid graphs for dataset '{target_folder}'."
		)

	holdout_keys = _load_holdout_group_keys(test_path, dataset_config)
	fold_rmses: list[float] = []
	for fold_idx in range(runtime_cfg["total_folds"]):
		fold_output = run_single_fold(
			val_idx=fold_idx,
			config=dataset_config,
			feature_context=feature_context,
			target_folder=target_folder,
			total_folds=runtime_cfg["total_folds"],
			batch_size=runtime_cfg["batch_size"],
			num_epochs=runtime_cfg["num_epochs"],
			print_every=runtime_cfg["print_every"],
			learning_rate=optimization_cfg["learning_rate"],
			weight_decay=optimization_cfg["weight_decay"],
			grad_clip_norm=optimization_cfg["grad_clip_norm"],
			early_stop_cfg=early_stop_cfg,
			scheduler_cfg=scheduler_cfg,
			save_every_n_epochs=0,
			target_std_cfg=target_std_cfg,
			use_target_standardization=use_target_standardization,
			dataset_label=Path(target_folder).name,
			run_synthetic_pretraining_stage=run_synthetic_pretraining_stage,
			build_model=build_model,
			model_class=GATModel,
			save_checkpoints=False,
		)
		fold_rmse = _evaluate_best_fold_on_holdout(
			config=dataset_config,
			feature_context=feature_context,
			raw_holdout_data=raw_holdout_data,
			holdout_keys=holdout_keys,
			fold_output=fold_output,
			use_target_standardization=use_target_standardization,
		)
		fold_rmses.append(fold_rmse)
		print(
			f"[Sweep][{Path(target_folder).name}][Fold {fold_idx}] "
			f"holdout molecule RMSE={fold_rmse:.4f}"
		)

	row: dict[str, Any] = {
		"Dataset": Path(target_folder).name,
		"FoldCount": len(fold_rmses),
		"Holdout_Molecule_RMSE_Mean": float(np.mean(fold_rmses)),
		"Holdout_Molecule_RMSE_Std": float(np.std(fold_rmses, ddof=0)),
		"Min_Fold_Holdout_Molecule_RMSE": float(np.min(fold_rmses)),
	}
	for fold_idx, fold_rmse in enumerate(fold_rmses):
		row[f"Fold_{fold_idx}_Holdout_Molecule_RMSE"] = float(fold_rmse)
	return row


def main() -> None:
	output_path = _resolve_output_path(str(SWEEP_CONFIG["output"]["csv"]))

	base_runtime_cfg = get_experiment_runtime_config(CONFIG)
	datasets = parse_target_folders(
		SWEEP_CONFIG["datasets"].get("target_folders", base_runtime_cfg["target_folders"])
	)
	holdout_file = SWEEP_CONFIG["datasets"].get("holdout_test_file", None)
	test_path = (
		Path(str(holdout_file))
		if holdout_file
		else WORKSPACE_ROOT / base_runtime_cfg["actual_test_file"]
	)
	if not test_path.is_absolute():
		test_path = WORKSPACE_ROOT / test_path
	if not test_path.exists():
		raise FileNotFoundError(f"Holdout test file not found: {test_path}")

	rows: list[dict[str, Any]] = []
	main_lrs = [float(v) for v in SWEEP_CONFIG["lrs"]["main"]]
	pretrain_lrs = [float(v) for v in SWEEP_CONFIG["lrs"]["pretrain"]]
	grid = list(product(pretrain_lrs, main_lrs))
	print(
		"Running LR sweep on holdout only for combinations: "
		+ ", ".join(
			f"(pretrain={pre_lr:.2e}, main={main_lr:.2e})" for pre_lr, main_lr in grid
		)
	)

	for combo_idx, (pretrain_lr, main_lr) in enumerate(grid, start=1):
		print(
			"\n============================================================\n"
			f"Sweep combo {combo_idx}/{len(grid)} | pretrain_lr={pretrain_lr:.2e} | main_lr={main_lr:.2e}\n"
			"============================================================"
		)
		combo_config = copy.deepcopy(CONFIG)
		combo_config["optimization"]["learning_rate"] = float(main_lr)
		combo_config["training"]["synthetic_pretraining"]["learning_rate"] = float(pretrain_lr)
		combo_config.setdefault("training", {}).setdefault("checkpointing", {})["save_every_n_epochs"] = 0

		combo_dataset_means: list[float] = []
		for dataset in datasets:
			print(f"\n--- Dataset {Path(dataset).name} ---")
			dataset_row = _run_single_dataset_sweep(
				base_config=combo_config,
				target_folder=dataset,
				test_path=test_path,
			)
			dataset_row["Pretrain_LR"] = float(pretrain_lr)
			dataset_row["Main_LR"] = float(main_lr)
			rows.append(dataset_row)
			combo_dataset_means.append(float(dataset_row["Holdout_Molecule_RMSE_Mean"]))

		rows.append(
			{
				"Pretrain_LR": float(pretrain_lr),
				"Main_LR": float(main_lr),
				"Dataset": "ALL_DATASETS",
				"FoldCount": len(datasets) * int(base_runtime_cfg["total_folds"]),
				"Holdout_Molecule_RMSE_Mean": float(np.mean(combo_dataset_means)),
				"Holdout_Molecule_RMSE_Std": float(np.std(combo_dataset_means, ddof=0)),
				"Min_Fold_Holdout_Molecule_RMSE": float(np.min(combo_dataset_means)),
			}
		)

	results_df = pd.DataFrame(rows)
	results_df = results_df.sort_values(
		by=["Holdout_Molecule_RMSE_Mean", "Dataset", "Pretrain_LR", "Main_LR"],
		ascending=[True, True, True, True],
	).reset_index(drop=True)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	results_df.to_csv(output_path, index=False)
	print(f"\nSaved sweep results to: {output_path}")
	print(results_df.to_string(index=False))


if __name__ == "__main__":
	main()