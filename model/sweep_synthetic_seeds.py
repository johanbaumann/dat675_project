from __future__ import annotations

import copy
import gc
import time
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
	infer_dataset_percent_label,
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
from gat_utils.training_helpers import build_model, evaluate


WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = WORKSPACE_ROOT / "gat_synthetic_sweep.csv"


SWEEP_CONFIG = {
	"datasets": {
		# One folder string or multiple folder strings are both supported.
		# Examples: ["67%"] or ["0%", "33%", "67%"]
		"target_folders": [
			"../data/combination_1300_molecules_and_0_%_synthetic",
			"../data/combination_1950_molecules_and_33_%_synthetic",
			"../data/combination_3900_molecules_and_67_%_synthetic",
		],
		# Set to None to use CONFIG["experiment"]["actual_test_file"].
		"holdout_test_file": None,
	},
	"conditions": [
		# Synthetic data used only during stage-1 pretraining.
		{
			"name": "pretrain_only",
			"use_synth_pretraining": True,
			"use_synth_finetuning": False,
		},
		# Synthetic data used only in finetuning train split (no synthetic pretraining).
		{
			"name": "finetune_only",
			"use_synth_pretraining": False,
			"use_synth_finetuning": True,
		},
	],
	"seeds": [7, 13, 23, 42, 123],
	"output": {
		# Relative paths are resolved from WORKSPACE_ROOT.
		"csv": str(DEFAULT_OUTPUT.name),
	},
	"runtime": {
		# Keep False for sweep runs to avoid overwriting best_model_fold_*.pth.
		"save_checkpoints": False,
	},
}


def _resolve_output_path(output_value: str) -> Path:
	path = Path(output_value)
	if path.is_absolute():
		return path
	return WORKSPACE_ROOT / path


def _safe_mean(values: list[float]) -> float:
	if not values:
		return float("nan")
	return float(np.mean(np.asarray(values, dtype=np.float64)))


def _safe_std(values: list[float], ddof: int = 1) -> float:
	if not values:
		return float("nan")
	arr = np.asarray(values, dtype=np.float64)
	if arr.size <= ddof:
		return 0.0
	return float(np.std(arr, ddof=ddof))


def _apply_synthetic_mode(
	config: dict[str, Any],
	*,
	use_synth_pretraining: bool,
	use_synth_finetuning: bool,
) -> None:
	data_cfg = config.setdefault("data", {})
	synth_cfg = data_cfg.setdefault("synthetic_cv", {})
	training_cfg = config.setdefault("training", {})
	pre_cfg = training_cfg.setdefault("synthetic_pretraining", {})

	pre_cfg["enabled"] = bool(use_synth_pretraining)
	synth_cfg["include_in_training"] = bool(use_synth_finetuning)

	# Keep validation synthetic disabled during this comparison unless already enabled.
	# This avoids mixing train/val policies while sweeping seeds.
	synth_cfg.setdefault("include_in_validation", False)


def _build_runtime_components(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], bool]:
	runtime_cfg = get_experiment_runtime_config(config)
	optimization_cfg = get_optimization_config(config)
	early_stop_cfg = get_early_stopping_config(config)
	scheduler_cfg = get_scheduler_config(config)
	target_std_cfg = get_target_standardization_config(config)
	use_target_standardization = target_std_cfg["enabled"]

	if scheduler_cfg["enabled"]:
		expected_mode = "min" if is_minimize_metric(scheduler_cfg["monitor_metric"]) else "max"
		if scheduler_cfg["mode"] != expected_mode:
			scheduler_cfg = {**scheduler_cfg, "mode": expected_mode}

	return (
		runtime_cfg,
		optimization_cfg,
		early_stop_cfg,
		scheduler_cfg,
		target_std_cfg,
		config,
		use_target_standardization,
	)


def _evaluate_fold_on_holdout(
	*,
	config: dict[str, Any],
	feature_context: dict[str, Any],
	raw_holdout_data: list,
	batch_size: int,
	fold_output: dict[str, Any],
	use_target_standardization: bool,
) -> dict[str, float]:
	model = build_model(GATModel, config, feature_context)
	model.load_state_dict(fold_output["best_model_weights"])

	holdout_data = copy.deepcopy(raw_holdout_data)
	apply_feature_scalers(holdout_data, feature_scalers=fold_output["feature_scalers"])
	loader = DataLoader(
		holdout_data,
		batch_size=batch_size,
		shuffle=False,
	)

	target_standardizer = None
	if use_target_standardization:
		target_standardizer = build_target_standardizer_from_stats(
			float(fold_output["train_target_mean"]),
			float(fold_output["train_target_std"]),
		)

	mse, rmse, mae, r2, rho, pearson = evaluate(
		model,
		loader,
		target_standardizer=target_standardizer,
	)

	del model
	del holdout_data
	del loader
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	gc.collect()

	return {
		"Holdout_MSE": float(mse),
		"Holdout_RMSE": float(rmse),
		"Holdout_MAE": float(mae),
		"Holdout_R2": float(r2),
		"Holdout_Rho": float(rho),
		"Holdout_Pearson": float(pearson),
	}


def _run_condition_seed_dataset(
	*,
	base_config: dict[str, Any],
	target_folder: str,
	artifact_folder: str,
	test_path: Path,
	condition: dict[str, Any],
	seed: int,
	save_checkpoints: bool,
) -> list[dict[str, Any]]:
	config = copy.deepcopy(base_config)
	config["experiment"]["target_folder"] = target_folder
	config["experiment"]["seed"] = int(seed)
	config.setdefault("training", {}).setdefault("checkpointing", {})["save_every_n_epochs"] = 0

	_apply_synthetic_mode(
		config,
		use_synth_pretraining=bool(condition["use_synth_pretraining"]),
		use_synth_finetuning=bool(condition["use_synth_finetuning"]),
	)

	set_seed(int(seed))
	feature_context = prepare_feature_config(config)
	(
		runtime_cfg,
		optimization_cfg,
		early_stop_cfg,
		scheduler_cfg,
		target_std_cfg,
		_,
		use_target_standardization,
	) = _build_runtime_components(config)

	raw_holdout_data = load_dataset(str(test_path), config, feature_context)
	if len(raw_holdout_data) == 0:
		raise RuntimeError(
			f"Holdout test set produced 0 valid graphs for dataset '{target_folder}'."
		)

	dataset_label = Path(target_folder).name
	rows: list[dict[str, Any]] = []
	for fold_idx in range(runtime_cfg["total_folds"]):
		fold_output = run_single_fold(
			val_idx=fold_idx,
			config=config,
			feature_context=feature_context,
			data_folder=target_folder,
			artifact_folder=artifact_folder,
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
			dataset_label=dataset_label,
			run_synthetic_pretraining_stage=run_synthetic_pretraining_stage,
			build_model=build_model,
			model_class=GATModel,
			save_checkpoints=save_checkpoints,
		)

		holdout_metrics = _evaluate_fold_on_holdout(
			config=config,
			feature_context=feature_context,
			raw_holdout_data=raw_holdout_data,
			batch_size=runtime_cfg["batch_size"],
			fold_output=fold_output,
			use_target_standardization=use_target_standardization,
		)

		val_metrics = fold_output["cv_result"]
		row = {
			"Level": "fold",
			"Condition": str(condition["name"]),
			"Dataset": dataset_label,
			"Seed": int(seed),
			"Fold": int(fold_idx),
			"SyntheticPretrainingEnabled": bool(condition["use_synth_pretraining"]),
			"SyntheticFinetuningEnabled": bool(condition["use_synth_finetuning"]),
			"Val_RMSE": float(val_metrics["val_rmse"]),
			"Val_MAE": float(val_metrics["val_mae"]),
			"Val_R2": float(val_metrics["val_r2"]),
			"Val_Rho": float(val_metrics["val_rho"]),
			"Val_Pearson": float(val_metrics["val_pearson"]),
		}
		row.update(holdout_metrics)
		rows.append(row)

		print(
			f"[SeedSweep][{condition['name']}][seed={seed}][{dataset_label}][fold={fold_idx}] "
			f"holdout_rmse={holdout_metrics['Holdout_RMSE']:.4f}"
		)

	return rows


def _build_seed_dataset_summaries(fold_df: pd.DataFrame) -> pd.DataFrame:
	metric_cols = [
		"Val_RMSE",
		"Val_MAE",
		"Val_R2",
		"Val_Rho",
		"Val_Pearson",
		"Holdout_MSE",
		"Holdout_RMSE",
		"Holdout_MAE",
		"Holdout_R2",
		"Holdout_Rho",
		"Holdout_Pearson",
	]

	rows: list[dict[str, Any]] = []
	for keys, group in fold_df.groupby(["Condition", "Dataset", "Seed"], sort=True):
		condition, dataset, seed = keys
		row: dict[str, Any] = {
			"Level": "seed_dataset_summary",
			"Condition": condition,
			"Dataset": dataset,
			"Seed": int(seed),
			"Fold": "",
			"SyntheticPretrainingEnabled": bool(group["SyntheticPretrainingEnabled"].iloc[0]),
			"SyntheticFinetuningEnabled": bool(group["SyntheticFinetuningEnabled"].iloc[0]),
			"FoldCount": int(len(group)),
		}
		for col in metric_cols:
			values = group[col].astype(float).to_list()
			row[f"{col}_Mean"] = _safe_mean(values)
			row[f"{col}_Std"] = _safe_std(values, ddof=1)
		rows.append(row)

	return pd.DataFrame(rows)


def _build_condition_dataset_summaries(seed_summary_df: pd.DataFrame) -> pd.DataFrame:
	metric_mean_cols = [
		"Val_RMSE_Mean",
		"Val_MAE_Mean",
		"Val_R2_Mean",
		"Val_Rho_Mean",
		"Val_Pearson_Mean",
		"Holdout_MSE_Mean",
		"Holdout_RMSE_Mean",
		"Holdout_MAE_Mean",
		"Holdout_R2_Mean",
		"Holdout_Rho_Mean",
		"Holdout_Pearson_Mean",
	]

	rows: list[dict[str, Any]] = []
	for keys, group in seed_summary_df.groupby(["Condition", "Dataset"], sort=True):
		condition, dataset = keys
		row: dict[str, Any] = {
			"Level": "condition_dataset_summary",
			"Condition": condition,
			"Dataset": dataset,
			"Seed": "",
			"Fold": "",
			"SyntheticPretrainingEnabled": bool(group["SyntheticPretrainingEnabled"].iloc[0]),
			"SyntheticFinetuningEnabled": bool(group["SyntheticFinetuningEnabled"].iloc[0]),
			"SeedCount": int(group["Seed"].nunique()),
		}
		for col in metric_mean_cols:
			values = group[col].astype(float).to_list()
			row[f"AcrossSeeds_{col}_Mean"] = _safe_mean(values)
			row[f"AcrossSeeds_{col}_Std"] = _safe_std(values, ddof=1)
		rows.append(row)

	return pd.DataFrame(rows)


def _build_condition_overall_summaries(seed_summary_df: pd.DataFrame) -> pd.DataFrame:
	rows: list[dict[str, Any]] = []
	for condition, group in seed_summary_df.groupby("Condition", sort=True):
		values = group["Holdout_RMSE_Mean"].astype(float).to_list()
		rows.append(
			{
				"Level": "condition_overall_summary",
				"Condition": condition,
				"Dataset": "ALL_DATASETS",
				"Seed": "",
				"Fold": "",
				"SyntheticPretrainingEnabled": bool(group["SyntheticPretrainingEnabled"].iloc[0]),
				"SyntheticFinetuningEnabled": bool(group["SyntheticFinetuningEnabled"].iloc[0]),
				"Entries": int(len(values)),
				"AcrossSeedsAndDatasets_Holdout_RMSE_Mean": _safe_mean(values),
				"AcrossSeedsAndDatasets_Holdout_RMSE_Std": _safe_std(values, ddof=1),
			}
		)
	return pd.DataFrame(rows)


def _build_compact_dataset_ranking(fold_df: pd.DataFrame) -> pd.DataFrame:
	"""Compact inter-dataset ranking using heldout fold-level metrics.

	Rows are aggregated over all fold evaluations (and all seeds) for each
	(Condition, Dataset) pair, then ranked by lowest mean holdout RMSE.
	"""
	heldout_cols = [
		"Holdout_MSE",
		"Holdout_RMSE",
		"Holdout_MAE",
		"Holdout_R2",
		"Holdout_Rho",
		"Holdout_Pearson",
	]

	agg_map: dict[str, list[str]] = {col: ["mean", "std"] for col in heldout_cols}
	agg_map["Fold"] = ["count"]
	ranking = (
		fold_df
		.groupby(["Condition", "Dataset"], sort=True)
		.agg(agg_map)
	)

	# Flatten MultiIndex columns into compact names like Holdout_RMSE_Mean.
	ranking.columns = [
		(f"{col}_{stat.title()}" if col != "Fold" else "N_Fold_Evals")
		for col, stat in ranking.columns
	]
	ranking = ranking.reset_index()

	# Keep std fields stable when there is only one value.
	std_cols = [c for c in ranking.columns if c.endswith("_Std")]
	if std_cols:
		ranking[std_cols] = ranking[std_cols].fillna(0.0)

	ranking["N_Seeds"] = (
		fold_df.groupby(["Condition", "Dataset"], sort=True)["Seed"]
		.nunique()
		.reset_index(drop=True)
	)

	# Rank per condition for direct inter-dataset comparison.
	ranking["Rank_In_Condition_By_RMSE"] = (
		ranking
		.groupby("Condition", sort=True)["Holdout_RMSE_Mean"]
		.rank(method="dense", ascending=True)
		.astype(int)
	)

	# Global rank across all condition+dataset entries.
	ranking = ranking.sort_values(
		by=["Holdout_RMSE_Mean", "Holdout_RMSE_Std", "Condition", "Dataset"],
		ascending=[True, True, True, True],
	).reset_index(drop=True)
	ranking["Global_Rank_By_RMSE"] = np.arange(1, len(ranking) + 1)

	# Make column order easy to scan.
	ordered_cols = [
		"Global_Rank_By_RMSE",
		"Rank_In_Condition_By_RMSE",
		"Condition",
		"Dataset",
		"N_Seeds",
		"N_Fold_Evals",
		"Holdout_RMSE_Mean",
		"Holdout_RMSE_Std",
		"Holdout_MAE_Mean",
		"Holdout_MAE_Std",
		"Holdout_R2_Mean",
		"Holdout_R2_Std",
		"Holdout_Rho_Mean",
		"Holdout_Rho_Std",
		"Holdout_Pearson_Mean",
		"Holdout_Pearson_Std",
		"Holdout_MSE_Mean",
		"Holdout_MSE_Std",
	]
	return ranking[ordered_cols]


def _build_condition_winner_summary(ranking_df: pd.DataFrame) -> pd.DataFrame:
	"""Pick one winning dataset per condition using lowest holdout RMSE mean."""
	if ranking_df.empty:
		return pd.DataFrame()

	winners = (
		ranking_df
		.sort_values(
			by=[
				"Condition",
				"Holdout_RMSE_Mean",
				"Holdout_RMSE_Std",
				"Dataset",
			],
			ascending=[True, True, True, True],
		)
		.groupby("Condition", as_index=False)
		.first()
	)

	return winners[
		[
			"Condition",
			"Dataset",
			"Holdout_RMSE_Mean",
			"Holdout_RMSE_Std",
			"Holdout_R2_Mean",
			"Holdout_R2_Std",
			"N_Seeds",
			"N_Fold_Evals",
		]
	]


def main() -> None:
	runtime_cfg = get_experiment_runtime_config(CONFIG)
	datasets = parse_target_folders(
		SWEEP_CONFIG["datasets"].get("target_folders", runtime_cfg["target_folders"])
	)
	holdout_file = SWEEP_CONFIG["datasets"].get("holdout_test_file", None)
	test_path = (
		Path(str(holdout_file))
		if holdout_file
		else WORKSPACE_ROOT / runtime_cfg["actual_test_file"]
	)
	if not test_path.is_absolute():
		test_path = WORKSPACE_ROOT / test_path
	if not test_path.exists():
		raise FileNotFoundError(f"Holdout test file not found: {test_path}")

	conditions = SWEEP_CONFIG.get("conditions", [])
	if not conditions:
		raise ValueError("SWEEP_CONFIG['conditions'] must contain at least one condition.")

	seeds = [int(s) for s in SWEEP_CONFIG.get("seeds", [])]
	if not seeds:
		raise ValueError("SWEEP_CONFIG['seeds'] must contain at least one seed.")

	output_path = _resolve_output_path(str(SWEEP_CONFIG["output"]["csv"]))
	save_checkpoints = bool(SWEEP_CONFIG.get("runtime", {}).get("save_checkpoints", False))

	print(
		"Seed sweep settings: "
		f"datasets={datasets}, conditions={[c['name'] for c in conditions]}, seeds={seeds}, "
		f"save_checkpoints={save_checkpoints}"
	)

	fold_rows: list[dict[str, Any]] = []
	total_runs = len(conditions) * len(seeds) * len(datasets)
	run_idx = 0
	global_start = time.perf_counter()
	run_durations_sec: list[float] = []
	for condition in conditions:
		for seed in seeds:
			for dataset in datasets:
				dataset_path = Path(dataset)
				dataset_name = dataset_path.name
				percent_label = infer_dataset_percent_label(dataset_name)
				artifact_folder = str(WORKSPACE_ROOT / (percent_label if percent_label else dataset_name))
				run_idx += 1
				now = time.perf_counter()
				elapsed_sec = now - global_start
				avg_run_sec = _safe_mean(run_durations_sec) if run_durations_sec else float("nan")
				remaining_runs = total_runs - run_idx + 1
				eta_sec = (
					avg_run_sec * remaining_runs
					if np.isfinite(avg_run_sec)
					else float("nan")
				)
				elapsed_min = elapsed_sec / 60.0
				eta_min = eta_sec / 60.0 if np.isfinite(eta_sec) else float("nan")
				print(
					"\n============================================================\n"
					f"Run {run_idx}/{total_runs} | condition={condition['name']} | seed={seed} | dataset={dataset_name} | "
					f"elapsed={elapsed_min:.1f}m"
					+ (
						f" | avg_run={avg_run_sec/60.0:.1f}m | ETA={eta_min:.1f}m"
						if np.isfinite(avg_run_sec)
						else " | avg_run=warming_up | ETA=estimating"
					)
					+ "\n"
					"============================================================"
				)
				run_start = time.perf_counter()
				rows = _run_condition_seed_dataset(
					base_config=CONFIG,
					target_folder=dataset,
					artifact_folder=artifact_folder,
					test_path=test_path,
					condition=condition,
					seed=seed,
					save_checkpoints=save_checkpoints,
				)
				run_time_sec = time.perf_counter() - run_start
				run_durations_sec.append(run_time_sec)
				print(
					f"Completed run {run_idx}/{total_runs} in {run_time_sec/60.0:.1f}m"
				)
				fold_rows.extend(rows)

	if not fold_rows:
		raise RuntimeError("No sweep rows were produced.")

	fold_df = pd.DataFrame(fold_rows)
	seed_summary_df = _build_seed_dataset_summaries(fold_df)
	condition_dataset_df = _build_condition_dataset_summaries(seed_summary_df)
	condition_overall_df = _build_condition_overall_summaries(seed_summary_df)
	ranking_df = _build_compact_dataset_ranking(fold_df)
	winners_df = _build_condition_winner_summary(ranking_df)

	combined_df = pd.concat(
		[
			fold_df,
			seed_summary_df,
			condition_dataset_df,
			condition_overall_df,
		],
		ignore_index=True,
		sort=False,
	)

	sort_cols = ["Level", "Condition", "Dataset", "Seed", "Fold"]
	for col in sort_cols:
		if col not in combined_df.columns:
			combined_df[col] = ""
	combined_df = combined_df.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	combined_df.to_csv(output_path, index=False)
	ranking_path = output_path.with_name(f"{output_path.stem}_ranking.csv")
	ranking_df.to_csv(ranking_path, index=False)
	winners_path = output_path.with_name(f"{output_path.stem}_winners.csv")
	winners_df.to_csv(winners_path, index=False)

	print(f"\nSaved sweep results to: {output_path}")
	print(f"Saved compact ranking table to: {ranking_path}")
	print(f"Saved condition winners table to: {winners_path}")
	print("\nCondition-level holdout RMSE across seeds and datasets:")
	if not condition_overall_df.empty:
		print(
			condition_overall_df[
				[
					"Condition",
					"AcrossSeedsAndDatasets_Holdout_RMSE_Mean",
					"AcrossSeedsAndDatasets_Holdout_RMSE_Std",
				]
			].to_string(index=False)
		)

	print("\nCompact inter-dataset ranking (heldout fold-level mean +/- std):")
	if not ranking_df.empty:
		print(
			ranking_df[
				[
					"Global_Rank_By_RMSE",
					"Rank_In_Condition_By_RMSE",
					"Condition",
					"Dataset",
					"N_Seeds",
					"N_Fold_Evals",
					"Holdout_RMSE_Mean",
					"Holdout_RMSE_Std",
					"Holdout_R2_Mean",
					"Holdout_R2_Std",
				]
			].to_string(index=False)
		)

	print("\nCondition winner summary (best dataset per condition):")
	if not winners_df.empty:
		for _, row in winners_df.iterrows():
			print(
				f"{row['Condition']}: winner={row['Dataset']} | "
				f"RMSE={float(row['Holdout_RMSE_Mean']):.4f} +/- {float(row['Holdout_RMSE_Std']):.4f} | "
				f"R2={float(row['Holdout_R2_Mean']):.4f} +/- {float(row['Holdout_R2_Std']):.4f} | "
				f"seeds={int(row['N_Seeds'])} | fold_evals={int(row['N_Fold_Evals'])}"
			)

	total_sec = time.perf_counter() - global_start
	print(f"\nTotal sweep time: {total_sec/60.0:.1f}m")


if __name__ == "__main__":
	main()
