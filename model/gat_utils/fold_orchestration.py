from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch_geometric.loader import DataLoader

from .checkpoints import get_epoch_checkpoint_path, get_fold_checkpoint_path, save_fold_checkpoint
from .config_helpers import is_minimize_metric
from .data_loading import (
	cap_synthetic_train_ratio,
	get_fold_files,
	get_synthetic_cv_config,
	get_synthetic_pretraining_files,
	get_targets_from_graphs,
	load_dataset,
	print_fold_data_summary,
)
from .features import apply_feature_scalers, fit_feature_scalers
from .target_scaling import build_target_standardizer, standardize_batch_targets
from .training_helpers import evaluate, is_improvement, metric_description, metric_from_name


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_checkpoint_payload(
	*,
	model_state_dict: dict[str, torch.Tensor],
	checkpoint_kind: str,
	dataset_label: str,
	fold_idx: int,
	epoch: int,
	monitor_metric: str,
	monitor_value: float,
	min_delta: float,
	minimum_improvement: float,
	target_standardizer: dict[str, Any] | None,
	train_target_mean: float,
	train_target_std: float,
	config: dict[str, Any],
	extra_metrics: dict[str, float],
) -> dict[str, Any]:
	feature_scaling_cfg = config.get("features", {}).get("feature_scaling", {})
	payload = {
		"state_dict": model_state_dict,
		"checkpoint_kind": checkpoint_kind,
		"dataset_label": dataset_label,
		"fold": fold_idx,
		"epoch": epoch,
		"monitor_metric": monitor_metric,
		"monitor_value": monitor_value,
		"min_delta": min_delta,
		"minimum_improvement": minimum_improvement,
		"target_standardization_enabled": bool(target_standardizer is not None),
		"target_mean": train_target_mean,
		"target_std": train_target_std,
		"feature_scaling_mode": feature_scaling_cfg.get("mode", "none"),
		"feature_scaling_fit_on": feature_scaling_cfg.get("fit_on", "real_plus_synthetic"),
	}
	payload.update(extra_metrics)
	return payload


def train_one_epoch(
	model: torch.nn.Module,
	train_loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	criterion: torch.nn.Module,
	*,
	target_standardizer: dict[str, Any] | None,
	grad_clip_norm: float,
) -> float:
	model.train()
	total_loss = 0.0
	for batch in train_loader:
		batch = batch.to(device)
		optimizer.zero_grad()
		out = model(batch).view(-1)
		y_true = standardize_batch_targets(batch.y.view(-1), target_standardizer=target_standardizer)
		loss = criterion(out, y_true)
		loss.backward()
		if grad_clip_norm and grad_clip_norm > 0:
			torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
		optimizer.step()
		total_loss += loss.item() * batch.num_graphs
	return total_loss


def run_single_fold(
	*,
	val_idx: int,
	config: dict[str, Any],
	feature_context: dict[str, Any],
	data_folder: str,
	artifact_folder: str,
	total_folds: int,
	batch_size: int,
	num_epochs: int,
	print_every: int,
	learning_rate: float,
	weight_decay: float,
	grad_clip_norm: float,
	early_stop_cfg: dict[str, Any],
	scheduler_cfg: dict[str, Any],
	save_every_n_epochs: int,
	target_std_cfg: dict[str, Any],
	use_target_standardization: bool,
	dataset_label: str,
	run_synthetic_pretraining_stage,
	build_model,
	model_class,
	save_checkpoints: bool = True,
) -> dict[str, Any]:
	print(f"\n-------------------- Fold {val_idx} (validation) --------------------")
	checkpoint_target_folder = artifact_folder or data_folder

	train_paths, val_paths = get_fold_files(data_folder, val_idx, config, total_folds=total_folds)
	train_data = load_dataset(train_paths, config, feature_context)
	val_data = load_dataset(val_paths, config, feature_context)

	pretrain_synth_paths = get_synthetic_pretraining_files(data_folder, val_idx, config)
	pretrain_synth_data = load_dataset(pretrain_synth_paths, config, feature_context) if pretrain_synth_paths else []

	model = build_model(model_class, config, feature_context)

	synth_policy = get_synthetic_cv_config(config)
	finetune_train_data = cap_synthetic_train_ratio(
		train_data,
		max_ratio=synth_policy["max_train_synth_to_real_ratio"],
		seed=config["experiment"]["seed"],
		fold_idx=val_idx,
	)
	if len(finetune_train_data) == 0:
		raise RuntimeError(f"Fold {val_idx} has 0 training graphs after preprocessing/filtering.")
	if len(val_data) == 0:
		raise RuntimeError(f"Fold {val_idx} has 0 validation graphs after preprocessing/filtering.")

	feature_scalers = fit_feature_scalers(finetune_train_data, feature_context=feature_context, config=config)
	apply_feature_scalers(finetune_train_data, feature_scalers=feature_scalers)
	apply_feature_scalers(val_data, feature_scalers=feature_scalers)
	apply_feature_scalers(pretrain_synth_data, feature_scalers=feature_scalers)

	run_synthetic_pretraining_stage(model, pretrain_synth_data, config=config, fold_idx=val_idx, target_std_cfg=target_std_cfg)

	print_fold_data_summary(val_idx, finetune_train_data, val_data)

	target_standardizer = None
	train_target_mean = 0.0
	train_target_std = 1.0
	if use_target_standardization:
		train_targets = get_targets_from_graphs(finetune_train_data)
		target_standardizer = build_target_standardizer(train_targets, epsilon=target_std_cfg["epsilon"])
		train_target_mean = target_standardizer["mean"]
		train_target_std = target_standardizer["std"]
		print(f"[Fold {val_idx}] target standardizer: mean={train_target_mean:.4f} std={train_target_std:.4f}")

	train_loader = DataLoader(finetune_train_data, batch_size=batch_size, shuffle=config["data"]["shuffle_train"])
	val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
	criterion = torch.nn.MSELoss()

	scheduler = None
	if scheduler_cfg["enabled"]:
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer,
			mode=scheduler_cfg["mode"],
			factor=scheduler_cfg["factor"],
			patience=scheduler_cfg["patience"],
		)

	print(
		f"[Fold {val_idx}] model params={model_class.count_parameters(model)} "
		f"| lr={learning_rate:.2e} | dropout={config['model']['dropout']:.2f} "
		f"| residual={config['model'].get('residual', True)} "
		f"| norm={str(config['model'].get('normalization', 'layernorm')).lower()} "
		f"| checkpoint_min_delta={early_stop_cfg['min_delta']:.6f} "
		f"| early_stop_min_improvement={early_stop_cfg['minimum_improvement']:.6f}"
	)

	best_val_mse = float("inf")
	best_val_rmse = float("inf")
	best_val_mae = float("inf")
	best_val_r2 = -float("inf")
	best_val_rho = -float("inf")
	best_val_pearson = -float("inf")
	best_epoch = 0
	best_model_weights = None
	best_monitor_value = float("inf") if is_minimize_metric(early_stop_cfg["monitor_metric"]) else -float("inf")
	early_stop_best_value = float("inf") if is_minimize_metric(early_stop_cfg["monitor_metric"]) else -float("inf")

	train_losses: list[float] = []
	val_losses: list[float] = []
	monitor_values: list[float] = []
	lr_history: list[float] = []
	epochs_wo_improv = 0

	for epoch in range(1, num_epochs + 1):
		train_loss_total = train_one_epoch(
			model,
			train_loader,
			optimizer,
			criterion,
			target_standardizer=target_standardizer,
			grad_clip_norm=grad_clip_norm,
		)
		train_loss = train_loss_total / len(finetune_train_data)
		val_mse, val_rmse, val_mae, val_r2, val_rho, val_pearson = evaluate(
			model,
			val_loader,
			target_standardizer=target_standardizer,
		)

		current_monitor = metric_from_name(
			early_stop_cfg["monitor_metric"],
			val_mse,
			val_rmse,
			val_mae,
			val_r2,
			val_rho,
			val_pearson,
		)

		checkpoint_improved = is_improvement(
			early_stop_cfg["monitor_metric"],
			current_monitor,
			best_monitor_value,
			early_stop_cfg["min_delta"],
		)
		early_stop_improved = is_improvement(
			early_stop_cfg["monitor_metric"],
			current_monitor,
			early_stop_best_value,
			early_stop_cfg["minimum_improvement"],
		)

		if checkpoint_improved:
			best_monitor_value = current_monitor
			best_val_mse = val_mse
			best_val_rmse = val_rmse
			best_val_mae = val_mae
			best_val_r2 = val_r2
			best_val_rho = val_rho
			best_val_pearson = val_pearson
			best_epoch = epoch
			best_model_weights = copy.deepcopy(model.state_dict())
			epochs_wo_improv = 0
			if save_checkpoints:
				save_fold_checkpoint(
					build_checkpoint_payload(
						model_state_dict=best_model_weights,
						checkpoint_kind="best",
						dataset_label=dataset_label,
						fold_idx=val_idx,
						epoch=epoch,
						monitor_metric=early_stop_cfg["monitor_metric"],
						monitor_value=current_monitor,
						min_delta=early_stop_cfg["min_delta"],
						minimum_improvement=early_stop_cfg["minimum_improvement"],
						target_standardizer=target_standardizer,
						train_target_mean=train_target_mean,
						train_target_std=train_target_std,
						config=config,
						extra_metrics={"val_rmse": val_rmse, "val_mae": val_mae, "val_pearson": val_pearson},
					),
					get_fold_checkpoint_path(checkpoint_target_folder, val_idx),
				)

		if early_stop_improved:
			early_stop_best_value = current_monitor
			epochs_wo_improv = 0
		else:
			epochs_wo_improv += 1

		should_save_epoch_checkpoint = (
			save_checkpoints
			and save_every_n_epochs > 0
			and (epoch % save_every_n_epochs == 0 or epoch == num_epochs)
		)
		if should_save_epoch_checkpoint:
			save_fold_checkpoint(
				build_checkpoint_payload(
					model_state_dict=copy.deepcopy(model.state_dict()),
					checkpoint_kind="epoch",
					dataset_label=dataset_label,
					fold_idx=val_idx,
					epoch=epoch,
					monitor_metric=early_stop_cfg["monitor_metric"],
					monitor_value=current_monitor,
					min_delta=early_stop_cfg["min_delta"],
					minimum_improvement=early_stop_cfg["minimum_improvement"],
					target_standardizer=target_standardizer,
					train_target_mean=train_target_mean,
					train_target_std=train_target_std,
					config=config,
					extra_metrics={
						"val_mse": val_mse,
						"val_rmse": val_rmse,
						"val_mae": val_mae,
						"val_r2": val_r2,
						"val_rho": val_rho,
						"val_pearson": val_pearson,
					},
				),
				get_epoch_checkpoint_path(checkpoint_target_folder, val_idx, epoch),
			)

		if scheduler is not None:
			scheduler_value = metric_from_name(
				scheduler_cfg["monitor_metric"],
				val_mse,
				val_rmse,
				val_mae,
				val_r2,
				val_rho,
				val_pearson,
			)
			scheduler.step(scheduler_value)

		train_losses.append(train_loss)
		val_losses.append(val_mse)
		monitor_values.append(current_monitor)
		lr_history.append(float(optimizer.param_groups[0]["lr"]))

		if epoch % print_every == 0:
			print(
				f"[Fold {val_idx}][Epoch {epoch:03d}] "
				f"train_mse={'(scaled) ' if use_target_standardization else ''}{train_loss:.4f} "
				f"val_mse={val_mse:.4f} val_rmse={val_rmse:.4f} val_mae={val_mae:.4f} "
				f"val_r2={val_r2:.4f} val_rho={val_rho:.4f} val_pearson={val_pearson:.4f} "
				f"lr={optimizer.param_groups[0]['lr']:.2e} "
				f"best_val_rmse={best_val_rmse:.4f} "
				f"no_improve={epochs_wo_improv}"
			)

		if early_stop_cfg["enabled"] and epochs_wo_improv >= early_stop_cfg["patience"]:
			print(
				f"[Fold {val_idx}] early stopping at epoch {epoch} "
				f"(monitor={early_stop_cfg['monitor_metric']}, patience={early_stop_cfg['patience']}, "
				f"minimum_improvement={early_stop_cfg['minimum_improvement']}, "
				f"checkpoint_min_delta={early_stop_cfg['min_delta']})"
			)
			break

	print(
		f"[Fold {val_idx}] best epoch={best_epoch} | "
		f"val_rmse={best_val_rmse:.4f} val_mae={best_val_mae:.4f} "
		f"val_r2={best_val_r2:.4f} "
		f"val_rho={best_val_rho:.4f} val_pearson={best_val_pearson:.4f}"
	)
	if best_model_weights is None:
		raise RuntimeError(
			f"Fold {val_idx} did not produce any best checkpoint updates. "
			"This usually means monitor metrics were NaN/invalid for all epochs."
		)

	return {
		"fold_idx": val_idx,
		"best_monitor_value": best_monitor_value,
		"best_model_weights": copy.deepcopy(best_model_weights),
		"train_target_mean": train_target_mean,
		"train_target_std": train_target_std,
		"feature_scalers": copy.deepcopy(feature_scalers),
		"cv_result": {
			"fold_idx": val_idx,
			"monitor_metric": early_stop_cfg["monitor_metric"],
			"monitor_metric_description": metric_description(early_stop_cfg["monitor_metric"]),
			"best_monitor_value": best_monitor_value,
			"val_loss_mse": best_val_mse,
			"val_mse": best_val_mse,
			"val_rmse": best_val_rmse,
			"val_mae": best_val_mae,
			"val_r2": best_val_r2,
			"val_rho": best_val_rho,
			"val_pearson": best_val_pearson,
		},
		"loss_curve": {
			"train": train_losses,
			"val": val_losses,
			"monitor": monitor_values,
			"lr": lr_history,
		},
	}


def run_cross_validation(
	*,
	config: dict[str, Any],
	feature_context: dict[str, Any],
	data_folder: str,
	artifact_folder: str,
	total_folds: int,
	batch_size: int,
	num_epochs: int,
	print_every: int,
	learning_rate: float,
	weight_decay: float,
	grad_clip_norm: float,
	early_stop_cfg: dict[str, Any],
	scheduler_cfg: dict[str, Any],
	save_every_n_epochs: int,
	target_std_cfg: dict[str, Any],
	use_target_standardization: bool,
	dataset_label: str,
	run_synthetic_pretraining_stage,
	build_model,
	model_class,
	save_checkpoints: bool = True,
) -> dict[str, Any]:
	if is_minimize_metric(early_stop_cfg["monitor_metric"]):
		global_best_monitor_value = float("inf")
	else:
		global_best_monitor_value = -float("inf")
	global_best_val_rmse = float("inf")
	global_best_weights = None
	best_fold_idx = -1
	global_best_target_mean = None
	global_best_target_std = None
	global_best_feature_scalers = None
	cv_results: list[dict[str, Any]] = []
	losses = {f"fold_{i}": {} for i in range(total_folds)}

	print("\n==================== 5-Fold Cross-Validation ====================")
	for val_idx in range(total_folds):
		fold_output = run_single_fold(
			val_idx=val_idx,
			config=config,
			feature_context=feature_context,
			data_folder=data_folder,
			artifact_folder=artifact_folder,
			total_folds=total_folds,
			batch_size=batch_size,
			num_epochs=num_epochs,
			print_every=print_every,
			learning_rate=learning_rate,
			weight_decay=weight_decay,
			grad_clip_norm=grad_clip_norm,
			early_stop_cfg=early_stop_cfg,
			scheduler_cfg=scheduler_cfg,
			save_every_n_epochs=save_every_n_epochs,
			target_std_cfg=target_std_cfg,
			use_target_standardization=use_target_standardization,
			dataset_label=dataset_label,
			run_synthetic_pretraining_stage=run_synthetic_pretraining_stage,
			build_model=build_model,
			model_class=model_class,
			save_checkpoints=save_checkpoints,
		)

		cv_results.append(fold_output["cv_result"])
		losses[f"fold_{val_idx}"] = fold_output["loss_curve"]

		if is_improvement(early_stop_cfg["monitor_metric"], fold_output["best_monitor_value"], global_best_monitor_value, 0.0):
			global_best_monitor_value = float(fold_output["best_monitor_value"])
			global_best_val_rmse = float(fold_output["cv_result"]["val_rmse"])
			global_best_weights = copy.deepcopy(fold_output["best_model_weights"])
			best_fold_idx = int(fold_output["fold_idx"])
			global_best_target_mean = float(fold_output["train_target_mean"])
			global_best_target_std = float(fold_output["train_target_std"])
			global_best_feature_scalers = copy.deepcopy(fold_output["feature_scalers"])
			print(
				f"[Fold {val_idx}] is current global best "
				f"(monitor={early_stop_cfg['monitor_metric']} value={fold_output['best_monitor_value']:.6f})."
			)

	return {
		"cv_results": cv_results,
		"losses": losses,
		"global_best_monitor_value": global_best_monitor_value,
		"global_best_val_rmse": global_best_val_rmse,
		"global_best_weights": global_best_weights,
		"best_fold_idx": best_fold_idx,
		"global_best_target_mean": global_best_target_mean,
		"global_best_target_std": global_best_target_std,
		"global_best_feature_scalers": global_best_feature_scalers,
	}
