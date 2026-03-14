from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch_geometric.loader import DataLoader

from .checkpoints import get_fold_checkpoint_path, save_fold_checkpoint
from .data_loading import (
	cap_synthetic_train_ratio,
	get_fold_files,
	get_synthetic_cv_config,
	get_synthetic_pretraining_config,
	get_synthetic_pretraining_files,
	get_targets_from_graphs,
	load_dataset,
	print_fold_data_summary,
)
from .features import prepare_feature_config, print_feature_summary, set_seed
from .target_scaling import (
	build_target_standardizer,
	build_target_standardizer_from_stats,
	get_target_standardization_config,
	standardize_batch_targets,
)
from .training_helpers import (
	build_model,
	build_summary_dataframe,
	evaluate,
	is_improvement,
	metric_description,
	metric_from_name,
	write_losses_file,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_optional(
	value: Any,
	fallback: Any,
):
	if value is None:
		return fallback
	return value


def _is_minimize_metric(metric_name: str) -> bool:
	return metric_name in {"mse", "rmse", "mae"}


def run_synthetic_pretraining_stage(
	model: torch.nn.Module,
	synthetic_data: list,
	*,
	config: dict[str, Any],
	fold_idx: int,
	target_std_cfg: dict[str, Any],
) -> None:
	pre_cfg = get_synthetic_pretraining_config(config)
	if not pre_cfg["enabled"] or pre_cfg["epochs"] <= 0:
		return

	if len(synthetic_data) < pre_cfg["min_synthetic_graphs"]:
		print(
			f"[Fold {fold_idx}] skipping synthetic pretraining: "
			f"only {len(synthetic_data)} synthetic graphs "
			f"(< min_synthetic_graphs={pre_cfg['min_synthetic_graphs']})."
		)
		return

	learning_rate = _resolve_optional(
		pre_cfg["learning_rate"],
		config["optimization"]["learning_rate"],
	)
	weight_decay = _resolve_optional(
		pre_cfg["weight_decay"],
		config["optimization"]["weight_decay"],
	)
	grad_clip_norm = _resolve_optional(
		pre_cfg["grad_clip_norm"],
		config["optimization"].get("grad_clip_norm", 0.0),
	)

	target_standardizer = None
	if pre_cfg["use_target_standardization"]:
		synth_targets = get_targets_from_graphs(synthetic_data)
		target_standardizer = build_target_standardizer(
			synth_targets,
			epsilon=target_std_cfg["epsilon"],
		)

	synth_loader = DataLoader(
		synthetic_data,
		batch_size=pre_cfg["batch_size"],
		shuffle=pre_cfg["shuffle"],
	)
	optimizer = torch.optim.Adam(
		model.parameters(),
		lr=float(learning_rate),
		weight_decay=float(weight_decay),
	)
	criterion = torch.nn.MSELoss()

	print(
		f"[Fold {fold_idx}] synthetic pretraining: "
		f"epochs={pre_cfg['epochs']} n={len(synthetic_data)} "
		f"lr={float(learning_rate):.2e} wd={float(weight_decay):.2e}"
	)
	for epoch in range(1, pre_cfg["epochs"] + 1):
		model.train()
		epoch_loss = 0.0
		for batch in synth_loader:
			batch = batch.to(device)
			optimizer.zero_grad()
			out = model(batch).view(-1)
			y_true = standardize_batch_targets(
				batch.y.view(-1),
				target_standardizer=target_standardizer,
			)
			loss = criterion(out, y_true)
			loss.backward()
			if grad_clip_norm and float(grad_clip_norm) > 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
			optimizer.step()
			epoch_loss += loss.item() * batch.num_graphs

		epoch_loss /= max(len(synthetic_data), 1)
		print_every = max(1, pre_cfg["epochs"] // 5)
		if epoch % print_every == 0 or epoch == pre_cfg["epochs"]:
			print(
				f"[Fold {fold_idx}][Synthetic pretrain][Epoch {epoch:03d}] "
				f"train_mse={'(scaled) ' if target_standardizer is not None else ''}{epoch_loss:.4f}"
			)


def run_training_pipeline(config: dict[str, Any], model_class) -> None:
	set_seed(config["experiment"]["seed"])
	feature_context = prepare_feature_config(config)
	print_feature_summary(config, feature_context)

	target_std_cfg = get_target_standardization_config(config)
	use_target_standardization = target_std_cfg["enabled"]
	if use_target_standardization:
		print(
			"\nTarget standardization: ENABLED (per fold) "
			f"| epsilon={target_std_cfg['epsilon']:.1e}"
		)
	else:
		print("\nTarget standardization: DISABLED")

	target_folder = config["experiment"]["target_folder"]
	actual_test_file = config["experiment"]["actual_test_file"]
	total_folds = config["experiment"]["total_folds"]
	batch_size = config["data"]["batch_size"]
	num_epochs = config["training"]["num_epochs"]
	print_every = config["training"]["print_every"]

	learning_rate = config["optimization"]["learning_rate"]
	weight_decay = config["optimization"]["weight_decay"]
	grad_clip_norm = config["optimization"]["grad_clip_norm"]

	early_stop_cfg = config["training"]["early_stopping"]
	early_stop_enabled = early_stop_cfg["enabled"]
	early_stop_metric = early_stop_cfg["monitor_metric"]
	early_stop_patience = early_stop_cfg["patience"]
	early_stop_min_delta = early_stop_cfg["min_delta"]

	scheduler_cfg = config["scheduler"]
	scheduler_enabled = scheduler_cfg["enabled"]
	scheduler_metric = scheduler_cfg["monitor_metric"]
	scheduler_mode = scheduler_cfg["mode"]
	if scheduler_enabled:
		expected_scheduler_mode = "min" if _is_minimize_metric(scheduler_metric) else "max"
		if scheduler_mode != expected_scheduler_mode:
			print(
				"[WARN] scheduler.mode does not match scheduler.monitor_metric "
				f"(mode='{scheduler_mode}', metric='{scheduler_metric}'). "
				f"Using mode='{expected_scheduler_mode}'."
			)
			scheduler_mode = expected_scheduler_mode

	print(f"\nLoading holdout test set: {actual_test_file}")
	test_data = load_dataset(actual_test_file, config, feature_context)
	if len(test_data) == 0:
		raise RuntimeError(
			"Holdout test set produced 0 valid graphs. "
			"Check smiles parsing and target column configuration."
		)
	test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

	if _is_minimize_metric(early_stop_metric):
		global_best_monitor_value = float("inf")
	else:
		global_best_monitor_value = -float("inf")
	global_best_val_rmse = float("inf")
	global_best_weights = None
	best_fold_idx = -1
	global_best_target_mean = None
	global_best_target_std = None
	cv_results = []

	losses = {f"fold_{i}": {} for i in range(total_folds)}

	ckpt_dir = Path(target_folder) / "checkpoints"
	ckpt_dir.mkdir(parents=True, exist_ok=True)

	print("\n==================== 5-Fold Cross-Validation ====================")
	for val_idx in range(total_folds):
		print(f"\n-------------------- Fold {val_idx} (validation) --------------------")

		train_paths, val_paths = get_fold_files(
			target_folder,
			val_idx,
			config,
			total_folds=total_folds,
		)
		train_data = load_dataset(train_paths, config, feature_context)
		val_data = load_dataset(val_paths, config, feature_context)

		pretrain_synth_paths = get_synthetic_pretraining_files(
			target_folder,
			val_idx,
			config,
		)
		pretrain_synth_data = []
		if pretrain_synth_paths:
			pretrain_synth_data = load_dataset(pretrain_synth_paths, config, feature_context)

		model = build_model(model_class, config, feature_context)

		run_synthetic_pretraining_stage(
			model,
			pretrain_synth_data,
			config=config,
			fold_idx=val_idx,
			target_std_cfg=target_std_cfg,
		)

		synth_policy = get_synthetic_cv_config(config)
		finetune_train_data = cap_synthetic_train_ratio(
			train_data,
			max_ratio=synth_policy["max_train_synth_to_real_ratio"],
			seed=config["experiment"]["seed"],
			fold_idx=val_idx,
		)
		if len(finetune_train_data) == 0:
			raise RuntimeError(
				f"Fold {val_idx} has 0 training graphs after preprocessing/filtering."
			)
		if len(val_data) == 0:
			raise RuntimeError(
				f"Fold {val_idx} has 0 validation graphs after preprocessing/filtering."
			)

		print_fold_data_summary(val_idx, finetune_train_data, val_data)

		target_standardizer = None
		train_target_mean = 0.0
		train_target_std = 1.0
		if use_target_standardization:
			train_targets = get_targets_from_graphs(finetune_train_data)
			target_standardizer = build_target_standardizer(
				train_targets,
				epsilon=target_std_cfg["epsilon"],
			)
			train_target_mean = target_standardizer["mean"]
			train_target_std = target_standardizer["std"]
			print(
				f"[Fold {val_idx}] target standardizer: "
				f"mean={train_target_mean:.4f} std={train_target_std:.4f}"
			)

		train_loader = DataLoader(
			finetune_train_data,
			batch_size=batch_size,
			shuffle=config["data"]["shuffle_train"],
		)
		val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

		optimizer = torch.optim.Adam(
			model.parameters(), lr=learning_rate, weight_decay=weight_decay
		)
		criterion = torch.nn.MSELoss()

		scheduler = None
		if scheduler_enabled:
			scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
				optimizer,
				mode=scheduler_mode,
				factor=scheduler_cfg["factor"],
				patience=scheduler_cfg["patience"],
			)

		print(
			f"[Fold {val_idx}] model params={model_class.count_parameters(model)} "
			f"| lr={learning_rate:.2e} | dropout={config['model']['dropout']:.2f} "
			f"| residual={config['model'].get('residual', True)} "
			f"| norm={str(config['model'].get('normalization', 'layernorm')).lower()}"
		)

		best_val_mse = float("inf")
		best_val_rmse = float("inf")
		best_val_mae = float("inf")
		best_val_r2 = -float("inf")
		best_val_rho = -float("inf")
		best_val_pearson = -float("inf")
		best_epoch = 0
		best_model_weights = None

		if _is_minimize_metric(early_stop_metric):
			best_monitor_value = float("inf")
		else:
			best_monitor_value = -float("inf")

		train_losses = []
		val_losses = []
		monitor_values = []
		lr_history = []
		epochs_wo_improv = 0

		for epoch in range(1, num_epochs + 1):
			model.train()
			train_loss = 0.0

			for batch in train_loader:
				batch = batch.to(device)
				optimizer.zero_grad()
				out = model(batch).view(-1)

				y_true = standardize_batch_targets(
					batch.y.view(-1),
					target_standardizer=target_standardizer,
				)
				loss = criterion(out, y_true)
				loss.backward()

				if grad_clip_norm and grad_clip_norm > 0:
					torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

				optimizer.step()
				train_loss += loss.item() * batch.num_graphs

			train_loss /= len(finetune_train_data)
			val_mse, val_rmse, val_mae, val_r2, val_rho, val_pearson = evaluate(
				model,
				val_loader,
				target_standardizer=target_standardizer,
			)

			# current monitor is the valye we compare aginst best value for early stopping and scheduler decisions
			current_monitor = metric_from_name(
				early_stop_metric, val_mse, val_rmse, val_mae, val_r2, val_rho, val_pearson
			)

			if is_improvement(
				early_stop_metric, current_monitor, best_monitor_value, early_stop_min_delta
			):
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
				save_fold_checkpoint(
					{
						"state_dict": best_model_weights,
						"fold": val_idx,
						"target_standardization_enabled": bool(target_standardizer is not None),
						"target_mean": train_target_mean,
						"target_std": train_target_std,
						"val_rmse": val_rmse,
						"val_mae": val_mae,
						"val_pearson": val_pearson,
					},
					get_fold_checkpoint_path(target_folder, val_idx),
				)
			else:
				epochs_wo_improv += 1

			if scheduler is not None:
				scheduler_value = metric_from_name(
					scheduler_metric, val_mse, val_rmse, val_mae, val_r2, val_rho, val_pearson
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

			if early_stop_enabled and epochs_wo_improv >= early_stop_patience:
				print(
					f"[Fold {val_idx}] early stopping at epoch {epoch} "
					f"(monitor={early_stop_metric}, patience={early_stop_patience}, "
					f"min_delta={early_stop_min_delta})"
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

		cv_results.append(
			{
				"fold_idx": val_idx,
				"monitor_metric": early_stop_metric,
				"monitor_metric_description": metric_description(early_stop_metric),
				"best_monitor_value": best_monitor_value,
				"val_loss_mse": best_val_mse,
				"val_mse": best_val_mse,
				"val_rmse": best_val_rmse,
				"val_mae": best_val_mae,
				"val_r2": best_val_r2,
				"val_rho": best_val_rho,
				"val_pearson": best_val_pearson,
			}
		)

		if is_improvement(
			early_stop_metric,
			best_monitor_value,
			global_best_monitor_value,
			0.0,
		):
			global_best_monitor_value = best_monitor_value
			global_best_val_rmse = best_val_rmse
			global_best_weights = copy.deepcopy(best_model_weights)
			best_fold_idx = val_idx
			global_best_target_mean = train_target_mean
			global_best_target_std = train_target_std
			print(
				f"[Fold {val_idx}] is current global best "
				f"(monitor={early_stop_metric} value={best_monitor_value:.6f})."
			)

		losses[f"fold_{val_idx}"] = {
			"train": train_losses,
			"val": val_losses,
			"monitor": monitor_values,
			"lr": lr_history,
		}

	print("\n-------------------- Holdout Evaluation --------------------")
	print(
		f"Loading global-best model from fold {best_fold_idx} "
		f"(monitor={early_stop_metric} value={global_best_monitor_value:.6f}, "
		f"val_rmse={global_best_val_rmse:.4f})"
	)
	if global_best_weights is None:
		raise RuntimeError("No best model weights were saved during cross-validation.")

	best_model = build_model(model_class, config, feature_context)
	best_model.load_state_dict(global_best_weights)
	# Evaluate the best model on the holdout test set

	best_target_standardizer = None
	if use_target_standardization:
		if global_best_target_mean is None or global_best_target_std is None:
			raise RuntimeError(
				"Target standardization is enabled, but best-fold stats were not captured."
			)
		best_target_standardizer = build_target_standardizer_from_stats(
			float(global_best_target_mean),
			float(global_best_target_std),
		)

	test_mse, test_rmse, test_mae, test_r2, test_rho, test_pearson = evaluate(
		best_model,
		test_loader,
		target_standardizer=best_target_standardizer,
	)
	print(
		f"Holdout -> MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, "
		f"R2: {test_r2:.4f}, Rho: {test_rho:.4f}, Pearson: {test_pearson:.4f}"
	)

	print("\n-------------------- Final CV Summary --------------------")
	df_results = build_summary_dataframe(
		cv_results,
		best_fold_idx,
		test_mse,
		test_rmse,
		test_mae,
		test_r2,
		test_rho,
		test_pearson,
	)
	print(df_results.to_string(index=False))

	target_path_obj = Path(target_folder)
	folder_name = target_path_obj.name if target_folder != "." else "default_experiment"

	save_path = target_path_obj / f"MPNN_results_{folder_name}.csv"
	df_results.to_csv(save_path, index=False)
	print(f"Saved results to: {save_path.resolve()}")

	loss_path = target_path_obj / f"MPNN_losses_{folder_name}.txt"
	write_losses_file(loss_path, losses)
	print(f"Saved learning curves to: {loss_path.resolve()}")
