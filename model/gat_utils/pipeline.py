from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch_geometric.loader import DataLoader

from .config_helpers import (
	get_checkpointing_config,
	get_early_stopping_config,
	get_experiment_runtime_config,
	get_optimization_config,
	get_scheduler_config,
	is_minimize_metric,
)
from .data_loading import get_synthetic_pretraining_config, get_targets_from_graphs, load_dataset
from .features import (
	apply_feature_scalers,
	prepare_feature_config,
	print_feature_summary,
	set_seed,
)
from .fold_orchestration import run_cross_validation
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
	criterion = torch.nn.MSELoss() # same as main train loop. 

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

	runtime_cfg = get_experiment_runtime_config(config)
	target_folder = runtime_cfg["target_folder"]
	actual_test_file = runtime_cfg["actual_test_file"]
	total_folds = runtime_cfg["total_folds"]
	batch_size = runtime_cfg["batch_size"]
	num_epochs = runtime_cfg["num_epochs"]
	print_every = runtime_cfg["print_every"]

	checkpoint_cfg = get_checkpointing_config(config)
	save_every_n_epochs = checkpoint_cfg["save_every_n_epochs"]

	optimization_cfg = get_optimization_config(config)
	learning_rate = optimization_cfg["learning_rate"]
	weight_decay = optimization_cfg["weight_decay"]
	grad_clip_norm = optimization_cfg["grad_clip_norm"]

	early_stop_cfg = get_early_stopping_config(config)
	early_stop_enabled = early_stop_cfg["enabled"]
	early_stop_metric = early_stop_cfg["monitor_metric"]
	early_stop_patience = early_stop_cfg["patience"]
	early_stop_min_delta = early_stop_cfg["min_delta"]
	early_stop_minimum_improvement = early_stop_cfg["minimum_improvement"]
	print(
		"Early stopping: "
		f"{'ENABLED' if early_stop_enabled else 'DISABLED'} "
		f"| monitor={early_stop_metric} | patience={early_stop_patience} "
		f"| checkpoint_min_delta={early_stop_min_delta} "
		f"| minimum_improvement={early_stop_minimum_improvement}"
	)

	scheduler_cfg = get_scheduler_config(config)
	scheduler_enabled = scheduler_cfg["enabled"]
	scheduler_metric = scheduler_cfg["monitor_metric"]
	scheduler_mode = scheduler_cfg["mode"]
	if scheduler_enabled:
		expected_scheduler_mode = "min" if is_minimize_metric(scheduler_metric) else "max"
		if scheduler_mode != expected_scheduler_mode:
			print(
				"[WARN] scheduler.mode does not match scheduler.monitor_metric "
				f"(mode='{scheduler_mode}', metric='{scheduler_metric}'). "
				f"Using mode='{expected_scheduler_mode}'."
			)
			scheduler_mode = expected_scheduler_mode

	print(f"\nLoading holdout test set: {actual_test_file}")
	raw_test_data = load_dataset(actual_test_file, config, feature_context)
	if len(raw_test_data) == 0:
		raise RuntimeError(
			"Holdout test set produced 0 valid graphs. "
			"Check smiles parsing and target column configuration."
		)

	ckpt_dir = Path(target_folder) / "checkpoints"
	ckpt_dir.mkdir(parents=True, exist_ok=True)
	dataset_label = Path(target_folder).name if target_folder != "." else "default_experiment"
	cv_output = run_cross_validation(
		config=config,
		feature_context=feature_context,
		target_folder=target_folder,
		total_folds=total_folds,
		batch_size=batch_size,
		num_epochs=num_epochs,
		print_every=print_every,
		learning_rate=learning_rate,
		weight_decay=weight_decay,
		grad_clip_norm=grad_clip_norm,
		early_stop_cfg=early_stop_cfg,
		scheduler_cfg={
			"enabled": scheduler_enabled,
			"monitor_metric": scheduler_metric,
			"mode": scheduler_mode,
			"factor": scheduler_cfg["factor"],
			"patience": scheduler_cfg["patience"],
		},
		save_every_n_epochs=save_every_n_epochs,
		target_std_cfg=target_std_cfg,
		use_target_standardization=use_target_standardization,
		dataset_label=dataset_label,
		run_synthetic_pretraining_stage=run_synthetic_pretraining_stage,
		build_model=build_model,
		model_class=model_class,
	)

	cv_results = cv_output["cv_results"]
	losses = cv_output["losses"]
	global_best_monitor_value = cv_output["global_best_monitor_value"]
	global_best_val_rmse = cv_output["global_best_val_rmse"]
	global_best_weights = cv_output["global_best_weights"]
	best_fold_idx = cv_output["best_fold_idx"]
	global_best_target_mean = cv_output["global_best_target_mean"]
	global_best_target_std = cv_output["global_best_target_std"]
	global_best_feature_scalers = cv_output["global_best_feature_scalers"]

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
	# Evaluate the best model on the holdout test set.
	# Feature scaling is fold-specific, so holdout features must be transformed
	# using the same scaler fitted on the selected best fold train split.
	test_data = copy.deepcopy(raw_test_data)
	apply_feature_scalers(test_data, feature_scalers=global_best_feature_scalers)
	test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

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
