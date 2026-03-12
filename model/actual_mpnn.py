from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from chemprop import featurizers
from chemprop.data import MoleculeDataset
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from mpnn_utils import (
    LossHistoryCallback,
    ReduceLROnPlateauCallback,
    _build_loader,
    _evaluate_best_checkpoint,
    _load_state_dict_from_checkpoint,
    _metric_description,
    _pick_best_fold,
    _safe_nanmean,
    _to_float,
    _write_losses_file,
    build_fold_paths,
    build_mpnn_model,
    configure_runtime,
    load_data,
    load_datapoints_from_csvs,
    set_global_seed,
)


# ==================== changelog ====================
# 2026-03-12
# - Replaced pseudocode with a runnable Chemprop MPNN training pipeline.
# - Added configurable 5-fold CV, fold summaries, and holdout evaluation.
# - Added robust target-column selection for real/synthetic CSVs.
# - Added checkpoint-safe loading for Torch 2.6+ (weights_only behavior).
# - Kept Chemprop metrics during training and sklearn RMSE/R2/Spearman at best-checkpoint evaluation.
# - Added config-driven RDKit randomized-SMILES oversampling for train split.
# - Added configurable ReduceLROnPlateau scheduling using the selected validation monitor metric.
# - Added ReduceLROnPlateau hyperparameters to DEFAULT_CONFIG optimization settings.
# - Added per-epoch learning-rate logging to exported MPNN loss files.
# - Refactored helper functions/callbacks into mpnn_utils.py so this file focuses on config + CV train loop.
# - Added module documentation in mpnn_utils.py describing responsibilities and preserved behavior.
# - Reduced script size while keeping output artifacts and fold-level reporting.


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "seed": 42,
        "total_folds": 5,
        "ratio_folders": ["0%", "33%", "67%"],
        "run_all_ratio_folders": False,
        "target_folder": "0%",
        "heldout_test_file": "heldout_testset.csv",
    },
    "data": {
        "batch_size": 128, # adjust based on GPU....
        "num_workers": 1,
        "real_target_column": "pIC50",
        "synthetic_target_column": "pred_pIC50",
        "fallback_target_columns": ["target_pIC50", "pred_pIC50", "pIC50"],
    },
    "oversampling": {
        "enabled": False,
        "duplicates_per_smiles": 3,
        "max_tries_per_duplicate": 8,
    },
    "model": {
        "d_h": 192, # hidden dimension for MPNN and FFN layers
        "depth": 3, # number of message-passing steps in the MPNN
        "dropout": 0.25,
        "ffn_hidden_mult": 1, # hidden dim = d_h * ffn_hidden_mult
        "ffn_n_layers": 2, # number of layers in the FFN
        "batch_norm": True,
        "use_sum_aggregation": False, # if false, it uses mean agg in MPNN readout layer. 
    },
    "optimization": {
        "warmup_epochs": 2,
        "init_lr": 1e-4,
        "max_lr": 8e-4,
        "final_lr": 1e-4,
        "reduce_lr_on_plateau": {
            "enabled": True,
            "mode": None,
            "factor": 0.75,
            "patience": 3,
            "threshold": 1e-4,
            "threshold_mode": "rel",
            "cooldown": 0,
            "min_lr": 1e-6,
            "eps": 1e-8, # small value to avoid zero division in LR scheduler
        },
    },
    "training": {
        "max_epochs": 50,
        "min_delta": 1e-4,
        "patience": 6,
        "accelerator": "auto",
        "deterministic": True,
        "test_each_fold": False,
        "checkpoint_monitor_metric": "val_loss",
        "checkpoint_monitor_mode": "min",
        "verbose_metric_logging": True,
    },
    "runtime": {
        "suppress_python_warnings": True,
        "suppress_rdkit_warnings": True,
    },
    "output": {
        "save_loss_curves": True,
    },
}


def train_chemprop_model_cv_it(
    train_csv_paths: list[Path],
    val_csv_path: Path,
    test_csv_path: Path,
    val_idx: int,
    ratio_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    should_test_each_fold = bool(config["training"]["test_each_fold"])
    effective_test_path = test_csv_path if should_test_each_fold else None
    monitor_metric = str(config["training"].get("checkpoint_monitor_metric", "val_loss"))
    monitor_mode = str(config["training"].get("checkpoint_monitor_mode", "min"))
    verbose_metric_logging = bool(config["training"].get("verbose_metric_logging", True))
    plateau_cfg = dict(config.get("optimization", {}).get("reduce_lr_on_plateau", {}))

    train_dset, val_dset, test_dset = load_data(train_csv_paths, val_csv_path, effective_test_path, config)
    train_loader = _build_loader(train_dset, config, shuffle=True)
    val_loader = _build_loader(val_dset, config, shuffle=False)
    test_loader = _build_loader(test_dset, config, shuffle=False) if test_dset is not None else None

    model = build_mpnn_model(config)
    history_cb = LossHistoryCallback(
        monitor_metric=monitor_metric,
        verbose_metric_logging=verbose_metric_logging,
    )

    print(
        f"Fold {val_idx}: monitoring '{monitor_metric}' ({monitor_mode}) "
        f"-> {_metric_description(monitor_metric)}"
    )

    checkpoint_dir = ratio_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=f"fold_{val_idx}_best",
        monitor=monitor_metric,
        mode=monitor_mode,
        save_top_k=1,
    )
    early_stop = EarlyStopping(
        monitor=monitor_metric,
        min_delta=float(config["training"]["min_delta"]),
        patience=int(config["training"]["patience"]),
        mode=monitor_mode,
        verbose=False,
    )

    logger = CSVLogger(save_dir=str(ratio_dir / "lightning_logs"), name="chemprop_mpnn", version=f"fold_{val_idx}" )

    reduce_lr_cb = None
    if bool(plateau_cfg.get("enabled", False)):
        reduce_lr_cb = ReduceLROnPlateauCallback(
            monitor_metric=monitor_metric,
            monitor_mode=monitor_mode,
            scheduler_cfg=plateau_cfg,
            verbose_metric_logging=verbose_metric_logging,
        )

    callbacks: list[Callback] = [early_stop, checkpoint_cb, history_cb]
    if reduce_lr_cb is not None:
        callbacks.append(reduce_lr_cb)

    trainer = pl.Trainer(
        max_epochs=int(config["training"]["max_epochs"]),
        accelerator=config["training"]["accelerator"],
        deterministic=bool(config["training"]["deterministic"]),
        enable_progress_bar=True,
        logger=logger,
        enable_checkpointing=True,
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, val_loader)

    best_ckpt_path = checkpoint_cb.best_model_path
    if not best_ckpt_path:
        raise RuntimeError(f"No best checkpoint produced for fold {val_idx}.")

    val_metrics = _evaluate_best_checkpoint(best_ckpt_path, val_loader, config)
    test_metrics = None
    if should_test_each_fold and test_loader is not None:
        test_metrics = _evaluate_best_checkpoint(best_ckpt_path, test_loader, config)

    best_pth_path = ratio_dir / f"best_model_iteration_{val_idx}.pth"
    torch.save(_load_state_dict_from_checkpoint(best_ckpt_path), best_pth_path)

    return {
        "fold_idx": val_idx,
        "monitor_metric": monitor_metric,
        "monitor_mode": monitor_mode,
        "monitor_metric_description": _metric_description(monitor_metric),
        "best_monitor_value": _to_float(checkpoint_cb.best_model_score),
        "best_ckpt_path": best_ckpt_path,
        "best_pth_path": str(best_pth_path),
        "val_loss": _to_float(checkpoint_cb.best_model_score),
        "val_loss_mse": float(val_metrics["mse"]),
        "val_mse": float(val_metrics["mse"]),
        "val_rmse": float(val_metrics["rmse"]),
        "val_r2": float(val_metrics["r2"]),
        "val_spearman": float(val_metrics["spearman"]),
        "val_metrics_sklearn": val_metrics,
        "test_metrics_sklearn": test_metrics,
        "train_losses": history_cb.train_losses,
        "monitor_values": history_cb.monitor_values,
        "monitor_metric_key_history": history_cb.monitor_metric_key_history,
        "val_losses": history_cb.val_losses,
        "val_loss_metric_key_history": history_cb.val_loss_metric_key_history,
        "chemprop_val_rmse_history": history_cb.chemprop_val_rmse_history,
        "chemprop_val_r2_history": history_cb.chemprop_val_r2_history,
        "lr_history": reduce_lr_cb.lr_history if reduce_lr_cb is not None else [],
    }


def run_allcv_iterations(config: dict[str, Any], ratio_dir: Path, workspace_root: Path) -> pd.DataFrame:
    exp_cfg = config["experiment"]
    total_folds = int(exp_cfg["total_folds"])
    ratio_name = ratio_dir.name
    heldout_path = workspace_root / exp_cfg["heldout_test_file"]
    if not heldout_path.exists():
        raise FileNotFoundError(f"Heldout test file not found: {heldout_path}")

    print(f"\n=== Running ratio folder: {ratio_name} ===")
    fold_results: list[dict[str, Any]] = []

    for val_idx in range(total_folds):
        print(f"\n--- Fold {val_idx} as validation ---")
        train_csvs, val_csv = build_fold_paths(ratio_dir, val_idx, total_folds)
        fold_result = train_chemprop_model_cv_it(
            train_csv_paths=train_csvs,
            val_csv_path=val_csv,
            test_csv_path=heldout_path,
            val_idx=val_idx,
            ratio_dir=ratio_dir,
            config=config,
        )
        fold_results.append(fold_result)
        print(
            f"Fold {val_idx} done | monitor={fold_result['monitor_metric']} "
            f"best_monitor_value={fold_result['best_monitor_value']} "
            f"val_loss_mse={fold_result['val_loss_mse']} "
            f"val_mse={fold_result['val_mse']} val_rmse={fold_result['val_rmse']} "
            f"val_r2={fold_result['val_r2']} val_spearman={fold_result['val_spearman']}"
        )

    best_fold = _pick_best_fold(fold_results)
    print(
        f"\nBest fold for {ratio_name}: fold_{best_fold['fold_idx']} "
        f"(val_rmse={best_fold['val_rmse']}, {best_fold['monitor_metric']}={best_fold['best_monitor_value']})"
    )

    test_data = load_datapoints_from_csvs([heldout_path], config, augment_train_smiles=False)
    test_dset = MoleculeDataset(test_data, featurizer=featurizers.SimpleMoleculeMolGraphFeaturizer())
    holdout_metrics = _evaluate_best_checkpoint(
        checkpoint_path=best_fold["best_ckpt_path"],
        dataloader=_build_loader(test_dset, config, shuffle=False),
        config=config,
    )

    summary_rows: list[dict[str, Any]] = [
        {
            "Stage": f"Fold_{fr['fold_idx']}_Val",
            "MonitorMetric": fr["monitor_metric"],
            "MonitorDescription": fr["monitor_metric_description"],
            "BestMonitorValue": fr["best_monitor_value"],
            "ValLoss_MSE": fr["val_loss_mse"],
            "MSE": fr["val_mse"],
            "RMSE": fr["val_rmse"],
            "R2": fr["val_r2"],
            "Rho": fr["val_spearman"],
        }
        for fr in fold_results
    ]
    summary_rows.append(
        {
            "Stage": "Average_Val",
            "MonitorMetric": fold_results[0]["monitor_metric"] if fold_results else "",
            "MonitorDescription": fold_results[0]["monitor_metric_description"] if fold_results else "",
            "BestMonitorValue": _safe_nanmean([fr["best_monitor_value"] for fr in fold_results]),
            "ValLoss_MSE": _safe_nanmean([fr["val_loss_mse"] for fr in fold_results]),
            "MSE": _safe_nanmean([fr["val_mse"] for fr in fold_results]),
            "RMSE": _safe_nanmean([fr["val_rmse"] for fr in fold_results]),
            "R2": _safe_nanmean([fr["val_r2"] for fr in fold_results]),
            "Rho": _safe_nanmean([fr["val_spearman"] for fr in fold_results]),
        }
    )
    summary_rows.append(
        {
            "Stage": f"Holdout_Test (Model from Fold_{best_fold['fold_idx']})",
            "MonitorMetric": "",
            "MonitorDescription": "",
            "BestMonitorValue": "",
            "ValLoss_MSE": "",
            "MSE": float(holdout_metrics["mse"]),
            "RMSE": float(holdout_metrics["rmse"]),
            "R2": float(holdout_metrics["r2"]),
            "Rho": float(holdout_metrics["spearman"]),
        }
    )

    summary_df = pd.DataFrame(summary_rows)
    results_path = ratio_dir / f"MPNN_results_{ratio_name}.csv"
    summary_df.to_csv(results_path, index=False)

    print("\nResult summary:")
    print(summary_df.to_string(index=False))
    print(f"Saved metrics to: {results_path}")

    if bool(config["output"].get("save_loss_curves", True)):
        _write_losses_file(ratio_dir, ratio_name, fold_results)

    return summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Chemprop MPNN with CV on ratio folders.")
    parser.add_argument("--folder", type=str, default=None, help="Run one ratio folder (0%, 33%, 67%).")
    parser.add_argument("--all", action="store_true", help="Override config and run all ratio folders.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DEFAULT_CONFIG
    workspace_root = Path(__file__).resolve().parent

    configure_runtime(config)
    set_global_seed(int(config["experiment"]["seed"]))

    run_all = bool(config["experiment"]["run_all_ratio_folders"])
    if args.all:
        run_all = True
    if args.folder is not None:
        run_all = False

    if run_all:
        target_dirs = [workspace_root / folder for folder in config["experiment"]["ratio_folders"]]
    else:
        selected = args.folder if args.folder is not None else config["experiment"]["target_folder"]
        target_dirs = [workspace_root / selected]

    for ratio_dir in target_dirs:
        if not ratio_dir.exists():
            raise FileNotFoundError(f"Ratio folder not found: {ratio_dir}")
        run_allcv_iterations(config=config, ratio_dir=ratio_dir, workspace_root=workspace_root)


if __name__ == "__main__":
    main()
