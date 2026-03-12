from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from chemprop import featurizers, models, nn
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.nn.metrics import R2Score, RMSE
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger


# ==================== changelog ====================
# 2026-03-12
# - Replaced pseudocode with a runnable Chemprop MPNN training pipeline.
# - Added configurable 5-fold CV with optional synthetic-data augmentation per fold.
# - Added robust target-column selection for real (pIC50) and synthetic (pred_pIC50) data.
# - Added best-checkpoint export to .pth, holdout evaluation of the global best fold,
#   and per-ratio result/loss artifacts.
# - Kept SimpleMoleculeMolGraphFeaturizer as the graph featurization backend.


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "seed": 42,
        "total_folds": 5,
        "ratio_folders": ["0%", "33%", "67%"],
        "run_all_ratio_folders": False,
        "target_folder": "0%",  # used when run_all_ratio_folders is False
        "heldout_test_file": "heldout_testset.csv",
    },
    "data": {
        "batch_size": 64,
        "real_target_column": "pIC50",
        "synthetic_target_column": "pred_pIC50",
        "fallback_target_columns": ["target_pIC50", "pred_pIC50", "pIC50"],
    },
    "model": {
        "d_h": 256,
        "depth": 6,
        "dropout": 0.2,
        "ffn_hidden_mult": 2, # multiplier for FFN hidden layer size relative to d_h
        "ffn_n_layers": 2, # number of layers in the FFN predictor
        "batch_norm": True,
        "use_sum_aggregation": True,
    },
    "optimization": {
        "warmup_epochs": 2,
        "init_lr": None,
        "max_lr": None,
        "final_lr": None,
    },
    "training": {
        "max_epochs": 60,
        "min_delta": 1e-5,
        "patience": 6,
        "accelerator": "auto",
        "deterministic": True,
        "test_each_fold": False,
    },
    "output": {
        "save_loss_curves": True,
    },
}


def set_global_seed(seed: int) -> None:
    pl.seed_everything(seed, workers=True)


def _metric_to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    return None


def _extract_metric(
    metrics: dict[str, Any],
    preferred_keys: list[str],
    key_contains_any: list[str],
) -> float | None:
    for key in preferred_keys:
        if key in metrics:
            value = _metric_to_float(metrics[key])
            if value is not None:
                return value

    for key, raw_value in metrics.items():
        low_key = str(key).lower()
        if any(fragment in low_key for fragment in key_contains_any):
            value = _metric_to_float(raw_value)
            if value is not None:
                return value

    return None


class LossHistoryCallback(Callback):
    """Collect per-epoch train/validation losses from Lightning callback metrics."""

    def __init__(self) -> None:
        super().__init__()
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        train_loss = _extract_metric(
            metrics=trainer.callback_metrics,
            preferred_keys=["train_loss", "loss"],
            key_contains_any=["train_loss"],
        )
        if train_loss is not None:
            self.train_losses.append(train_loss)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        val_loss = _extract_metric(
            metrics=trainer.callback_metrics,
            preferred_keys=["val_loss"],
            key_contains_any=["val_loss"],
        )
        if val_loss is not None:
            self.val_losses.append(val_loss)


def choose_target_column(df: pd.DataFrame, csv_path: Path, config: dict[str, Any]) -> str:
    is_synthetic = csv_path.name.startswith("synthetic_data_iteration_")
    data_cfg = config["data"]

    if is_synthetic:
        priorities = [
            data_cfg["synthetic_target_column"],
            *data_cfg["fallback_target_columns"],
            data_cfg["real_target_column"],
        ]
    else:
        priorities = [
            data_cfg["real_target_column"],
            *data_cfg["fallback_target_columns"],
        ]

    for col in priorities:
        if col in df.columns:
            return col

    raise ValueError(
        f"No supported target column found in {csv_path}. "
        f"Tried: {priorities}"
    )


def load_datapoints_from_csvs(
    csv_paths: list[Path],
    config: dict[str, Any],
) -> list[MoleculeDatapoint]:
    datapoints: list[MoleculeDatapoint] = []

    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing expected file: {csv_path}")

        df = pd.read_csv(csv_path)
        if "smiles" not in df.columns:
            raise ValueError(f"Missing required 'smiles' column in {csv_path}")

        target_col = choose_target_column(df, csv_path, config)
        skipped_rows = 0

        for _, row in df.iterrows():
            smiles = row["smiles"]
            target = row[target_col]

            if pd.isna(smiles) or pd.isna(target):
                skipped_rows += 1
                continue

            try:
                datapoints.append(MoleculeDatapoint.from_smi(str(smiles), [float(target)]))
            except Exception:
                skipped_rows += 1

        print(
            f"Loaded {csv_path.name}: rows={len(df)} kept={len(df) - skipped_rows} "
            f"skipped={skipped_rows} target_col={target_col}"
        )

    if not datapoints:
        raise ValueError("No valid molecules were loaded. Check CSV files and target columns.")

    return datapoints


def load_data(
    train_csv_paths: list[Path],
    val_csv_path: Path,
    test_csv_path: Path | None,
    config: dict[str, Any],
) -> tuple[MoleculeDataset, MoleculeDataset, MoleculeDataset | None]:
    train_data = load_datapoints_from_csvs(train_csv_paths, config)
    val_data = load_datapoints_from_csvs([val_csv_path], config)
    test_data = load_datapoints_from_csvs([test_csv_path], config) if test_csv_path is not None else None

    # These datasets apply the same graph featurizer to all splits so train/val/test
    # use identical atom/bond graph feature construction during batching.
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dset = MoleculeDataset(train_data, featurizer=featurizer)
    val_dset = MoleculeDataset(val_data, featurizer=featurizer)
    test_dset = MoleculeDataset(test_data, featurizer=featurizer) if test_data is not None else None
    return train_dset, val_dset, test_dset


def build_fold_paths(ratio_dir: Path, val_idx: int, total_folds: int) -> tuple[list[Path], Path]:
    val_csv = ratio_dir / f"fold_{val_idx}.csv"
    train_csvs = [ratio_dir / f"fold_{i}.csv" for i in range(total_folds) if i != val_idx]

    synthetic_csv = ratio_dir / f"synthetic_data_iteration_{val_idx}.csv"
    if synthetic_csv.exists():
        train_csvs.append(synthetic_csv)
        print(f"Fold {val_idx}: synthetic augmentation enabled via {synthetic_csv.name}")
    else:
        print(f"Fold {val_idx}: no synthetic file, using only real fold CSVs")

    return train_csvs, val_csv


def build_mpnn_model(config: dict[str, Any]) -> models.MPNN:
    model_cfg = config["model"]
    optim_cfg = config["optimization"]

    d_h = int(model_cfg["d_h"])
    dropout = float(model_cfg["dropout"])
    ffn_hidden_dim = d_h * int(model_cfg["ffn_hidden_mult"])

    message_passing = nn.BondMessagePassing(
        d_h=d_h,
        depth=int(model_cfg["depth"]),
        dropout=dropout,
    )
    aggregation = nn.SumAggregation() if model_cfg["use_sum_aggregation"] else nn.MeanAggregation()
    predictor = nn.RegressionFFN(
        input_dim=d_h,
        hidden_dim=ffn_hidden_dim,
        n_layers=int(model_cfg["ffn_n_layers"]),
        dropout=dropout,
    )

    mpnn_kwargs: dict[str, Any] = {
        "message_passing": message_passing,
        "agg": aggregation,
        "predictor": predictor,
        "batch_norm": bool(model_cfg["batch_norm"]),
        "metrics": [RMSE(), R2Score()],
        "warmup_epochs": int(optim_cfg["warmup_epochs"]),
    }

    for lr_key in ["init_lr", "max_lr", "final_lr"]:
        lr_value = optim_cfg.get(lr_key)
        if lr_value is not None:
            mpnn_kwargs[lr_key] = float(lr_value)

    return models.MPNN(**mpnn_kwargs)


def _trainer_metric_dict(metric_output: list[dict[str, Any]]) -> dict[str, Any]:
    return metric_output[0] if metric_output else {}


def _safe_nanmean(values: list[float | None]) -> float:
    numeric_values = [np.nan if value is None else float(value) for value in values]
    return float(np.nanmean(numeric_values))


def _evaluate_best_checkpoint(
    checkpoint_path: str,
    dataloader,
    config: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    model = build_mpnn_model(config)

    # Torch 2.6 defaults torch.load(..., weights_only=True), which can fail for
    # Lightning checkpoints that include class references in metadata.
    # We trust checkpoints produced locally by this script, so we explicitly load
    # with weights_only=False and then run validate/test without ckpt_path restore.
    try:
        checkpoint_obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")

    state_dict = (
        checkpoint_obj["state_dict"]
        if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj
        else checkpoint_obj
    )
    model.load_state_dict(state_dict, strict=True)

    trainer = pl.Trainer(
        accelerator=config["training"]["accelerator"],
        logger=False,
        enable_checkpointing=False,
        deterministic=bool(config["training"]["deterministic"]),
        enable_progress_bar=False,
    )

    if mode == "validate":
        metrics = trainer.validate(model=model, dataloaders=dataloader, verbose=False)
    elif mode == "test":
        metrics = trainer.test(model=model, dataloaders=dataloader, verbose=False)
    else:
        raise ValueError(f"Unsupported evaluation mode: {mode}")

    return _trainer_metric_dict(metrics)


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
    train_dset, val_dset, test_dset = load_data(train_csv_paths, val_csv_path, effective_test_path, config)

    batch_size = int(config["data"]["batch_size"])
    train_loader = build_dataloader(train_dset, batch_size=batch_size, shuffle=True)
    val_loader = build_dataloader(val_dset, batch_size=batch_size, shuffle=False)
    test_loader = build_dataloader(test_dset, batch_size=batch_size, shuffle=False) if test_dset is not None else None

    model = build_mpnn_model(config)
    loss_history_callback = LossHistoryCallback()

    checkpoint_dir = ratio_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=f"fold_{val_idx}_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=float(config["training"]["min_delta"]),
        patience=int(config["training"]["patience"]),
        mode="min",
        verbose=False,
    )
    logger = CSVLogger(save_dir=str(ratio_dir / "lightning_logs"), name="chemprop_mpnn", version=f"fold_{val_idx}")

    trainer = pl.Trainer(
        max_epochs=int(config["training"]["max_epochs"]),
        accelerator=config["training"]["accelerator"],
        deterministic=bool(config["training"]["deterministic"]),
        enable_progress_bar=True,
        logger=logger,
        enable_checkpointing=True,
        callbacks=[early_stop, checkpoint_callback, loss_history_callback],
    )

    trainer.fit(model, train_loader, val_loader)

    best_ckpt_path = checkpoint_callback.best_model_path
    if not best_ckpt_path:
        raise RuntimeError(f"No best checkpoint produced for fold {val_idx}.")

    val_metrics_raw = _evaluate_best_checkpoint(best_ckpt_path, val_loader, config, mode="validate")
    val_loss = _extract_metric(val_metrics_raw, ["val_loss"], ["val_loss"])
    val_rmse = _extract_metric(val_metrics_raw, ["val_rmse"], ["rmse", "val/rmse"])
    val_r2 = _extract_metric(val_metrics_raw, ["val_r2"], ["r2", "val/r2"])

    test_metrics_raw: dict[str, Any] | None = None
    if should_test_each_fold and test_loader is not None:
        test_metrics_raw = _evaluate_best_checkpoint(best_ckpt_path, test_loader, config, mode="test")

    # Export lightweight state_dict for reuse without Lightning checkpoint metadata.
    best_pth_path = ratio_dir / f"best_model_iteration_{val_idx}.pth"
    try:
        checkpoint_obj = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint_obj = torch.load(best_ckpt_path, map_location="cpu")
    state_dict = checkpoint_obj["state_dict"] if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj else checkpoint_obj
    torch.save(state_dict, best_pth_path)

    return {
        "fold_idx": val_idx,
        "best_ckpt_path": best_ckpt_path,
        "best_pth_path": str(best_pth_path),
        "val_loss": val_loss,
        "val_rmse": val_rmse,
        "val_r2": val_r2,
        "val_metrics_raw": {k: _metric_to_float(v) for k, v in val_metrics_raw.items()},
        "test_metrics_raw": {k: _metric_to_float(v) for k, v in test_metrics_raw.items()} if test_metrics_raw else None,
        "train_losses": loss_history_callback.train_losses,
        "val_losses": loss_history_callback.val_losses,
    }


def _pick_best_fold(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    with_rmse = [fr for fr in fold_results if fr["val_rmse"] is not None]
    if with_rmse:
        return min(with_rmse, key=lambda fr: fr["val_rmse"])

    with_loss = [fr for fr in fold_results if fr["val_loss"] is not None]
    if with_loss:
        return min(with_loss, key=lambda fr: fr["val_loss"])

    raise RuntimeError("Unable to pick best fold: no fold has val_rmse or val_loss.")


def _write_losses_file(
    ratio_dir: Path,
    ratio_name: str,
    fold_results: list[dict[str, Any]],
) -> None:
    loss_path = ratio_dir / f"MPNN_losses_{ratio_name}.txt"
    with open(loss_path, "w", encoding="utf-8") as f:
        for fold_res in fold_results:
            f.write(f"fold_{fold_res['fold_idx']}\n")
            f.write("epoch,train_loss,val_loss\n")

            max_len = max(len(fold_res["train_losses"]), len(fold_res["val_losses"]))
            for epoch in range(max_len):
                train_loss = fold_res["train_losses"][epoch] if epoch < len(fold_res["train_losses"]) else ""
                val_loss = fold_res["val_losses"][epoch] if epoch < len(fold_res["val_losses"]) else ""
                f.write(f"{epoch + 1},{train_loss},{val_loss}\n")
            f.write("\n")

    print(f"Saved losses to: {loss_path}")


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
            f"Fold {val_idx} done | val_loss={fold_result['val_loss']} "
            f"val_rmse={fold_result['val_rmse']} val_r2={fold_result['val_r2']}"
        )

    best_fold = _pick_best_fold(fold_results)
    print(
        f"\nBest fold for {ratio_name}: fold_{best_fold['fold_idx']} "
        f"(val_rmse={best_fold['val_rmse']}, val_loss={best_fold['val_loss']})"
    )

    # Evaluate only the globally best fold on holdout, matching the original workflow.
    test_data = load_datapoints_from_csvs([heldout_path], config)
    test_featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    test_dset = MoleculeDataset(test_data, featurizer=test_featurizer)
    test_loader = build_dataloader(test_dset, batch_size=int(config["data"]["batch_size"]), shuffle=False)
    holdout_metrics_raw = _evaluate_best_checkpoint(
        checkpoint_path=best_fold["best_ckpt_path"],
        dataloader=test_loader,
        config=config,
        mode="test",
    )

    holdout_rmse = _extract_metric(holdout_metrics_raw, ["test_rmse"], ["rmse", "test/rmse"])
    holdout_r2 = _extract_metric(holdout_metrics_raw, ["test_r2"], ["r2", "test/r2"])
    holdout_loss = _extract_metric(
        holdout_metrics_raw,
        ["test_loss", "test_mse"],
        ["test_loss", "/loss", "_loss", "test/mse", "_mse"],
    )

    summary_rows: list[dict[str, Any]] = []
    for fr in fold_results:
        summary_rows.append(
            {
                "Stage": f"Fold_{fr['fold_idx']}_Val",
                "Loss": fr["val_loss"],
                "RMSE": fr["val_rmse"],
                "R2": fr["val_r2"],
            }
        )

    avg_loss = _safe_nanmean([fr["val_loss"] for fr in fold_results])
    avg_rmse = _safe_nanmean([fr["val_rmse"] for fr in fold_results])
    avg_r2 = _safe_nanmean([fr["val_r2"] for fr in fold_results])

    summary_rows.append(
        {
            "Stage": "Average_Val",
            "Loss": avg_loss,
            "RMSE": avg_rmse,
            "R2": avg_r2,
        }
    )
    summary_rows.append(
        {
            "Stage": f"Holdout_Test (Model from Fold_{best_fold['fold_idx']})",
            "Loss": holdout_loss,
            "RMSE": holdout_rmse,
            "R2": holdout_r2,
        }
    )

    summary_df = pd.DataFrame(summary_rows)
    results_path = ratio_dir / f"MPNN_results_{ratio_name}.csv"
    summary_df.to_csv(results_path, index=False)

    print("\nResult summary:")
    print(summary_df.to_string(index=False))
    print(f"Saved metrics to: {results_path}")

    if bool(config["output"]["save_loss_curves"]):
        _write_losses_file(ratio_dir, ratio_name, fold_results)

    return summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Chemprop MPNN with CV on ratio folders.")
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Run only one ratio folder (example: 0%, 33%, 67%).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Override config and run all ratio folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DEFAULT_CONFIG
    workspace_root = Path(__file__).resolve().parent

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
