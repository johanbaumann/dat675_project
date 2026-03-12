from __future__ import annotations

import argparse
import warnings
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
from rdkit import Chem, RDLogger
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score


# ==================== changelog ====================
# 2026-03-12
# - Replaced pseudocode with a runnable Chemprop MPNN training pipeline.
# - Added configurable 5-fold CV, fold summaries, and holdout evaluation.
# - Added robust target-column selection for real/synthetic CSVs.
# - Added checkpoint-safe loading for Torch 2.6+ (weights_only behavior).
# - Added sklearn RMSE/R2/Spearman metric tracking and per-epoch histories.
# - Added config-driven RDKit randomized-SMILES oversampling for train split.
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
        "batch_size": 64,
        "num_workers": 4,
        "real_target_column": "pIC50",
        "synthetic_target_column": "pred_pIC50",
        "fallback_target_columns": ["target_pIC50", "pred_pIC50", "pIC50"],
    },
    "oversampling": {
        "enabled": False,
        "duplicates_per_smiles": 2,
        "max_tries_per_duplicate": 8,
    },
    "model": {
        "d_h": 256,
        "depth": 6,
        "dropout": 0.2,
        "ffn_hidden_mult": 2,
        "ffn_n_layers": 3,
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
    "runtime": {
        "suppress_python_warnings": True,
        "suppress_rdkit_warnings": True,
    },
    "output": {
        "save_loss_curves": True,
    },
}


def set_global_seed(seed: int) -> None:
    pl.seed_everything(seed, workers=True)


def configure_runtime(config: dict[str, Any]) -> None:
    runtime_cfg = config.get("runtime", {})
    if bool(runtime_cfg.get("suppress_python_warnings", True)):
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
        warnings.filterwarnings("ignore", category=UserWarning, module="lightning")
        warnings.filterwarnings("ignore", category=UserWarning, module="torch")
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    if bool(runtime_cfg.get("suppress_rdkit_warnings", True)):
        try:
            RDLogger.DisableLog("rdApp.*")
        except Exception:
            pass


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    return None


def _safe_nanmean(values: list[float | None]) -> float:
    numeric = [np.nan if value is None else float(value) for value in values]
    return float(np.nanmean(numeric))


def _extract_metric(metrics: dict[str, Any], preferred: list[str], contains_any: list[str]) -> float | None:
    for key in preferred:
        if key in metrics:
            value = _to_float(metrics[key])
            if value is not None:
                return value
    for key, raw in metrics.items():
        if any(fragment in str(key).lower() for fragment in contains_any):
            value = _to_float(raw)
            if value is not None:
                return value
    return None


def choose_target_column(df: pd.DataFrame, csv_path: Path, config: dict[str, Any]) -> str:
    data_cfg = config["data"]
    is_synthetic = csv_path.name.startswith("synthetic_data_iteration_")
    if is_synthetic:
        priorities = [
            data_cfg["synthetic_target_column"],
            *data_cfg["fallback_target_columns"],
            data_cfg["real_target_column"],
        ]
    else:
        priorities = [data_cfg["real_target_column"], *data_cfg["fallback_target_columns"]]
    for col in priorities:
        if col in df.columns:
            return col
    raise ValueError(f"No supported target column found in {csv_path}. Tried: {priorities}")


def _randomized_smiles_variants(smiles: str, duplicates: int, max_tries_per_dup: int) -> list[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or duplicates <= 0:
        return []
    variants: list[str] = []
    seen = {smiles}
    max_tries = max(duplicates * max_tries_per_dup, duplicates)
    for _ in range(max_tries):
        if len(variants) >= duplicates:
            break
        try:
            candidate = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        except Exception:
            continue
        if candidate and candidate not in seen:
            seen.add(candidate)
            variants.append(candidate)
    return variants


def load_datapoints_from_csvs(
    csv_paths: list[Path],
    config: dict[str, Any],
    augment_train_smiles: bool = False,
) -> list[MoleculeDatapoint]:
    over_cfg = config.get("oversampling", {})
    use_oversampling = bool(over_cfg.get("enabled", False)) and augment_train_smiles
    duplicates = int(over_cfg.get("duplicates_per_smiles", 0))
    max_tries_per_dup = int(over_cfg.get("max_tries_per_duplicate", 8))

    datapoints: list[MoleculeDatapoint] = []
    total_aug_added = 0

    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing expected file: {csv_path}")

        df = pd.read_csv(csv_path)
        if "smiles" not in df.columns:
            raise ValueError(f"Missing required 'smiles' column in {csv_path}")

        target_col = choose_target_column(df, csv_path, config)
        skipped = 0
        aug_added = 0

        for _, row in df.iterrows():
            smiles = row["smiles"]
            target = row[target_col]
            if pd.isna(smiles) or pd.isna(target):
                skipped += 1
                continue

            base_smiles = str(smiles)
            try:
                target_value = float(target)
                datapoints.append(MoleculeDatapoint.from_smi(base_smiles, [target_value]))
            except Exception:
                skipped += 1
                continue

            if use_oversampling and duplicates > 0:
                for aug_smiles in _randomized_smiles_variants(base_smiles, duplicates, max_tries_per_dup):
                    try:
                        datapoints.append(MoleculeDatapoint.from_smi(aug_smiles, [target_value]))
                        aug_added += 1
                    except Exception:
                        continue

        total_aug_added += aug_added
        print(
            f"Loaded {csv_path.name}: rows={len(df)} kept={len(df) - skipped} "
            f"skipped={skipped} target_col={target_col} aug_added={aug_added}"
        )

    if not datapoints:
        raise ValueError("No valid molecules were loaded. Check CSV files and target columns.")
    if use_oversampling and duplicates > 0:
        print(f"Oversampling summary: total_augmented_smiles_added={total_aug_added}")
    return datapoints


def load_data(
    train_csv_paths: list[Path],
    val_csv_path: Path,
    test_csv_path: Path | None,
    config: dict[str, Any],
) -> tuple[MoleculeDataset, MoleculeDataset, MoleculeDataset | None]:
    train_data = load_datapoints_from_csvs(train_csv_paths, config, augment_train_smiles=True)
    val_data = load_datapoints_from_csvs([val_csv_path], config, augment_train_smiles=False)
    test_data = (
        load_datapoints_from_csvs([test_csv_path], config, augment_train_smiles=False)
        if test_csv_path is not None
        else None
    )
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


def _build_loader(dataset: MoleculeDataset, config: dict[str, Any], shuffle: bool):
    data_cfg = config["data"]
    return build_dataloader(
        dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0)),
    )


def build_mpnn_model(config: dict[str, Any]) -> models.MPNN:
    model_cfg, optim_cfg = config["model"], config["optimization"]
    d_h = int(model_cfg["d_h"])
    dropout = float(model_cfg["dropout"])
    message_passing = nn.BondMessagePassing(d_h=d_h, depth=int(model_cfg["depth"]), dropout=dropout)
    aggregation = nn.SumAggregation() if model_cfg["use_sum_aggregation"] else nn.MeanAggregation()
    predictor = nn.RegressionFFN(
        input_dim=d_h,
        hidden_dim=d_h * int(model_cfg["ffn_hidden_mult"]),
        n_layers=int(model_cfg["ffn_n_layers"]),
        dropout=dropout,
    )

    kwargs: dict[str, Any] = {
        "message_passing": message_passing,
        "agg": aggregation,
        "predictor": predictor,
        "batch_norm": bool(model_cfg["batch_norm"]),
        "metrics": [RMSE(), R2Score()],
        "warmup_epochs": int(optim_cfg["warmup_epochs"]),
    }
    for key in ["init_lr", "max_lr", "final_lr"]:
        value = optim_cfg.get(key)
        if value is not None:
            kwargs[key] = float(value)
    return models.MPNN(**kwargs)


def _flatten_prediction_output(pred_output) -> list[float]:
    if isinstance(pred_output, torch.Tensor):
        return [float(v) for v in pred_output.detach().cpu().reshape(-1).tolist()]
    if isinstance(pred_output, (list, tuple)):
        values: list[float] = []
        for item in pred_output:
            values.extend(_flatten_prediction_output(item))
        return values
    raise TypeError(f"Unsupported prediction output type: {type(pred_output)}")


def _extract_targets_from_dataloader(dataloader) -> list[float]:
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        raise ValueError("Dataloader has no dataset attribute; cannot extract targets.")
    targets: list[float] = []
    for datum in dataset:
        y_value = getattr(datum, "y", None)
        if y_value is None or len(y_value) == 0:
            raise ValueError("Encountered datapoint without target 'y'.")
        targets.append(float(y_value[0]))
    return targets


def _compute_regression_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    if len(y_true) < 2:
        r2 = float("nan")
    else:
        try:
            r2 = float(r2_score(y_true, y_pred))
        except ValueError:
            r2 = float("nan")

    if len(y_true) < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        rho = float("nan")
    else:
        rho_raw = spearmanr(y_true, y_pred)[0]
        rho = float(rho_raw) if rho_raw is not None and not np.isnan(rho_raw) else float("nan")

    return {"mse": mse, "rmse": rmse, "r2": r2, "spearman": rho}


def _predict_with_lightning_module(pl_module: pl.LightningModule, dataloader) -> list[float]:
    was_training = pl_module.training
    predictions: list[float] = []
    pl_module.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = pl_module.transfer_batch_to_device(batch, pl_module.device, 0)
            batch_pred = pl_module.predict_step(batch, batch_idx)
            predictions.extend(_flatten_prediction_output(batch_pred))
    if was_training:
        pl_module.train()
    return predictions


def _resolve_single_dataloader(dataloaders):
    if isinstance(dataloaders, (list, tuple)):
        return dataloaders[0] if len(dataloaders) > 0 else None
    return dataloaders


class LossHistoryCallback(Callback):
    """Collect per-epoch train/validation losses and sklearn validation metrics."""

    def __init__(self) -> None:
        super().__init__()
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.val_mse_history: list[float] = []
        self.val_rmse_history: list[float] = []
        self.val_r2_history: list[float] = []
        self.val_spearman_history: list[float] = []
        self._cached_val_targets: list[float] | None = None

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        value = _extract_metric(trainer.callback_metrics, ["train_loss", "loss"], ["train_loss"])
        if value is not None:
            self.train_losses.append(value)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return

        value = _extract_metric(trainer.callback_metrics, ["val_loss"], ["val_loss"])
        if value is not None:
            self.val_losses.append(value)

        val_loader = _resolve_single_dataloader(trainer.val_dataloaders)
        if val_loader is None:
            return

        if self._cached_val_targets is None:
            self._cached_val_targets = _extract_targets_from_dataloader(val_loader)

        y_pred = _predict_with_lightning_module(pl_module, val_loader)
        if len(y_pred) != len(self._cached_val_targets):
            return

        metrics = _compute_regression_metrics(self._cached_val_targets, y_pred)
        self.val_mse_history.append(metrics["mse"])
        self.val_rmse_history.append(metrics["rmse"])
        self.val_r2_history.append(metrics["r2"])
        self.val_spearman_history.append(metrics["spearman"])


def _load_state_dict_from_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    try:
        checkpoint_obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        return checkpoint_obj["state_dict"]
    return checkpoint_obj


def _evaluate_best_checkpoint(checkpoint_path: str, dataloader, config: dict[str, Any]) -> dict[str, Any]:
    model = build_mpnn_model(config)
    model.load_state_dict(_load_state_dict_from_checkpoint(checkpoint_path), strict=True)

    trainer = pl.Trainer(
        accelerator=config["training"]["accelerator"],
        logger=False,
        enable_checkpointing=False,
        deterministic=bool(config["training"]["deterministic"]),
        enable_progress_bar=False,
    )
    pred_batches = trainer.predict(model=model, dataloaders=dataloader)
    y_pred = [value for batch in pred_batches for value in _flatten_prediction_output(batch)]
    y_true = _extract_targets_from_dataloader(dataloader)
    if len(y_true) != len(y_pred):
        raise RuntimeError(
            f"Prediction/target length mismatch ({len(y_pred)} vs {len(y_true)}) while evaluating {checkpoint_path}."
        )

    metrics = _compute_regression_metrics(y_true, y_pred)
    metrics["n_samples"] = float(len(y_true))
    return metrics


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
    train_loader = _build_loader(train_dset, config, shuffle=True)
    val_loader = _build_loader(val_dset, config, shuffle=False)
    test_loader = _build_loader(test_dset, config, shuffle=False) if test_dset is not None else None

    model = build_mpnn_model(config)
    history_cb = LossHistoryCallback()

    checkpoint_dir = ratio_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
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
    logger = CSVLogger(
        save_dir=str(ratio_dir / "lightning_logs"),
        name="chemprop_mpnn",
        version=f"fold_{val_idx}",
    )

    trainer = pl.Trainer(
        max_epochs=int(config["training"]["max_epochs"]),
        accelerator=config["training"]["accelerator"],
        deterministic=bool(config["training"]["deterministic"]),
        enable_progress_bar=True,
        logger=logger,
        enable_checkpointing=True,
        callbacks=[early_stop, checkpoint_cb, history_cb],
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
        "best_ckpt_path": best_ckpt_path,
        "best_pth_path": str(best_pth_path),
        "val_loss": _to_float(checkpoint_cb.best_model_score),
        "val_mse": float(val_metrics["mse"]),
        "val_rmse": float(val_metrics["rmse"]),
        "val_r2": float(val_metrics["r2"]),
        "val_spearman": float(val_metrics["spearman"]),
        "val_metrics_sklearn": val_metrics,
        "test_metrics_sklearn": test_metrics,
        "train_losses": history_cb.train_losses,
        "val_losses": history_cb.val_losses,
        "val_mse_history": history_cb.val_mse_history,
        "val_rmse_history": history_cb.val_rmse_history,
        "val_r2_history": history_cb.val_r2_history,
        "val_spearman_history": history_cb.val_spearman_history,
    }


def _pick_best_fold(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    with_rmse = [fr for fr in fold_results if fr["val_rmse"] is not None]
    if with_rmse:
        return min(with_rmse, key=lambda fr: fr["val_rmse"])
    with_loss = [fr for fr in fold_results if fr["val_loss"] is not None]
    if with_loss:
        return min(with_loss, key=lambda fr: fr["val_loss"])
    raise RuntimeError("Unable to pick best fold: no fold has val_rmse or val_loss.")


def _write_losses_file(ratio_dir: Path, ratio_name: str, fold_results: list[dict[str, Any]]) -> None:
    loss_path = ratio_dir / f"MPNN_losses_{ratio_name}.txt"
    with open(loss_path, "w", encoding="utf-8") as f:
        for fold_res in fold_results:
            f.write(f"fold_{fold_res['fold_idx']}\n")
            f.write("epoch,train_loss,val_loss,val_mse,val_rmse,val_r2,val_spearman\n")
            max_len = max(
                len(fold_res["train_losses"]),
                len(fold_res["val_losses"]),
                len(fold_res.get("val_mse_history", [])),
                len(fold_res.get("val_rmse_history", [])),
                len(fold_res.get("val_r2_history", [])),
                len(fold_res.get("val_spearman_history", [])),
            )
            for epoch in range(max_len):
                train_loss = fold_res["train_losses"][epoch] if epoch < len(fold_res["train_losses"]) else ""
                val_loss = fold_res["val_losses"][epoch] if epoch < len(fold_res["val_losses"]) else ""
                val_mse = fold_res["val_mse_history"][epoch] if epoch < len(fold_res["val_mse_history"]) else ""
                val_rmse = fold_res["val_rmse_history"][epoch] if epoch < len(fold_res["val_rmse_history"]) else ""
                val_r2 = fold_res["val_r2_history"][epoch] if epoch < len(fold_res["val_r2_history"]) else ""
                val_rho = fold_res["val_spearman_history"][epoch] if epoch < len(fold_res["val_spearman_history"]) else ""
                f.write(f"{epoch + 1},{train_loss},{val_loss},{val_mse},{val_rmse},{val_r2},{val_rho}\n")
            f.write(
                "final_summary,"
                f"{''},{fold_res.get('val_loss', '')},{fold_res.get('val_mse', '')},"
                f"{fold_res.get('val_rmse', '')},{fold_res.get('val_r2', '')},{fold_res.get('val_spearman', '')}\n\n"
            )
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
            f"Fold {val_idx} done | best_val_loss={fold_result['val_loss']} "
            f"val_mse={fold_result['val_mse']} val_rmse={fold_result['val_rmse']} "
            f"val_r2={fold_result['val_r2']} val_spearman={fold_result['val_spearman']}"
        )

    best_fold = _pick_best_fold(fold_results)
    print(
        f"\nBest fold for {ratio_name}: fold_{best_fold['fold_idx']} "
        f"(val_rmse={best_fold['val_rmse']}, best_val_loss={best_fold['val_loss']})"
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
            "ValLoss": fr["val_loss"],
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
            "ValLoss": _safe_nanmean([fr["val_loss"] for fr in fold_results]),
            "MSE": _safe_nanmean([fr["val_mse"] for fr in fold_results]),
            "RMSE": _safe_nanmean([fr["val_rmse"] for fr in fold_results]),
            "R2": _safe_nanmean([fr["val_r2"] for fr in fold_results]),
            "Rho": _safe_nanmean([fr["val_spearman"] for fr in fold_results]),
        }
    )
    summary_rows.append(
        {
            "Stage": f"Holdout_Test (Model from Fold_{best_fold['fold_idx']})",
            "ValLoss": "",
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
