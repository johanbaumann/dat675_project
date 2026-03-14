from __future__ import annotations

"""Utility functions and callbacks for the Chemprop MPNN training pipeline.

This module centralizes shared logic used by the training entrypoint:
- Runtime setup and seeding
- Data loading / target-column selection / optional SMILES oversampling
- Model construction
- Metric extraction and best-checkpoint evaluation
- Lightning callbacks (loss history and ReduceLROnPlateau)
- Fold selection and loss-file export

Refactoring goal: keep the main training script focused on configuration and
CV loop orchestration while preserving behavior.
"""

import warnings
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from chemprop import featurizers, models, nn
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.nn.metrics import R2Score, RMSE
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from rdkit import Chem, RDLogger
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score


METRIC_DESCRIPTIONS: dict[str, str] = {
    "val_loss": "Chemprop validation optimization loss (MSE on target scale; lower is better).",
    "val/rmse": "Chemprop validation RMSE metric (sqrt of MSE; lower is better).",
    "val/r2": "Chemprop validation R2 metric (higher is better).",
    "train_loss": "Chemprop training optimization loss (typically batch/epoch MSE).",
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
            disable_log = getattr(RDLogger, "DisableLog", None)
            if callable(disable_log):
                disable_log("rdApp.*")
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
    _, value = _extract_metric_with_key(metrics, preferred, contains_any)
    return value


def _extract_metric_with_key(
    metrics: dict[str, Any],
    preferred: list[str],
    contains_any: list[str],
) -> tuple[str | None, float | None]:
    for key in preferred:
        if key in metrics:
            value = _to_float(metrics[key])
            if value is not None:
                return str(key), value
    lowered_fragments = [fragment.lower() for fragment in contains_any]
    for key, raw in metrics.items():
        if any(fragment in str(key).lower() for fragment in lowered_fragments):
            value = _to_float(raw)
            if value is not None:
                return str(key), value
    return None, None


def _metric_description(metric_name: str) -> str:
    return METRIC_DESCRIPTIONS.get(metric_name, "Custom metric from Lightning/Chemprop logger.")


def _format_metric(value: float | None) -> str:
    if value is None:
        return "nan"
    if isinstance(value, float) and np.isnan(value):
        return "nan"
    return f"{value:.6f}"


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
    """
    The purpose of this function is to generate randomized SMILES variants for a given input SMILES string.
    It uses RDKit's ability to produce non-canonical SMILES by randomizing the atom order.
    This can help using data augmentation.

    """
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
                datapoints.append(cast(MoleculeDatapoint, MoleculeDatapoint.from_smi(base_smiles, [target_value])))
            except Exception:
                skipped += 1
                continue

            if use_oversampling and duplicates > 0:
                for aug_smiles in _randomized_smiles_variants(base_smiles, duplicates, max_tries_per_dup):
                    try:
                        datapoints.append(cast(MoleculeDatapoint, MoleculeDatapoint.from_smi(aug_smiles, [target_value])))
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

    """
    Builds an chemprop MPNN, model based on provided config.

    Bondmessage passing is used as the message passing mechanism, with configurable depth and dropout.
    Sum aggregation is used by default.


    """

    message_passing = nn.BondMessagePassing(d_h=d_h, depth=int(model_cfg["depth"]), dropout=dropout)  # type: ignore[abstract]
    aggregation = nn.SumAggregation() if model_cfg["use_sum_aggregation"] else nn.MeanAggregation()
    predictor = nn.RegressionFFN(input_dim=d_h, hidden_dim=d_h * int(model_cfg["ffn_hidden_mult"]), n_layers=int(model_cfg["ffn_n_layers"]), dropout=dropout)  # type: ignore[abstract]

    # Kwars used to int the MPNN.
    # it includes the:
    # Message passing modules
    # Aggregation mechanism
    # Predictor architecture (Regression FFNN in this case)
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
        rho_result = spearmanr(y_true, y_pred)
        rho_candidate = getattr(rho_result, "statistic", None)
        if rho_candidate is None and isinstance(rho_result, (tuple, list)) and len(rho_result) > 0:
            rho_candidate = rho_result[0]

        if rho_candidate is None or not isinstance(rho_candidate, (int, float, np.integer, np.floating)):
            rho = float("nan")
        else:
            rho_value = float(rho_candidate)
            rho = rho_value if not np.isnan(rho_value) else float("nan")

    return {"mse": mse, "rmse": rmse, "r2": r2, "spearman": rho}


class LossHistoryCallback(Callback):
    """Collect per-epoch train/validation losses and Chemprop validation metrics."""

    def __init__(self, monitor_metric: str, verbose_metric_logging: bool) -> None:
        super().__init__()
        self.monitor_metric = monitor_metric
        self.verbose_metric_logging = verbose_metric_logging

        self.train_losses: list[float] = []
        self.monitor_values: list[float] = []
        self.monitor_metric_key_history: list[str] = []

        self.val_losses: list[float] = []
        self.val_loss_metric_key_history: list[str] = []
        self.chemprop_val_rmse_history: list[float] = []
        self.chemprop_val_r2_history: list[float] = []

        self._metric_keys_reported = False

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        value = _extract_metric(
            trainer.callback_metrics,
            ["train_loss_epoch", "train_loss", "loss"],
            ["train_loss_epoch", "train_loss"],
        )
        if value is not None:
            self.train_losses.append(value)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return

        callback_metrics = trainer.callback_metrics

        if self.verbose_metric_logging and not self._metric_keys_reported:
            val_keys = sorted([str(k) for k in callback_metrics if "val" in str(k).lower()])
            if val_keys:
                print(f"Validation metrics seen in Lightning callback_metrics: {val_keys}")
            print(
                f"Monitoring metric '{self.monitor_metric}' -> {_metric_description(self.monitor_metric)}"
            )
            self._metric_keys_reported = True

        monitor_key, monitor_value = _extract_metric_with_key(
            callback_metrics,
            [self.monitor_metric],
            [self.monitor_metric],
        )
        self.monitor_metric_key_history.append(monitor_key or "")
        self.monitor_values.append(float("nan") if monitor_value is None else monitor_value)

        val_loss_key, val_loss_value = _extract_metric_with_key(callback_metrics, ["val_loss"], ["val_loss"])
        self.val_loss_metric_key_history.append(val_loss_key or "")
        self.val_losses.append(float("nan") if val_loss_value is None else val_loss_value)

        chemprop_rmse = _extract_metric(callback_metrics, ["val/rmse"], ["val/rmse", "rmse"])
        chemprop_r2 = _extract_metric(callback_metrics, ["val/r2"], ["val/r2", "r2"])
        self.chemprop_val_rmse_history.append(float("nan") if chemprop_rmse is None else chemprop_rmse)
        self.chemprop_val_r2_history.append(float("nan") if chemprop_r2 is None else chemprop_r2)

        if self.verbose_metric_logging:
            print(
                f"[Fold val epoch {trainer.current_epoch}] "
                f"monitor({monitor_key or self.monitor_metric})={_format_metric(monitor_value)} | "
                f"val_loss(MSE)={_format_metric(val_loss_value)} | "
                f"chemprop_val_rmse={_format_metric(chemprop_rmse)}"
            )


class ReduceLROnPlateauCallback(Callback):
    """Apply ReduceLROnPlateau to the first optimizer using a validation monitor metric."""

    def __init__(
        self,
        monitor_metric: str,
        monitor_mode: str,
        scheduler_cfg: dict[str, Any],
        verbose_metric_logging: bool,
    ) -> None:
        super().__init__()
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.scheduler_cfg = scheduler_cfg
        self.verbose_metric_logging = verbose_metric_logging

        self.enabled = bool(scheduler_cfg.get("enabled", False))
        self.lr_history: list[float] = []

        self._optimizer = None
        self._scheduler: ReduceLROnPlateau | None = None
        self._warned_missing_metric = False

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.enabled or self._scheduler is not None:
            return
        if not trainer.optimizers:
            print("ReduceLROnPlateau disabled: trainer has no optimizers.")
            return

        self._optimizer = trainer.optimizers[0]
        mode_candidate = str(self.scheduler_cfg.get("mode") or self.monitor_mode).lower()
        mode: Literal["min", "max"] = "max" if mode_candidate == "max" else "min"

        min_lr_cfg = self.scheduler_cfg.get("min_lr", 1e-6)
        if isinstance(min_lr_cfg, (list, tuple)):
            min_lr: float | list[float] = [float(v) for v in min_lr_cfg]
        else:
            min_lr = float(min_lr_cfg)

        threshold_mode_candidate = str(self.scheduler_cfg.get("threshold_mode", "rel")).lower()
        threshold_mode: Literal["rel", "abs"] = "abs" if threshold_mode_candidate == "abs" else "rel"

        self._scheduler = ReduceLROnPlateau(
            optimizer=self._optimizer,
            mode=mode,
            factor=float(self.scheduler_cfg.get("factor", 0.5)),
            patience=int(self.scheduler_cfg.get("patience", 2)),
            threshold=float(self.scheduler_cfg.get("threshold", 1e-4)),
            threshold_mode=threshold_mode,
            cooldown=int(self.scheduler_cfg.get("cooldown", 0)),
            min_lr=min_lr,
            eps=float(self.scheduler_cfg.get("eps", 1e-8)),
        )

        if self.verbose_metric_logging:
            print(
                "ReduceLROnPlateau enabled "
                f"(monitor={self.monitor_metric}, mode={mode}, "
                f"factor={self.scheduler_cfg.get('factor', 0.5)}, "
                f"patience={self.scheduler_cfg.get('patience', 2)})"
            )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking or self._scheduler is None or self._optimizer is None:
            return

        _, monitor_value = _extract_metric_with_key(
            trainer.callback_metrics,
            [self.monitor_metric],
            [self.monitor_metric],
        )
        if monitor_value is None:
            if self.verbose_metric_logging and not self._warned_missing_metric:
                print(
                    "ReduceLROnPlateau step skipped: "
                    f"monitor metric '{self.monitor_metric}' not found in callback_metrics."
                )
                self._warned_missing_metric = True
            return

        before_lrs = [float(pg.get("lr", 0.0)) for pg in self._optimizer.param_groups]
        self._scheduler.step(monitor_value)
        after_lrs = [float(pg.get("lr", 0.0)) for pg in self._optimizer.param_groups]

        if after_lrs:
            self.lr_history.append(after_lrs[0])

        if (
            self.verbose_metric_logging
            and before_lrs
            and after_lrs
            and not np.isclose(before_lrs[0], after_lrs[0])
        ):
            print(
                f"ReduceLROnPlateau: lr reduced from {before_lrs[0]:.6e} to {after_lrs[0]:.6e} "
                f"at epoch={trainer.current_epoch} using {self.monitor_metric}={_format_metric(monitor_value)}"
            )


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
    if pred_batches is None:
        raise RuntimeError("Prediction returned no batches while evaluating best checkpoint.")
    y_pred = [value for batch in pred_batches for value in _flatten_prediction_output(batch)]
    y_true = _extract_targets_from_dataloader(dataloader)
    if len(y_true) != len(y_pred):
        raise RuntimeError(
            f"Prediction/target length mismatch ({len(y_pred)} vs {len(y_true)}) while evaluating {checkpoint_path}."
        )

    metrics = _compute_regression_metrics(y_true, y_pred)
    metrics["n_samples"] = float(len(y_true))
    return metrics


def _pick_best_fold(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    with_rmse = [fr for fr in fold_results if fr["val_rmse"] is not None]
    if with_rmse:
        return min(with_rmse, key=lambda fr: fr["val_rmse"])
    with_monitor = [fr for fr in fold_results if fr.get("best_monitor_value") is not None]
    if with_monitor:
        mode = str(with_monitor[0].get("monitor_mode", "min")).lower()
        if mode == "max":
            return max(with_monitor, key=lambda fr: fr["best_monitor_value"])
        return min(with_monitor, key=lambda fr: fr["best_monitor_value"])
    raise RuntimeError("Unable to pick best fold: no fold has val_rmse or val_loss.")


def _write_losses_file(ratio_dir: Path, ratio_name: str, fold_results: list[dict[str, Any]]) -> None:
    loss_path = ratio_dir / f"MPNN_losses_{ratio_name}.txt"
    with open(loss_path, "w", encoding="utf-8") as f:
        for fold_res in fold_results:
            f.write(f"fold_{fold_res['fold_idx']}\n")
            f.write(
                "epoch,train_loss_epoch,monitor_metric,monitor_value,lr,val_loss_key,val_loss_mse,"
                "chemprop_val_rmse,chemprop_val_r2\n"
            )
            max_len = max(
                len(fold_res["train_losses"]),
                len(fold_res.get("monitor_values", [])),
                len(fold_res.get("lr_history", [])),
                len(fold_res["val_losses"]),
                len(fold_res.get("chemprop_val_rmse_history", [])),
                len(fold_res.get("chemprop_val_r2_history", [])),
            )
            for epoch in range(max_len):
                train_loss = fold_res["train_losses"][epoch] if epoch < len(fold_res["train_losses"]) else ""
                monitor_metric = fold_res.get("monitor_metric", "")
                monitor_value = fold_res["monitor_values"][epoch] if epoch < len(fold_res.get("monitor_values", [])) else ""
                lr_value = fold_res["lr_history"][epoch] if epoch < len(fold_res.get("lr_history", [])) else ""
                val_loss_key = (
                    fold_res["val_loss_metric_key_history"][epoch]
                    if epoch < len(fold_res.get("val_loss_metric_key_history", []))
                    else ""
                )
                val_loss = fold_res["val_losses"][epoch] if epoch < len(fold_res["val_losses"]) else ""
                chemprop_rmse = (
                    fold_res["chemprop_val_rmse_history"][epoch]
                    if epoch < len(fold_res.get("chemprop_val_rmse_history", []))
                    else ""
                )
                chemprop_r2 = (
                    fold_res["chemprop_val_r2_history"][epoch]
                    if epoch < len(fold_res.get("chemprop_val_r2_history", []))
                    else ""
                )
                f.write(
                    f"{epoch + 1},{train_loss},{monitor_metric},{monitor_value},{lr_value},{val_loss_key},{val_loss},"
                    f"{chemprop_rmse},{chemprop_r2}\n"
                )
            f.write(
                "final_summary,"
                f"{''},{fold_res.get('monitor_metric', '')},{fold_res.get('best_monitor_value', '')},"
                f"best_ckpt_metrics,{fold_res.get('val_loss_mse', '')},{fold_res.get('val_rmse', '')},"
                f"{fold_res.get('val_r2', '')}\n\n"
            )
    print(f"Saved losses to: {loss_path}")


__all__ = [
    "LossHistoryCallback",
    "ReduceLROnPlateauCallback",
    "build_fold_paths",
    "build_mpnn_model",
    "configure_runtime",
    "load_data",
    "load_datapoints_from_csvs",
    "set_global_seed",
    "_build_loader",
    "_evaluate_best_checkpoint",
    "_load_state_dict_from_checkpoint",
    "_metric_description",
    "_pick_best_fold",
    "_safe_nanmean",
    "_to_float",
    "_write_losses_file",
]
