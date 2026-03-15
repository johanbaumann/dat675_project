"""GAT holdout evaluation script.

For every dataset folder that has saved fold checkpoints this script loads
the configured checkpoint for each fold, evaluates it on the held-out test set, and writes
all results to gat_results_heldout.csv in the workspace root (configurable via
OUTPUT_FILENAME below or the --output CLI flag).

This script is inference-only - it does not train any models.
Training is done separately via gat_predictor.py (one run per dataset).

Usage
-----
    # Evaluate all datasets that have checkpoints:
    python run_gat.py

    # Evaluate a single dataset only:
    python run_gat.py --folder 0%

    # Override the holdout CSV path:
    python run_gat.py --test-file path/to/heldout_testset.csv

Output
------
    gat_results_heldout.csv  (in the same folder as this script; change OUTPUT_FILENAME
    or pass --output to override)

    Columns: Dataset | Fold | Checkpoint | Val_RMSE | Val_MAE | Val_Pearson |
             Holdout_MSE | Holdout_RMSE | Holdout_MAE | Holdout_R2 |
             Holdout_Rho | Holdout_Pearson

    A second file, GAT_predictions_heldout_set.csv, contains per-molecule
    True_pIC50 vs Pred_pIC50 for every fold * dataset combination.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch
import pandas as pd
from torch_geometric.loader import DataLoader

# Import the canonical CONFIG and GATModel from the training entry point.
# The module-level code in MPNN_predictor only defines CONFIG + GATModel;
# training is guarded by 'if __name__ == "__main__"', so this import is safe.
from gat_predictor import CONFIG, GATModel
from gat_utils.checkpoints import load_fold_checkpoint, resolve_checkpoint_path
from gat_utils.data_loading import (
    cap_synthetic_train_ratio,
    get_fold_files,
    get_synthetic_cv_config,
    load_dataset,
)
from gat_utils.features import apply_feature_scalers, fit_feature_scalers, prepare_feature_config
from gat_utils.training_helpers import evaluate, predict

WORKSPACE_ROOT = Path(__file__).resolve().parent
DATASETS = ["0%", "33%", "67%"]

# Change this string to rename the output file, or pass --output at the command line.
OUTPUT_FILENAME = "gat_results_heldout.csv"

OUTPUT_CSV = WORKSPACE_ROOT / OUTPUT_FILENAME
OUTPUT_PRED_CSV = WORKSPACE_ROOT / "GAT_predictions_heldout_set.csv"

# Optional per-dataset/per-fold checkpoint overrides.
# Set a value to either:
# - None: use the default best_model_fold_<k>.pth checkpoint
# - an int epoch shorthand, e.g. 22 -> model_<dataset>_cv_iteration_<fold>_epoch_22.pth
# - a filename inside that dataset's checkpoints/ folder, e.g.
#   "model_0_percent_cv_iteration_0_epoch_25.pth"
# - a relative path from the workspace root
# - an absolute path
#CHECKPOINT_SELECTIONS: dict[str, dict[int, int | str | None]] = {
#    "0%": {
#        0: 32,
#        1: 30,
#        2: 25,
#        3: 30,
#        4: 20,
#    },
#    "33%": {
#        0: 25,
#        1: 35,
#        2: 21,
#        3: 22,
#        4: 15,
#    },
#    "67%": {
#        0: 19,
#        1: 16,
#        2: 28,
#        3: 30,
#        4: 20,
#    },
#}

CHECKPOINT_SELECTIONS: dict[str, dict[int, int | str | None]] = {
    "0%": {
        0: None,
        1: None,
        2: None,
        3: None,
        4: None,
    },
    "33%": {
        0: None,
        1: None,
        2: None,
        3: None,
        4: None,
    },
    "67%": {
        0: None,
        1: None,
        2: None,
        3: None,
        4: None,
    },
}

# ==================== helpers ====================

def _load_holdout_smiles_df(test_path: Path, config: dict[str, Any]) -> pd.DataFrame | None:
    """Return the holdout CSV rows that load_dataset would process, in order.

    Only the same NaN filter applied by load_dataset is replicated here
    (rows where 'smiles' or the real target column is NaN are dropped).
    The caller should verify that len(returned_df) == len(test_data) before
    trusting SMILES alignment — a mismatch means at least one SMILES failed
    smiles_to_graph() and was silently dropped by load_dataset.
    """
    real_target_col = config["data"]["real_target_column"]
    try:
        df = pd.read_csv(test_path)
        needed = ["smiles", real_target_col]
        if not all(col in df.columns for col in needed):
            return None
        df = df[df["smiles"].notna() & df[real_target_col].notna()].reset_index(drop=True)
        return df
    except Exception:
        return None


def _resolve_selected_checkpoint(dataset_label: str, fold_idx: int, target_folder: str) -> Path:
    dataset_map: dict[int, int | str | None] = CHECKPOINT_SELECTIONS.get(dataset_label, {})
    selected_checkpoint = dataset_map.get(fold_idx)
    return resolve_checkpoint_path(
        target_folder,
        fold_idx,
        selected_checkpoint,
        workspace_root=WORKSPACE_ROOT,
    )


# ==================== per-dataset evaluation ====================

def evaluate_dataset(
    dataset_label: str,
    base_config: dict[str, Any],
    test_path: Path,
    holdout_smiles_df: pd.DataFrame | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Load every fold checkpoint for *dataset_label* and evaluate on holdout.

    Returns:
        (metric_results, predictions_df) where metric_results is a list of dicts
        (one per fold that has a checkpoint) and predictions_df contains per-molecule
        True_pIC50 vs Pred_pIC50 for every fold, de-standardized.

    Missing checkpoints are skipped with a warning.
    """
    config = copy.deepcopy(base_config)
    config["experiment"]["target_folder"] = str(WORKSPACE_ROOT / dataset_label)

    feature_context = prepare_feature_config(config)

    raw_test_data = load_dataset(str(test_path), config, feature_context)

    total_folds = config["experiment"]["total_folds"]
    results: list[dict[str, Any]] = []
    pred_rows: list[pd.DataFrame] = []

    for fold_idx in range(total_folds):
        ckpt_path = _resolve_selected_checkpoint(
            dataset_label,
            fold_idx,
            config["experiment"]["target_folder"],
        )
        if not ckpt_path.exists():
            print(
                f"  [WARN] Checkpoint missing for {dataset_label} fold {fold_idx}: "
                f"{ckpt_path}"
            )
            continue
        print(f"  Using checkpoint for {dataset_label} fold {fold_idx}: {ckpt_path.name}")

        # Rebuild fold-specific feature scalers from the training split so
        # inference uses the same transform policy as training for this fold.
        train_paths, _ = get_fold_files(
            config["experiment"]["target_folder"],
            fold_idx,
            config,
            total_folds=total_folds,
        )
        train_data = load_dataset(train_paths, config, feature_context)
        synth_policy = get_synthetic_cv_config(config)
        finetune_train_data = cap_synthetic_train_ratio(
            train_data,
            max_ratio=synth_policy["max_train_synth_to_real_ratio"],
            seed=config["experiment"]["seed"],
            fold_idx=fold_idx,
        )
        feature_scalers = fit_feature_scalers(
            finetune_train_data,
            feature_context=feature_context,
            config=config,
        )

        model, target_standardizer = load_fold_checkpoint(
            ckpt_path, config, GATModel, feature_context
        )

        fold_test_data = copy.deepcopy(raw_test_data)
        apply_feature_scalers(fold_test_data, feature_scalers=feature_scalers)
        test_loader = DataLoader(
            fold_test_data,
            batch_size=config["data"]["batch_size"],
            shuffle=False,
        )

        # Read stored validation metrics from the checkpoint for reporting.
        raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        val_rmse = float(raw_ckpt.get("val_rmse", float("nan"))) if isinstance(raw_ckpt, dict) else float("nan")
        val_mae = float(raw_ckpt.get("val_mae", float("nan"))) if isinstance(raw_ckpt, dict) else float("nan")
        val_pearson = float(raw_ckpt.get("val_pearson", float("nan"))) if isinstance(raw_ckpt, dict) else float("nan")

        # Older checkpoints do not store MAE or Pearson, so recompute validation
        # metrics from the same fold split used during training when needed.
        if (
            torch.isnan(torch.tensor(val_rmse))
            or torch.isnan(torch.tensor(val_mae))
            or torch.isnan(torch.tensor(val_pearson))
        ):
            _, val_paths = get_fold_files(
                config["experiment"]["target_folder"],
                fold_idx,
                config,
                total_folds=total_folds,
            )
            val_data = load_dataset(val_paths, config, feature_context)
            apply_feature_scalers(val_data, feature_scalers=feature_scalers)
            val_loader = DataLoader(
                val_data,
                batch_size=config["data"]["batch_size"],
                shuffle=False,
            )
            _, val_rmse, val_mae, _, _, val_pearson = evaluate(
                model,
                val_loader,
                target_standardizer=target_standardizer,
            )

        mse, rmse, mae, r2, rho, pearson = evaluate(
            model, test_loader, target_standardizer=target_standardizer
        )

        print(
            f"  [{dataset_label}][Fold {fold_idx}]  "
            f"val_rmse={val_rmse:.4f}  val_mae={val_mae:.4f}  "
            f"val_pearson={val_pearson:.4f}  "
            f"holdout -> MSE={mse:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  "
            f"R2={r2:.4f}  Rho={rho:.4f}  Pearson={pearson:.4f}"
        )

        results.append(
            {
                "Dataset": dataset_label,
                "Fold": fold_idx,
                "Checkpoint": ckpt_path.name,
                "Val_RMSE": val_rmse,
                "Val_MAE": val_mae,
                "Val_Pearson": val_pearson,
                "Holdout_MSE": mse,
                "Holdout_RMSE": rmse,
                "Holdout_MAE": mae,
                "Holdout_R2": r2,
                "Holdout_Rho": rho,
                "Holdout_Pearson": pearson,
            }
        )

        # Collect per-molecule predictions for this fold (de-standardized).
        y_true_arr, y_pred_arr = predict(
            model, test_loader, target_standardizer=target_standardizer
        )
        n = len(y_true_arr)
        # Use SMILES from the pre-loaded holdout DataFrame when the row count
        # matches exactly; fall back to None if load_dataset dropped any rows
        # due to invalid SMILES (smiles_to_graph returned None).
        if holdout_smiles_df is not None and len(holdout_smiles_df) == n:
            smiles_col = holdout_smiles_df["smiles"].values.tolist()
        else:
            if holdout_smiles_df is not None:
                print(
                    f"  [WARN] SMILES count mismatch for {dataset_label} fold {fold_idx}: "
                    f"CSV has {len(holdout_smiles_df)} rows, model produced {n} predictions. "
                    "SMILES column will be None in predictions CSV."
                )
            smiles_col = [None] * n
        pred_rows.append(
            pd.DataFrame(
                {
                    "Dataset": dataset_label,
                    "Fold": fold_idx,
                    "Checkpoint": ckpt_path.name,
                    "smiles": smiles_col,
                    "True_pIC50": y_true_arr,
                    "Pred_pIC50": y_pred_arr,
                }
            )
        )

    pred_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    return results, pred_df


# ==================== main ====================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved GAT fold models on the held-out test set."
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        metavar="DATASET",
        help="Evaluate only one dataset folder, e.g. '0%%', '33%%', '67%%'. "
             "Omit to evaluate all datasets.",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to holdout CSV. Defaults to CONFIG['experiment']['actual_test_file'] "
             "resolved from the workspace root.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="FILENAME",
        help="Override the output CSV filename (e.g. my_results.csv). "
             f"Defaults to OUTPUT_FILENAME='{OUTPUT_FILENAME}'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_csv = WORKSPACE_ROOT / (args.output if args.output else OUTPUT_FILENAME)

    test_path = (
        Path(args.test_file)
        if args.test_file
        else WORKSPACE_ROOT / CONFIG["experiment"]["actual_test_file"]
    )
    if not test_path.exists():
        raise FileNotFoundError(f"Holdout test file not found: {test_path}")

    datasets = [args.folder] if args.folder else DATASETS

    all_results: list[dict[str, Any]] = []
    all_pred_rows: list[pd.DataFrame] = []
    for dataset_label in datasets:
        folder_path = WORKSPACE_ROOT / dataset_label
        if not folder_path.exists():
            print(f"[SKIP] Dataset folder not found: {folder_path}")
            continue
        print(f"\n=== Evaluating dataset: {dataset_label} ===")
        holdout_smiles_df = _load_holdout_smiles_df(test_path, CONFIG)
        metric_results, pred_df = evaluate_dataset(
            dataset_label, CONFIG, test_path, holdout_smiles_df
        )
        all_results.extend(metric_results)
        if not pred_df.empty:
            all_pred_rows.append(pred_df)

    if not all_results:
        print("\nNo results produced — are checkpoints present?")
        print(
            "Train models first with gat_predictor.py, then re-run this script."
        )
        return

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved holdout results ({len(df)} rows) to: {output_csv}")
    print(df.to_string(index=False))

    if all_pred_rows:
        pred_all = pd.concat(all_pred_rows, ignore_index=True)
        pred_all.to_csv(OUTPUT_PRED_CSV, index=False)
        print(f"\nSaved per-fold predictions ({len(pred_all)} rows) to: {OUTPUT_PRED_CSV}")
        print(pred_all.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
