"""GAT holdout evaluation script.

For every dataset folder that has saved fold checkpoints this script loads
each fold's best model, evaluates it on the held-out test set, and writes
all results to MPNN_results_heldout_set.csv in the workspace root.

This script is inference-only — it does not train any models.
Training is done separately via MPNN_predictor.py (one run per dataset).

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
    MPNN_results_heldout_set.csv  (in the same folder as this script)

    Columns: Dataset | Fold | Val_RMSE | Val_Pearson | Holdout_MSE | Holdout_RMSE |
             Holdout_R2 | Holdout_Rho | Holdout_Pearson
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
# training is guarded by `if __name__ == "__main__"`, so this import is safe.
from gat_predictor import CONFIG, GATModel
from gat_utils import (
    evaluate,
    get_fold_checkpoint_path,
    load_dataset,
    load_fold_checkpoint,
    prepare_feature_config,
)
from gat_utils.data_loading import get_fold_files

WORKSPACE_ROOT = Path(__file__).resolve().parent
DATASETS = ["0%", "33%", "67%"]
OUTPUT_CSV = WORKSPACE_ROOT / "MPNN_results_heldout_set.csv"


# ==================== per-dataset evaluation ====================

def evaluate_dataset(
    dataset_label: str,
    base_config: dict[str, Any],
    test_path: Path,
) -> list[dict[str, Any]]:
    """Load every fold checkpoint for *dataset_label* and evaluate on holdout.

    Returns a list of result dicts (one per fold that has a checkpoint).
    Missing checkpoints are skipped with a warning.
    """
    config = copy.deepcopy(base_config)
    config["experiment"]["target_folder"] = str(WORKSPACE_ROOT / dataset_label)

    feature_context = prepare_feature_config(config)

    test_data = load_dataset(str(test_path), config, feature_context)
    test_loader = DataLoader(
        test_data,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
    )

    total_folds = config["experiment"]["total_folds"]
    results: list[dict[str, Any]] = []

    for fold_idx in range(total_folds):
        ckpt_path = get_fold_checkpoint_path(
            config["experiment"]["target_folder"], fold_idx
        )
        if not ckpt_path.exists():
            print(
                f"  [WARN] Checkpoint missing for {dataset_label} fold {fold_idx}: "
                f"{ckpt_path}"
            )
            continue

        model, target_standardizer = load_fold_checkpoint(
            ckpt_path, config, GATModel, feature_context
        )

        # Read stored validation metrics from the checkpoint for reporting.
        raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        val_rmse = float(raw_ckpt.get("val_rmse", float("nan"))) if isinstance(raw_ckpt, dict) else float("nan")
        val_pearson = float(raw_ckpt.get("val_pearson", float("nan"))) if isinstance(raw_ckpt, dict) else float("nan")

        # Older checkpoints do not store Pearson, so recompute validation metrics
        # from the same fold split used during training when needed.
        if torch.isnan(torch.tensor(val_rmse)) or torch.isnan(torch.tensor(val_pearson)):
            _, val_paths = get_fold_files(
                config["experiment"]["target_folder"],
                fold_idx,
                config,
                total_folds=total_folds,
            )
            val_data = load_dataset(val_paths, config, feature_context)
            val_loader = DataLoader(
                val_data,
                batch_size=config["data"]["batch_size"],
                shuffle=False,
            )
            _, val_rmse, _, _, val_pearson = evaluate(
                model,
                val_loader,
                target_standardizer=target_standardizer,
            )

        mse, rmse, r2, rho, pearson = evaluate(
            model, test_loader, target_standardizer=target_standardizer
        )

        print(
            f"  [{dataset_label}][Fold {fold_idx}]  "
            f"val_rmse={val_rmse:.4f}  "
            f"val_pearson={val_pearson:.4f}  "
            f"holdout -> MSE={mse:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  "
            f"Rho={rho:.4f}  Pearson={pearson:.4f}"
        )

        results.append(
            {
                "Dataset": dataset_label,
                "Fold": fold_idx,
                "Val_RMSE": val_rmse,
                "Val_Pearson": val_pearson,
                "Holdout_MSE": mse,
                "Holdout_RMSE": rmse,
                "Holdout_R2": r2,
                "Holdout_Rho": rho,
                "Holdout_Pearson": pearson,
            }
        )

    return results


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    test_path = (
        Path(args.test_file)
        if args.test_file
        else WORKSPACE_ROOT / CONFIG["experiment"]["actual_test_file"]
    )
    if not test_path.exists():
        raise FileNotFoundError(f"Holdout test file not found: {test_path}")

    datasets = [args.folder] if args.folder else DATASETS

    all_results: list[dict[str, Any]] = []
    for dataset_label in datasets:
        folder_path = WORKSPACE_ROOT / dataset_label
        if not folder_path.exists():
            print(f"[SKIP] Dataset folder not found: {folder_path}")
            continue
        print(f"\n=== Evaluating dataset: {dataset_label} ===")
        results = evaluate_dataset(dataset_label, CONFIG, test_path)
        all_results.extend(results)

    if not all_results:
        print("\nNo results produced — are checkpoints present?")
        print(
            "Train models first with MPNN_predictor.py, then re-run this script."
        )
        return

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved holdout results ({len(df)} rows) to: {OUTPUT_CSV}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
