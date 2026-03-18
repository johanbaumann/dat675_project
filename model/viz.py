# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import time as t 
import random as rnd

seed = 42
rnd.seed(seed)
np.random.seed(seed)

print(f"Imported at: {t.ctime()}")

# %%
results = "gat_results_heldout.csv"
df_results = pd.read_csv(results)
print(df_results.head())

names = ["0%", "33%", "67%"]

"""
Columns are: ["Dataset","Fold","Val_RMSE", "Val_MAE", "Val_Pearson", "Holdout_MSE", "Holdout_RMSE", "Holdout_MAE", "Holdout_R2", "Holdout_Rho", "Holdout_Pearson"]
What presentage synthetic they are from is in:

Dataset: "0%", "33%", "67%"
"""

zero_percent = df_results[df_results["Dataset"] == "0%"]
thirtythree_percent = df_results[df_results["Dataset"] == "33%"]
sixtyseven_percent = df_results[df_results["Dataset"] == "67%"]



# %%
import os
import io
import re
import pandas as pd
import matplotlib.pyplot as plt


folders = ["./0%", "./33%", "./67%"]
path_in_folder = ["MPNN_losses_0%.txt", "MPNN_losses_33%.txt", "MPNN_losses_67%.txt"]


def load_fold_loss_file(file_path: str) -> dict[int, pd.DataFrame]:
    """
    Read a custom loss file with sections like:

    fold_0:
       train      val  monitor       lr
    5.560307 2.158475 1.469175 0.001000
    ...

    fold_1:
       train      val  monitor       lr
    ...

    Returns:
        dict mapping fold index -> DataFrame with columns:
        ['train', 'val', 'monitor', 'lr']
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split on lines like "fold_0:", "fold_1:", ...
    parts = re.split(r"(fold_\d+:)\s*", text)

    fold_data = {}

    # parts looks like:
    # [text_before, 'fold_0:', content0, 'fold_1:', content1, ...]
    for i in range(1, len(parts), 2):
        fold_header = parts[i].strip()
        fold_block = parts[i + 1].strip()

        match = re.match(r"fold_(\d+):", fold_header)
        if not match:
            continue

        fold_idx = int(match.group(1))

        # Remove empty lines inside block
        lines = [line.rstrip() for line in fold_block.splitlines() if line.strip()]
        if not lines:
            continue

        # Rebuild mini-table text for pandas
        block_text = "\n".join(lines)

        # Parse whitespace-separated mini table
        df = pd.read_csv(io.StringIO(block_text), sep=r"\s+")


        fold_data[fold_idx] = df

    return fold_data


def plot_loss_curves_subplots(
    folder: str,
    filename: str,
    max_folds: int = 5,
    figsize: tuple[int, int] = (18, 10),
):
    """
    Plot one dataset in one large figure with 5 smaller subplots,
    one subplot per fold.

    Parameters
    ----------
    folder : str
        Folder containing the loss file, for example "./0%".
    filename : str
        Name of the loss file, for example "MPNN_losses_0%.txt".
    max_folds : int
        Number of folds to plot.
    figsize : tuple[int, int]
        Figure size.
    """
    file_path = os.path.join(folder, filename)

    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Skipping this dataset.")
        return

    try:
        fold_data = load_fold_loss_file(file_path)
    except Exception as e:
        print(f"Warning: failed to parse {file_path}: {e}")
        return

    if not fold_data:
        print(f"No valid fold data found in {file_path}.")
        return

    dataset_label = os.path.basename(os.path.normpath(folder))

    # Create 2x3 grid, hide the last subplot if only 5 folds
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for fold_idx in range(max_folds):
        ax = axes[fold_idx]

        if fold_idx not in fold_data:
            ax.set_title(f"Fold {fold_idx} (missing)")
            ax.axis("off")
            continue

        df = fold_data[fold_idx]
        epochs = range(1, len(df) + 1)

        ax.plot(epochs, df["train"], label="Train", linewidth=2)
        ax.plot(epochs, df["val"], label="Validation", linestyle="--", linewidth=2)

        ax.set_title(f"Fold {fold_idx}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Hide unused axes
    for i in range(max_folds, len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        f"Training and Validation Loss Curves Across Folds, dataset {dataset_label}",
        fontsize=16
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def iterate_datasets(
    folders: list[str],
    filenames: list[str],
    max_folds: int = 5,
    figsize: tuple[int, int] = (18, 10),
):
    """
    Iterate over datasets and plot one figure per dataset.

    Parameters
    ----------
    folders : list[str]
        Dataset folders, for example ["./0%", "./33%", "./67%"].
    filenames : list[str]
        Matching filenames for each folder.
    max_folds : int
        Number of folds to plot.
    figsize : tuple[int, int]
        Figure size.
    """
    if len(folders) != len(filenames):
        raise ValueError("folders and filenames must have the same length.")

    for folder, filename in zip(folders, filenames):
        print(f"Plotting dataset from folder={folder}, file={filename}")
        plot_loss_curves_subplots(
            folder=folder,
            filename=filename,
            max_folds=max_folds,
            figsize=figsize,
        )


iterate_datasets(folders, path_in_folder)

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


results = "gat_results_heldout.csv"
df_results = pd.read_csv(results)

print(df_results.head())
print(df_results.columns.tolist())

names = ["0%", "33%", "67%"]


def simple_metric_boxplots(df: pd.DataFrame, metrics: str | list[str], title: str = "Boxplots"):
    if isinstance(metrics, str):
        metrics = [metrics]

    amount = len(metrics)
    n_per_row = min(3, amount)  # max 3 plots per row for better aesthetics
    rows = (amount + n_per_row - 1) // n_per_row  # calculate number of rows needed
    fig, axs = plt.subplots(rows, n_per_row, figsize=(5 * n_per_row, 5 * rows)) # Adjust figure size based on number of plots

    if len(metrics) == 1:
        axs = [axs]
    
    

    axs = np.atleast_1d(axs).flatten()
    plot_df = df[df["Dataset"].isin(names)].copy()

    colors = ['#FF0000', '#00FF00', '#0000FF']  # Red, Green, Blue for the three datasets
    #colors = sns.color_palette("tab10", n_colors=len(metrics))  # Seaborn Set2 palette for better aesthetics
    idx = 0
    for ax, metric in zip(axs, metrics):
        if metric not in plot_df.columns:
            ax.set_visible(False)
            print(f"Warning: metric '{metric}' not found in dataframe columns.")
            continue

        
        mean_handle = Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor='white',
            markeredgecolor='black',
            markersize=8,
            label='Mean'
        )

        median_handle = Line2D(
            [0], [0],
            color='black',
            linewidth=4,
            label='Median'
        )
        sns.boxplot(
            x="Dataset",
            y=metric,
            data=plot_df,
            order=names,
            ax=ax,
            showfliers=True,
            palette=colors,
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 8},
            medianprops={
                "linewidth": 4,
                "color": "black"
            }
        )
        
        ax.set_title(metric)
        ax.set_xlabel("Dataset")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)
        for label in ax.get_xticklabels():
            label.set_fontsize(12)
            label.set_fontweight('bold')
    # Hide any unused subplot axes
    for ax in axs[len(metrics):]:
        ax.set_visible(False)
    fig.legend(
        handles=[mean_handle, median_handle],
        loc='upper left',
        ncol=3,
        fontsize=20,
        frameon=True
    )
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300)  # Save the figure as a high-resolution PNG file
    plt.show()


"""
Columns are: ["Dataset","Fold","Val_RMSE", "Val_MAE", "Val_Pearson", "Holdout_MSE", "Holdout_RMSE", "Holdout_MAE", "Holdout_R2", "Holdout_Rho", "Holdout_Pearson"]
What presentage synthetic they are from is in:

Dataset: "0%", "33%", "67%"
"""

simple_metric_boxplots(
    df_results,
    ["Holdout_R2", "Holdout_Rho", "Holdout_MSE","Holdout_RMSE", "Holdout_MAE", "Holdout_Pearson"],
    title="Boxplots synthetic data generation metrics"
)

# %%
import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


# ==============================================================================
# Configuration
# ==============================================================================

PREDICTIONS_CSV = "GAT_predictions_heldout_set.csv"
DATASET_ORDER = ["0%", "33%", "67%"]
OUTPUT_DIR = "pretrain"

# Keep only the main performance metrics
MOLECULE_LEVEL_METRICS = [
    "mean_abs_error",
    "rmse",
]


# ==============================================================================
# Utility functions
# ==============================================================================

def holm_correction(p_values: list[float]) -> list[float]:
    """
    Apply Holm correction to a list of p-values.

    Returns corrected p-values in the original order.
    We have one p-value per pairwise Wilcoxon test within each metric, so we apply Holm correction separately for each metric.
    Holm correction steps:
    1. Sort p-values in ascending order.
    2. For each p-value, compute adjusted p = min((n - i) * p, 1.0) where:
        - n is the total number of tests (length of p_values).
        - i is the index in the sorted list (starting at 0).

    3. Take the cumulative maximum of the adjusted p-values to ensure monotonicity.

    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted = np.empty(n, dtype=float)

    running_max = 0.0
    for i, p in enumerate(sorted_p):
        adjusted_value = min((n - i) * p, 1.0)
        running_max = max(running_max, adjusted_value)
        adjusted[i] = running_max

    corrected = np.empty(n, dtype=float)
    corrected[order] = adjusted
    return corrected.tolist()


def load_and_prepare_predictions(predictions_csv: str) -> pd.DataFrame:
    """
    Load prediction CSV and compute row-level error quantities.
    """
    df = pd.read_csv(predictions_csv)

    required_cols = ["Dataset", "Fold", "smiles", "True_pIC50", "Pred_pIC50"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["Fold"] = pd.to_numeric(df["Fold"], errors="raise")

    df["residual"] = df["Pred_pIC50"] - df["True_pIC50"]
    df["abs_error"] = df["residual"].abs()
    df["squared_error"] = df["residual"] ** 2

    return df


def summarize_row_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute compact row-level summaries by dataset.
    """
    row_summary = (
        df.groupby("Dataset", as_index=False)
        .agg(
            n_rows=("smiles", "size"),
            n_unique_molecules=("smiles", "nunique"),
            mean_abs_error=("abs_error", "mean"),
            rmse=("squared_error", lambda x: np.sqrt(np.mean(x))),
        )
    )

    return row_summary


def aggregate_per_molecule_per_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate across folds to one row per molecule per dataset.
    """
    mol_dataset = (
        df.groupby(["smiles", "Dataset"], as_index=False)
        .agg(
            True_pIC50=("True_pIC50", "first"),
            mean_abs_error=("abs_error", "mean"),
            rmse=("squared_error", lambda x: np.sqrt(np.mean(x))),
        )
    )

    return mol_dataset


def build_molecule_level_summary(mol_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize molecule-level metrics by dataset.
    """
    mol_summary = (
        mol_dataset.groupby("Dataset", as_index=False)
        .agg(
            n_molecules=("smiles", "nunique"),
            mean_mae=("mean_abs_error", "mean"),
            median_mae=("mean_abs_error", "median"),
            mean_rmse=("rmse", "mean"),
            median_rmse=("rmse", "median"),
        )
    )

    return mol_summary


def run_friedman_and_wilcoxon(
    mol_dataset: pd.DataFrame,
    dataset_order: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    """
    For each molecule-level metric:
    1. Pivot to matched molecules across datasets
    2. Run Friedman test across all datasets
    3. Run pairwise Wilcoxon signed-rank tests
    4. Apply Holm correction within each metric
    """
    results = []

    for metric in metrics:
        pivot = mol_dataset.pivot(index="smiles", columns="Dataset", values=metric)
        pivot = pivot[dataset_order].dropna()

        if len(pivot) == 0:
            print(f"[warning] No matched molecules found for metric: {metric}")
            continue

        friedman_stat, friedman_p = friedmanchisquare(
            *[pivot[dataset_name].to_numpy() for dataset_name in dataset_order]
        )

        metric_rows = []

        for dataset_a, dataset_b in combinations(dataset_order, 2):
            x = pivot[dataset_a].to_numpy()
            y = pivot[dataset_b].to_numpy()
            diff = x - y

            try:
                wilcoxon_stat, wilcoxon_p = wilcoxon(
                    x,
                    y,
                    alternative="two-sided",
                    zero_method="wilcox",
                    correction=False,
                    mode="auto",
                )
            except ValueError:
                wilcoxon_stat = np.nan
                wilcoxon_p = np.nan

            metric_rows.append(
                {
                    "Metric": metric,
                    "Dataset_A": dataset_a,
                    "Dataset_B": dataset_b,
                    "n_molecules": len(pivot),
                    "Mean_A": np.mean(x),
                    "Mean_B": np.mean(y),
                    "Median_A": np.median(x),
                    "Median_B": np.median(y),
                    "Mean_diff_A_minus_B": np.mean(diff),
                    "Median_diff_A_minus_B": np.median(diff),
                    "Wilcoxon_stat": wilcoxon_stat,
                    "Wilcoxon_p": wilcoxon_p,
                    "Friedman_stat": friedman_stat,
                    "Friedman_p": friedman_p,
                }
            )

        corrected_p_values = holm_correction([row["Wilcoxon_p"] for row in metric_rows])

        for row, corrected_p in zip(metric_rows, corrected_p_values):
            row["Wilcoxon_p_holm"] = corrected_p
            results.append(row)

    return pd.DataFrame(results)


def build_improvement_table(
    mol_dataset: pd.DataFrame,
    metric: str,
    dataset_order: list[str],
) -> pd.DataFrame:
    """
    Build a compact per-molecule improvement table for one metric.
    Positive values mean improvement because lower error is better.
    """
    pivot = mol_dataset.pivot(index="smiles", columns="Dataset", values=metric)
    pivot = pivot[dataset_order].dropna().copy()

    pivot["improvement_33_vs_0"] = pivot["0%"] - pivot["33%"]
    pivot["improvement_67_vs_0"] = pivot["0%"] - pivot["67%"]
    pivot["improvement_67_vs_33"] = pivot["33%"] - pivot["67%"]

    return pivot.reset_index()


def save_dataframe(df: pd.DataFrame, output_dir: str, filename: str) -> None:
    """
    Save a DataFrame to CSV.
    """
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def plot_metric_boxplot(
    mol_dataset: pd.DataFrame,
    dataset_order: list[str],
    metric: str,
    ylabel: str,
    title: str,
    output_path: str | None = None,
    show_plot: bool = True,
) -> None:
    """
    Plot molecule-level distribution for one metric and optionally save it.
    """
    data = [
        mol_dataset.loc[mol_dataset["Dataset"] == dataset_name, metric].to_numpy()
        for dataset_name in dataset_order
    ]

    plt.figure(figsize=(7, 5))
    plt.boxplot(data, tick_labels=dataset_order, showfliers=False)
    plt.xlabel("Dataset")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


# ==============================================================================
# Main analysis
# ==============================================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_and_prepare_predictions(PREDICTIONS_CSV)

    print("=" * 80)
    print("Loaded prediction file")
    print("=" * 80)
    print("Shape:", df.shape)
    print("Unique datasets:", sorted(df["Dataset"].unique().tolist()))
    print("Unique folds:", sorted(df["Fold"].unique().tolist()))
    print("Unique molecules:", df["smiles"].nunique())

    # --------------------------------------------------------------------------
    # Row-level summary
    # --------------------------------------------------------------------------
    row_summary = summarize_row_level(df)
    save_dataframe(row_summary, OUTPUT_DIR, "row_summary.csv")

    print("\n" + "=" * 80)
    print("Row-level summary by dataset")
    print("=" * 80)
    print(row_summary.to_string(index=False))

    # --------------------------------------------------------------------------
    # Aggregate to one row per molecule per dataset
    # --------------------------------------------------------------------------
    mol_dataset = aggregate_per_molecule_per_dataset(df)
    save_dataframe(mol_dataset, OUTPUT_DIR, "molecule_level_dataset.csv")

    mol_summary = build_molecule_level_summary(mol_dataset)
    save_dataframe(mol_summary, OUTPUT_DIR, "molecule_summary.csv")

    print("\n" + "=" * 80)
    print("Molecule-level dataset summary")
    print("=" * 80)
    print(mol_summary.to_string(index=False))

    # --------------------------------------------------------------------------
    # Statistical tests
    # --------------------------------------------------------------------------
    stats_df = run_friedman_and_wilcoxon(
        mol_dataset=mol_dataset,
        dataset_order=DATASET_ORDER,
        metrics=MOLECULE_LEVEL_METRICS,
    )
    save_dataframe(stats_df, OUTPUT_DIR, "statistical_tests.csv")

    print("\n" + "=" * 80)
    print("Statistical tests")
    print("=" * 80)
    print(stats_df.to_string(index=False))

    # --------------------------------------------------------------------------
    # Improvement table for RMSE only
    # --------------------------------------------------------------------------
    rmse_improvement_df = build_improvement_table(
        mol_dataset=mol_dataset,
        metric="rmse",
        dataset_order=DATASET_ORDER,
    )
    save_dataframe(rmse_improvement_df, OUTPUT_DIR, "rmse_improvement_per_molecule.csv")

    # --------------------------------------------------------------------------
    # Save run metadata
    # --------------------------------------------------------------------------
    run_config_df = pd.DataFrame(
        {
            "PREDICTIONS_CSV": [PREDICTIONS_CSV],
            "OUTPUT_DIR": [OUTPUT_DIR],
            "DATASET_ORDER": [",".join(DATASET_ORDER)],
            "MOLECULE_LEVEL_METRICS": [",".join(MOLECULE_LEVEL_METRICS)],
        }
    )
    save_dataframe(run_config_df, OUTPUT_DIR, "run_config.csv")

    # --------------------------------------------------------------------------
    # Plots
    # --------------------------------------------------------------------------
    plot_metric_boxplot(
        mol_dataset=mol_dataset,
        dataset_order=DATASET_ORDER,
        metric="rmse",
        ylabel="Per-molecule RMSE across folds",
        title="Per-molecule RMSE by dataset",
        output_path=os.path.join(OUTPUT_DIR, "boxplot_rmse.png"),
        show_plot=True,
    )

    plot_metric_boxplot(
        mol_dataset=mol_dataset,
        dataset_order=DATASET_ORDER,
        metric="mean_abs_error",
        ylabel="Per-molecule mean absolute error across folds",
        title="Per-molecule mean absolute error by dataset",
        output_path=os.path.join(OUTPUT_DIR, "boxplot_mean_abs_error.png"),
        show_plot=True,
    )


if __name__ == "__main__":
    main()

# %%
import os
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, ttest_rel, wilcoxon


# ==============================================================================
# Configuration
# ==============================================================================

PRETRAIN_CSV = os.path.join("pretrain", "molecule_level_dataset.csv")
NO_PRETRAIN_CSV = os.path.join("no_pretrain", "molecule_level_dataset.csv")
OUTPUT_CSV = "pretrain_vs_no_pretrain_molecule_level_tests.csv"

MOLECULE_METRICS = [
    "mean_abs_error",
    "rmse",
]


# ==============================================================================
# Utility
# ==============================================================================

def holm_correction(p_values: list[float]) -> list[float]:
    """
    Apply Holm correction to a list of p-values.

    Returns corrected p-values in the original order.
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted = np.empty(n, dtype=float)

    running_max = 0.0
    for i, p in enumerate(sorted_p):
        adjusted_value = min((n - i) * p, 1.0)
        running_max = max(running_max, adjusted_value)
        adjusted[i] = running_max

    corrected = np.empty(n, dtype=float)
    corrected[order] = adjusted
    return corrected.tolist()


def load_and_align_molecule_level(
    pretrain_csv: str,
    no_pretrain_csv: str,
) -> pd.DataFrame:
    """
    Load the molecule-level files and align them by smiles and Dataset.
    """
    df_pre = pd.read_csv(pretrain_csv).copy()
    df_no = pd.read_csv(no_pretrain_csv).copy()

    required_cols = ["smiles", "Dataset"] + MOLECULE_METRICS
    for col in required_cols:
        if col not in df_pre.columns:
            raise ValueError(f"Missing column '{col}' in pretrain molecule-level file")
        if col not in df_no.columns:
            raise ValueError(f"Missing column '{col}' in no_pretrain molecule-level file")

    merged = df_pre.merge(
        df_no,
        on=["smiles", "Dataset"],
        suffixes=("_pretrain", "_no_pretrain"),
        how="inner",
    )

    if len(merged) == 0:
        raise ValueError("No matched (smiles, Dataset) rows found between files")

    return merged


def run_tests(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Run:
    - Wilcoxon signed-rank test
    - paired t-test
    - Friedman test on paired differences across dataset conditions

    Friedman here is run on the paired differences (pretrain - no_pretrain)
    across the 3 dataset conditions (0%, 33%, 67%) for each molecule.
    """
    results = []

    wilcoxon_raw_p_values = []
    ttest_raw_p_values = []

    for metric in MOLECULE_METRICS:
        x = merged[f"{metric}_pretrain"].to_numpy(dtype=float)
        y = merged[f"{metric}_no_pretrain"].to_numpy(dtype=float)
        diff = x - y

        # ----------------------------------------------------------------------
        # Wilcoxon signed-rank test
        # ----------------------------------------------------------------------
        try:
            wilcoxon_stat, wilcoxon_p = wilcoxon(
                x,
                y,
                alternative="two-sided",
                zero_method="wilcox",
                correction=False,
                mode="auto",
            )
        except ValueError:
            wilcoxon_stat = np.nan
            wilcoxon_p = np.nan

        # ----------------------------------------------------------------------
        # Paired t-test
        # ----------------------------------------------------------------------
        try:
            t_stat, t_p = ttest_rel(x, y, nan_policy="omit")
        except ValueError:
            t_stat = np.nan
            t_p = np.nan

        # ----------------------------------------------------------------------
        # Friedman test on paired differences across dataset conditions
        # ----------------------------------------------------------------------
        diff_df = merged[["smiles", "Dataset"]].copy()
        diff_df["diff"] = merged[f"{metric}_pretrain"] - merged[f"{metric}_no_pretrain"]

        diff_pivot = diff_df.pivot(index="smiles", columns="Dataset", values="diff")

        expected_datasets = ["0%", "33%", "67%"]
        if all(dataset_name in diff_pivot.columns for dataset_name in expected_datasets):
            diff_pivot = diff_pivot[expected_datasets].dropna()

            if len(diff_pivot) > 0:
                try:
                    friedman_stat, friedman_p = friedmanchisquare(
                        diff_pivot["0%"].to_numpy(),
                        diff_pivot["33%"].to_numpy(),
                        diff_pivot["67%"].to_numpy(),
                    )
                    n_friedman = len(diff_pivot)
                except ValueError:
                    friedman_stat = np.nan
                    friedman_p = np.nan
                    n_friedman = 0
            else:
                friedman_stat = np.nan
                friedman_p = np.nan
                n_friedman = 0
        else:
            friedman_stat = np.nan
            friedman_p = np.nan
            n_friedman = 0

        wilcoxon_raw_p_values.append(wilcoxon_p)
        ttest_raw_p_values.append(t_p)

        results.append(
            {
                "Metric": metric,
                "n_pairs": len(x),
                "n_molecules_friedman": n_friedman,
                "Pretrain_mean": np.mean(x),
                "NoPretrain_mean": np.mean(y),
                "Mean_diff_pretrain_minus_no_pretrain": np.mean(diff),
                "Median_diff_pretrain_minus_no_pretrain": np.median(diff),
                "Std_diff_pretrain_minus_no_pretrain": np.std(diff, ddof=1),
                "Fraction_pretrain_better": np.mean(diff < 0),
                "Fraction_equal": np.mean(diff == 0),
                "Fraction_no_pretrain_better": np.mean(diff > 0),
                "Wilcoxon_stat": wilcoxon_stat,
                "Wilcoxon_p": wilcoxon_p,
                "Ttest_stat": t_stat,
                "Ttest_p": t_p,
                "Friedman_stat_on_diff_across_datasets": friedman_stat,
                "Friedman_p_on_diff_across_datasets": friedman_p,
            }
        )

    wilcoxon_corrected = holm_correction(wilcoxon_raw_p_values)
    ttest_corrected = holm_correction(ttest_raw_p_values)

    for row, p_corr_w, p_corr_t in zip(results, wilcoxon_corrected, ttest_corrected):
        row["Wilcoxon_p_holm"] = p_corr_w
        row["Ttest_p_holm"] = p_corr_t

    return pd.DataFrame(results)


def main() -> None:
    merged = load_and_align_molecule_level(PRETRAIN_CSV, NO_PRETRAIN_CSV)

    print("=" * 80)
    print("Matched molecule-level rows")
    print("=" * 80)
    print("Number of matched rows:", len(merged))

    result_df = run_tests(merged)

    print("\n" + "=" * 80)
    print("Pretrain vs no_pretrain molecule-level paired tests")
    print("=" * 80)
    print(result_df.to_string(index=False))

    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()


