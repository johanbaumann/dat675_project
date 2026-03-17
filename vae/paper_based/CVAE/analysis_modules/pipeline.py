from __future__ import annotations

import json
import math
import os
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .chem_utils import (
    canonicalize_smiles,
    max_tanimoto_to_reference,
    morgan_fp,
    morgan_fp_generator,
    safe_mol_from_smiles,
    safe_murcko_scaffold_smiles,
    scaffold_counts,
    smiles_list_to_descriptor_matrix,
    smiles_list_to_fp_matrix,
)
from .config import AnalysisConfig
from .io_utils import load_generated_dataframe, load_train_dataframe


def _apply_projector_plot_style() -> None:
    """Use larger, bolder text so saved figures are readable on projectors."""
    plt.rcParams.update(
        {
            'font.size': 14,
            'font.weight': 'bold',
            'axes.titlesize': 17,
            'axes.titleweight': 'bold',
            'axes.labelsize': 15,
            'axes.labelweight': 'bold',
            'xtick.labelsize': 13,
            'ytick.labelsize': 13,
            'xtick.major.size': 7,
            'ytick.major.size': 7,
            'xtick.major.width': 1.4,
            'ytick.major.width': 1.4,
            'legend.fontsize': 12,
            'legend.title_fontsize': 13,
            'figure.titlesize': 18,
            'figure.titleweight': 'bold',
        }
    )


_apply_projector_plot_style()


def _debug_log(cfg: AnalysisConfig, message: str) -> None:
    if bool(cfg.debug):
        print(f'[analysis:debug] {message}')


def _get_scatter_edgecolor_kwargs(cfg: AnalysisConfig) -> dict:
    """Return edgecolor kwargs for scatter plots if enabled, else empty dict."""
    if bool(cfg.embedding_point_edgecolors_enabled):
        return {
            'edgecolors': str(cfg.embedding_edge_color),
            'linewidths': float(cfg.embedding_edge_width),
        }
    return {}


def _save_figure(cfg: AnalysisConfig, filename: str, *, dpi: int = 180) -> str:
    out_path = os.path.join(cfg.output_dir, filename)
    plt.savefig(out_path, dpi=dpi)
    _debug_log(cfg, f'Wrote figure: {out_path}')
    return out_path


def _resolve_by_aliases(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        found = lower_map.get(alias.lower())
        if found is not None:
            return found
    return None


def _extract_numeric_series(df: pd.DataFrame, col: str | None) -> np.ndarray:
    if not col or col not in df.columns:
        return np.asarray([], dtype=float)
    vals = pd.to_numeric(df[col], errors='coerce').dropna().to_numpy(dtype=float)
    return vals if vals.size > 0 else np.asarray([], dtype=float)


def _coerce_int(value) -> int | None:
    if value is None:
        return None
    try:
        raw = str(value).strip()
        if raw == '':
            return None
        return int(float(raw))
    except Exception:
        return None


def _coerce_float(value) -> float | None:
    if value is None:
        return None
    try:
        raw = str(value).strip()
        if raw == '':
            return None
        return float(raw)
    except Exception:
        return None


def _compute_vun_from_counts(
    *,
    total_generated: int,
    invalid_or_empty: int,
    discarded_cleanup: int,
    in_training: int,
    duplicate: int,
    accepted: int,
) -> dict:
    total = total_generated
    if total <= 0:
        return {
            'validity': 0.0,
            'uniqueness': 0.0,
            'novelty': 0.0,
            'acceptance_rate': 0.0,
            'valid_count': 0,
            'unique_count': 0,
            'novel_count': 0,
        }

    valid_count = total - invalid_or_empty - discarded_cleanup
    unique_count = total - duplicate
    novel_count = total - in_training
    accepted_count = accepted

    return {
        'validity': float(valid_count) / float(total),
        'uniqueness': float(unique_count) / float(total),
        'novelty': float(novel_count) / float(total),
        'acceptance_rate': float(accepted_count) / float(total),
        'valid_count': valid_count,
        'unique_count': unique_count,
        'novel_count': novel_count,
    }


def _try_read_vun_from_quality_summary_csv(path: str | None) -> dict | None:
    if not path:
        return None
    if not os.path.isfile(path):
        return None

    try:
        quality_df = pd.read_csv(path)
    except Exception:
        return None
    if len(quality_df) == 0:
        return None

    row = quality_df.iloc[0].to_dict()
    counts = {
        'total_generated': _coerce_int(row.get('total_generated')),
        'invalid_or_empty': _coerce_int(row.get('invalid_or_empty')),
        'discarded_cleanup': _coerce_int(row.get('discarded_cleanup')),
        'in_training': _coerce_int(row.get('in_training')),
        'duplicate': _coerce_int(row.get('duplicate')),
        'accepted': _coerce_int(row.get('accepted')),
        'rejected_by_filter': _coerce_int(row.get('rejected_by_filter')),
        'salt_stripped': _coerce_int(row.get('salt_stripped')),
        'tautomer_canonicalized': _coerce_int(row.get('tautomer_canonicalized')),
    }
    required = ('total_generated', 'invalid_or_empty', 'discarded_cleanup', 'in_training', 'duplicate', 'accepted')
    has_required_counts = all(counts.get(k) is not None for k in required)

    if has_required_counts:
        # All required counts are guaranteed to be non-None at this point
        total_generated: int = counts['total_generated']  # type: ignore
        invalid_or_empty: int = counts['invalid_or_empty']  # type: ignore
        discarded_cleanup: int = counts['discarded_cleanup']  # type: ignore
        in_training: int = counts['in_training']  # type: ignore
        duplicate: int = counts['duplicate']  # type: ignore
        accepted: int = counts['accepted']  # type: ignore
        
        vun = _compute_vun_from_counts(
            total_generated=total_generated,
            invalid_or_empty=invalid_or_empty,
            discarded_cleanup=discarded_cleanup,
            in_training=in_training,
            duplicate=duplicate,
            accepted=accepted,
        )
        return {
            'source': 'quality_summary_csv',
            'quality_summary_csv_path': os.path.abspath(path),
            'quality_run_scope': row.get('run_scope'),
            'quality_counts': {k: v for k, v in counts.items() if v is not None},
            **vun,
        }

    validity = _coerce_float(row.get('validity'))
    uniqueness = _coerce_float(row.get('uniqueness'))
    novelty = _coerce_float(row.get('novelty'))
    acceptance_rate = _coerce_float(row.get('acceptance_rate'))
    if validity is None or uniqueness is None or novelty is None:
        return None

    return {
        'source': 'quality_summary_csv_metrics_only',
        'quality_summary_csv_path': os.path.abspath(path),
        'quality_run_scope': row.get('run_scope'),
        'quality_counts': {k: int(v) for k, v in counts.items() if v is not None},
        'validity': float(validity),
        'uniqueness': float(uniqueness),
        'novelty': float(novelty),
        'acceptance_rate': (None if acceptance_rate is None else float(acceptance_rate)),
        'valid_count': _coerce_int(row.get('valid_count')),
        'unique_count': _coerce_int(row.get('unique_count')),
        'novel_count': _coerce_int(row.get('novel_count')),
    }


def _compute_vun_from_loaded_data(
    *,
    train_smiles: list[str],
    gen_df: pd.DataFrame,
) -> dict:

    """
    Compute the V.U.N. metrics (validity, uniqueness, novelty) based on the provided training SMILES and generated DataFrame.

    - C_s = canonical valid smiles
    - n_s = total generated smiles
    - D = training set canonical smiles

    V = |C_s| / n_s
    U = set(C_s) / n_s
    N = (1- (|C_s ∩ D|/|C_s|))

    V = valid_count / total_generated; 
    U = unique_count / total_generated
    N = novel_count / total_generated

    Where:
    - valid_count: Number of generated molecules that are valid (can be parsed and are not empty).
    - unique_count: Number of unique valid generated molecules not in the generated set more than once.
    - novel_count: Number of unique valid generated molecules that are not in the training set.

    
    """



    train_canonical = {
        c
        for c in (canonicalize_smiles(s) for s in train_smiles)
        if c is not None and c != ''
    }

    is_valid_mask = gen_df['is_valid'].to_numpy(dtype=bool)
    canonical_generated = gen_df['canonical_smiles'].astype(str).tolist()
    valid_canonical = [
        s
        for s, is_valid in zip(canonical_generated, is_valid_mask)
        if bool(is_valid) and s and s != 'None' and s != 'nan'
    ]

    unique_generated = set(valid_canonical)
    novel_generated = {s for s in unique_generated if s not in train_canonical}

    total = int(len(gen_df))
    valid_count = int(np.sum(is_valid_mask))
    unique_count = int(len(unique_generated))
    novel_count = int(len(novel_generated))

    if total <= 0:
        validity = 0.0
        uniqueness = 0.0
        novelty = 0.0
    else:
        validity = float(valid_count) / float(total) # validity = valid_count / total_generated
        uniqueness = float(unique_count) / float(total) # uniqueness = unique_count / total_generated
        novelty = float(novel_count) / float(total) # novelty = novel_count / total_generated

    return {
        'source': 'computed_from_analysis_inputs',
        'quality_summary_csv_path': None,
        'quality_run_scope': None,
        'quality_counts': {
            'total_generated': int(total),
            'invalid_or_empty': int(max(0, total - valid_count)),
            'discarded_cleanup': 0,
            'in_training': int(max(0, total - novel_count)),
            'duplicate': int(max(0, total - unique_count)),
            'accepted': int(valid_count),
        },
        'validity': float(validity),
        'uniqueness': float(uniqueness),
        'novelty': float(novelty),
        'acceptance_rate': float(validity),
        'valid_count': int(valid_count),
        'unique_count': int(unique_count),
        'novel_count': int(novel_count),
    }


def _compute_mw_from_smiles(df: pd.DataFrame, smiles_col: str) -> np.ndarray:
    if smiles_col not in df.columns:
        return np.asarray([], dtype=float)
    out = []
    for s in df[smiles_col].astype(str).tolist():
        mol = safe_mol_from_smiles(s)
        if mol is None:
            continue
        try:
            out.append(float(Descriptors.MolWt(mol)))
        except Exception:
            continue
    if len(out) == 0:
        return np.asarray([], dtype=float)
    return np.asarray(out, dtype=float)


def _maybe_plot_property_distributions(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame | None,
    gen_df: pd.DataFrame,
    cfg: AnalysisConfig,
) -> None:
    if not bool(cfg.save_distribution_plot):
        _debug_log(cfg, 'Skipping property distribution plots (save_distribution_plot=False).')
        return

    os.makedirs(cfg.output_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))

    train_target_col = _resolve_property_column(train_df, cfg.target_property_column, role='target')
    val_target_col = (
        _resolve_property_column(validation_df, cfg.target_property_column, role='target')
        if validation_df is not None
        else None
    )
    gen_pred_col = _resolve_property_column(gen_df, cfg.predicted_property_column, role='pred')
    gen_target_col = _resolve_property_column(gen_df, cfg.target_property_column, role='target')

    train_prop_vals = _extract_numeric_series(train_df, train_target_col)
    val_prop_vals = _extract_numeric_series(validation_df, val_target_col) if validation_df is not None else np.asarray([], dtype=float)
    gen_prop_col_used = gen_pred_col if gen_pred_col else gen_target_col
    gen_prop_vals = _extract_numeric_series(gen_df, gen_prop_col_used)
    _debug_log(
        cfg,
        f'Property distribution columns -> train_target={train_target_col!r}, '
        f'validation_target={val_target_col!r}, generated_property={gen_prop_col_used!r}',
    )

    all_vals = np.concatenate(
        [vals for vals in (train_prop_vals, val_prop_vals, gen_prop_vals) if vals.size > 0],
        axis=0,
    ) if (train_prop_vals.size > 0 or val_prop_vals.size > 0 or gen_prop_vals.size > 0) else np.asarray([], dtype=float)

    if all_vals.size == 0:
        _debug_log(cfg, 'Skipping property distribution plot (no numeric property values found).')
        plt.close(fig)
        return

    bins = np.linspace(float(np.min(all_vals)), float(np.max(all_vals)), num=70)
    if np.allclose(bins[0], bins[-1]):
        bins = 40

    # Draw generated first with partial transparency; overlay train/validation as
    # high-contrast outlines so they remain visible even when generated dominates.

    only_train_vals = False
    if not only_train_vals:
        if gen_prop_vals.size > 0:
            ax.hist(
                gen_prop_vals,
                bins=bins,
                histtype='stepfilled',
                alpha=0.30,
                linewidth=1.0,
                edgecolor='#8A1C0F',
                color='#E8751A',
                label=f'Generated ({gen_prop_col_used or "predicted"})',
                zorder=1,
            )
            ax.hist(
                gen_prop_vals,
                bins=bins,
                histtype='step',
                alpha=0.95,
                linewidth=1.6,
                color='#B45309',
                zorder=2,
            )
            if train_prop_vals.size > 0:
                ax.hist(
                    train_prop_vals,
                    bins=bins,
                    histtype='step',
                    alpha=0.98,
                    linewidth=2.0,
                    color='#1E3A8A',
                    label=f'Train ({train_target_col or "target"})',
                    zorder=4,
                )

    if only_train_vals:
        # in case of only 
        ax.hist(
            train_prop_vals,
            bins=bins,
            histtype='stepfilled',
            alpha=0.3,
            linewidth=1.0,
            edgecolor="#FF1515",
            color='#FF1515',
            label=f'Train ({train_target_col or "target"})',
            zorder=4,
        )
        ax.hist(
            train_prop_vals,
            bins=bins,
            histtype='step',
            alpha=0.98,
            linewidth=2.0,
            color='#FF1515',
            zorder=4,
        )

    if val_prop_vals.size > 0:
        ax.hist(
            val_prop_vals,
            bins=bins,
            histtype='step',
            alpha=0.98,
            linewidth=2.0,
            color="#00FF15",
            label=f'Validation ({val_target_col or "target"})',
            zorder=5,
        )

    x_label = gen_prop_col_used or train_target_col or 'Property value'
    title_label = 'pIC50' if 'pic50' in str(x_label).lower() else str(x_label)
    
    title_lab = f'{title_label} distribution: train/val'
    if not only_train_vals:
        title_lab += '/generated'
    
    
    ax.set_title(title_lab)
    
    
    
    ax.set_xlabel(x_label)
    ax.set_ylabel('Count')
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.35)
    ax.legend(frameon=True, framealpha=0.9)

    plt.tight_layout()
    _save_figure(cfg, cfg.distribution_plot_filename, dpi=180)
    plt.close(fig)


def _maybe_plot_train_loss(cfg: AnalysisConfig) -> None:
    if not bool(cfg.run_train_loss_plot):
        _debug_log(cfg, 'Skipping train-loss plot (run_train_loss_plot=False).')
        return
    history_path = os.path.join(cfg.train_folder, 'history.csv')
    if not os.path.exists(history_path):
        _debug_log(cfg, f'Skipping train-loss plot (history file not found): {history_path}')
        return

    df = pd.read_csv(history_path)
    if 'train_loss' not in df.columns or 'test_loss' not in df.columns:
        _debug_log(cfg, 'Skipping train-loss plot (train_loss/test_loss columns missing in history.csv).')
        return

    x = np.arange(len(df), dtype=np.int32)
    train_loss = pd.to_numeric(df['train_loss'], errors='coerce').to_numpy(dtype=float)
    test_loss = pd.to_numeric(df['test_loss'], errors='coerce').to_numpy(dtype=float)
    fig = plt.figure(figsize=(10, 5))
    plt.plot(x, train_loss, label='Train Loss')
    plt.plot(x, test_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    _save_figure(cfg, cfg.train_loss_plot_filename, dpi=180)
    plt.close(fig)


def _maybe_plot_tanimoto_histogram(gen_df: pd.DataFrame, cfg: AnalysisConfig) -> None:
    if not bool(cfg.run_tanimoto_histogram):
        _debug_log(cfg, 'Skipping Tanimoto histogram (run_tanimoto_histogram=False).')
        return
    if 'tanimoto_max_to_ref' not in gen_df.columns:
        _debug_log(cfg, "Skipping Tanimoto histogram ('tanimoto_max_to_ref' column missing).")
        return

    vals = gen_df['tanimoto_max_to_ref'].dropna().to_numpy(dtype=float)
    if vals.size == 0:
        _debug_log(cfg, 'Skipping Tanimoto histogram (no valid tanimoto values).')
        return

    fig = plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=100, alpha=1.0, color='#01FF22')
    plt.title('Average Tanimoto Similarity Distribution')
    plt.xlabel('Max Tanimoto Similarity to train Set')
    plt.ylabel('Frequency')
    plt.tight_layout()
    _save_figure(cfg, cfg.tanimoto_histogram_filename, dpi=180)
    plt.close(fig)


def _resolve_property_column(df: pd.DataFrame, requested: str | None, *, role: str) -> str | None:
    if requested and requested in df.columns:
        return requested
    if not requested:
        return None

    req = str(requested)
    candidates = [
        req,
        req.lower(),
        req.upper(),
        f'{role}_{req}',
        f'{role}_{req.lower()}',
        f'{role}_{req.upper()}',
    ]
    for col in df.columns:
        if col in candidates:
            return col
    for col in df.columns:
        if col.lower() in {c.lower() for c in candidates}:
            return col
    return None


def _maybe_plot_prediction_errors(gen_df: pd.DataFrame, cfg: AnalysisConfig) -> dict:
    if not bool(cfg.run_prediction_error_plot):
        _debug_log(cfg, 'Skipping prediction-error plot (run_prediction_error_plot=False).')
        return {}
    target_col = _resolve_property_column(gen_df, cfg.target_property_column, role='target')
    pred_col = _resolve_property_column(gen_df, cfg.predicted_property_column, role='pred')
    _debug_log(cfg, f'Prediction error columns -> target={target_col!r}, prediction={pred_col!r}')
    if not target_col or not pred_col:
        return {}

    gt = pd.to_numeric(gen_df[target_col], errors='coerce').to_numpy(dtype=float)
    pred = pd.to_numeric(gen_df[pred_col], errors='coerce').to_numpy(dtype=float)
    valid_mask = np.isfinite(gt) & np.isfinite(pred)
    if int(np.sum(valid_mask)) == 0:
        _debug_log(cfg, 'Skipping prediction-error plot (no finite target/prediction pairs).')
        return {}

    gt = gt[valid_mask]
    pred = pred[valid_mask]
    abs_err = np.abs(gt - pred)
    mse = float(np.mean((gt - pred) ** 2))
    mae = float(np.mean(abs_err))
    medae = float(np.median(abs_err))
    stdae = float(np.std(abs_err))

    fig = plt.figure(figsize=(10, 5))
    edge_kwargs = _get_scatter_edgecolor_kwargs(cfg)
    plt.scatter(gt, abs_err, alpha=0.5, s=16, **edge_kwargs)
    plt.xlabel(f'Ground Truth {target_col}')
    plt.ylabel('Absolute Error')
    plt.title(f'Absolute Error vs Ground Truth ({target_col})')
    plt.tight_layout()
    _save_figure(cfg, cfg.prediction_error_plot_filename, dpi=180)
    plt.close(fig)

    return {
        'prediction_error_count': int(gt.shape[0]),
        'prediction_target_column': target_col,
        'prediction_column': pred_col,
        'prediction_mse': mse,
        'prediction_mae': mae,
        'prediction_median_ae': medae,
        'prediction_std_ae': stdae,
    }


def _safe_tsne_perplexity(desired: float, n_samples: int) -> float:
    if n_samples <= 3:
        return 2.0
    return float(max(2.0, min(float(desired), float(n_samples - 1))))


def _sample_smiles_for_embeddings(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame | None,
    gen_df: pd.DataFrame,
    cfg: AnalysisConfig,
) -> tuple[list[str], list[str], list[str], np.ndarray | None]:
    train_sample = train_df.sample(n=min(int(cfg.embedding_train_sample), len(train_df)), random_state=int(cfg.random_seed))
    validation_sample = None
    if validation_df is not None and len(validation_df) > 0:
        validation_sample = validation_df.sample(
            n=min(int(cfg.embedding_validation_sample), len(validation_df)),
            random_state=int(cfg.random_seed),
        )
    gen_sample = gen_df.sample(n=min(int(cfg.embedding_generated_sample), len(gen_df)), random_state=int(cfg.random_seed)).copy()

    gen_smiles_col = 'canonical_smiles' if 'canonical_smiles' in gen_sample.columns else cfg.smiles_column
    train_smiles = train_sample[cfg.smiles_column].astype(str).tolist()
    validation_smiles = (
        validation_sample[cfg.smiles_column].astype(str).tolist()
        if validation_sample is not None
        else []
    )
    gen_smiles = gen_sample[gen_smiles_col].astype(str).tolist()
    gen_tanimoto = (
        gen_sample['tanimoto_max_to_ref'].to_numpy(dtype=float)
        if 'tanimoto_max_to_ref' in gen_sample.columns
        else None
    )
    return train_smiles, validation_smiles, gen_smiles, gen_tanimoto


def _run_chemical_space_embedding(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame | None,
    gen_df: pd.DataFrame,
    cfg: AnalysisConfig,
) -> None:
    if not bool(cfg.run_chemical_space):
        _debug_log(cfg, 'Skipping chemical-space embedding (run_chemical_space=False).')
        return

    train_smiles, validation_smiles, gen_smiles, gen_tanimoto = _sample_smiles_for_embeddings(
        train_df,
        validation_df,
        gen_df,
        cfg,
    )
    fp_gen = morgan_fp_generator(radius=cfg.tanimoto_radius, n_bits=cfg.tanimoto_n_bits)

    x_train, _ = smiles_list_to_fp_matrix(fp_gen, train_smiles, dtype=np.int8)
    x_validation = np.zeros((0, int(cfg.tanimoto_n_bits)), dtype=np.int8)
    if len(validation_smiles) > 0:
        x_validation, _ = smiles_list_to_fp_matrix(fp_gen, validation_smiles, dtype=np.int8)
    x_gen, gen_valid_mask = smiles_list_to_fp_matrix(fp_gen, gen_smiles, dtype=np.int8)
    if x_train.shape[0] == 0 or x_gen.shape[0] == 0:
        _debug_log(cfg, 'Skipping chemical-space embedding (no valid fingerprint rows in train or generated sample).')
        return

    _debug_log(
        cfg,
        f'Chemical-space molecules -> train_valid={x_train.shape[0]}, '
        f'validation_valid={x_validation.shape[0]}, generated_valid={x_gen.shape[0]}',
    )

    if gen_tanimoto is not None:
        gen_tanimoto = gen_tanimoto[gen_valid_mask]

    blocks = [x_train]
    if x_validation.shape[0] > 0:
        blocks.append(x_validation)
    blocks.append(x_gen)
    x_all = np.vstack(blocks)

    idx_train_end = int(x_train.shape[0])
    idx_validation_end = int(idx_train_end + x_validation.shape[0])

    pca_vis = PCA(n_components=2, random_state=int(cfg.random_seed))
    x_pca_2d = pca_vis.fit_transform(x_all)
    fig = plt.figure(figsize=(8, 6))
    edge_kwargs = _get_scatter_edgecolor_kwargs(cfg)
    plt.scatter(
        x_pca_2d[:idx_train_end, 0],
        x_pca_2d[:idx_train_end, 1],
        s=int(cfg.embedding_point_size_train),
        alpha=0.70,
        label='Train',
        **edge_kwargs,
    )
    if x_validation.shape[0] > 0:
        plt.scatter(
            x_pca_2d[idx_train_end:idx_validation_end, 0],
            x_pca_2d[idx_train_end:idx_validation_end, 1],
            s=int(cfg.embedding_point_size_validation),
            alpha=0.70,
            label='Validation',
            **edge_kwargs,
        )
    plt.scatter(
        x_pca_2d[idx_validation_end:, 0],
        x_pca_2d[idx_validation_end:, 1],
        s=int(cfg.embedding_point_size_generated),
        alpha=0.70,
        label='Generated',
        **edge_kwargs,
    )
    plt.legend()
    plt.title('PCA (2D) on Morgan fingerprints: train/val/generated')
    plt.tight_layout()
    _save_figure(cfg, cfg.chemical_pca_plot_filename, dpi=180)
    plt.close(fig)

    pre_dim = int(cfg.chemical_pca_pre_dim)
    if pre_dim > 0 and pre_dim < x_all.shape[1] and pre_dim < x_all.shape[0]:
        x_pre = PCA(n_components=pre_dim, random_state=int(cfg.random_seed)).fit_transform(x_all)
    else:
        x_pre = x_all

    perpl = _safe_tsne_perplexity(cfg.chemical_tsne_perplexity, x_pre.shape[0])
    tsne = TSNE(
        n_components=2,
        random_state=int(cfg.random_seed),
        perplexity=perpl,
        init='random',
        learning_rate='auto',
    )
    x_tsne = tsne.fit_transform(x_pre)
    train_tsne = x_tsne[:idx_train_end]
    validation_tsne = x_tsne[idx_train_end:idx_validation_end]
    gen_tsne = x_tsne[idx_validation_end:]

    fig = plt.figure(figsize=(8, 6))
    edge_kwargs = _get_scatter_edgecolor_kwargs(cfg)
    plt.scatter(train_tsne[:, 0], train_tsne[:, 1], s=int(cfg.embedding_point_size_train), alpha=float(cfg.embedding_alpha), label='Train', **edge_kwargs)
    if validation_tsne.shape[0] > 0:
        plt.scatter(
            validation_tsne[:, 0],
            validation_tsne[:, 1],
            s=int(cfg.embedding_point_size_validation),
            alpha=float(cfg.embedding_alpha),
            label='Validation',
            **edge_kwargs,
        )
    plt.scatter(gen_tsne[:, 0], gen_tsne[:, 1], s=int(cfg.embedding_point_size_generated), alpha=float(cfg.embedding_alpha), label='Generated', **edge_kwargs)
    plt.legend()
    plt.title(f't-SNE on Morgan-fps (train/val/gen PCA-{pre_dim} preprojection)')
    plt.tight_layout()
    _save_figure(cfg, cfg.chemical_tsne_plot_filename, dpi=300)
    plt.close(fig)

    if gen_tanimoto is not None and gen_tanimoto.shape[0] == gen_tsne.shape[0]:
        cmap = LinearSegmentedColormap.from_list('custom', ['red', 'yellow', 'green'], N=256)
        fig = plt.figure(figsize=(8, 6))
        edge_kwargs = _get_scatter_edgecolor_kwargs(cfg)
        sc2 = plt.scatter(
            gen_tsne[:, 0],
            gen_tsne[:, 1],
            s=int(cfg.embedding_point_size_generated),
            alpha=0.85,
            c=gen_tanimoto,
            vmin=0.0,
            vmax=1.0,
            cmap=cmap,
            **edge_kwargs,
        )
        plt.colorbar(sc2, label='Max Tanimoto to train')
        plt.title('Generated t-SNE colored by max Tanimoto to train')
        plt.tight_layout()
        _save_figure(cfg, cfg.chemical_tsne_tanimoto_plot_filename, dpi=300)
        plt.close(fig)


def _run_descriptor_space_embedding(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame | None,
    gen_df: pd.DataFrame,
    cfg: AnalysisConfig,
) -> None:
    if not bool(cfg.run_descriptor_space):
        _debug_log(cfg, 'Skipping descriptor-space embedding (run_descriptor_space=False).')
        return

    train_smiles, validation_smiles, gen_smiles, gen_tanimoto = _sample_smiles_for_embeddings(
        train_df,
        validation_df,
        gen_df,
        cfg,
    )

    x_train_desc, _ = smiles_list_to_descriptor_matrix(train_smiles, cfg.descriptor_names)
    x_validation_desc = np.zeros((0, len(cfg.descriptor_names)), dtype=np.float32)
    if len(validation_smiles) > 0:
        x_validation_desc, _ = smiles_list_to_descriptor_matrix(validation_smiles, cfg.descriptor_names)
    x_gen_desc, gen_desc_valid_mask = smiles_list_to_descriptor_matrix(gen_smiles, cfg.descriptor_names)
    if x_train_desc.shape[0] == 0 or x_gen_desc.shape[0] == 0:
        _debug_log(cfg, 'Skipping descriptor-space embedding (no valid descriptor rows in train or generated sample).')
        return

    _debug_log(
        cfg,
        f'Descriptor-space molecules -> train_valid={x_train_desc.shape[0]}, '
        f'validation_valid={x_validation_desc.shape[0]}, generated_valid={x_gen_desc.shape[0]}',
    )

    if gen_tanimoto is not None:
        gen_tanimoto = gen_tanimoto[gen_desc_valid_mask]

    blocks = [x_train_desc]
    if x_validation_desc.shape[0] > 0:
        blocks.append(x_validation_desc)
    blocks.append(x_gen_desc)
    x_all_desc = np.vstack(blocks)
    x_all_scaled = StandardScaler().fit_transform(x_all_desc)
    idx_train_end = int(x_train_desc.shape[0])
    idx_validation_end = int(idx_train_end + x_validation_desc.shape[0])

    pca2 = PCA(n_components=2, random_state=int(cfg.random_seed)).fit_transform(x_all_scaled)
    fig = plt.figure(figsize=(8, 6))
    edge_kwargs = _get_scatter_edgecolor_kwargs(cfg)
    plt.scatter(
        pca2[:idx_train_end, 0],
        pca2[:idx_train_end, 1],
        s=int(cfg.embedding_point_size_train),
        alpha=0.70,
        label='Train',
        **edge_kwargs,
    )
    if x_validation_desc.shape[0] > 0:
        plt.scatter(
            pca2[idx_train_end:idx_validation_end, 0],
            pca2[idx_train_end:idx_validation_end, 1],
            s=int(cfg.embedding_point_size_validation),
            alpha=0.70,
            label='Validation',
            **edge_kwargs,
        )
    plt.scatter(
        pca2[idx_validation_end:, 0],
        pca2[idx_validation_end:, 1],
        s=int(cfg.embedding_point_size_generated),
        alpha=0.70,
        label='Generated',
        **edge_kwargs,
    )
    plt.legend()
    plt.title('PCA (2D) on RDKit descriptors: train vs validation vs generated')
    plt.tight_layout()
    _save_figure(cfg, cfg.descriptor_pca_plot_filename, dpi=300)
    plt.close(fig)

    pre_dim = int(cfg.descriptor_pca_pre_dim)
    if pre_dim > 0 and pre_dim < x_all_scaled.shape[1] and pre_dim < x_all_scaled.shape[0]:
        x_pre = PCA(n_components=pre_dim, random_state=int(cfg.random_seed)).fit_transform(x_all_scaled)
    else:
        x_pre = x_all_scaled

    perpl = _safe_tsne_perplexity(cfg.descriptor_tsne_perplexity, x_pre.shape[0])
    x_tsne = TSNE(
        n_components=2,
        random_state=int(cfg.random_seed),
        perplexity=perpl,
        init='random',
        learning_rate='auto',
    ).fit_transform(x_pre)
    train_tsne = x_tsne[:idx_train_end]
    validation_tsne = x_tsne[idx_train_end:idx_validation_end]
    gen_tsne = x_tsne[idx_validation_end:]

    fig = plt.figure(figsize=(8, 6))
    edge_kwargs = _get_scatter_edgecolor_kwargs(cfg)
    plt.scatter(train_tsne[:, 0], train_tsne[:, 1], s=int(cfg.embedding_point_size_train), alpha=float(cfg.embedding_alpha), label='Train', **edge_kwargs)
    if validation_tsne.shape[0] > 0:
        plt.scatter(
            validation_tsne[:, 0],
            validation_tsne[:, 1],
            s=int(cfg.embedding_point_size_validation),
            alpha=float(cfg.embedding_alpha),
            label='Validation',
            **edge_kwargs,
        )
    plt.scatter(gen_tsne[:, 0], gen_tsne[:, 1], s=int(cfg.embedding_point_size_generated), alpha=float(cfg.embedding_alpha), label='Generated', **edge_kwargs)
    plt.legend()
    plt.title('t-SNE on RDKit descriptors (scaled): train vs validation vs generated')
    plt.tight_layout()
    _save_figure(cfg, cfg.descriptor_tsne_plot_filename, dpi=180)
    plt.close(fig)

    if gen_tanimoto is not None and gen_tanimoto.shape[0] == gen_tsne.shape[0]:
        cmap = LinearSegmentedColormap.from_list('custom', ['red', 'yellow', 'green'], N=256)
        fig = plt.figure(figsize=(8, 6))
        edge_kwargs = _get_scatter_edgecolor_kwargs(cfg)
        sc2 = plt.scatter(
            gen_tsne[:, 0],
            gen_tsne[:, 1],
            s=int(cfg.embedding_point_size_generated),
            alpha=0.85,
            c=gen_tanimoto,
            vmin=0.0,
            vmax=1.0,
            cmap=cmap,
            **edge_kwargs,
        )
        plt.colorbar(sc2, label='Max Tanimoto to train')
        plt.title('Generated descriptor t-SNE colored by max Tanimoto to train')
        plt.tight_layout()
        _save_figure(cfg, cfg.descriptor_tsne_tanimoto_plot_filename, dpi=180)
        plt.close(fig)


def _maybe_plot_scaffold_distribution(
    train_scaffolds: list,
    validation_scaffolds: list,
    gen_scaffolds: list,
    cfg: AnalysisConfig,
) -> None:
    if not bool(cfg.save_scaffold_plot):
        _debug_log(cfg, 'Skipping scaffold distribution plot (save_scaffold_plot=False).')
        return

    train_counts = scaffold_counts(train_scaffolds)
    validation_counts = scaffold_counts(validation_scaffolds)
    gen_counts = scaffold_counts(gen_scaffolds)
    keys = list(set(train_counts.keys()) | set(validation_counts.keys()) | set(gen_counts.keys()))
    keys_sorted = sorted(
        keys,
        key=lambda k: train_counts.get(k, 0) + validation_counts.get(k, 0) + gen_counts.get(k, 0),
        reverse=True,
    )[:20]

    if len(keys_sorted) == 0:
        _debug_log(cfg, 'Skipping scaffold distribution plot (no scaffold keys to plot).')
        return

    os.makedirs(cfg.output_dir, exist_ok=True)
    x = np.arange(len(keys_sorted))
    train_vals = [train_counts.get(k, 0) for k in keys_sorted]
    validation_vals = [validation_counts.get(k, 0) for k in keys_sorted]
    gen_vals = [gen_counts.get(k, 0) for k in keys_sorted]

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.25
    ax.bar(x - width, train_vals, width=width, label='Train')
    ax.bar(x, validation_vals, width=width, label='Validation')
    ax.bar(x + width, gen_vals, width=width, label='Generated')
    ax.set_xticks(x)
    ax.set_xticklabels(keys_sorted, rotation=90)
    ax.set_title('Top scaffolds: train vs validation vs generated')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    _save_figure(cfg, cfg.scaffold_plot_filename, dpi=180)
    plt.close(fig)


def _draw_scaffold_grid(scaffold_counts_dict: dict[str, int], out_path: str, n_top: int, n_cols: int, title: str, cfg: AnalysisConfig) -> dict:
    items = sorted(scaffold_counts_dict.items(), key=lambda x: x[1], reverse=True)[: max(0, int(n_top))]
    mols = []
    legends = []
    for scaffold_smiles, count in items:
        mol = safe_mol_from_smiles(scaffold_smiles)
        if mol is None:
            continue
        mols.append(mol)
        legends.append(f'count={int(count)}')

    if len(mols) == 0:
        return {'saved': False, 'num_drawn': 0}

    n_cols_eff = max(1, int(n_cols))
    n_rows = int(math.ceil(len(mols) / float(n_cols_eff)))
    fig_w = n_cols_eff * 2.5
    fig_h = n_rows * 2.5

    fig, axes = plt.subplots(n_rows, n_cols_eff, figsize=(fig_w, fig_h))
    if isinstance(axes, np.ndarray):
        axes_list = axes.flatten().tolist()
    else:
        axes_list = [axes]

    for ax in axes_list:
        ax.axis('off')

    for i, (mol, legend) in enumerate(zip(mols, legends)):
        img = Draw.MolToImage(mol, size=(250, 250))
        axes_list[i].imshow(img)
        axes_list[i].set_title(legend, fontsize=13, fontweight='bold')

    fig.suptitle(title, fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    _debug_log(cfg, f'Wrote scaffold grid: {out_path}')
    return {'saved': True, 'num_drawn': int(len(mols))}


def _maybe_save_scaffold_grids_and_stats(
    train_scaffolds: list,
    validation_scaffolds: list,
    gen_scaffolds: list,
    cfg: AnalysisConfig,
) -> dict:
    train_counts = scaffold_counts(train_scaffolds)
    validation_counts = scaffold_counts(validation_scaffolds)
    gen_counts = scaffold_counts(gen_scaffolds)
    train_set = set(train_counts.keys())
    validation_set = set(validation_counts.keys())
    gen_set = set(gen_counts.keys())

    overlap = int(len(train_set & gen_set))
    overlap_validation_generated = int(len(validation_set & gen_set))
    overlap_train_validation = int(len(train_set & validation_set))
    novel = int(len(gen_set - train_set))
    novel_vs_train_and_validation = int(len(gen_set - (train_set | validation_set)))

    novel_gen_counts = {k: int(v) for k, v in gen_counts.items() if k not in train_set}
    novel_gen_counts_vs_train_validation = {
        k: int(v) for k, v in gen_counts.items() if k not in (train_set | validation_set)
    }

    scaffold_stats = {
        'unique_train_scaffolds': int(len(train_set)),
        'unique_validation_scaffolds': int(len(validation_set)),
        'unique_generated_scaffolds': int(len(gen_set)),
        'overlap_scaffolds': overlap,
        'overlap_train_validation_scaffolds': overlap_train_validation,
        'overlap_validation_generated_scaffolds': overlap_validation_generated,
        'novel_generated_scaffolds': novel,
        'novel_generated_scaffolds_vs_train_validation': novel_vs_train_and_validation,
        'top_train_scaffolds': [
            {'scaffold': str(k), 'count': int(v)} for k, v in train_counts.most_common(int(cfg.scaffold_grid_top_n))
        ],
        'top_validation_scaffolds': [
            {'scaffold': str(k), 'count': int(v)} for k, v in validation_counts.most_common(int(cfg.scaffold_grid_top_n))
        ],
        'top_generated_scaffolds': [
            {'scaffold': str(k), 'count': int(v)} for k, v in gen_counts.most_common(int(cfg.scaffold_grid_top_n))
        ],
        'top_novel_generated_scaffolds': [
            {'scaffold': str(k), 'count': int(v)}
            for k, v in sorted(novel_gen_counts.items(), key=lambda x: x[1], reverse=True)[: int(cfg.scaffold_grid_top_n)]
        ],
        'top_novel_generated_scaffolds_vs_train_validation': [
            {'scaffold': str(k), 'count': int(v)}
            for k, v in sorted(novel_gen_counts_vs_train_validation.items(), key=lambda x: x[1], reverse=True)[
                : int(cfg.scaffold_grid_top_n)
            ]
        ],
    }

    stats_path = os.path.join(cfg.output_dir, cfg.scaffold_stats_filename)
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(scaffold_stats, f, indent=2)
    _debug_log(cfg, f'Wrote scaffold stats: {stats_path}')

    grid_meta = {
        'train_top_scaffold_grid': {'saved': False, 'num_drawn': 0},
        'generated_top_scaffold_grid': {'saved': False, 'num_drawn': 0},
        'novel_top_scaffold_grid': {'saved': False, 'num_drawn': 0},
    }

    if bool(cfg.save_scaffold_grids):
        grid_meta['train_top_scaffold_grid'] = _draw_scaffold_grid(
            dict(train_counts),
            os.path.join(cfg.output_dir, cfg.train_scaffold_grid_filename),
            n_top=int(cfg.scaffold_grid_top_n),
            n_cols=int(cfg.scaffold_grid_n_cols),
            title='Train set',
            cfg=cfg,
        )
        grid_meta['generated_top_scaffold_grid'] = _draw_scaffold_grid(
            dict(gen_counts),
            os.path.join(cfg.output_dir, cfg.generated_scaffold_grid_filename),
            n_top=int(cfg.scaffold_grid_top_n),
            n_cols=int(cfg.scaffold_grid_n_cols),
            title='Generated set',
            cfg=cfg,
        )
        grid_meta['novel_top_scaffold_grid'] = _draw_scaffold_grid(
            novel_gen_counts,
            os.path.join(cfg.output_dir, cfg.novel_scaffold_grid_filename),
            n_top=int(cfg.scaffold_grid_top_n),
            n_cols=int(cfg.scaffold_grid_n_cols),
            title='Novel generated scaffolds',
            cfg=cfg,
        )

    scaffold_stats['scaffold_stats_file'] = stats_path
    scaffold_stats.update(grid_meta)
    return scaffold_stats


def _subset_df(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= int(max_rows):
        return df.reset_index(drop=True)
    return df.sample(n=int(max_rows), random_state=int(seed)).reset_index(drop=True)


def run_analysis_pipeline(cfg: AnalysisConfig) -> dict:
    os.makedirs(cfg.output_dir, exist_ok=True)
    _debug_log(cfg, f'Output directory ready: {cfg.output_dir}')
    _debug_log(
        cfg,
        f'Inputs -> train_data_path={cfg.train_data_path}, '
        f'validation_data_path={cfg.validation_data_path}, generated_data_path={cfg.generated_data_path}',
    )
    _debug_log(cfg, f'Target settings -> target_property_column={cfg.target_property_column!r}, predicted_property_column={cfg.predicted_property_column!r}')
    _maybe_plot_train_loss(cfg)

    train_df = load_train_dataframe(cfg.train_data_path, smiles_column=cfg.smiles_column, sep=cfg.train_sep)
    validation_df = None
    if cfg.validation_data_path:
        validation_df = load_train_dataframe(
            cfg.validation_data_path,
            smiles_column=cfg.smiles_column,
            sep=cfg.validation_sep,
        )
    gen_df = load_generated_dataframe(cfg.generated_data_path, sep=cfg.generated_sep)
    _debug_log(
        cfg,
        f'Loaded rows -> train={len(train_df)}, '
        f'validation={(0 if validation_df is None else len(validation_df))}, generated={len(gen_df)}',
    )

    if cfg.smiles_column not in train_df.columns:
        raise ValueError(f"Train data is missing smiles column '{cfg.smiles_column}'")
    if validation_df is not None and cfg.smiles_column not in validation_df.columns:
        raise ValueError(f"Validation data is missing smiles column '{cfg.smiles_column}'")
    if cfg.smiles_column not in gen_df.columns:
        raise ValueError(f"Generated data is missing smiles column '{cfg.smiles_column}'")

    train_df = _subset_df(train_df, cfg.train_max, cfg.random_seed)
    if validation_df is not None:
        validation_df = _subset_df(validation_df, cfg.validation_max, cfg.random_seed)
    gen_df = _subset_df(gen_df, cfg.generated_max, cfg.random_seed)
    _debug_log(
        cfg,
        f'Subset rows -> train={len(train_df)} (max={cfg.train_max}), '
        f'validation={(0 if validation_df is None else len(validation_df))} (max={cfg.validation_max}), '
        f'generated={len(gen_df)} (max={cfg.generated_max})',
    )

    train_smiles = train_df[cfg.smiles_column].astype(str).tolist()
    validation_smiles = (
        validation_df[cfg.smiles_column].astype(str).tolist() if validation_df is not None else []
    )
    gen_smiles = gen_df[cfg.smiles_column].astype(str).tolist()

    train_mols = [safe_mol_from_smiles(s) for s in train_smiles]
    validation_mols = [safe_mol_from_smiles(s) for s in validation_smiles]
    gen_mols = [safe_mol_from_smiles(s) for s in gen_smiles]

    gen_df = gen_df.copy()
    gen_df['is_valid'] = [m is not None for m in gen_mols]
    gen_df['canonical_smiles'] = [canonicalize_smiles(s) for s in gen_smiles]
    _debug_log(cfg, f'Validity progress -> valid_generated={int(np.sum(gen_df["is_valid"].to_numpy(dtype=bool)))}/{len(gen_df)}')

    train_scaffolds = [safe_murcko_scaffold_smiles(m) for m in train_mols]
    validation_scaffolds = [safe_murcko_scaffold_smiles(m) for m in validation_mols]
    gen_scaffolds = [safe_murcko_scaffold_smiles(m) for m in gen_mols]
    gen_df['murcko_scaffold_smiles'] = gen_scaffolds

    fp_gen = morgan_fp_generator(radius=cfg.tanimoto_radius, n_bits=cfg.tanimoto_n_bits)
    ref_fps = [morgan_fp(fp_gen, m) for m in train_mols if m is not None]
    query_valid_idx = [i for i, m in enumerate(gen_mols) if m is not None]
    query_fps = [morgan_fp(fp_gen, gen_mols[i]) for i in query_valid_idx]

    max_sims_valid, mean_similarity = max_tanimoto_to_reference(query_fps, ref_fps)
    max_sim = np.full((len(gen_df),), np.nan, dtype=np.float32)
    for j, idx in enumerate(query_valid_idx):
        max_sim[idx] = max_sims_valid[j]
    gen_df['tanimoto_max_to_ref'] = max_sim


    #======================================================
    # NOTE: Diversity is defined as 1 - mean_similarity, where:
    # mean similarity is the avg of max Tanimoto sim of gen vs train
    # ========================================================

    diversity = float(1.0 - mean_similarity) if not np.isnan(mean_similarity) else 0.0
    gen_df['diversity_score'] = diversity
    _debug_log(cfg, f'Similarity stats -> mean_tanimoto={mean_similarity}, diversity_score={diversity}')

    target_col = _resolve_property_column(gen_df, cfg.target_property_column, role='target')
    pred_col = _resolve_property_column(gen_df, cfg.predicted_property_column, role='pred')
    _debug_log(cfg, f'Resolved generated columns -> target={target_col!r}, prediction={pred_col!r}')
    if target_col and pred_col:
        if target_col in gen_df.columns and pred_col in gen_df.columns:
            diff = gen_df[target_col].astype(float) - gen_df[pred_col].astype(float)
            gen_df['abs_prediction_error'] = diff.abs()
            _debug_log(cfg, "Computed 'abs_prediction_error' column.")

    processed_csv_path = os.path.join(cfg.output_dir, cfg.processed_csv_filename)
    gen_df.to_csv(processed_csv_path, index=False)
    _debug_log(cfg, f'Wrote processed CSV: {processed_csv_path}')

    _maybe_plot_property_distributions(train_df, validation_df, gen_df, cfg)
    _maybe_plot_scaffold_distribution(train_scaffolds, validation_scaffolds, gen_scaffolds, cfg)
    _maybe_plot_tanimoto_histogram(gen_df, cfg)
    pred_error_stats = _maybe_plot_prediction_errors(gen_df, cfg)
    _run_chemical_space_embedding(train_df, validation_df, gen_df, cfg)
    _run_descriptor_space_embedding(train_df, validation_df, gen_df, cfg)

    scaffold_stats = _maybe_save_scaffold_grids_and_stats(
        train_scaffolds,
        validation_scaffolds,
        gen_scaffolds,
        cfg,
    )

    train_scaf_set = set(s for s in train_scaffolds if s is not None)
    validation_scaf_set = set(s for s in validation_scaffolds if s is not None)
    gen_scaf_set = set(s for s in gen_scaffolds if s is not None)
    overlap = len(train_scaf_set & gen_scaf_set)
    overlap_train_validation = len(train_scaf_set & validation_scaf_set)
    overlap_validation_generated = len(validation_scaf_set & gen_scaf_set)
    novel = len(gen_scaf_set - train_scaf_set)
    novel_vs_train_validation = len(gen_scaf_set - (train_scaf_set | validation_scaf_set))

    vun_metrics = _try_read_vun_from_quality_summary_csv(cfg.quality_summary_data_path)
    if vun_metrics is None:
        vun_metrics = _compute_vun_from_loaded_data(
            train_smiles=train_smiles,
            gen_df=gen_df,
        )
    _debug_log(
        cfg,
        'V.U.N summary -> '
        f"source={vun_metrics.get('source')}, "
        f"validity={vun_metrics.get('validity')}, "
        f"uniqueness={vun_metrics.get('uniqueness')}, "
        f"novelty={vun_metrics.get('novelty')}",
    )
    #first part of the needle. 

    summary = {
        'profile_name': cfg.profile_name,
        'train_folder': cfg.train_folder,
        'train_data_path': cfg.train_data_path,
        'validation_data_path': cfg.validation_data_path,
        'generated_data_path': cfg.generated_data_path,
        'output_dir': cfg.output_dir,
        'num_train_rows': int(len(train_df)),
        'num_train_valid': int(sum(1 for m in train_mols if m is not None)),
        'num_validation_rows': int(0 if validation_df is None else len(validation_df)),
        'num_validation_valid': int(sum(1 for m in validation_mols if m is not None)),
        'num_generated_rows': int(len(gen_df)),
        'num_generated_valid': int(int(gen_df['is_valid'].sum())),
        'vun': {
            'source': str(vun_metrics.get('source')),
            'quality_summary_csv_path': vun_metrics.get('quality_summary_csv_path'),
            'quality_run_scope': vun_metrics.get('quality_run_scope'),
            'quality_counts': dict(vun_metrics.get('quality_counts', {})),
            'validity': float(vun_metrics.get('validity', 0.0)),
            'uniqueness': float(vun_metrics.get('uniqueness', 0.0)),
            'novelty': float(vun_metrics.get('novelty', 0.0)),
            'acceptance_rate': (
                None
                if vun_metrics.get('acceptance_rate') is None
                else float(vun_metrics.get('acceptance_rate'))
            ),
            'valid_count': int(vun_metrics.get('valid_count', 0)),
            'unique_count': int(vun_metrics.get('unique_count', 0)),
            'novel_count': int(vun_metrics.get('novel_count', 0)),
        },
        'diversity_score': float(diversity),
        'mean_tanimoto_all_pairs': float(mean_similarity) if not np.isnan(mean_similarity) else None,
        'unique_train_scaffolds': int(len(train_scaf_set)),
        'unique_validation_scaffolds': int(len(validation_scaf_set)),
        'unique_generated_scaffolds': int(len(gen_scaf_set)),
        'overlap_scaffolds': int(overlap),
        'overlap_train_validation_scaffolds': int(overlap_train_validation),
        'overlap_validation_generated_scaffolds': int(overlap_validation_generated),
        'novel_generated_scaffolds': int(novel),
        'novel_generated_scaffolds_vs_train_validation': int(novel_vs_train_validation),
        'processed_csv': processed_csv_path,
        'scaffold_stats_file': scaffold_stats.get('scaffold_stats_file'),
        'train_top_scaffold_grid': scaffold_stats.get('train_top_scaffold_grid', {}).get('saved', False),
        'generated_top_scaffold_grid': scaffold_stats.get('generated_top_scaffold_grid', {}).get('saved', False),
        'novel_top_scaffold_grid': scaffold_stats.get('novel_top_scaffold_grid', {}).get('saved', False),
    }
    summary.update(pred_error_stats)

    summary_path = os.path.join(cfg.output_dir, cfg.summary_json_filename)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({'config': asdict(cfg), 'summary': summary}, f, indent=2)
    _debug_log(cfg, f'Wrote summary JSON: {summary_path}')

    return summary
