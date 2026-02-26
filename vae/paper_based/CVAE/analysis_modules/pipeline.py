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


def _maybe_plot_property_distributions(train_df: pd.DataFrame, gen_df: pd.DataFrame, cfg: AnalysisConfig) -> None:
    if not bool(cfg.save_distribution_plot):
        return

    os.makedirs(cfg.output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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

    train_mw_col = _resolve_by_aliases(train_df, ('MW', 'MolWt', 'molecular_weight', 'mw'))
    gen_mw_col = _resolve_by_aliases(gen_df, ('MW', 'MolWt', 'molecular_weight', 'mw'))

    train_mw_vals = _extract_numeric_series(train_df, train_mw_col)
    gen_mw_vals = _extract_numeric_series(gen_df, gen_mw_col)

    if train_mw_vals.size == 0:
        train_mw_vals = _compute_mw_from_smiles(train_df, cfg.smiles_column)
    if gen_mw_vals.size == 0:
        gen_mw_vals = _compute_mw_from_smiles(gen_df, cfg.smiles_column)

    if train_mw_vals.size > 0:
        axes[0].hist(train_mw_vals, bins=80, alpha=0.65, label='Original train')
    if gen_mw_vals.size > 0:
        axes[0].hist(gen_mw_vals, bins=80, alpha=0.65, label='Generated')
    axes[0].set_title('MW distribution')
    axes[0].set_xlabel('MW')
    axes[0].set_ylabel('Count')
    if len(axes[0].patches) > 0:
        axes[0].legend()

    train_target_col = _resolve_property_column(train_df, cfg.target_property_column, role='target')
    gen_pred_col = _resolve_property_column(gen_df, cfg.predicted_property_column, role='pred')
    gen_target_col = _resolve_property_column(gen_df, cfg.target_property_column, role='target')

    train_prop_vals = _extract_numeric_series(train_df, train_target_col)
    gen_prop_col_used = gen_pred_col if gen_pred_col else gen_target_col
    gen_prop_vals = _extract_numeric_series(gen_df, gen_prop_col_used)

    if train_prop_vals.size > 0:
        axes[1].hist(
            train_prop_vals,
            bins=80,
            alpha=0.65,
            label=f'Original train ({train_target_col or "target"})',
            color='#0046AB',
        )

    if gen_prop_vals.size > 0:
        axes[1].hist(
            gen_prop_vals,
            bins=80,
            alpha=0.65,
            label=f'Generated ({gen_prop_col_used or "predicted"})',
            color='#FF8400',
        )

    axes[1].set_title('Property distribution: original train vs generated')
    axes[1].set_xlabel(gen_prop_col_used or train_target_col or 'Property value')
    axes[1].set_ylabel('Count')
    if len(axes[1].patches) > 0:
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, cfg.distribution_plot_filename), dpi=180)
    plt.close(fig)

    if train_mw_vals.size > 0 and gen_mw_vals.size > 0:
        mw_min = float(min(np.min(train_mw_vals), np.min(gen_mw_vals)))
        mw_max = float(max(np.max(train_mw_vals), np.max(gen_mw_vals)))
        if mw_max > mw_min:
            bins = np.linspace(mw_min, mw_max, num=81)
            train_counts, edges = np.histogram(train_mw_vals, bins=bins)
            gen_counts, _ = np.histogram(gen_mw_vals, bins=bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            diff = train_counts - gen_counts

            fig = plt.figure(figsize=(10, 5))
            plt.axhline(0.0, color='black', linewidth=1.0)
            plt.plot(centers, diff, color='#6A1B9A', linewidth=1.5)
            plt.fill_between(centers, diff, 0.0, alpha=0.25, color='#6A1B9A')
            plt.title('MW distribution difference (train - generated)')
            plt.xlabel('MW')
            plt.ylabel('Count difference per bin')
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.output_dir, cfg.mw_distribution_diff_plot_filename), dpi=180)
            plt.close(fig)


def _maybe_plot_train_loss(cfg: AnalysisConfig) -> None:
    if not bool(cfg.run_train_loss_plot):
        return
    history_path = os.path.join(cfg.train_folder, 'history.csv')
    if not os.path.exists(history_path):
        return

    df = pd.read_csv(history_path)
    if 'train_loss' not in df.columns or 'test_loss' not in df.columns:
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
    plt.savefig(os.path.join(cfg.output_dir, cfg.train_loss_plot_filename), dpi=180)
    plt.close(fig)


def _maybe_plot_tanimoto_histogram(gen_df: pd.DataFrame, cfg: AnalysisConfig) -> None:
    if not bool(cfg.run_tanimoto_histogram):
        return
    if 'tanimoto_max_to_ref' not in gen_df.columns:
        return

    vals = gen_df['tanimoto_max_to_ref'].dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return

    fig = plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=100, alpha=1.0, color='#01FF22')
    plt.title('Average Tanimoto Similarity Distribution')
    plt.xlabel('Max Tanimoto Similarity to train Set')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, cfg.tanimoto_histogram_filename), dpi=180)
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
        return {}
    target_col = _resolve_property_column(gen_df, cfg.target_property_column, role='target')
    pred_col = _resolve_property_column(gen_df, cfg.predicted_property_column, role='pred')
    if not target_col or not pred_col:
        return {}

    gt = pd.to_numeric(gen_df[target_col], errors='coerce').to_numpy(dtype=float)
    pred = pd.to_numeric(gen_df[pred_col], errors='coerce').to_numpy(dtype=float)
    valid_mask = np.isfinite(gt) & np.isfinite(pred)
    if int(np.sum(valid_mask)) == 0:
        return {}

    gt = gt[valid_mask]
    pred = pred[valid_mask]
    abs_err = np.abs(gt - pred)
    mse = float(np.mean((gt - pred) ** 2))
    mae = float(np.mean(abs_err))
    medae = float(np.median(abs_err))
    stdae = float(np.std(abs_err))

    fig = plt.figure(figsize=(10, 5))
    plt.scatter(gt, abs_err, alpha=0.5, s=8)
    plt.xlabel(f'Ground Truth {target_col}')
    plt.ylabel('Absolute Error')
    plt.title(f'Absolute Error vs Ground Truth ({target_col})')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, cfg.prediction_error_plot_filename), dpi=180)
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


def _sample_smiles_for_embeddings(train_df: pd.DataFrame, gen_df: pd.DataFrame, cfg: AnalysisConfig) -> tuple[list[str], pd.DataFrame, list[str], np.ndarray | None]:
    train_sample = train_df.sample(n=min(int(cfg.embedding_train_sample), len(train_df)), random_state=int(cfg.random_seed))
    gen_sample = gen_df.sample(n=min(int(cfg.embedding_generated_sample), len(gen_df)), random_state=int(cfg.random_seed)).copy()

    gen_smiles_col = 'canonical_smiles' if 'canonical_smiles' in gen_sample.columns else cfg.smiles_column
    train_smiles = train_sample[cfg.smiles_column].astype(str).tolist()
    gen_smiles = gen_sample[gen_smiles_col].astype(str).tolist()
    gen_tanimoto = (
        gen_sample['tanimoto_max_to_ref'].to_numpy(dtype=float)
        if 'tanimoto_max_to_ref' in gen_sample.columns
        else None
    )
    return train_smiles, gen_sample, gen_smiles, gen_tanimoto


def _run_chemical_space_embedding(train_df: pd.DataFrame, gen_df: pd.DataFrame, cfg: AnalysisConfig) -> None:
    if not bool(cfg.run_chemical_space):
        return

    train_smiles, _, gen_smiles, gen_tanimoto = _sample_smiles_for_embeddings(train_df, gen_df, cfg)
    fp_gen = morgan_fp_generator(radius=cfg.tanimoto_radius, n_bits=cfg.tanimoto_n_bits)

    x_train, _ = smiles_list_to_fp_matrix(fp_gen, train_smiles, dtype=np.int8)
    x_gen, gen_valid_mask = smiles_list_to_fp_matrix(fp_gen, gen_smiles, dtype=np.int8)
    if x_train.shape[0] == 0 or x_gen.shape[0] == 0:
        return

    if gen_tanimoto is not None:
        gen_tanimoto = gen_tanimoto[gen_valid_mask]

    x_all = np.vstack([x_train, x_gen])
    y_class = np.concatenate([
        np.zeros((x_train.shape[0],), dtype=np.int32),
        np.ones((x_gen.shape[0],), dtype=np.int32),
    ])

    pca_vis = PCA(n_components=2, random_state=int(cfg.random_seed))
    x_pca_2d = pca_vis.fit_transform(x_all)
    fig = plt.figure(figsize=(8, 6))
    sc = plt.scatter(x_pca_2d[:, 0], x_pca_2d[:, 1], s=int(cfg.embedding_point_size), alpha=0.7, c=y_class, cmap='coolwarm')
    plt.colorbar(sc, label='Class (0=train, 1=generated)')
    plt.title('PCA (2D) on Morgan fingerprints colored by Class')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, cfg.chemical_pca_plot_filename), dpi=180)
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
    train_tsne = x_tsne[: x_train.shape[0]]
    gen_tsne = x_tsne[x_train.shape[0] :]

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(train_tsne[:, 0], train_tsne[:, 1], s=int(cfg.embedding_point_size), alpha=float(cfg.embedding_alpha), label='Train')
    plt.scatter(gen_tsne[:, 0], gen_tsne[:, 1], s=int(cfg.embedding_point_size), alpha=float(cfg.embedding_alpha), label='Generated')
    plt.legend()
    plt.title(f't-SNE on Morgan fingerprints (PCA-{pre_dim} preprojection)')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, cfg.chemical_tsne_plot_filename), dpi=180)
    plt.close(fig)

    if gen_tanimoto is not None and gen_tanimoto.shape[0] == gen_tsne.shape[0]:
        cmap = LinearSegmentedColormap.from_list('custom', ['red', 'yellow', 'green'], N=256)
        fig = plt.figure(figsize=(8, 6))
        sc2 = plt.scatter(
            gen_tsne[:, 0],
            gen_tsne[:, 1],
            s=int(cfg.embedding_point_size),
            alpha=0.85,
            c=gen_tanimoto,
            vmin=0.0,
            vmax=1.0,
            cmap=cmap,
        )
        plt.colorbar(sc2, label='Max Tanimoto to train')
        plt.title('Generated t-SNE colored by max Tanimoto to train')
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.output_dir, cfg.chemical_tsne_tanimoto_plot_filename), dpi=180)
        plt.close(fig)


def _run_descriptor_space_embedding(train_df: pd.DataFrame, gen_df: pd.DataFrame, cfg: AnalysisConfig) -> None:
    if not bool(cfg.run_descriptor_space):
        return

    train_smiles, _, gen_smiles, gen_tanimoto = _sample_smiles_for_embeddings(train_df, gen_df, cfg)

    x_train_desc, _ = smiles_list_to_descriptor_matrix(train_smiles, cfg.descriptor_names)
    x_gen_desc, gen_desc_valid_mask = smiles_list_to_descriptor_matrix(gen_smiles, cfg.descriptor_names)
    if x_train_desc.shape[0] == 0 or x_gen_desc.shape[0] == 0:
        return

    if gen_tanimoto is not None:
        gen_tanimoto = gen_tanimoto[gen_desc_valid_mask]

    x_all_desc = np.vstack([x_train_desc, x_gen_desc])
    x_all_scaled = StandardScaler().fit_transform(x_all_desc)
    y_class = np.concatenate([
        np.zeros((x_train_desc.shape[0],), dtype=np.int32),
        np.ones((x_gen_desc.shape[0],), dtype=np.int32),
    ])

    pca2 = PCA(n_components=2, random_state=int(cfg.random_seed)).fit_transform(x_all_scaled)
    fig = plt.figure(figsize=(8, 6))
    sc = plt.scatter(pca2[:, 0], pca2[:, 1], s=int(cfg.embedding_point_size), alpha=0.7, c=y_class, cmap='coolwarm')
    plt.colorbar(sc, label='Class (0=train, 1=generated)')
    plt.title('PCA (2D) on RDKit descriptors (scaled)')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, cfg.descriptor_pca_plot_filename), dpi=180)
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
    train_tsne = x_tsne[: x_train_desc.shape[0]]
    gen_tsne = x_tsne[x_train_desc.shape[0] :]

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(train_tsne[:, 0], train_tsne[:, 1], s=int(cfg.embedding_point_size), alpha=float(cfg.embedding_alpha), label='Train')
    plt.scatter(gen_tsne[:, 0], gen_tsne[:, 1], s=int(cfg.embedding_point_size), alpha=float(cfg.embedding_alpha), label='Generated')
    plt.legend()
    plt.title('t-SNE on RDKit descriptors (scaled)')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, cfg.descriptor_tsne_plot_filename), dpi=180)
    plt.close(fig)

    if gen_tanimoto is not None and gen_tanimoto.shape[0] == gen_tsne.shape[0]:
        cmap = LinearSegmentedColormap.from_list('custom', ['red', 'yellow', 'green'], N=256)
        fig = plt.figure(figsize=(8, 6))
        sc2 = plt.scatter(
            gen_tsne[:, 0],
            gen_tsne[:, 1],
            s=int(cfg.embedding_point_size),
            alpha=0.85,
            c=gen_tanimoto,
            vmin=0.0,
            vmax=1.0,
            cmap=cmap,
        )
        plt.colorbar(sc2, label='Max Tanimoto to train')
        plt.title('Generated descriptor t-SNE colored by max Tanimoto to train')
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.output_dir, cfg.descriptor_tsne_tanimoto_plot_filename), dpi=180)
        plt.close(fig)


def _maybe_plot_scaffold_distribution(train_scaffolds: list, gen_scaffolds: list, cfg: AnalysisConfig) -> None:
    if not bool(cfg.save_scaffold_plot):
        return

    train_counts = scaffold_counts(train_scaffolds)
    gen_counts = scaffold_counts(gen_scaffolds)
    keys = list(set(train_counts.keys()) | set(gen_counts.keys()))
    keys_sorted = sorted(keys, key=lambda k: train_counts.get(k, 0) + gen_counts.get(k, 0), reverse=True)[:20]

    if len(keys_sorted) == 0:
        return

    os.makedirs(cfg.output_dir, exist_ok=True)
    x = np.arange(len(keys_sorted))
    train_vals = [train_counts.get(k, 0) for k in keys_sorted]
    gen_vals = [gen_counts.get(k, 0) for k in keys_sorted]

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.4
    ax.bar(x - width / 2.0, train_vals, width=width, label='Train')
    ax.bar(x + width / 2.0, gen_vals, width=width, label='Generated')
    ax.set_xticks(x)
    ax.set_xticklabels(keys_sorted, rotation=90)
    ax.set_title('Top scaffolds: train vs generated')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, cfg.scaffold_plot_filename), dpi=180)
    plt.close(fig)


def _draw_scaffold_grid(scaffold_counts_dict: dict[str, int], out_path: str, n_top: int, n_cols: int, title: str) -> dict:
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
        axes_list[i].set_title(legend, fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return {'saved': True, 'num_drawn': int(len(mols))}


def _maybe_save_scaffold_grids_and_stats(train_scaffolds: list, gen_scaffolds: list, cfg: AnalysisConfig) -> dict:
    train_counts = scaffold_counts(train_scaffolds)
    gen_counts = scaffold_counts(gen_scaffolds)
    train_set = set(train_counts.keys())
    gen_set = set(gen_counts.keys())

    overlap = int(len(train_set & gen_set))
    novel = int(len(gen_set - train_set))

    novel_gen_counts = {k: int(v) for k, v in gen_counts.items() if k not in train_set}

    scaffold_stats = {
        'unique_train_scaffolds': int(len(train_set)),
        'unique_generated_scaffolds': int(len(gen_set)),
        'overlap_scaffolds': overlap,
        'novel_generated_scaffolds': novel,
        'top_train_scaffolds': [
            {'scaffold': str(k), 'count': int(v)} for k, v in train_counts.most_common(int(cfg.scaffold_grid_top_n))
        ],
        'top_generated_scaffolds': [
            {'scaffold': str(k), 'count': int(v)} for k, v in gen_counts.most_common(int(cfg.scaffold_grid_top_n))
        ],
        'top_novel_generated_scaffolds': [
            {'scaffold': str(k), 'count': int(v)}
            for k, v in sorted(novel_gen_counts.items(), key=lambda x: x[1], reverse=True)[: int(cfg.scaffold_grid_top_n)]
        ],
    }

    stats_path = os.path.join(cfg.output_dir, cfg.scaffold_stats_filename)
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(scaffold_stats, f, indent=2)

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
        )
        grid_meta['generated_top_scaffold_grid'] = _draw_scaffold_grid(
            dict(gen_counts),
            os.path.join(cfg.output_dir, cfg.generated_scaffold_grid_filename),
            n_top=int(cfg.scaffold_grid_top_n),
            n_cols=int(cfg.scaffold_grid_n_cols),
            title='Generated set',
        )
        grid_meta['novel_top_scaffold_grid'] = _draw_scaffold_grid(
            novel_gen_counts,
            os.path.join(cfg.output_dir, cfg.novel_scaffold_grid_filename),
            n_top=int(cfg.scaffold_grid_top_n),
            n_cols=int(cfg.scaffold_grid_n_cols),
            title='Novel generated scaffolds',
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
    _maybe_plot_train_loss(cfg)

    train_df = load_train_dataframe(cfg.train_data_path, smiles_column=cfg.smiles_column, sep=cfg.train_sep)
    gen_df = load_generated_dataframe(cfg.generated_data_path, sep=cfg.generated_sep)

    if cfg.smiles_column not in train_df.columns:
        raise ValueError(f"Train data is missing smiles column '{cfg.smiles_column}'")
    if cfg.smiles_column not in gen_df.columns:
        raise ValueError(f"Generated data is missing smiles column '{cfg.smiles_column}'")

    train_df = _subset_df(train_df, cfg.train_max, cfg.random_seed)
    gen_df = _subset_df(gen_df, cfg.generated_max, cfg.random_seed)

    train_smiles = train_df[cfg.smiles_column].astype(str).tolist()
    gen_smiles = gen_df[cfg.smiles_column].astype(str).tolist()

    train_mols = [safe_mol_from_smiles(s) for s in train_smiles]
    gen_mols = [safe_mol_from_smiles(s) for s in gen_smiles]

    gen_df = gen_df.copy()
    gen_df['is_valid'] = [m is not None for m in gen_mols]
    gen_df['canonical_smiles'] = [canonicalize_smiles(s) for s in gen_smiles]

    train_scaffolds = [safe_murcko_scaffold_smiles(m) for m in train_mols]
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

    diversity = float(1.0 - mean_similarity) if not np.isnan(mean_similarity) else 0.0
    gen_df['diversity_score'] = diversity

    target_col = _resolve_property_column(gen_df, cfg.target_property_column, role='target')
    pred_col = _resolve_property_column(gen_df, cfg.predicted_property_column, role='pred')
    if target_col and pred_col:
        if target_col in gen_df.columns and pred_col in gen_df.columns:
            diff = gen_df[target_col].astype(float) - gen_df[pred_col].astype(float)
            gen_df['abs_prediction_error'] = diff.abs()

    processed_csv_path = os.path.join(cfg.output_dir, cfg.processed_csv_filename)
    gen_df.to_csv(processed_csv_path, index=False)

    _maybe_plot_property_distributions(train_df, gen_df, cfg)
    _maybe_plot_scaffold_distribution(train_scaffolds, gen_scaffolds, cfg)
    _maybe_plot_tanimoto_histogram(gen_df, cfg)
    pred_error_stats = _maybe_plot_prediction_errors(gen_df, cfg)
    _run_chemical_space_embedding(train_df, gen_df, cfg)
    _run_descriptor_space_embedding(train_df, gen_df, cfg)

    scaffold_stats = _maybe_save_scaffold_grids_and_stats(train_scaffolds, gen_scaffolds, cfg)

    train_scaf_set = set(s for s in train_scaffolds if s is not None)
    gen_scaf_set = set(s for s in gen_scaffolds if s is not None)
    overlap = len(train_scaf_set & gen_scaf_set)
    novel = len(gen_scaf_set - train_scaf_set)

    summary = {
        'profile_name': cfg.profile_name,
        'train_folder': cfg.train_folder,
        'train_data_path': cfg.train_data_path,
        'generated_data_path': cfg.generated_data_path,
        'output_dir': cfg.output_dir,
        'num_train_rows': int(len(train_df)),
        'num_generated_rows': int(len(gen_df)),
        'num_generated_valid': int(int(gen_df['is_valid'].sum())),
        'diversity_score': float(diversity),
        'mean_tanimoto_all_pairs': float(mean_similarity) if not np.isnan(mean_similarity) else None,
        'unique_train_scaffolds': int(len(train_scaf_set)),
        'unique_generated_scaffolds': int(len(gen_scaf_set)),
        'overlap_scaffolds': int(overlap),
        'novel_generated_scaffolds': int(novel),
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

    return summary
