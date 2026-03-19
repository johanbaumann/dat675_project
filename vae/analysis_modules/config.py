from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional




"""
Helper module for defining and loading analysis configuration.
Will be used in combo with the fold_pipeline outputs to run the analysis scripts in a standardized way.

"""

@dataclass(frozen=True)
class AnalysisConfig:
    profile_name: str
    train_folder: str
    train_data_path: str
    validation_data_path: Optional[str]
    generated_data_path: str
    output_dir: str
    quality_summary_data_path: Optional[str] = None

    smiles_column: str = 'smiles'
    train_sep: Optional[str] = None
    validation_sep: Optional[str] = None
    generated_sep: str = ','

    target_property_column: Optional[str] = None
    predicted_property_column: Optional[str] = None

    train_max: int = 50_000
    validation_max: int = 50_000
    generated_max: int = 50_000
    random_seed: int = 42

    tanimoto_radius: int = 2
    tanimoto_n_bits: int = 2048
    tanimoto_cache_filename: str = 'tanimoto_cache.bin'
    tanimoto_flush_every: int = 1000
    internal_diversity_enabled: bool = True
    internal_diversity_max_pairs: int = 200_000
    internal_diversity_random_seed: int = 42

    save_distribution_plot: bool = True
    save_scaffold_plot: bool = True
    save_scaffold_grids: bool = True
    run_prediction_error_plot: bool = True

    distribution_plot_filename: str = 'property_distribution.png'
    mw_distribution_diff_plot_filename: str = 'mw_distribution_diff_train_minus_generated.png'
    scaffold_plot_filename: str = 'scaffold_distribution_train_vs_gen.png'
    train_scaffold_grid_filename: str = 'train_top_scaffolds_grid.png'
    generated_scaffold_grid_filename: str = 'generated_top_scaffolds_grid.png'
    novel_scaffold_grid_filename: str = 'novel_top_scaffolds_grid.png'
    scaffold_stats_filename: str = 'scaffold_stats.json'
    train_loss_plot_filename: str = 'train_vs_test_loss.png'
    tanimoto_histogram_filename: str = 'tanimoto_similarity_distribution.png'
    prediction_error_plot_filename: str = 'prediction_error_vs_ground_truth.png'
    processed_csv_filename: str = 'generated_subset_with_similarity.csv'
    summary_json_filename: str = 'analysis_summary.json'

    run_train_loss_plot: bool = True
    run_tanimoto_histogram: bool = True
    run_chemical_space: bool = True
    run_descriptor_space: bool = True
    debug: bool = False

    embedding_train_sample: int = 10_000
    embedding_validation_sample: int = 10_000
    embedding_generated_sample: int = 10_000

    chemical_tsne_perplexity: float = 30.0
    chemical_pca_pre_dim: int = 50
    descriptor_tsne_perplexity: float = 30.0
    descriptor_pca_pre_dim: int = 0
    embedding_point_size: int = 6
    embedding_point_size_train: int = 14
    embedding_point_size_validation: int = 14
    embedding_point_size_generated: int = 14
    embedding_point_edgecolors_enabled: bool = True
    embedding_edge_color: str = 'black'
    embedding_edge_width: float = 0.5
    embedding_alpha: float = 0.5
    scaffold_grid_top_n: int = 24
    scaffold_grid_n_cols: int = 6

    descriptor_names: tuple[str, ...] = (
        'HeavyAtomCount',
        'RingCount',
        'MolLogP',
        'MolWt',
        'NumHAcceptors',
        'NumHDonors',
    )

    chemical_pca_plot_filename: str = 'pca2_train_vs_generated.png'
    chemical_tsne_plot_filename: str = 'tsne_train_vs_generated.png'
    chemical_tsne_tanimoto_plot_filename: str = 'tsne_generated_colored_by_tanimoto.png'
    descriptor_pca_plot_filename: str = 'pca2_descriptors_train_vs_generated.png'
    descriptor_tsne_plot_filename: str = 'tsne_descriptors_train_vs_generated.png'
    descriptor_tsne_tanimoto_plot_filename: str = 'tsne_descriptors_generated_colored_by_tanimoto.png'


# ============================================================
# ACTIVE PROFILE (BACE beta-CVAE workflow)
# ============================================================
DEFAULT_BACE_PROFILE = AnalysisConfig(
    profile_name='bace_pic50_10k',
    train_folder='save/fold_pipeline_runs/cv_iteration_0/training',
    train_data_path='bace_pic50.txt',
    validation_data_path=None,
    generated_data_path='fold_pipeline_outputs/cv_iteration_0/generated/generated.csv',
    output_dir='fold_pipeline_outputs/cv_iteration_0/analysis_refreshed',
    target_property_column='pIC50',
    predicted_property_column='pred_pIC50',
)
"""
Default active profile for BACE pIC50 beta-CVAE analysis.
Points to validated workspace artifacts (fold_pipeline outputs).
This is the ONLY officially supported profile for current development.
"""


def _validate_analysis_config_paths(cfg: AnalysisConfig) -> None:
    """
    Validate that all required config paths exist in the workspace.
    
    Checks train_data_path, generated_data_path, and optional validation_data_path.
    
    Args:
        cfg: AnalysisConfig to validate
    
    Raises:
        FileNotFoundError: if any required path doesn't exist, with actionable message
    """
    missing: list[str] = []
    if not os.path.exists(cfg.train_data_path):
        missing.append(f'train_data_path: {cfg.train_data_path}')
    if not os.path.exists(cfg.generated_data_path):
        missing.append(f'generated_data_path: {cfg.generated_data_path}')
    if cfg.validation_data_path and not os.path.exists(cfg.validation_data_path):
        missing.append(f'validation_data_path: {cfg.validation_data_path}')
    if missing:
        joined = '; '.join(missing)
        raise FileNotFoundError(
            'Analysis config contains missing input paths. '
            f'Fix these fields in analysis_run_config.json: {joined}'
        )


def build_profile_config(profile: str = 'bace_pic50_10k', **overrides) -> AnalysisConfig:
    """
    Build AnalysisConfig from a profile name and optional overrides.
    
    Supported profiles:
    - 'bace_pic50_10k': BACE pIC50 beta-CVAE workflow (default, active)
    - 'bace', 'bace_pic50': aliases for bace_pic50_10k
    
    Args:
        profile: Profile name (default 'bace_pic50_10k')
        **overrides: Config fields to override (e.g., output_dir='custom_dir')
    
    Returns:
        AnalysisConfig instance with merged settings
    
    Raises:
        ValueError: if profile is not recognized
    """
    normalized = str(profile).strip().lower()
    if normalized in ('bace', 'bace_pic50', 'bace_pic50_10k'):
        base = DEFAULT_BACE_PROFILE
    else:
        raise ValueError("profile must be one of: 'bace_pic50_10k', 'bace', 'bace_pic50'")

    payload = dict(base.__dict__)
    payload.update(overrides)
    return AnalysisConfig(**payload)


def load_analysis_config_from_file(path: str) -> AnalysisConfig:
    """
    Load and validate AnalysisConfig from a JSON file.
    
    The JSON file can include:
    - 'profile': profile name (default 'bace_pic50_10k')
    - 'overrides': object with config field overrides
    - Or directly include override fields at top level
    
    All input paths are validated; FileNotFoundError raised if missing.
    
    Args:
        path: Path to config JSON file
    
    Returns:
        AnalysisConfig instance with validated paths
    
    Raises:
        FileNotFoundError: if required data paths don't exist
        ValueError: if config JSON structure is invalid
    """
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError('Config JSON must be an object.')

    profile = str(payload.get('profile', 'bace_pic50_10k'))
    overrides = payload.get('overrides', {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError('Config key "overrides" must be an object when provided.')

    for key, value in payload.items():
        if key not in ('profile', 'overrides'):
            overrides[key] = value

    if 'descriptor_names' in overrides and isinstance(overrides['descriptor_names'], list):
        overrides['descriptor_names'] = tuple(str(x) for x in overrides['descriptor_names'])

    # Backward/forward compatibility: allow extra keys in config JSON (for runner
    # toggles or future extensions) without breaking AnalysisConfig construction.
    valid_keys = set(AnalysisConfig.__dataclass_fields__.keys())
    overrides = {k: v for k, v in overrides.items() if k in valid_keys}

    cfg = build_profile_config(profile=profile, **overrides)
    _validate_analysis_config_paths(cfg)
    return cfg
