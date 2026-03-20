from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator

from utils import canonicalize_for_filtering, safe_mol_from_smiles, safe_murcko_scaffold_smiles


def canonicalize_smiles(smiles: str, *, strip_salts: bool = True, decharge: bool = True, canonicalize_tautomer: bool = False) -> Optional[str]:
    can, _, _ = canonicalize_for_filtering(
        smiles,
        strip_salts=bool(strip_salts),
        decharge=bool(decharge),
        canonicalize_tautomer=bool(canonicalize_tautomer),
    )
    return can


def morgan_fp_generator(radius: int = 2, n_bits: int = 2048):
    return rdFingerprintGenerator.GetMorganGenerator(radius=int(radius), fpSize=int(n_bits))


def morgan_fp(gen, mol: Chem.Mol):
    if mol is None:
        return None
    return gen.GetFingerprint(mol)


def max_tanimoto_to_reference(query_fps: list, ref_fps: list) -> tuple[np.ndarray, float]:
    if len(query_fps) == 0 or len(ref_fps) == 0:
        return np.full((len(query_fps),), np.nan, dtype=np.float32), float('nan')

    out = np.empty((len(query_fps),), dtype=np.float32)
    sim_sum = 0.0
    sim_count = 0
    for i, qfp in enumerate(query_fps):
        sims = DataStructs.BulkTanimotoSimilarity(qfp, ref_fps)
        if len(sims) == 0:
            out[i] = np.nan
            continue
        arr = np.asarray(sims, dtype=np.float32)
        sim_sum += float(arr.sum(dtype=np.float64))
        sim_count += int(arr.size)
        out[i] = float(np.max(arr))

    mean_similarity = (sim_sum / float(sim_count)) if sim_count > 0 else float('nan')
    return out, float(mean_similarity)


def mean_pairwise_tanimoto(
    fps: list,
    *,
    max_pairs: Optional[int] = None,
    random_seed: int = 42,
) -> tuple[float, int, int, str]:
    """Compute mean pairwise Tanimoto across a fingerprint list.

    Returns:
      (mean_similarity, num_pairs_used, num_pairs_total, mode)
      - mode is one of: 'insufficient_data', 'exact', 'sampled'
    """
    n = int(len(fps))
    total_pairs = int((n * (n - 1)) // 2)
    if n < 2 or total_pairs <= 0:
        return float('nan'), 0, total_pairs, 'insufficient_data'

    # Exact path when pair count is modest or no cap is set.
    if max_pairs is None or int(max_pairs) <= 0 or total_pairs <= int(max_pairs):
        sim_sum = 0.0
        count = 0
        for i in range(n - 1):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
            if len(sims) == 0:
                continue
            arr = np.asarray(sims, dtype=np.float32)
            sim_sum += float(arr.sum(dtype=np.float64))
            count += int(arr.size)
        mean_similarity = (sim_sum / float(count)) if count > 0 else float('nan')
        return float(mean_similarity), int(count), total_pairs, 'exact'

    # Sampled path for very large generated sets.
    target_pairs = min(int(max_pairs), total_pairs)
    rng = np.random.default_rng(int(random_seed))

    sampled_pairs: set[tuple[int, int]] = set()
    sim_sum = 0.0
    count = 0
    max_attempts = int(target_pairs * 20)
    attempts = 0

    while count < target_pairs and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        if i > j:
            i, j = j, i
        pair = (i, j)
        if pair in sampled_pairs:
            continue
        sampled_pairs.add(pair)
        sim_sum += float(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
        count += 1

    mean_similarity = (sim_sum / float(count)) if count > 0 else float('nan')
    return float(mean_similarity), int(count), total_pairs, 'sampled'


def scaffold_counts(scaffolds: Sequence[Optional[str]]) -> Counter:
    return Counter(s for s in scaffolds if s is not None)


def smiles_list_to_fp_matrix(fpgen, smiles_list: list[str], *, dtype=np.int8) -> tuple[np.ndarray, np.ndarray]:
    n_bits = int(fpgen.GetOptions().fpSize)
    fps = []
    valid_mask = np.zeros((len(smiles_list),), dtype=bool)

    for i, s in enumerate(smiles_list):
        mol = safe_mol_from_smiles(s)
        if mol is None:
            continue
        fp = fpgen.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=dtype)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
        valid_mask[i] = True

    if len(fps) == 0:
        return np.zeros((0, n_bits), dtype=dtype), valid_mask
    return np.vstack(fps), valid_mask


def smiles_list_to_descriptor_matrix(smiles_list: list[str], descriptor_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    valid_mask = np.zeros((len(smiles_list),), dtype=bool)

    for i, s in enumerate(smiles_list):
        mol = safe_mol_from_smiles(s)
        if mol is None:
            continue
        try:
            vec = np.asarray([float(getattr(Descriptors, name)(mol)) for name in descriptor_names], dtype=np.float32)
        except Exception:
            continue
        rows.append(vec)
        valid_mask[i] = True

    if len(rows) == 0:
        return np.zeros((0, len(descriptor_names)), dtype=np.float32), valid_mask
    return np.vstack(rows).astype(np.float32), valid_mask
