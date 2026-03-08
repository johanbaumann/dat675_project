# %%
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import time as t
import os 
import sys
import pandas as pd


def _apply_projector_plot_style() -> None:
    """Make all plots easier to read on low-quality projectors."""
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


import rdkit
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA

# rdkitdraw
from rdkit.Chem import Draw
import py3Dmol as dmol
from rdkit.Chem import AllChem,Descriptors,Draw

from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import DataStructs
print(f"imported at: {t.ctime()}")

# %% [markdown]
# # For THE transformer:
# ### Stored at: run_20260219_230438
# ### CVAE_transformer_300k_test.txt

# %%
save_folder = "save/"

file = "history.csv"

final_lstm = "run_20260224_160237"
final_transformer = "run_20260224_205844"

df = pd.read_csv(save_folder+final_lstm+"/"+file)
print(df)

def plot_history(df:pd.DataFrame, save_folder:str, file:str):
    x = np.linspace(0, len(df)-1, len(df))
    plt.figure(figsize=(10, 5))
    plt.plot(x, df['train_loss'], label='Train Loss')
    plt.plot(x, df['test_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')


    plt.savefig(save_folder + file + "/history_plot.png")
    plt.show()

histories = [final_lstm, final_transformer]
for his in histories:
    df = pd.read_csv(save_folder+his+"/"+file)

    plot_history(df, save_folder, his)



# %%
from json import load


#file = "label_CVAE_transformer_300k_test.txt"
#file = "label_CVAE_transformer_100k_test.txt"
#file = "temp_100k_test.txt"
#file = 'temp_transformer_100k_test.txt'
file = 'train_dist_temp_transformer_300k_test.txt'

df = pd.read_csv(file, sep=",")

# make 0th row the header
print(df.columns)

train_file = '250k_zinc_clean.txt'

train_df = pd.read_csv(train_file, sep=",")



def stats_logP(df:pd.DataFrame) -> dict[str, float]:
    ground_truth = df['LogP'].values
    predicted = df['pred_LogP'].values
    diff = np.abs(ground_truth - predicted)
    absolute_diff = np.abs(diff)
    avg_error = np.mean(absolute_diff)
    median_err = np.median(absolute_diff)
    std = np.std(absolute_diff)
    stats = {
        "ground_truth": ground_truth,
        "predicted": predicted,
        "absolute_diff": absolute_diff,
        "avg_error": avg_error,
        "median_error": median_err,
        "std": std
    }
    return stats
def plot_errors(ground_truth:np.ndarray, error_as_function_of_logp:np.ndarray):
    plt.figure(figsize=(10, 5))
    edge_kwargs = {'edgecolors': point_edge_color, 'linewidths': point_edge_width} if point_edgecolors_enabled else {}
    plt.scatter(ground_truth, error_as_function_of_logp, alpha=0.5, s=14, **edge_kwargs)
    plt.xlabel('Ground Truth LogP')
    plt.ylabel('Absolute Error')

    plt.title('Absolute Error vs rdKit LogP')
    plt.show()

def mse_logp(df:pd.DataFrame,) -> float:
    ground_truth = df['LogP'].values
    predicted = df['pred_LogP'].values
    diff = ground_truth - predicted
    mse = np.mean(diff**2)
    return mse


stats = stats_logP(df)

ground_truth = stats["ground_truth"]
predicted = stats["predicted"]
avg_error = stats["avg_error"]
std = stats["std"]
median_error = stats["median_error"]



error_as_function_of_logp = np.abs(ground_truth - predicted)

print(f"Average error: {avg_error}, p/m {std}")
print(f"median_error: {median_error}")

plot_errors(ground_truth, error_as_function_of_logp)

# %%

def read_train_file(train_path:str):
    smiles_list = []
    properties_list = []
    
    with open(train_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 3:
                smiles_list.append(parts[0])
                properties_list.append((float(parts[1]), float(parts[2])))  # Assuming MW and LogP are in the second and third columns
    
    return smiles_list, properties_list

def read_resultant_file(resultant_path:str):
    """
    This file has column headers, so we need to skip the first line when reading the file.
    The first column is SMILES, the second column is MW, and the third column is LogP. the 8th column is pred_LogP, should be included in the properties list as well.
    
    """
    gen_smiles = []
    gen_properties = []
    
    with open(resultant_path, 'r') as file:
        first_line = True
        #for line in file:
        #    parts = line.strip().split()
        #    if first_line:
        #        first_line = False
        #        continue  # Skip the header line
        #    if len(parts) >= 3:
        #        gen_smiles.append(parts[0])
        #        gen_properties.append((float(parts[1]), float(parts[2])))  # Assuming MW and LogP are in the second and third columns
        # just print all results:
        
        first_line = True
        for line in file:
            if first_line:
                first_line = False
                continue  # Skip the header line
            #print(line.strip())
            # split at ','
            parts = line.strip().split(',')
            if len(parts) >= 3:
                gen_smiles.append(parts[0])
                gen_properties.append((float(parts[1]), float(parts[2]), float(parts[7])))  # Assuming MW and LogP are in the second and third columns, pred_LogP is in the 8th column (index 7)
        
    return gen_smiles, gen_properties



def plot_prop_distribution(df:pd.DataFrame, save_folder:str,top_title_str:str="Property Distribution"):
    plt.figure(figsize=(12, 5))

    nice_blue = '#0046AB'
    nice_orange = "#FF8400"

    plt.subplot(1, 2, 1)
    plt.hist(df['MW'], bins=100, alpha=1.0, color=nice_blue)
    plt.title(top_title_str + ' - Molecular Weight Distribution')
    plt.xlabel('Molecular Weight')
    plt.ylabel('Frequency')

    if 'pred_LogP' in df.columns:
        plt.subplot(1, 2, 2)
        plt.hist(df['pred_LogP'], bins=100, alpha=1.0, color=nice_orange)
        plt.title(top_title_str + ' - Predicted LogP Distribution')
        plt.xlabel('Predicted LogP')
    else:
        plt.subplot(1, 2, 2)
        plt.hist(df['LogP'], bins=100, alpha=1.0, color=nice_orange)
        plt.title(top_title_str + ' - LogP Distribution')
        plt.xlabel('LogP')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(save_folder + top_title_str + "_property_distribution.png")
    plt.show()


# %%




# first column is smiles,
# second column is: MW, third column is LogP

train_path = '250k_zinc_clean.txt'




smiles_list, properties_list = read_train_file(train_path)
#

#NOTE: df_train has the training data, with columns: smiles, MW, LogP
df_train = pd.DataFrame({
    'smiles': smiles_list,
    'MW': [prop[0] for prop in properties_list],
    'LogP': [prop[1] for prop in properties_list]
})


resultant_path = 'train_dist_temp_transformer_300k_test.txt'

gen_smiles, gen_prop = read_resultant_file(resultant_path)
gen_df = pd.DataFrame({
    'smiles': gen_smiles,
    'MW': [prop[0] for prop in gen_prop],
    'LogP': [prop[1] for prop in gen_prop],
    'pred_LogP': [prop[2] for prop in gen_prop]
})

plot_prop_distribution(df_train, save_folder, top_title_str="Original Dataset")
print(80*"=")
print("Generated Dataset:")


plot_prop_distribution(gen_df, save_folder, top_title_str="Generated Dataset")


def check_validity_clean(smiles_list:list[str]) -> float:
    """
    Checks how many of the generated molecules are clean, and valid

    A clean and valid molecule is one that is decharged, and not a radical.
    No ions, no radicals, no charged species. Only neutral molecules.




    """
    valid_count = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            # Check if molecule is clean (no charges)
            is_clean = all(atom.GetFormalCharge() == 0 for atom in mol.GetAtoms())
            # Check if molecule is not a radical (no unpaired electrons)
            is_not_radical = all(atom.GetNumRadicalElectrons() == 0 for atom in mol.GetAtoms())
            if is_clean and is_not_radical:
                valid_count += 1
    validity_percentage = (valid_count / len(smiles_list)) * 100
    return validity_percentage

validity_percentage = check_validity_clean(gen_smiles)
print(f"Validity percentage (clean and valid): {validity_percentage:.2f}%")

# %%
# read molecules from file
# has columns: smiles, MW, LogP, TPSA
df = pd.read_csv('CVAE_transformer_300k_test.txt')



ms = [Chem.MolFromSmiles(s) for s in df['smiles']]
def avg_mv(mols):
    return sum([ExactMolWt(m) for m in mols])/len(mols)


samples = 10


def draw_samples(mols: list, samples: int) -> None:
    sample_mols = rnd.sample(mols, samples)
    img = Draw.MolsToGridImage(sample_mols, molsPerRow=5, subImgSize=(200, 200))
    display(img)



def draw_3d_samples(mols: list, samples: int) -> None:
    sample_mols = rnd.sample(mols, samples)
    for mol in sample_mols:
        # add hydrogens and embed the molecule in 3D space
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        # optimize the molecule
        # for better visualization
        AllChem.MMFFOptimizeMolecule(mol)
        view = dmol.view(width=400, height=400)
        view.addModel(Chem.MolToMolBlock(mol), 'sdf')
        view.setStyle({'stick': {}})
        view.zoomTo()
        display(view.show())


draw_samples(ms, samples)

draw_3d_samples(ms, samples)



# %%
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import os
import random as rnd

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator


# =============================================================================
# CHANGELOG (2026-02-25)
# - Added an on-disk cache file for Tanimoto similarities so we do not keep all
#   similarities in RAM.
# - Added periodic flushing so intermediate results survive crashes and memory
#   stays bounded.
# - Diversity is now computed from streaming mean similarity, which matches:
#     Diversity = 1 - mean(Tanimoto)
#   but does not require storing all pairwise similarities in a Python list.
# - Kept all existing comments; only added new ones.
# =============================================================================


class _TanimotoCacheWriter:
    """
    Append-only writer for all Tanimoto similarities.

    Stores raw float32 values in a binary file:
      - write: np.asarray(sims, np.float32).tofile(file_handle)
      - read:  np.fromfile(path, dtype=np.float32)

    A small sidecar meta file is written to "<path>.meta.txt" containing count.
    This is intentionally simple and robust.
    """

    def __init__(self, path: str):
        self.path = str(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._fh = open(self.path, "ab")  # append binary
        self.count = 0
        self.meta_path = self.path + ".meta.txt"
        # initialize meta file
        self._write_meta()

    def _write_meta(self) -> None:
        with open(self.meta_path, "w", encoding="utf-8") as f:
            f.write("dtype=float32\n")
            f.write("format=raw_float32_stream\n")
            f.write(f"count={self.count}\n")

    def write(self, sims: List[float] | np.ndarray) -> None:
        if sims is None:
            return
        arr = np.asarray(sims, dtype=np.float32)
        if arr.size == 0:
            return
        arr.tofile(self._fh)
        self.count += int(arr.size)

    def flush(self) -> None:
        self._fh.flush()
        self._write_meta()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._fh.close()


def safe_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """
    Parse a SMILES into an RDKit Mol.
    Returns None if parsing or sanitization fails.
    """
    if smiles is None:
        return None
    s = str(smiles).strip()
    if len(s) == 0:
        return None
    try:
        mol = Chem.MolFromSmiles(s, sanitize=True)
        return mol
    except Exception:
        return None


def fpgen_fp_size(fpgen) -> int:
    """
    RDKit fingerprint generators expose options that include fpSize.
    This is the correct fixed length for ExplicitBitVect and dense count vectors.
    """
    try:
        return int(fpgen.GetOptions().fpSize)
    except Exception:
        # Fallback: make one fingerprint and query its size
        m = Chem.MolFromSmiles("CC")
        fp = fpgen.GetFingerprint(m)
        return int(fp.GetNumBits())


def safe_canonical_smiles(mol: Chem.Mol) -> Optional[str]:
    """
    Canonical SMILES for a valid RDKit Mol. Returns None on failure.
    """
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception:
        return None


def safe_murcko_scaffold_smiles(mol: Chem.Mol) -> Optional[str]:
    """
    Murcko scaffold SMILES for a valid RDKit Mol.
    Returns None if scaffold extraction fails (common for weird structures).
    """
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None:
            return None
        # Canonical scaffold smiles
        return Chem.MolToSmiles(scaf, isomericSmiles=False, canonical=True)
    except Exception:
        return None


def morgan_fp(morgan_gen, mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> Optional[DataStructs.ExplicitBitVect]:
    """
    Morgan fingerprint as an RDKit ExplicitBitVect.
    Returns None if mol is None.
    """
    if mol is None:
        return None
    # Using Morgan bit vector is standard for Tanimoto similarity
    return morgan_gen.GetFingerprint(mol)


def mol_to_fp_array(fpgen, mol: Chem.Mol, *, use_counts: bool = False, dtype=np.int8) -> Optional[np.ndarray]:
    """
    Convert an RDKit mol to a numpy fingerprint array of shape (fpSize,).

    If use_counts=False:
        uses GetFingerprint(mol) -> ExplicitBitVect -> values in {0,1}

    If use_counts=True:
        uses GetCountFingerprint(mol) -> UIntSparseIntVect (dense-exported) -> nonnegative integer counts
        dtype should then be something like np.int16 or np.int32.
    """
    if mol is None:
        return None

    n_bits = int(fpgen.GetOptions().fpSize)  # fpgen_fp_size(fpgen)

    if not use_counts:
        fp = fpgen.GetFingerprint(mol)  # ExplicitBitVect, length n_bits
        arr = np.zeros((n_bits,), dtype=dtype)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    # Count fingerprints: values can exceed 1, so do not use int8 unless one is shure
    fp = fpgen.GetCountFingerprint(mol)  # count vector, fixed length n_bits in the new generator API
    arr = np.zeros((n_bits,), dtype=dtype)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_list_to_fp_matrix(
    fpgen,
    smiles_list: List[str],
    *,
    use_counts: bool = False,
    dtype=np.int8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert list of SMILES to (X, valid_mask).
    X has shape [num_valid, fpSize].
    valid_mask has shape [len(smiles_list)] and indicates which were valid.
    """
    # n_bits = fpgen_fp_size(fpgen)
    n_bits = int(fpgen.GetOptions().fpSize)
    fps = []
    valid_mask = np.zeros((len(smiles_list),), dtype=bool)

    for i, s in enumerate(smiles_list):
        mol = safe_mol_from_smiles(s)
        if mol is None:
            continue
        arr = mol_to_fp_array(fpgen, mol, use_counts=use_counts, dtype=dtype)
        if arr is None:
            continue
        fps.append(arr)
        valid_mask[i] = True

    if len(fps) == 0:
        return np.zeros((0, n_bits), dtype=dtype), valid_mask

    X = np.vstack(fps)
    return X, valid_mask


def chunked_max_tanimoto(
    morgan_gen,
    query_fps: List,
    ref_fps: List,
    chunk_size: int = 5000,
    *,
    tanimoto_cache_path: Optional[str] = None,
    flush_every_queries: int = 2500,
) -> Tuple[np.ndarray, dict]:
    """
    For each query fingerprint, compute the maximum Tanimoto similarity
    to any fingerprint in ref_fps.

    Note: If ref_fps is huge, this can still be very expensive.
     (for example 10k to 50k molecules is reasonable...)
     This is since complexity is O(len(query_fps) * len(ref_fps)) and BulkTanimotoSimilarity is fast but not that fast.
     (fast is constant factor, but the O() is still quadratic in the number of molecules...)

    tanimoto sim is: IOU = (A cap B) / (A cup B) = (A and B) / (A or B)

    args:
        morgan_gen: RDKit Morgan fingerprint generator, passed in to avoid re-creating it multiple times.
        query_fps: List of RDKit ExplicitBitVect fingerprints for the query molecules.
        ref_fps: List of RDKit ExplicitBitVect fingerprints for the reference molecules.
        chunk_size: Number of query_fps to process at once to manage memory and speed.

        tanimoto_cache_path: If provided, all computed similarities are appended to this file as float32.
        flush_every_queries: Flush the cache file every N query fingerprints.

    returns:
        out: np.ndarray of shape (len(query_fps),) with the max Tanimoto similarity to ref_fps for each query_fp.
        stats: dict with streaming stats for computing diversity without holding all sims in memory.
            stats contains:
              - tanimoto_count: number of similarities written/aggregated
              - tanimoto_sum: sum of similarities
              - tanimoto_mean: mean similarity
              - cache_path: cache path used (or None)
    """
    len_query = len(query_fps)
    if len(ref_fps) == 0:
        out = np.full((len_query,), np.nan, dtype=np.float32)
        stats = {
            "tanimoto_count": 0,
            "tanimoto_sum": 0.0,
            "tanimoto_mean": np.nan,
            "cache_path": tanimoto_cache_path,
        }
        return out, stats

    out = np.empty((len_query,), dtype=np.float32)

    # Pre-store ref_fps as list, BulkTanimotoSimilarity is implemented in C++ and is fast
    # for i0 in range of query_fps with step chunk_size, compute similarities to all ref_fps and take max for each query_fp in the chunk
    print_every = 2500
    # i0 is the start index of the chunk, i1 is the end index (exclusive)

    # all tanimotos will be used for diversity score!
    # Instead of storing everything in a python list, we cache to disk and track streaming mean.

    cache_writer: Optional[_TanimotoCacheWriter] = None
    if tanimoto_cache_path is not None:
        cache_writer = _TanimotoCacheWriter(tanimoto_cache_path)

    tanimoto_sum = 0.0
    tanimoto_count = 0

    try:
        # for all fps
        for i0 in range(0, len_query, chunk_size):
            # what part of of the query fp:s are processing...
            i1 = min(i0 + chunk_size, len_query)
            # for all fp in chunk, compute similarity to all ref_fps and take max!
            for j, qfp in enumerate(query_fps[i0:i1]):
                qi = i0 + j
                if qi % print_every == 0:
                    print(f"Computing similarity for query fingerprint {qi} of {len_query}")

                sims = DataStructs.BulkTanimotoSimilarity(qfp, ref_fps)

                # cache sims instead of keeping everything in RAM
                if len(sims) > 0:
                    sims_arr = np.asarray(sims, dtype=np.float32)
                    tanimoto_sum += float(sims_arr.sum(dtype=np.float64))
                    tanimoto_count += int(sims_arr.size)

                    if cache_writer is not None:
                        cache_writer.write(sims_arr)

                out[qi] = float(max(sims)) if len(sims) > 0 else np.nan

                # occasional flush so one does not have to keep everything in memory
                if cache_writer is not None and flush_every_queries > 0:
                    if (qi > 0) and (qi % int(flush_every_queries) == 0):
                        cache_writer.flush()

    finally:
        if cache_writer is not None:
            cache_writer.close()

    tanimoto_mean = (tanimoto_sum / float(tanimoto_count)) if tanimoto_count > 0 else np.nan
    stats = {
        "tanimoto_count": int(tanimoto_count),
        "tanimoto_sum": float(tanimoto_sum),
        "tanimoto_mean": float(tanimoto_mean) if not np.isnan(tanimoto_mean) else np.nan,
        "cache_path": tanimoto_cache_path,
    }

    return out, stats


def diversity_score(distances: list) -> float:
    """
    The diversity score, which is:
    Diversity (D) = 1- (1/N) * sum_{i!=j} Tanimoto(fp_i,fp_j)
    Resuses the chuncked max
    """
    len_dist = len(distances)
    if len_dist <= 1:
        return 0.0
    np_dist = np.array(distances)
    mean_sim = np.mean(np_dist)
    diversity = 1.0 - mean_sim
    return diversity


def diversity_score_from_mean(mean_similarity: float) -> float:
    """
    Same diversity definition as diversity_score, but uses the mean similarity
    directly, so we do not need to store all pairwise tanimoto values.
    """
    if mean_similarity is None or np.isnan(mean_similarity):
        return 0.0
    return float(1.0 - float(mean_similarity))


def postprocess_and_save(
    # morgan_gen,
    gen_df: pd.DataFrame,
    ref_mols: List[Chem.Mol] | pd.DataFrame,
    out_csv_path: str,
    gen_max: int = 50_000,
    ref_max: int = 30000,
    radius: int = 2,
    n_bits: int = 2048,
    random: bool = True,
    rand_seed: int = 42,
    *,
    tanimoto_cache_path: Optional[str] = None,
    flush_every_queries: int = 2500,
) -> pd.DataFrame:
    """
    Postprocess generated SMILES and save results to CSV.
    Returns the processed DataFrame.
    Needs to be random selection of results since the result file is for each of the 10*10 generation sweeps.
    Just choosing the first n:k molecules would be biased since the sampling is done by linspace of MW and LogP.
    Has both random and deterministic options for selecting reference molecules for similarity calculation.

    ref_max limits how many reference molecules are used for similarity. This is a limit of
    runtime and memory for the similarity calculation, which is O(len(gen_df) * ref_max) in the worst case.

    args:
        gen_df: DataFrame with at least a 'smiles' column for the generated molecules
        ref_mols: List of RDKit Mol objects for the reference molecules to compare against.
        out_csv_path: Path to save the output CSV with postprocessed results.
        gen_max: Maximum number of generated molecules to process (for runtime management).
        ref_max: Maximum number of reference molecules to use for similarity (for runtime management).
            radius: Radius for Morgan fingerprint.
            n_bits: Number of bits for Morgan fingerprint.
            random: Whether to randomly select reference molecules if ref_mols is larger than ref_max.
            rand_seed: Random seed for reproducibility if random selection is used.

        tanimoto_cache_path: If provided, stores all tanimoto sims to disk (float32 stream).
        flush_every_queries: Flush the cache file every N query fingerprints.



    """
    # NOTE: WIll be reused by a bunch of diffrent functions, so we create it here and pass it in.
    mg_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    # Build a reference set (cap size to keep runtime sane)
    if isinstance(ref_mols, pd.DataFrame):
        orig_ref = ref_mols.copy()
        ref_mols = ref_mols["mol"].tolist()

    ref_mols_valid = [m for m in ref_mols if m is not None]
    len_train = len(ref_mols)
    num_valid = len(ref_mols_valid)
    print(f"Number of valid reference molecules: {num_valid}, filtered out of {len_train} total reference molecules.")

    if len(ref_mols_valid) > ref_max:
        # deterministic subset to avoid importing random; one can randomize if desired
        if random:
            rnd.seed(rand_seed)
            rnd.shuffle(ref_mols_valid)
            # choose the first ref_max after shuffling
            ref_mols_valid = ref_mols_valid[:ref_max]
        else:
            ref_mols_valid = ref_mols_valid[:ref_max]

    if len(gen_df) > gen_max:
        if random:
            gen_df = gen_df.sample(n=gen_max, random_state=rand_seed).reset_index(drop=True)
            gen_df = gen_df.iloc[:gen_max]
        else:
            gen_df = gen_df.iloc[:gen_max]

    ref_fps = [morgan_fp(mg_gen, m, radius=radius, n_bits=n_bits) for m in ref_mols_valid]
    ref_fps = [fp for fp in ref_fps if fp is not None]

    smiles_list = gen_df["smiles"].astype(str).tolist()

    mols: List[Optional[Chem.Mol]] = []
    canon: List[Optional[str]] = []
    scaff: List[Optional[str]] = []
    fps = []

    len_smiles = len(smiles_list)

    print(f"Processing {len_smiles} generated molecules...")

    print_every = 10_000
    # Parse and featurize
    for i, s in enumerate(smiles_list):
        if i % print_every == 0:
            print(f"Processing molecule {i} of {len_smiles}")
        mol = safe_mol_from_smiles(s)
        mols.append(mol)

        c = safe_canonical_smiles(mol)
        canon.append(c)

        sc = safe_murcko_scaffold_smiles(mol)
        scaff.append(sc)

        fp = morgan_fp(mg_gen, mol, radius=radius, n_bits=n_bits)
        fps.append(fp)

    print("Finished parsing and featurizing generated molecules.")
    is_valid = [m is not None for m in mols]

    # Similarity only for valid molecules (keeps BulkTanimotoSimilarity clean)
    valid_idx = [i for i, ok in enumerate(is_valid) if ok]
    valid_fps = [fps[i] for i in valid_idx if fps[i] is not None]

    print(f"Computing Tanimoto similarity of {len(valid_fps)} valid generated mols")
    # chunks must divide the valid_fps, otherwise last chunck need to be smaller
    chunk_s = 5000
    if len(valid_fps) % chunk_s != 0:
        print(
            f"Warning: chunk_size {chunk_s} does not divide the number of valid fingerprints {len(valid_fps)}. "
            f"The last chunk will be smaller."
        )

    # max_sim_valid: per generated molecule, max similarity to reference set
    # tan_stats: streaming statistics (mean similarity) and cache meta info
    max_sim_valid, tan_stats = chunked_max_tanimoto(
        mg_gen,
        valid_fps,
        ref_fps,
        chunk_size=chunk_s,
        tanimoto_cache_path=tanimoto_cache_path,
        flush_every_queries=flush_every_queries,
    )

    # If you still want the old behavior (keeping all sims in RAM), you can use:
    # diversity = diversity_score(all_tanimotos)
    # but now we compute diversity without storing all_tanimotos.
    diversity = diversity_score_from_mean(tan_stats["tanimoto_mean"])
    print(f"Diversity score: {diversity:.4f}")

    if tanimoto_cache_path is not None:
        print(
            f"Tanimoto cache written to {tanimoto_cache_path} "
            f"(count={tan_stats['tanimoto_count']}, mean={tan_stats['tanimoto_mean']})"
        )

    len_gen = len(gen_df)
    # Scatter back into full-length array
    print("converting back into full-length array...")
    max_sim = np.full((len_gen,), np.nan, dtype=np.float32)
    k = 0

    for i in valid_idx:
        if i % print_every == 0:
            print(f"Scattering similarity for molecule {i} of {len_gen}")
        if fps[i] is None:
            continue
        max_sim[i] = max_sim_valid[k]
        k += 1

    out_df = gen_df.copy()
    out_df["is_valid"] = is_valid
    out_df["canonical_smiles"] = canon
    out_df["murcko_scaffold_smiles"] = scaff
    out_df["tanimoto_max_to_ref"] = max_sim
    out_df["diversity_score"] = diversity

    out_df.to_csv(out_csv_path, index=False)
    print(f"Saved postprocessed results to {out_csv_path}")
    return out_df


def load_cached_tanimotos(path: str) -> np.ndarray:
    """
    Helper to load cached tanimoto sims later if you want histograms, quantiles, etc.
    The cache is written by _TanimotoCacheWriter as a raw float32 stream.
    """
    return np.fromfile(path, dtype=np.float32)

# %% [markdown]
# ## SLOW PART (Do not auto run!)

# %%

cal_subset_IOU = True

resultant_path = 'train_dist_temp_transformer_300k_test.txt'

#resultant_path = 'CVAE_transformer_300k_test.txt'

gen_smiles, gen_prop = read_resultant_file(resultant_path)
gen_df = pd.DataFrame({
    'smiles': gen_smiles,
    'MW': [prop[0] for prop in gen_prop],
    'LogP': [prop[1] for prop in gen_prop],
    'pred_LogP': [prop[2] for prop in gen_prop]
})

if cal_subset_IOU:


    # will take amount_to_process random molecules
    # of both the generated and reference molecules....
    amount_to_process = 50_000

    train_df = pd.DataFrame({
        'smiles': smiles_list,
        'MW': [prop[0] for prop in properties_list],
        'LogP': [prop[1] for prop in properties_list]
    })
    train_smiles = train_df['smiles'].tolist()
    train_mols = [Chem.MolFromSmiles(s) for s in train_smiles]


    processed = postprocess_and_save(gen_df=gen_df,gen_max=amount_to_process,ref_max=amount_to_process, ref_mols=train_mols,tanimoto_cache_path="save/tanimoto_cache.bin", flush_every_queries=1000, out_csv_path="generated_subset_with_similarity.csv")

# %%








def distribution_avg_tanimoto(df: pd.DataFrame, title_str: str = "Average Tanimoto Similarity Distribution"):
    plt.figure(figsize=(8, 5))
    plt.hist(df['tanimoto_max_to_ref'], bins=100, alpha=1.0, color="#01FF22")
    # add trend line
    plt.title(title_str)
    plt.xlabel('Max Tanimoto Similarity to train Set')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(save_folder + run_name + "/tanimoto_similarity_distribution.png")
    plt.show()



csv_path = "generated_subset_with_similarity.csv"
processed_df = pd.read_csv(csv_path)



# %% [markdown]
# ## CHEMICAL SPACE t-SNE!
# 

# %%
# =============================================================================
# Train vs Generated: PCA (2D) and t-SNE (2D) on Morgan fingerprints
#
# Assumptions:
# - train_df has column "smiles"
# - processed_df has column "smiles" and optionally "canonical_smiles"
# - processed_df optionally has column "tanimoto_max_to_ref" (float in [0,1])
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors


# -------------------------
# Config
# -------------------------
rand_num = 42
n_train = 10_000
n_gen = 10_000

radius = 2
fp_size = 2048

pca_pre_dim = 50          # preprojection dimension before t-SNE
tsne_perplexity = 30      # typical range: 5 to 50, must be < n_samples
tsne_alpha = 0.5 # point transparency for t-SNE plot
point_size = 14 # point size for scatter plots (increased for projector visibility)
point_size_train = 14  # separate size for train points
point_size_generated = 14  # separate size for generated points
point_edge_color = 'black'  # edge color for points
point_edge_width = 0.5  # edge width for points
point_edgecolors_enabled = True  # enable/disable point outlines


# -------------------------
# Sample SMILES
# -------------------------
train_smiles_sample = train_df["smiles"].sample(
    n=min(n_train, len(train_df)),
    random_state=rand_num
).tolist()

gen_smiles_col = "canonical_smiles" if "canonical_smiles" in processed_df.columns else "smiles"
gen_sample_df = processed_df.sample(
    n=min(n_gen, len(processed_df)),
    random_state=rand_num
).copy()

gen_smiles_sample = gen_sample_df[gen_smiles_col].tolist()
gen_tanimoto = (
    gen_sample_df["tanimoto_max_to_ref"].to_numpy(dtype=float)
    if "tanimoto_max_to_ref" in gen_sample_df.columns else None
)


# -------------------------
# Fingerprints -> matrices
# -------------------------
mp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

X_train, train_valid_mask = smiles_list_to_fp_matrix(
    mp_gen, train_smiles_sample, use_counts=False, dtype=np.int8
)
X_gen, gen_valid_mask = smiles_list_to_fp_matrix(
    mp_gen, gen_smiles_sample, use_counts=False, dtype=np.int8
)

# Keep tanimoto aligned with *valid* generated rows only
if gen_tanimoto is not None:
    gen_tanimoto = gen_tanimoto[gen_valid_mask]

print("Train valid:", X_train.shape[0], "of", len(train_smiles_sample))
print("Gen valid:", X_gen.shape[0], "of", len(gen_smiles_sample))

if X_train.shape[0] == 0 or X_gen.shape[0] == 0:
    raise RuntimeError("No valid molecules found in train or generated sample. Check SMILES parsing.")

# Combine for shared embedding space (train first, then gen)
X_all = np.vstack([X_train, X_gen])

# Class labels for coloring (0=train, 1=generated)
y_class = np.concatenate([
    np.zeros((X_train.shape[0],), dtype=np.int32),
    np.ones((X_gen.shape[0],), dtype=np.int32),
])

colors = np.array(["#0046AB", "#CC5979"])
# turn colors into cmap
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
colors = ListedColormap(colors)

# -------------------------
# Make sure output folder exists
# -------------------------
out_dir = os.path.join(save_folder, run_name)
os.makedirs(out_dir, exist_ok=True)


# -------------------------
# PCA (2D) visualization
# -------------------------
pca_vis = PCA(n_components=2, random_state=rand_num)
X_pca_2d = pca_vis.fit_transform(X_all)

def plot_pca_2d(X_pca_2d:np.ndarray, y_class:np.ndarray, colors:ListedColormap, out_dir:str) -> None:
    plt.figure(figsize=(8, 6))
    edge_kwargs = {'edgecolors': point_edge_color, 'linewidths': point_edge_width} if point_edgecolors_enabled else {}
    sc = plt.scatter(
        X_pca_2d[:, 0], X_pca_2d[:, 1],
        s=point_size, alpha=0.7, c=y_class, cmap=colors,
        **edge_kwargs
    )
    plt.colorbar(sc, label="Class (0=train, 1=generated)")
    plt.title("PCA (2D) on Morgan fingerprints colored by Class")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca2_train_vs_generated.png"), dpi=200)
    plt.show()

plot_pca_2d(X_pca_2d, y_class, colors, out_dir)

# -------------------------
# PCA-50 preprojection then t-SNE (2D)
# Note: when you already preproject with PCA, prefer init='random' in sklearn TSNE.
# -------------------------
pca_pre = PCA(n_components=pca_pre_dim, random_state=rand_num)
X_pca_pre = pca_pre.fit_transform(X_all)

# Perplexity must be < n_samples, and practically should be smaller than about n_samples/3
n_total = X_pca_pre.shape[0]
if tsne_perplexity >= n_total:
    raise ValueError(f"t-SNE perplexity ({tsne_perplexity}) must be < number of samples ({n_total}).")

tsne = TSNE(
    n_components=2,
    random_state=rand_num,
    perplexity=tsne_perplexity,
    init="random",
    learning_rate="auto",
)
X_tsne = tsne.fit_transform(X_pca_pre)

# Split back to train/gen
train_tsne = X_tsne[: X_train.shape[0]]
gen_tsne = X_tsne[X_train.shape[0] :]

def plot_tsne_2d(train_tsne:np.ndarray, gen_tsne:np.ndarray, point_size:int, tsne_alpha:float, out_dir:str) -> None:
    # Plot: train vs generated (two colors, with legend)
    plt.figure(figsize=(8, 6))
    edge_kwargs = {'edgecolors': point_edge_color, 'linewidths': point_edge_width} if point_edgecolors_enabled else {}
    plt.scatter(train_tsne[:, 0], train_tsne[:, 1], s=point_size_train, alpha=tsne_alpha, label="Train", **edge_kwargs)
    plt.scatter(gen_tsne[:, 0], gen_tsne[:, 1], s=point_size_generated, alpha=tsne_alpha, label="Generated", **edge_kwargs)
    plt.legend()
    plt.title(f"t-SNE on Morgan fingerprints (PCA-{pca_pre_dim} preprojection)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tsne_train_vs_generated.png"), dpi=200)
    plt.show()

plot_tsne_2d(train_tsne, gen_tsne, point_size, tsne_alpha, out_dir)

# -------------------------
# Plot: generated colored by tanimoto_max_to_ref (optional)
# -------------------------

# cholor sceme should be red to yellow to green, with red at 1 and green at 0, since higher tanimoto means more similar to train, and we want that to be red

def plot_tsne_colored_by_tanimoto(gen_tsne:np.ndarray, gen_tanimoto:np.ndarray, point_size:int, out_dir:str) -> None:
    colors = ["red", "yellow", "green"]
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)

    if gen_tanimoto is not None:
        if gen_tanimoto.shape[0] != gen_tsne.shape[0]:
            raise ValueError(
                "gen_tanimoto and gen_tsne are misaligned after validity filtering. "
                f"gen_tanimoto: {gen_tanimoto.shape[0]}, gen_tsne: {gen_tsne.shape[0]}"
            )

        plt.figure(figsize=(8, 6))
        edge_kwargs = {'edgecolors': point_edge_color, 'linewidths': point_edge_width} if point_edgecolors_enabled else {}
        sc = plt.scatter(
            gen_tsne[:, 0], gen_tsne[:, 1],
            s=point_size_generated, alpha=0.85, c=gen_tanimoto,
            vmin=0.0, vmax=1.0, cmap=cmap,
            **edge_kwargs
        )
        plt.colorbar(sc, label="Max Tanimoto to train")
        plt.title("Generated t-SNE colored by max Tanimoto to train")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "tsne_generated_colored_by_tanimoto.png"), dpi=200)
        plt.show()
plot_tsne_colored_by_tanimoto(gen_tsne, gen_tanimoto, point_size, out_dir)
print(80*"=")
print("Scaffold analysis...")

# %% [markdown]
# ---------------

# %% [markdown]
# ## Scaffold analysis!
# 

# %%
# =============================================================================
# Scaffold analysis (train vs generated) with deterministic sampling + zero-padding for missing scaffolds, and plotting the distribution of scaffold frequencies.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import math
import os
from collections import Counter

import numpy as np
import pandas as pd

# Required imports for plotting + RDKit drawing:
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold

# ---------------------------------------------------------------------
# Config containers
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class SampleConfig:
    rand_num: int = 42
    n_train: int = 100_000
    n_gen: int = 100_000
    index_npz_path: str = "train_random_indexes.npz"


@dataclass(frozen=True)
class DataPaths:
    train_path: str = "250k_zinc_clean.txt"
    gen_path: str = "CVAE_transformer_300k_test.txt"


@dataclass(frozen=True)
class OutputPaths:
    out_dir: str
    scaffold_dist_png: str = "scaffold_distribution_train_vs_gen.png"


# ---------------------------------------------------------------------
# I/O helpers (single definition)
# ---------------------------------------------------------------------


def read_smiles_one_per_line(path: str) -> List[str]:
    smiles: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                smiles.append(s)
    return smiles


def load_train_df(train_path: str) -> pd.DataFrame:
    train_smiles = read_smiles_one_per_line(train_path)
    return pd.DataFrame({"smiles": train_smiles})


def load_generated_df(gen_path: str) -> pd.DataFrame:
    # Wants path from config, and returns a dataframe with columns: smiles, MW, LogP
    gen_smiles, gen_prop = read_resultant_file(gen_path)
    return pd.DataFrame(
        {
            "smiles": gen_smiles,
            "MW": [p[0] for p in gen_prop],
            "LogP": [p[1] for p in gen_prop],
        }
    )


# ---------------------------------------------------------------------
# Sampling helpers, and pre-gen random indicies
# ---------------------------------------------------------------------

def generate_or_load_indexes_npz(
    file_name: str,
    n_train: int,
    n_gen: int,
    rand_num: int,
    n_train_pop: int,
    n_gen_pop: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-generate random indices for sampling train and generated populations.
    This is since generating random indicies is slow

    If the file already exists and matches the config, it will be loaded.
    """
    if os.path.exists(file_name):
        data = np.load(file_name, allow_pickle=False)

        meta_ok = (
            int(data["seed_train"]) == int(rand_num)
            and int(data["seed_gen"]) == int(rand_num) + 1
            and int(data["n_train"]) == int(n_train)
            and int(data["n_gen"]) == int(n_gen)
            and int(data["n_train_pop"]) == int(n_train_pop)
            and int(data["n_gen_pop"]) == int(n_gen_pop)
        )

        if meta_ok:
            print("Indexes already exist and match config. Loading them.")
            train_idx = data["train_idx"]
            gen_idx = data["gen_idx"]
            return train_idx, gen_idx

        print("NPZ exists but config differs. Regenerating.")
    else:
        print("Index file not found. Generating new random indices.")

    n_train_eff = min(int(n_train), int(n_train_pop))
    n_gen_eff = min(int(n_gen), int(n_gen_pop))

    rng_train = np.random.default_rng(int(rand_num))
    rng_gen = np.random.default_rng(int(rand_num) + 1)

    train_idx = rng_train.choice(n_train_pop, size=n_train_eff, replace=False).astype(np.int64)
    gen_idx = rng_gen.choice(n_gen_pop, size=n_gen_eff, replace=False).astype(np.int64)

    np.savez(
        file_name,
        train_idx=train_idx,
        gen_idx=gen_idx,
        seed_train=int(rand_num),
        seed_gen=int(rand_num) + 1,
        n_train=int(n_train),
        n_gen=int(n_gen),
        n_train_pop=int(n_train_pop),
        n_gen_pop=int(n_gen_pop),
    )

    print(f"Saved indices to: {file_name}")
    return train_idx, gen_idx


def sample_smiles(
    train_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    cfg: SampleConfig,
) -> Tuple[List[str], pd.DataFrame, List[str]]:
    train_idx, gen_idx = generate_or_load_indexes_npz(
        file_name=cfg.index_npz_path,
        n_train=cfg.n_train,
        n_gen=cfg.n_gen,
        rand_num=cfg.rand_num,
        n_train_pop=len(train_df),
        n_gen_pop=len(gen_df),
    )

    train_smiles_sample = train_df["smiles"].iloc[train_idx].to_list()
    gen_sample_df = gen_df.iloc[gen_idx].copy()
    gen_smiles_sample = gen_sample_df["smiles"].to_list()

    return train_smiles_sample, gen_sample_df, gen_smiles_sample


# ---------------------------------------------------------------------
# Scaffolds + novelty analysis (single unified implementation)
# ---------------------------------------------------------------------


def compute_scaffolds(smiles_list: Sequence[str]) -> List[Optional[str]]:
    scaffolds: List[Optional[str]] = []
    for s in smiles_list:
        mol = safe_mol_from_smiles(s)
        scaf = safe_murcko_scaffold_smiles(mol)
        scaffolds.append(scaf)
    return scaffolds


def scaffold_set(scaffolds: Sequence[Optional[str]]) -> set[str]:
    return set(s for s in scaffolds if s is not None)


def scaffold_counts(scaffolds: Sequence[Optional[str]]) -> Counter:
    return Counter(s for s in scaffolds if s is not None)


def novel_scaffold_counts(
    train_scaffolds: Sequence[Optional[str]],
    gen_scaffolds: Sequence[Optional[str]],
) -> Counter:
    train = scaffold_set(train_scaffolds)
    gen_counts = scaffold_counts(gen_scaffolds)
    return Counter({s: c for s, c in gen_counts.items() if s not in train})


def top_n_items(counts: Counter, n_top: int) -> List[Tuple[str, int]]:
    return counts.most_common(int(n_top))
def read_resultant_file(resultant_path:str):
    """
    This file has column headers, so we need to skip the first line when reading the file.
    The first column is SMILES, the second column is MW, and the third column is LogP.
    
    """
    gen_smiles = []
    gen_properties = []
    
    with open(resultant_path, 'r') as file:
        first_line = True
        #for line in file:
        #    parts = line.strip().split()
        #    if first_line:
        #        first_line = False
        #        continue  # Skip the header line
        #    if len(parts) >= 3:
        #        gen_smiles.append(parts[0])
        #        gen_properties.append((float(parts[1]), float(parts[2])))  # Assuming MW and LogP are in the second and third columns
        # just print all results:
        
        first_line = True
        for line in file:
            if first_line:
                first_line = False
                continue  # Skip the header line
            #print(line.strip())
            # split at ','
            parts = line.strip().split(',')
            if len(parts) >= 3:
                gen_smiles.append(parts[0])
                gen_properties.append((float(parts[1]), float(parts[2])))  # Assuming MW and LogP are in the second and third columns
        
    return gen_smiles, gen_properties

def safe_murcko_scaffold_smiles(mol: Chem.Mol) -> Optional[str]:
    """
    Murcko scaffold SMILES for a valid RDKit Mol.
    Returns None if scaffold extraction fails (common for weird structures).
    """
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None:
            return None
        # Canonical scaffold smiles
        return Chem.MolToSmiles(scaf, isomericSmiles=False, canonical=True)
    except Exception:
        return None

def safe_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """
    Parse a SMILES into an RDKit Mol.
    Returns None if parsing or sanitization fails.
    """
    if smiles is None:
        return None
    s = str(smiles).strip()
    if len(s) == 0:
        return None
    try:
        mol = Chem.MolFromSmiles(s, sanitize=True)
        return mol
    except Exception:
        return None


def unify_top_scaffolds_for_plot(
    train_scaffolds: Sequence[Optional[str]],
    gen_scaffolds: Sequence[Optional[str]],
    n_top: int = 20,
) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    train_top = dict(top_n_items(scaffold_counts(train_scaffolds), n_top))
    gen_top = dict(top_n_items(scaffold_counts(gen_scaffolds), n_top))

    keys = list(set(train_top.keys()) | set(gen_top.keys()))
    keys_sorted = sorted(
        keys,
        key=lambda s: train_top.get(s, 0) + gen_top.get(s, 0),
        reverse=True,
    )
    return keys_sorted, train_top, gen_top


def summarize_scaffold_overlap(
    train_scaffolds: Sequence[Optional[str]],
    gen_scaffolds: Sequence[Optional[str]],
) -> Dict[str, int]:
    train_set = scaffold_set(train_scaffolds)
    gen_set = scaffold_set(gen_scaffolds)

    overlap = len(train_set & gen_set)
    novel = len(gen_set - train_set)
    return {
        "unique_train_scaffolds": len(train_set),
        "unique_gen_scaffolds": len(gen_set),
        "overlap_scaffolds": overlap,
        "novel_gen_scaffolds": novel,
    }


# ---------------------------------------------------------------------
# Plotting + drawing (single definitions)
# ---------------------------------------------------------------------


def plot_scaffold_dist_train_vs_gen(
    train_scaffolds: Sequence[Optional[str]],
    gen_scaffolds: Sequence[Optional[str]],
    out_path: str,
    n_top: int = 20,
) -> None:
    labels, train_counts, gen_counts = unify_top_scaffolds_for_plot(
        train_scaffolds=train_scaffolds,
        gen_scaffolds=gen_scaffolds,
        n_top=n_top,
    )

    train_freqs = [train_counts.get(s, 0) for s in labels]
    gen_freqs = [gen_counts.get(s, 0) for s in labels]

    x = np.arange(len(labels))
    width = 0.35

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, train_freqs, width=width, label="Train")
    plt.bar(x + width / 2, gen_freqs, width=width, label="Generated")

    plt.xticks(x, labels, rotation=90)
    plt.xlabel("Scaffold (SMILES)")
    plt.ylabel("Frequency in Sample")
    plt.title("Top Scaffolds in Train vs Generated Samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()


def draw_top_novel_scaffolds_grid(
    train_scaffolds: Sequence[Optional[str]],
    gen_scaffolds: Sequence[Optional[str]],
    n_top: int = 24,
    n_cols: int = 6,
    mol_size: Tuple[int, int] = (250, 250),
    seed: int = 0,
    sort_by: str = "count",
) -> None:
    """
    Draw the most common novel scaffolds (present in gen but absent in train).
    """
    novel_counts = novel_scaffold_counts(train_scaffolds, gen_scaffolds)
    if len(novel_counts) == 0:
        print("No novel scaffolds found (gen contains no scaffolds absent from train).")
        return

    top_novel = top_n_items(novel_counts, n_top)
    if len(top_novel) == 0:
        print("No novel scaffolds found after filtering.")
        return

    if sort_by not in {"count", "random"}:
        raise ValueError("sort_by must be 'count' or 'random'")

    if sort_by == "random":
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(len(top_novel))
        top_novel = [top_novel[i] for i in perm]

    mols: List[Chem.Mol] = []
    legends: List[str] = []
    bad = 0

    for scaf_smiles, c in top_novel:
        mol = Chem.MolFromSmiles(scaf_smiles)
        if mol is None:
            bad += 1
            continue
        mols.append(mol)
        legends.append(f"count={c}")

    if len(mols) == 0:
        print("All top novel scaffold SMILES failed to parse with RDKit.")
        return

    if bad > 0:
        print(f"Warning: {bad} scaffold SMILES failed to parse and were skipped.")

    n = len(mols)
    n_cols_eff = max(1, int(n_cols))
    n_rows = int(math.ceil(n / n_cols_eff))

    fig_w = n_cols_eff * (mol_size[0] / 100.0)
    fig_h = n_rows * (mol_size[1] / 100.0)
    fig, axes = plt.subplots(n_rows, n_cols_eff, figsize=(fig_w, fig_h))

    if isinstance(axes, np.ndarray):
        axes_list = axes.flatten().tolist()
    else:
        axes_list = [axes]

    for ax in axes_list:
        ax.axis("off")

    for i, (mol, legend) in enumerate(zip(mols, legends)):
        ax = axes_list[i]
        img = Draw.MolToImage(mol, size=mol_size)
        ax.imshow(img)
        ax.set_title(legend, fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.show()

def draw_most_common_scaffolds_grid(
    scaffolds: Sequence[Optional[str]],
    n_top: int = 24,
    n_cols: int = 6,
    mol_size: Tuple[int, int] = (250, 250),
    title: Optional[str] = None,
) -> None:
    """
    Draw the most common scaffolds in a single set.
    This can be used for either train or generated scaffolds, but is not comparing them, just showing the most common ones in that set.
    Will be used as rough reps to see how complex the scaffolds are in each set! (Sanity check for whether the generated scaffolds are much simpler than the train ones, which could be a failure mode of generation.)
    """
    counts = scaffold_counts(scaffolds)
    top = top_n_items(counts, n_top)

    mols: List[Chem.Mol] = []
    legends: List[str] = []
    bad = 0

    for scaf_smiles, c in top:
        mol = Chem.MolFromSmiles(scaf_smiles)
        if mol is None:
            bad += 1
            continue
        mols.append(mol)
        legends.append(f"count={c}")

    if len(mols) == 0:
        print("No scaffold SMILES parsed successfully.")
        return

    if bad > 0:
        print(f"Warning: {bad} scaffold SMILES failed to parse and were skipped.")

    n = len(mols)
    n_cols_eff = max(1, int(n_cols))
    n_rows = int(math.ceil(n / n_cols_eff))

    fig_w = n_cols_eff * (mol_size[0] / 100.0)
    fig_h = n_rows * (mol_size[1] / 100.0)
    fig, axes = plt.subplots(n_rows, n_cols_eff, figsize=(fig_w, fig_h))
    # add title if provided
    if title is not None:
        plt.suptitle(title, fontsize=18,fontweight='bold')

    if isinstance(axes, np.ndarray):
        axes_list = axes.flatten().tolist()
    else:
        axes_list = [axes]

    for ax in axes_list:
        ax.axis("off")

    for i, (mol, legend) in enumerate(zip(mols, legends)):
        ax = axes_list[i]
        img = Draw.MolToImage(mol, size=mol_size)
        ax.imshow(img)
        ax.set_title(legend, fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# Orchestrator (no duplicated logic)
# ---------------------------------------------------------------------


def run_scaffold_analysis(
    *,
    data_paths: DataPaths,
    sample_cfg: SampleConfig,
    out_paths: OutputPaths,
    n_top_plot: int = 20,
    n_top_novel_grid: int = 24,
) -> None:
    print("Loading train SMILES from:", data_paths.train_path)
    train_df = load_train_df(data_paths.train_path)

    print("Loading generated data from:", data_paths.gen_path)
    gen_df = load_generated_df(data_paths.gen_path)

    print("\n=== Sanity checks ===")
    print("len(train_df):", len(train_df))
    print("len(gen_df):  ", len(gen_df))
    print("n_train:", sample_cfg.n_train)
    print("n_gen:  ", sample_cfg.n_gen)

    train_smiles_sample, gen_sample_df, gen_smiles_sample = sample_smiles(
        train_df=train_df,
        gen_df=gen_df,
        cfg=sample_cfg,
    )

    print("\n=== Sample sizes ===")
    print("Train sample size:", len(train_smiles_sample))
    print("Gen sample size:  ", len(gen_smiles_sample))

    # Compute scaffolds exactly once.
    train_scaffolds = compute_scaffolds(train_smiles_sample)
    gen_scaffolds = compute_scaffolds(gen_smiles_sample)

    stats = summarize_scaffold_overlap(train_scaffolds, gen_scaffolds)
    print("\n=== Scaffold statistics ===")
    print(f"Unique scaffolds in train sample:     {stats['unique_train_scaffolds']}")
    print(f"Unique scaffolds in generated sample: {stats['unique_gen_scaffolds']}")
    print(f"Generated scaffolds also in train:    {stats['overlap_scaffolds']}")
    print(f"Generated scaffolds NOT in train:     {stats['novel_gen_scaffolds']}")

    os.makedirs(out_paths.out_dir, exist_ok=True)

    plot_scaffold_dist_train_vs_gen(
        train_scaffolds=train_scaffolds,
        gen_scaffolds=gen_scaffolds,
        out_path=os.path.join(out_paths.out_dir, out_paths.scaffold_dist_png),
        n_top=n_top_plot,
    )

    draw_top_novel_scaffolds_grid(
        train_scaffolds=train_scaffolds,
        gen_scaffolds=gen_scaffolds,
        n_top=n_top_novel_grid,
        n_cols=6,
        mol_size=(250, 250),
        seed=sample_cfg.rand_num,
        sort_by="count",
    )
    print(80*"=")
    print("Most common scaffolds in train set!!!!")

    draw_most_common_scaffolds_grid(
        scaffolds=train_scaffolds,
        n_top=n_top_novel_grid,
        n_cols=6,
        mol_size=(250, 250),
        title="Train set",
    )
    print(80*"=")
    print("Most common scaffolds in generated set!!!!")
    draw_most_common_scaffolds_grid(
        scaffolds=gen_scaffolds,
        n_top=n_top_novel_grid,
        n_cols=6,
        mol_size=(250, 250),
        title="Generated set",
    )


# ---------------------------------------------------------------------
# Main exec!
# ---------------------------------------------------------------------





if __name__ == "__main__":
    
    mode: str = "transformer" #'lstm' or 'transformer', 
    if mode not in {"lstm", "transformer"}:
        raise ValueError("mode must be 'lstm' or 'transformer'")
    

  

    # Run of the transformer!
    save_folder = "save/"
    #run_name = "run_20260219_230438"
    run_name = "huge_generation_lstm"


    if mode == "transformer":
        run_name = "run_20260219_230438"
        gen_pth = "CVAE_transformer_300k_test.txt"
    elif mode == "lstm":
        run_name = "huge_generation_lstm"
        gen_pth = "CVAE_lstm_300k_test.txt"
    out_dir = os.path.join(save_folder, run_name)

    # mul for easy controll
    rand_num = 42
    mul = 15

    orig = 10_000

    num = orig * mul
    
    print(f"model: {mode}, num_samples: {num}, rand_num: {rand_num}")

    npz_path = f"train_random_indexes_{mode}_{num}_samples.npz"

    cfg = SampleConfig(
        rand_num=rand_num,
        n_train= num,
        n_gen= num,
        index_npz_path=npz_path,
    )

    run_scaffold_analysis(
        data_paths=DataPaths(
            train_path="250k_zinc_clean.txt",
            #gen_path="CVAE_transformer_300k_test.txt",
            gen_path=gen_pth,
        ),
        sample_cfg=cfg,
        out_paths=OutputPaths(out_dir=out_dir),
        n_top_plot=20,
        n_top_novel_grid=24,
    )
    

# %%
print("len(train_df):", len(train_df))
print("len(processed_df):", len(processed_df))
print("n_train:", n_train)
print("n_gen:", n_gen)

# %% [markdown]
# ## Descriptor space t-SNE
# 

# %%
# =============================================================================
# Train vs Generated: t-SNE (2D) on RDKit DESCRIPTORS (scaled)
#
# Mirrors the idea from dat675-recitation-02.ipynb:
# - compute a descriptor feature matrix
# - z-score normalize descriptors (StandardScaler)
# - optional PCA preprojection then t-SNE
#
# Assumptions:
# - train_df has column "smiles"
# - processed_df has column "smiles" and optionally "canonical_smiles"
# =============================================================================

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -------------------------
# Descriptor config
# -------------------------
# Use the same core set as in the recitation by default.
# You can extend this list (must match names in rdkit.Chem.Descriptors).
DESCRIPTOR_NAMES = [
    "HeavyAtomCount",
    "RingCount",
    "MolLogP",
    "MolWt",
    "NumHAcceptors",
    "NumHDonors",
]

desc_pca_pre_dim = 0           # 0 or None means "no PCA preprojection" for descriptors
desc_tsne_perplexity = 30      # must be < n_samples
desc_tsne_alpha = 0.5
desc_point_size = point_size   # reuse your scatter size
desc_point_size_train = point_size_train  # separate size for descriptor train points
desc_point_size_generated = point_size_generated  # separate size for descriptor generated points


def mol_to_descriptor_array(mol: Chem.Mol, descriptor_names: List[str]) -> np.ndarray:
    """
    Compute a fixed-length descriptor vector for one RDKit mol.
    Raises AttributeError if a descriptor name does not exist in RDKit.
    """
    out = np.empty((len(descriptor_names),), dtype=np.float32)
    for i, name in enumerate(descriptor_names):
        fn = getattr(Descriptors, name)
        out[i] = float(fn(mol))
    return out


def smiles_list_to_desc_matrix(
    smiles_list: List[str],
    descriptor_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Convert list of SMILES to (X_desc, valid_mask).
    X_desc has shape [num_valid, num_descriptors].
    valid_mask has shape [len(smiles_list)] and indicates which were valid.
    '''
    X = []
    valid_mask = np.zeros((len(smiles_list),), dtype=bool)

    for i, s in enumerate(smiles_list):
        mol = safe_mol_from_smiles(s)
        if mol is None:
            continue
        try:
            vec = mol_to_descriptor_array(mol, descriptor_names)
        except Exception:
            continue
        X.append(vec)
        valid_mask[i] = True

    if len(X) == 0:
        return np.zeros((0, len(descriptor_names)), dtype=np.float32), valid_mask

    return np.vstack(X).astype(np.float32), valid_mask


# -------------------------
# Build descriptor matrices
# -------------------------
X_train_desc, train_desc_valid_mask = smiles_list_to_desc_matrix(train_smiles_sample, DESCRIPTOR_NAMES)
X_gen_desc, gen_desc_valid_mask = smiles_list_to_desc_matrix(gen_smiles_sample, DESCRIPTOR_NAMES)

print("Train valid (desc):", X_train_desc.shape[0], "of", len(train_smiles_sample))
print("Gen valid (desc):", X_gen_desc.shape[0], "of", len(gen_smiles_sample))

if X_train_desc.shape[0] == 0 or X_gen_desc.shape[0] == 0:
    raise RuntimeError("No valid molecules found in train or generated sample for descriptor embedding.")

# Combine for shared embedding space
X_all_desc = np.vstack([X_train_desc, X_gen_desc])

# -------------------------
# Scale descriptors (z-score)
# -------------------------
scaler = StandardScaler()
X_all_desc_scaled = scaler.fit_transform(X_all_desc)

# -------------------------
# Optional PCA preprojection then t-SNE
# -------------------------
if desc_pca_pre_dim is not None and desc_pca_pre_dim > 0 and desc_pca_pre_dim < X_all_desc_scaled.shape[1]:
    pca_pre_desc = PCA(n_components=desc_pca_pre_dim, random_state=rand_num)
    X_pre_desc = pca_pre_desc.fit_transform(X_all_desc_scaled)
else:
    X_pre_desc = X_all_desc_scaled

n_total_desc = X_pre_desc.shape[0]
if desc_tsne_perplexity >= n_total_desc:
    raise ValueError(
        f"Descriptor t-SNE perplexity ({desc_tsne_perplexity}) must be < number of samples ({n_total_desc})."
    )

tsne_desc = TSNE(
    n_components=2,
    random_state=rand_num,
    perplexity=desc_tsne_perplexity,
    init="random",
    learning_rate="auto",
)
X_tsne_desc = tsne_desc.fit_transform(X_pre_desc)

# Split back to train/gen
train_tsne_desc = X_tsne_desc[: X_train_desc.shape[0]]
gen_tsne_desc = X_tsne_desc[X_train_desc.shape[0] :]


def train_to_gen_desc_plot(train_tsne_desc:np.ndarray, gen_tsne_desc:np.ndarray,desc_point_size,alpha:float=desc_tsne_alpha) -> None:
    # Plot: train vs generated in descriptor space
    plt.figure(figsize=(8, 6))
    edge_kwargs = {'edgecolors': point_edge_color, 'linewidths': point_edge_width} if point_edgecolors_enabled else {}
    plt.scatter(train_tsne_desc[:, 0], train_tsne_desc[:, 1], s=desc_point_size_train, alpha=alpha, label="Train", **edge_kwargs)
    plt.scatter(gen_tsne_desc[:, 0], gen_tsne_desc[:, 1], s=desc_point_size_generated, alpha=alpha, label="Generated", **edge_kwargs)
    plt.legend()
    plt.title("t-SNE on RDKit descriptors (scaled)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tsne_descriptors_train_vs_generated.png"), dpi=200)
    plt.show()

train_to_gen_desc_plot(train_tsne_desc, gen_tsne_desc, desc_point_size, alpha=desc_tsne_alpha)



def gen_desc_colored_by_tanimoto_plot(gen_tsne_desc:np.ndarray, gen_tanimoto_desc:np.ndarray, desc_point_size:int) -> None:
    # Plot: generated colored by max Tanimoto to train in descriptor space
    plt.figure(figsize=(8, 6))
    edge_kwargs = {'edgecolors': point_edge_color, 'linewidths': point_edge_width} if point_edgecolors_enabled else {}
    sc = plt.scatter(
        gen_tsne_desc[:, 0], gen_tsne_desc[:, 1],
        s=desc_point_size_generated, alpha=0.85, c=gen_tanimoto_desc,
        vmin=0.0, vmax=1.0, cmap=cmap,
        **edge_kwargs
    )
    plt.colorbar(sc, label="Max Tanimoto to train")
    plt.title("Generated t-SNE on RDKit descriptors colored by max Tanimoto to train")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tsne_descriptors_generated_colored_by_tanimoto.png"), dpi=200)
    plt.show()

if gen_tanimoto is not None:
    if np.array_equal(gen_valid_mask, gen_desc_valid_mask):
        gen_tanimoto_desc = gen_tanimoto
        gen_desc_colored_by_tanimoto_plot(gen_tsne_desc, gen_tanimoto_desc, desc_point_size)
        #plt.figure(figsize=(8, 6))
        #sc = plt.scatter(
        #    gen_tsne_desc[:, 0], gen_tsne_desc[:, 1],
        #    s=desc_point_size, alpha=0.85, c=gen_tanimoto_desc,
        #    vmin=0.0, vmax=1.0, cmap=cmap
        #)
        #plt.colorbar(sc, label="Max Tanimoto to train")
        #plt.title("Descriptor t-SNE colored by max Tanimoto to train")
        #plt.tight_layout()
        #plt.savefig(os.path.join(out_dir, "tsne_descriptors_generated_colored_by_tanimoto.png"), dpi=200)
        #plt.show()
    else:
        print(
            "Skipping descriptor t-SNE coloring by tanimoto: "
            "descriptor validity mask differs from fingerprint validity mask."
        )



