import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Set, Dict
from collections import defaultdict
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')



# Standard de-salting and de-duplication.
# De-duplication is done using a dictionary, which in python is a hash set and so very efficient.
def clean_and_process_dataframe(dataframe : pd.DataFrame, smiles_key : str, target_key : str) -> dict:
    resulting_dataset = {}
    for i, row in dataframe.iterrows():
        mol = Chem.MolFromSmiles(row["mol"])

        lfg = rdMolStandardize.LargestFragmentChooser()
        mol = lfg.choose(mol)

        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)

        smiles = Chem.MolToSmiles(mol)

        smiles_value = resulting_dataset.get(smiles)
        if smiles_value == None:
            resulting_dataset[smiles] = row["pIC50"]

    return resulting_dataset


# Associate molecules by their scaffolds.
# The output is a dictionary where the key is a scaffold, and where the values are the associated smiles, pIC50 pairs.
# Code is generic and so pIC50 could be replaced by any other target.
def sort_dataset_by_scaffolds(dataset : dict) -> dict:
    sorted_dataset = defaultdict(list)
    for smiles, target in dataset.items():
        mol = Chem.MolFromSmiles(smiles)

        try: # Remove?
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        except Chem.AtomValenceException:
            continue

        scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)

        if scaffold_smiles != '' and scaffold.GetNumAtoms() > 0:
            sorted_dataset[scaffold_smiles].append((smiles, target))

    return sorted_dataset


# Split the data into hold out testset and 5 folds.
# Is done so by scaffold splitting, in which allocations are made with molecules associated to scaffolds.
# The allocation is procedural: Each bin's (hold out set and folds) ability to contain the scaffolds associated molecules is check sequentially.
# Code is again generic and so the size of the hold out testset and the amount of folds can be changed.
def split_data_by_scaffolds_dataset(scaffold_sorted_dataset : dict, mol_count : int,
                                    holdout_size : int, fold_count : int
                                    ) -> tuple[dict, list[dict]]:
    data_bins = []
    bins_size_targets = []

    data_bins.append([])
    bins_size_targets.append(holdout_size)

    for i in range(fold_count):
        data_bins.append([])
        bins_size_targets.append((mol_count-holdout_size) / fold_count)

    for scaffold, molecules in scaffold_sorted_dataset.items():
        for bin_index in range(len(data_bins)):
            if len(data_bins[bin_index]) + len(molecules) <= bins_size_targets[bin_index]:
                data_bins[bin_index].extend(molecules)
                break
        else:
            data_bins[len(data_bins)-1].extend(molecules)
    
    return (data_bins[0], data_bins[1:])



bace_df = pd.read_csv("data/bace.csv").sample(frac=1)
clean_dataset = clean_and_process_dataframe(bace_df, "mol", "pIC50")
sorted_dataset = sort_dataset_by_scaffolds(clean_dataset)
holdout_set, data_folds = split_data_by_scaffolds_dataset(sorted_dataset, len(clean_dataset), 200, 5)

directory = f"data/heldout_datasets"
if not os.path.exists(directory): os.makedirs(directory)
saving_df = pd.DataFrame(data=holdout_set, columns=["smiles", "pIC50"])
saving_df.to_csv(f"{directory}/heldout_testset.csv")

directory = f"data/combination_1300_molecules_and_0_%_synthetic"
if not os.path.exists(directory): os.makedirs(directory)
for fold_index in range(len(data_folds)):
    saving_df = pd.DataFrame(data=data_folds[fold_index], columns=["smiles", "pIC50"])
    saving_df.to_csv(f"{directory}/original_fold_{fold_index}.csv")
