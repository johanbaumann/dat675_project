import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Set, Dict
from collections import defaultdict
import os
import shutil

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')



def clone_to_folders(destinations : list[str], source : str) -> dict:
    for dst_dir in destinations:
        shutil.copytree(source, dst_dir, dirs_exist_ok=True)


def add_synthetic_molecules(destinations : list[str], desired_counts : list[int], source : str) -> tuple[dict, list[dict]]:
    for i in range(len(destinations)):
        for iteration in range(5):
            iteration_df = pd.read_csv(f"{source}/cv_iteration_{iteration}/generated/generated.csv").sample(n=desired_counts[i], ignore_index=True)
            iteration_df.to_csv(f"{destinations[i]}/synthetic_data_iteration_{iteration}.csv")


clone_to_folders(["data/combination_1950_molecules_and_33_%_synthetic", "data/combination_3900_molecules_and_67_%_synthetic"],
                 "data/combination_1300_molecules_and_0_%_synthetic")

add_synthetic_molecules(["data/combination_1950_molecules_and_33_%_synthetic", "data/combination_3900_molecules_and_67_%_synthetic"],
                        [650, 2600],
                        "vae/paper_based/CVAE/fold_pipeline_outputs")
