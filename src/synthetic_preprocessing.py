import pandas as pd
import numpy as np
import shutil

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)



# Duplicated a folder's contents into a list of new folders.
def clone_to_folders(destinations : list[str], source : str) -> dict:
    for dst_dir in destinations:
        shutil.copytree(source, dst_dir, dirs_exist_ok=True)


# Add a desired_counts[i] amount of synthetic molecules from source to destination[i], for each fold iteration
def add_synthetic_molecules(destinations : list[str], desired_counts : list[int], fold_count : int, source : str) -> tuple[dict, list[dict]]:
    for i in range(len(destinations)):
        for iteration in range(fold_count):
            iteration_df = pd.read_csv(f"{source}/cv_iteration_{iteration}/generated/generated.csv").sample(n=desired_counts[i], ignore_index=True)
            iteration_df.to_csv(f"{destinations[i]}/synthetic_data_iteration_{iteration}.csv")


clone_to_folders(["data/combination_1950_molecules_and_33_%_synthetic", "data/combination_3900_molecules_and_67_%_synthetic"],
                 "data/combination_1300_molecules_and_0_%_synthetic")

add_synthetic_molecules(["data/combination_1950_molecules_and_33_%_synthetic", "data/combination_3900_molecules_and_67_%_synthetic"],
                        [650, 2600],
                        5,
                        "vae/paper_based/CVAE/fold_pipeline_outputs")
