import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
import os

# fixed parameters for all groups (for fair comparison)
FIXED_PARAMS = {
    'n_estimators': 500,
    'max_depth': 12,     
    'max_features': 'sqrt',
    'min_samples_leaf': 10,
    'random_state': 42,
    'n_jobs': -1
}

WORKSPACE_ROOT = os.path.dirname(__file__)
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(WORKSPACE_ROOT)), "data")
RESULTS_ROOT = os.path.join(os.path.dirname(os.path.dirname(WORKSPACE_ROOT)), "results")

GROUP_FOLDER = "combination_3900_molecules_and_67_%_synthetic"   #change this for different groups!!!
USE_SYNTHETIC = True
HOLDOUT_FILE = "heldout_datasets/heldout_testset.csv"


# ================= feature engineering =================
def generate_combined_features(smiles_list, radius=2, n_bits=2048):
    fps = []
    valid_indices = []
    
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    
    calc_list = [x[1] for x in Descriptors.descList]
    
    for idx, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = mfpgen.GetFingerprint(mol)
                arr_fp = np.zeros((0,), dtype=np.int8)
                Chem.DataStructs.ConvertToNumpyArray(fp, arr_fp)
                arr_desc = []
                for calc in calc_list:
                    try:
                        val = calc(mol)
                    except:
                        val = 0.0 
                    arr_desc.append(val)
                
                combined_features = np.concatenate([arr_fp, arr_desc])
                
                fps.append(combined_features)
                valid_indices.append(idx)
        except:
            pass

    fps_array = np.array(fps, dtype=np.float32)
    fps_array = np.nan_to_num(fps_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    return fps_array, valid_indices

# ================= read & 5-fold cross-validation =================

# We load all data sets and folds, and generate the features for the SMILES.
# The features and targets are returned separately, aswell as separated by original and synthetic data.
def load_and_process_data(folder : str, contains_synthetic : bool) -> tuple[list, list, list, list]:
    folds_X = []
    folds_y = []

    for i in range(5):
        print(f"Loading fold {i}")
        csv_name = f'original_fold_{i}.csv' 
        filename = os.path.join(folder, csv_name)
        
        if not os.path.exists(filename):
            print(f"can not find {filename}")
            exit()
        df = pd.read_csv(filename)
        X, idx = generate_combined_features(df['smiles'])
        y = df.iloc[idx]['pIC50'].values
        folds_X.append(X)
        folds_y.append(y)
    
    if contains_synthetic:
        syn_X = []
        syn_y = []

        for i in range(5):
            print(f"Loading synthetic iteration {i}")
            csv_name = f'synthetic_data_iteration_{i}.csv'
            filename = os.path.join(folder, csv_name)
            
            if not os.path.exists(filename):
                print(f"can not find {filename}")
                exit()
            df = pd.read_csv(filename)
            X, idx = generate_combined_features(df['smiles'])
            y = df.iloc[idx]['pred_pIC50'].values 
            
            syn_X.append(X)
            syn_y.append(y)
    else:
        syn_X = [None, None, None, None, None]
        syn_y = [None, None, None, None, None]
    
    print("Done loading data")
    return folds_X, folds_y, syn_X, syn_y

# We combine the folds that aren't part of the validation set for each fold iteration,
# we further add the corresponding synthetic set, if available, to the combined training set.
def compile_training_fold_iterations(folds_X, folds_y, syn_X, syn_y) -> list[tuple[list, list]]:
    iterations = []
    for i in range(5):
        iteration_X_list = [folds_X[j] for j in range(5) if j != i]
        iteration_y_list = [folds_y[j] for j in range(5) if j != i]

        if syn_X[i] is not None:
            iteration_X_list.append(syn_X[i])
            iteration_y_list.append(syn_y[i]) 

        iteration_X = np.vstack(iteration_X_list)
        iteration_y = np.concatenate(iteration_y_list)

        iterations.append((iteration_X, iteration_y))

    return iterations

# We structure the folds into their validation sets.
def compile_validation_sets(folds_X, folds_y) -> list[tuple[list, list]]:
    validation_sets = []
    for i in range(5):
        validation_sets.append((folds_X[i], folds_y[i]))

    return validation_sets

# We train the random forset models on the fold iterations.
def train_RF_models(fold_iterations : list[tuple[list, list]]) -> list[RandomForestRegressor]:
    models = []
    for iteration_X, iteration_y in fold_iterations:
        model = RandomForestRegressor(**FIXED_PARAMS) 
        model.fit(iteration_X, iteration_y)
        models.append(model)
    
    print("Training models done")

    return models

# We evaluate the trained models on R2 for the training and validation sets, RMSE on the validation set, and rho on the validation set.
def evaluate_models(models : list[RandomForestRegressor], fold_iterations : list[tuple[list, list]], validation_sets : list[tuple[list, list]]) -> dict:
    cv_metrics = {
        'train_r2': [], 'val_r2': [], 
        'val_rmse': [], 'val_rho': []
    }
    
    for i, model in enumerate(models):
        training_X, training_y = fold_iterations[i]
        validation_X, validation_y = validation_sets[i]

        # evaluate
        train_preds = model.predict(training_X)
        val_preds = model.predict(validation_X)
        
        # metrics
        tr_r2 = r2_score(training_y, train_preds)
        val_r2 = r2_score(validation_y, val_preds)
        val_rmse = np.sqrt(mean_squared_error(validation_y, val_preds))
        val_rho = spearmanr(validation_y, val_preds)[0]
        
        cv_metrics['train_r2'].append(tr_r2)
        cv_metrics['val_r2'].append(val_r2)
        cv_metrics['val_rmse'].append(val_rmse)
        cv_metrics['val_rho'].append(val_rho)
        
        print(f"  Fold {i}: Train R2={tr_r2:.3f} | Val R2={val_r2:.3f} | Gap={tr_r2-val_r2:.3f}")
    
    return cv_metrics



print(f"--- Evaluating the data set {GROUP_FOLDER} ---")


# read fold data
folds_X, folds_y, syn_X, syn_y = load_and_process_data(os.path.join(DATA_ROOT, GROUP_FOLDER), USE_SYNTHETIC)

print("starting 5-Fold CV (using fixed parameters)...")

training_sets = compile_training_fold_iterations(folds_X, folds_y, syn_X, syn_y)
validation_sets = compile_validation_sets(folds_X, folds_y)

trained_models = train_RF_models(training_sets)

cv_metrics = evaluate_models(trained_models, training_sets, validation_sets)


if not os.path.exists(os.path.join(RESULTS_ROOT, "RF_model", GROUP_FOLDER)): os.makedirs(os.path.join(RESULTS_ROOT, "RF_model", GROUP_FOLDER))


# CV summary
mean_train_r2 = np.mean(cv_metrics['train_r2'])
mean_val_r2 = np.mean(cv_metrics['val_r2'])
gap = mean_train_r2 - mean_val_r2

print(" [CV summary]:")
print(f"   1. (Val R2):  {mean_val_r2:.4f} ")
print(f"   2. (Gap):     {gap:.4f} ")
print(f"   3. (Std Dev):   {np.std(cv_metrics['val_r2']):.4f} ")
print(f"   4. (Rho):     {np.mean(cv_metrics['val_rho']):.4f} ")

# =================  Holdout test =================
print("using fixed parameters for final full-training...")
X_all_list = folds_X + [x for x in syn_X if x is not None]
y_all_list = folds_y + [y for y in syn_y if y is not None]

X_all = np.vstack(X_all_list)
y_all = np.concatenate(y_all_list)

final_model = RandomForestRegressor(**FIXED_PARAMS)
final_model.fit(X_all, y_all)
full_train_score = final_model.score(X_all, y_all)

if os.path.exists(os.path.join(DATA_ROOT, HOLDOUT_FILE)):
    holdout_df = pd.read_csv(os.path.join(DATA_ROOT, HOLDOUT_FILE))
    X_h, idx_h = generate_combined_features(holdout_df['smiles'])
    y_h = holdout_df.iloc[idx_h]['pIC50'].values
    
    preds_h = final_model.predict(X_h)
    
    h_r2 = r2_score(y_h, preds_h)
    h_rmse = np.sqrt(mean_squared_error(y_h, preds_h))
    h_rho = spearmanr(y_h, preds_h)[0]
    final_gap = full_train_score - h_r2
    
    print(f" final Holdout report:")
    print(f"   Train R2 (Full): {full_train_score:.4f}")
    print(f"   Test R2 (Hold):  {h_r2:.4f}")
    print(f"   Gap (Overfit):   {final_gap:.4f}")
    print(f"   Test RMSE:       {h_rmse:.4f}")
    print(f"   Test Rho:        {h_rho:.4f}")
    
    # whtether overfitting or not?
    if final_gap > 0.2:
        print("overfitting.")
    elif final_gap < 0.1:
        print("excellent.")
    else:
        print("normal")

else:
    print("can not find Holdout file")

# =================  save outcomes =================

result_data = {
    'Folder_Name': GROUP_FOLDER,  
    'CV_Mean_Val_R2': mean_val_r2,
    'CV_Gap': gap,
    'CV_Val_R2_Std': np.std(cv_metrics['val_r2']),
    'CV_Mean_Rho': np.mean(cv_metrics['val_rho']),
    'Holdout_Train_R2(Full)': full_train_score,
    'Holdout_Test_R2': h_r2,
    'Holdout_Gap': final_gap,
    'Holdout_RMSE': h_rmse,
    'Holdout_Rho': h_rho
}

df_result = pd.DataFrame([result_data])

csv_save_path = os.path.join(RESULTS_ROOT, "RF_model", GROUP_FOLDER, 'RF_metrics_summary.csv')

df_result.to_csv(csv_save_path, index=False)

print(f"Metrics saved to: {csv_save_path} ---")




# ================= visualization =================
sns.set_theme(style="whitegrid")
print("creating visualizations...")

# ----------------- figure 1: Actual vs Predicted scatter plot -----------------
plt.figure(figsize=(8, 8))
plt.scatter(y_h, preds_h, alpha=0.7, edgecolors='w', s=80, color='#4C72B0')

min_val = min(np.min(y_h), np.min(preds_h))
max_val = max(np.max(y_h), np.max(preds_h))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal (y=x)')

plt.title(f'Holdout Set: Actual vs Predicted pIC50\n$R^2$ = {h_r2:.3f}, RMSE = {h_rmse:.3f}', fontsize=14)
plt.xlabel('Actual pIC50', fontsize=12)
plt.ylabel('Predicted pIC50', fontsize=12)
plt.legend(loc='upper left', fontsize=12)

scatter_path = os.path.join(RESULTS_ROOT, "RF_model", GROUP_FOLDER, 'RF_plot_actual_vs_predicted.png')
plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
plt.close() 

# ----------------- figure 2: Residual Plot -----------------
plt.figure(figsize=(8, 6))
residuals = y_h - preds_h
plt.scatter(preds_h, residuals, alpha=0.7, edgecolors='w', s=80, color='#55A868')

plt.axhline(y=0, color='r', linestyle='--', lw=2)

plt.title('Holdout Set: Residual Plot', fontsize=14)
plt.xlabel('Predicted pIC50', fontsize=12)
plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)

residual_path = os.path.join(RESULTS_ROOT, "RF_model", GROUP_FOLDER, 'RF_plot_residuals.png')
plt.savefig(residual_path, dpi=300, bbox_inches='tight')
plt.close()

# ----------------- figure 3: Rank Correlation Plot -----------------
plt.figure(figsize=(8, 8))


actual_ranks = rankdata(y_h)
predicted_ranks = rankdata(preds_h)

plt.scatter(actual_ranks, predicted_ranks, alpha=0.7, edgecolors='w', s=80, color='#937860')

max_rank = len(y_h)
plt.plot([1, max_rank], [1, max_rank], 'r--', lw=2, label='Perfect Ranking')

plt.title(f'Holdout Set: Actual Rank vs Predicted Rank\nSpearman $\\rho$ = {h_rho:.3f}', fontsize=14)
plt.xlabel('Actual pIC50 Rank', fontsize=12)
plt.ylabel('Predicted pIC50 Rank', fontsize=12)
plt.legend(loc='upper left', fontsize=12)

rank_path = os.path.join(RESULTS_ROOT, "RF_model", GROUP_FOLDER, 'RF_plot_rank_correlation.png')
plt.savefig(rank_path, dpi=300, bbox_inches='tight')
plt.close()


print(f"--- Charts generated and saved in {GROUP_FOLDER} folder ---")