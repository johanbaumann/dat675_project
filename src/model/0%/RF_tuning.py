import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
import os


DATA_FILE_PREFIX = 'fold_' 


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

print("reading data...")

X_list = []
y_list = []
fold_indices = [] 

for i in range(5):
    filename = f'{DATA_FILE_PREFIX}{i}.csv'
    
    if not os.path.exists(filename):
        print(f"can not find {filename}")
        exit()
        
    df = pd.read_csv(filename)
    
    X, idx = generate_combined_features(df['smiles'])
    y = df.iloc[idx]['pIC50'].values
    
    X_list.append(X)
    y_list.append(y)

    fold_indices.extend([i] * len(y))
    
    print(f"  -> Fold {i} finished, total {len(y)} samples")

X_all = np.vstack(X_list)
y_all = np.concatenate(y_list)
ps = PredefinedSplit(test_fold=fold_indices)

# ================= define search space =================
# serch space for Grid Search
param_grid = {
    'n_estimators': [300, 500],      # number of trees in the forest
    'max_depth': [8, 12, 15],          # maximum depth of the tree
    'max_features': ['sqrt', 'log2',0.1],     # feature selection strategy
    'min_samples_leaf': [5, 10, 15]            # parameter to prevent overfitting
}

print(f"Starting Grid Search...")
print(f"   (Will automatically perform 5-fold cross-validation, testing {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['max_features']) * len(param_grid['min_samples_leaf'])} parameter combinations)")

# ================= search =================
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=ps,
    scoring='neg_root_mean_squared_error', 
    return_train_score=True, 
    verbose=1,
    n_jobs=1 
)

grid_search.fit(X_all, y_all)

# ==================== output results& penalization ====================
print("\n" + "="*50)
print("search completed, calculating penalized scores...")

results = pd.DataFrame(grid_search.cv_results_)

results['mean_train_rmse'] = -results['mean_train_score']
results['mean_test_rmse'] = -results['mean_test_score']
results['gap'] = results['mean_test_rmse'] - results['mean_train_rmse']

alpha = 1.2
results['penalized_rmse'] = results['mean_test_rmse'] + alpha * results['gap'].apply(lambda x: max(0, x))

best_index = results['penalized_rmse'].idxmin()
best_params_penalized = results.loc[best_index, 'params']

print(grid_search.best_params_)
print(f"Test RMSE: {-grid_search.best_score_:.4f}")

print(best_params_penalized)
print(f"Test RMSE:       {results.loc[best_index, 'mean_test_rmse']:.4f}")
print(f"Train RMSE:      {results.loc[best_index, 'mean_train_rmse']:.4f}")
print(f"Gap:      {results.loc[best_index, 'gap']:.4f}")
print(f"Penalized Score: {results.loc[best_index, 'penalized_rmse']:.4f}")
print("="*50)

# ==================== rank top-5 models ====================
top_n = 5
best_models = results.sort_values(by='penalized_rmse', ascending=True).head(top_n)

print(f" {top_n} best robust parameter combinations：\n")
print("="*60)

for rank, (index, row) in enumerate(best_models.iterrows(), 1):
    print(f" {rank}")
    print(f"params combination: {row['params']}")
    print(f"Test RMSE  (test set error): {row['mean_test_rmse']:.4f}")
    print(f"Train RMSE (train set error): {row['mean_train_rmse']:.4f}")
    print(f"Gap        (overfitting gap): {row['gap']:.4f}")
    print(f"Final Score: {row['penalized_rmse']:.4f}")
    print("-" * 60)

# ==================== selectfinal model  ====================

best_robust_params = best_models.iloc[0]['params']
print(f"selected robust parameters: {best_robust_params}")
final_rf_model = RandomForestRegressor(random_state=42, n_jobs=-1, **best_robust_params)
final_rf_model.fit(X_all, y_all)

print("final model trained with the best robust parameters on all data.")