import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

import integrate_select_unmapped_features
import fun_biased_libsvm
from datetime import datetime
import os
from pathlib import Path
from joblib import dump

# -----------------------
# Paths (cross-platform)
# -----------------------
# Go up 2 levels from this file’s directory
BASE_DIR = Path(__file__).resolve().parent
absolute_path = (BASE_DIR / ".." / "..").resolve()

# Where results will be written
result_path = absolute_path / "performance"

# Helper: ensure a subdirectory exists and return its Path
def ensure_dir(*parts: str) -> Path:
    p = result_path.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p

# -----------------------
# Config
# -----------------------
features = ['aaFeature', 'binaryFeature', 'ProbabilityFeature', 'C1SAAP',
            'C2SAAP', 'C3SAAP', 'C4SAAP', 'C5SAAP']

# -----------------------
# I/O helpers
# -----------------------
def result(res_m, feature_name, clf_type):
    res_m = pd.DataFrame(res_m)
    out_dir = ensure_dir(feature_name)
    if clf_type in {"yes", "y"}:
        out_file = out_dir / "res_Optimized.csv"
    else:
        out_file = out_dir / "res_LibSVM.csv"
    res_m.to_csv(out_file, index=None, header=['Aim','Cov','Acc','Abs-true','Abs_false'])

def Backup(y_test, y_pred, feature_name, clf_type):
    y_test = pd.concat(y_test, axis=1)
    y_pred = pd.concat(y_pred, axis=1)
    # Drop second level if present
    if isinstance(y_test.columns, pd.MultiIndex):
        y_test.columns = y_test.columns.droplevel(1)
    if isinstance(y_pred.columns, pd.MultiIndex):
        y_pred.columns = y_pred.columns.droplevel(1)

    out_dir = ensure_dir(feature_name)
    if clf_type in {"yes", "y"}:
        y_test.to_csv(out_dir / "Y_Test_Optimized.csv", index=None)
        y_pred.to_csv(out_dir / "Y_Pred_Optimized.csv", index=None)
    else:
        y_test.to_csv(out_dir / "Y_Test_LibSVM.csv", index=None)
        y_pred.to_csv(out_dir / "Y_Pred_LibSVM.csv", index=None)

def save_indices(indices, feature_name):
    indices = pd.concat(indices, axis=1)
    if isinstance(indices.columns, pd.MultiIndex):
        indices.columns = indices.columns.droplevel(1)
    out_dir = ensure_dir(feature_name)
    indices.to_csv(out_dir / "FeatureIndices.csv", index=None)

def get_indices():
    feature_name = "Selection"
    in_file = result_path / feature_name / "FeatureIndices.csv"
    return pd.read_csv(in_file)

# -----------------------
# Main routines
# -----------------------
def Optimized():
    no = int(input(
        "0:aaFeature\n1:binaryFeature\n2:ProbabilityFeature\n3:C1SAAP\n4:C2SAAP\n"
        "5:C3SAAP\n6:C4SAAP\n7:C5SAAP\n8:all\n9:selection\n10:train for deployment\n\nChoose an option: "
    ))

    ptm_type = {'Ace': 4154, 'Cro': 208, 'Met': 325, 'Suc': 1253, 'Glut': 236}

    y_test_all = {}
    y_pred_all = {}
    indices_all = {}

    for key, value in ptm_type.items():
        if no == 8:
            feature_name_list = ['aaFeature', 'ProbabilityFeature', 'binaryFeature', 'C5SAAP']
            X, y = integrate_select_unmapped_features.OptimizedReadFeatures(absolute_path, key, value, feature_name_list)
            feature_name = 'Integrated_all'
            clf_type = 'no'

        elif no == 9:
            feature_name_list = ['aaFeature', 'ProbabilityFeature', 'binaryFeature', 'C5SAAP']
            X, y = integrate_select_unmapped_features.OptimizedReadFeatures(absolute_path, key, value, feature_name_list)
            print(X.shape)
            K = 100
            if K >= 3527:
                X, indices = integrate_select_unmapped_features.int_and_sel('all', X, y)
            else:
                X, indices = integrate_select_unmapped_features.int_and_sel(K, X, y)
            indices_all[key] = pd.DataFrame(indices)
            feature_name = 'Selection'
            clf_type = "no"
            print("Model with LibSVM")
        elif no == 10:
            feature_name_list = ['aaFeature', 'ProbabilityFeature', 'binaryFeature', 'C5SAAP']
            X, y = integrate_select_unmapped_features.OptimizedReadFeatures(absolute_path, key, value, feature_name_list)
            print(X.shape)
            K = 100
            if K >= 3527:
                X, indices = integrate_select_unmapped_features.int_and_sel('all', X, y)
            else:
                X, indices = integrate_select_unmapped_features.int_and_sel(K, X, y)

            scaler = StandardScaler()
            scaler.fit(X)  # Fit on training data only

            # Save the scaler
            scaler_path = os.path.join(absolute_path, "performance", "models", f"scaler_{key}.sav")
            joblib.dump(scaler, scaler_path)
            print(f"Saved scaler for {key}")
            indices_all[key] = pd.DataFrame(indices)
            feature_name = 'Selection'
            clf_type = "no"
            print("Model with LibSVM")
            clf = fun_biased_libsvm.Train_Full(X, key="Ace", value=4154, clf_type="no")
            # Save the trained model
            model_dir = os.path.join(absolute_path, "performance", "models")
            os.makedirs(model_dir, exist_ok=True)
            dump(clf, os.path.join(model_dir, f"{key}_full_model.joblib"))
            print(f"Saved model to {os.path.join(model_dir, f'{key}_full_model.joblib')}")
            save_indices(indices_all, feature_name)
            if key.lower() == "ace":
                return
            continue
        else:
            feature_name = features[no]
            X, y = integrate_select_unmapped_features.OptimizedReadFeatures(absolute_path, key, value, feature_name)
            clf_type = 'no'

        y_test, y_pred = fun_biased_libsvm.Optimized_CLF(X, key, value, clf_type)
        y_test_all[key], y_pred_all[key] = pd.DataFrame(y_test), pd.DataFrame(y_pred)
        Backup(y_test_all, y_pred_all, feature_name, clf_type)

    y_test_all = pd.concat(y_test_all, axis=1, ignore_index=True)
    y_pred_all = pd.concat(y_pred_all, axis=1, ignore_index=True)

    results = fun_biased_libsvm.OptimizedPred(y_test_all.values, y_pred_all.values)
    result(results, feature_name, clf_type)
    if indices_all:
        save_indices(indices_all, feature_name)

def GS_temp_res(result_vals, ptm_type):
    now = datetime.now()
    time_str = now.strftime("%H_%M_%S")
    df = pd.DataFrame(result_vals)
    out_dir = ensure_dir("GridSearch")
    out_file = out_dir / f"6_res_{ptm_type}_{time_str}.csv"
    df.to_csv(out_file, index=None, header=['C','gamma','AUC'])

# def GS():
#     now = datetime.now()
#     date_str = now.strftime("%d_%m_%Y")
#     ptm_in = input("Enter PTM type for grid-search: ").lower()
#     mapping = {
#         "acetylation": ("Ace", 3725), "ace": ("Ace", 3725),
#         "crotonylation": ("Cro", 183), "cro": ("Cro", 183),
#         "methylation": ("Met", 309), "met": ("Met", 309),
#         "succinylation": ("Suc", 1143), "suc": ("Suc", 1143),
#         "glutarylation": ("Glut", 215), "glut": ("Glut", 215),
#     }
#     if ptm_in not in mapping:
#         print("Wrong keywords!")
#         return
#     ptm_type, value = mapping[ptm_in]
#     feature_name_list = ['aaFeature','ProbabilityFeature','binaryFeature','C5SAAP']
#     X, y = integrate_select_unmapped_features.OptimizedReadFeatures(absolute_path, ptm_type, value, feature_name_list)
#     indices = get_indices()
#     X = X[:, indices[ptm_type]]
#     res = fun_biased_libsvm.GS_CLF(X, ptm_type, value)
#     df = pd.DataFrame(res)
#     out_dir = ensure_dir("GridSearch")
#     out_file = out_dir / f"6_res_{ptm_type}_{date_str}.csv"
#     df.to_csv(out_file, index=None, header=['C','gamma','AUC'])

def main():
    Optimized()
    # GS()

if __name__ == "__main__":
    main()
