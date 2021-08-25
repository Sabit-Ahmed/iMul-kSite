import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest


def OptimizedReadFeatures(absolute_path, key, value, feature_name):
    path = absolute_path+'\\extracted_dataset_'+key.lower()
    X_combined = {}
    if type(feature_name) is list:
        for i in feature_name:
            dataset = pd.read_csv(path+'\\'+i+'.csv', header = None)
            X_combined[i] = dataset
        X = pd.concat(X_combined, axis=1)
        X = X.values
    else:
        dataset = pd.read_csv(path+'\\'+feature_name+'.csv', header = None)
        X = dataset.values
    
    y = []
    for i in range(value):
        y.append(1);
    for i in range(X.shape[0]-value):
        y.append(-1);
    y = np.array(y, dtype=np.int64)
    print(f"Features reading done for {key}")
    return X, y


def int_and_sel(K,X,y):
    sel = SelectKBest(f_classif, k=K)
    X_train = sel.fit(X, y)
    cols = sel.get_support(indices=True)
    features = X[:,cols]
    
    return features, cols
