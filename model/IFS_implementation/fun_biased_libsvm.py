import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os


absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
train_index = pd.read_csv(os.path.join(absolute_path, 'cross_val_index', 'train_index.csv'), header=None)
test_index  = pd.read_csv(os.path.join(absolute_path, 'cross_val_index', 'test_index.csv'), header=None)

def weird_division(n, d):
    return n / d if d else 1

def normal_division(m, c):
    return m / c


def Optimized_CLF(X, key, value):
    dataset_index_path = os.path.join(absolute_path, 'Dataset', 'Combined_Preprocessing')

    pos_ind = pd.read_csv(os.path.join(dataset_index_path, f"{key.lower()}_pos_ind.csv"), header=None)
    neg_ind = pd.read_csv(os.path.join(dataset_index_path, f"{key.lower()}_neg_ind.csv"), header=None)
    ptm_dataset = pd.read_csv(os.path.join(dataset_index_path, "S13_PTM_Dataset.csv"), index_col=[0])
    y_train = np.array(ptm_dataset[key], dtype=np.int64)

    sc = StandardScaler()
    X = sc.fit_transform(X)
    pos_ind = np.array(pos_ind, dtype=np.int64)
    neg_ind = np.array(neg_ind, dtype=np.int64)

    X_train = np.zeros([X.shape[0], X.shape[1]], dtype=np.float64)

    for c in range(len(X)):
        if c < value:
            X_train[pos_ind[c]] = X[c]
        else:
            X_train[neg_ind[c - value]] = X[c]

    wp = X_train.shape[0] / (2 * pos_ind.shape[0])
    wn = X_train.shape[0] / (2 * neg_ind.shape[0])
    weight = {0: wn, 1: wp}

    g = 1 / X_train.shape[1]
    clf = SVC(C=1, kernel='rbf', gamma=g, class_weight=weight, cache_size=500, random_state=0)

    y_test_all = []
    y_pred_all = []
    for fold in range(train_index.shape[1]):

        train_ind = pd.DataFrame(train_index.iloc[:, fold].values).dropna()
        train_ind = np.array(train_ind, dtype=np.int64).reshape(-1)

        test_ind = pd.DataFrame(test_index.iloc[:, fold].values).dropna()
        test_ind = np.array(test_ind, dtype=np.int64).reshape(-1)

        X_train_split = X_train[train_ind]
        y_train_split = y_train[train_ind]
        clf.fit(X_train_split, y_train_split)

        X_test_split = X_train[test_ind]
        y_test_split = y_train[test_ind]
        y_pred = clf.predict(X_test_split)

        # Safe concatenation when starting from empty lists
        y_test_all = np.concatenate([y_test_all, y_test_split], axis=0) if len(y_test_all) else y_test_split
        y_pred_all = np.concatenate([y_pred_all, y_pred], axis=0) if len(y_pred_all) else y_pred

        print(f"Fold: {fold + 1}")

    print(f"Model Fitted for {key}")

    return y_test_all, y_pred_all



def OptimizedPred(y_test, y_pred):
    ###########  Multi-Label New  ###########
    intersect = (y_test == 1) & (y_pred == 1)
    y = y_test == 1
    z = y_pred == 1
    abs_false = 0
    accuracy = 0
    joint = 0
    sample = y_test.shape[0]
    label = 6
    for k in range(sample):
        intersection = np.sum(intersect[k])  #####  INTERSECTION  #####

        ############   UNION   ###########
        if np.sum(y[k]) == 0 and np.sum(z[k]) == 0:
            joint = 0
        elif np.sum(y[k]) == 0 or np.sum(z[k]) == 0:
            numTrue = 0
            numPred = 0
            if np.sum(y[k]) == 0:
                numTrue = 1
            if np.sum(z[k]) == 0:
                numPred = 1
            joint = ((np.sum(y[k]) + numTrue + np.sum(z[k]) + numPred) - intersection)
        else:
            joint = ((np.sum(y[k]) + np.sum(z[k])) - intersection)

        ###########################################################

        abs_false += (joint - intersection)  ####### ABSOLUTE-FALSE #######
        accuracy += weird_division(intersection, joint)  #########    ACCURACY  #########

    abs_false = abs_false / (sample * label)
    accuracy /= sample

    return accuracy, abs_false