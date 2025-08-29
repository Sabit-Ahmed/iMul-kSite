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


def Optimized_CLF(X, key, value, clf_type):
    import math
    import sklearn

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

    if clf_type in {"yes", "y"}:
        if key == "Ace":
            C = np.array([math.pow(2, 1)] * 5)
            gamma = np.array([math.pow(2, -8)] * 5)

        elif key == "Cro":
            C = np.array([math.pow(2, 9), math.pow(2, 3), math.pow(2, 2), math.pow(2, 5), math.pow(2, 4)])
            gamma = np.array([math.pow(2, -10)] * 5)

        elif key == "Met":
            C = np.array([math.pow(2, 3), math.pow(2, 3), math.pow(2, 1), math.pow(2, 2), math.pow(2, 1)])
            gamma = np.array([math.pow(2, -8), math.pow(2, -7), math.pow(2, -7), math.pow(2, -8), math.pow(2, -8)])

        elif key == "Suc":
            C = np.array([math.pow(2, 3)] * 5)
            gamma = np.array([math.pow(2, -10)] * 5)

        elif key == "Glut":
            C = np.array([math.pow(2, 5), math.pow(2, 3), math.pow(2, 2), math.pow(2, 1), math.pow(2, 1)])
            gamma = np.array([math.pow(2, -8)] * 5)

    elif clf_type in {"no", "n"}:
        C = np.array([1, 1, 1, 1, 1])
        g = 1 / X_train.shape[1]
        gamma = np.array([g] * 5)

    C = np.asarray(C, dtype=float).ravel()
    gamma = np.asarray(gamma, dtype=float).ravel()
    i = 1
    time = 0
    y_test_all = []
    y_pred_all = []

    for fold in range(train_index.shape[1]):
        train_ind = pd.DataFrame(train_index.iloc[:, fold].values).dropna()
        train_ind = np.array(train_ind, dtype=np.int64).reshape(-1)

        test_ind = pd.DataFrame(test_index.iloc[:, fold].values).dropna()
        test_ind = np.array(test_ind, dtype=np.int64).reshape(-1)

        if (i - 1) % 5 == 0:
            time = int((i - 1) / 5)
            print("\n\n")
            print("Iteration: " + str(time))

        X_train_split = X_train[train_ind]
        y_train_split = y_train[train_ind]
        clf = SVC(C=C[time], kernel='rbf', gamma=gamma[time], class_weight=weight,
                  probability=True, cache_size=500, random_state=0)
        clf.fit(X_train_split, y_train_split)
        X_test_split = X_train[test_ind]
        y_test_split = y_train[test_ind]
        y_pred = clf.predict(X_test_split)

        y_test_all = np.concatenate([y_test_all, y_test_split], axis=0) if len(y_test_all) else y_test_split
        y_pred_all = np.concatenate([y_pred_all, y_pred], axis=0) if len(y_pred_all) else y_pred

        print(f"Fold: {fold + 1}")
        i += 1

    print(f"Model Fitted for {key}")

    return y_test_all, y_pred_all


def GS_CLF(X, key, value):
    import sklearn.metrics
    import Main

    dataset_index_path = os.path.join(absolute_path, 'Dataset', 'Combined_Preprocessing')
    print(key)
    pos_ind = pd.read_csv(os.path.join(dataset_index_path, f"{key.lower()}_pos_ind.csv"), header=None)
    neg_ind = pd.read_csv(os.path.join(dataset_index_path, f"{key.lower()}_neg_ind.csv"), header=None)
    ptm_dataset = pd.read_csv(os.path.join(dataset_index_path, "S12_PTM_Dataset.csv"), index_col=[0])
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

    j = 1
    res = []
    y_test_all = []
    y_pred_score_all = []

    if key == "Ace":
        c_arr = [8]
        g_arr = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
    elif key == "Cro":
        c_arr = list(range(11))
        g_arr = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
    elif key == "Met":
        c_arr = list(range(11))
        g_arr = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
    elif key == "Suc":
        c_arr = [8]
        g_arr = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
    elif key == "Glut":
        c_arr = list(range(11))
        g_arr = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10]

    for cp in c_arr:
        c_val = 2 ** cp
        print(cp)
        for gp in g_arr:
            g_val = 2 ** gp
            print(gp)

            i = 1
            print("fold " + str(i))
            for fold in range(train_index.shape[1]):

                train_ind = pd.DataFrame(train_index.iloc[:, fold].values).dropna()
                train_ind = np.array(train_ind, dtype=np.int64).reshape(-1)

                test_ind = pd.DataFrame(test_index.iloc[:, fold].values).dropna()
                test_ind = np.array(test_ind, dtype=np.int64).reshape(-1)

                X_train_split = X_train[train_ind]
                y_train_split = y_train[train_ind]
                clf = SVC(C=c_val, kernel='rbf', gamma=g_val, class_weight=weight,
                          probability=True, cache_size=500, random_state=0)
                X_test_split = X_train[test_ind]
                y_test_split = y_train[test_ind]
                y_pred_score = clf.fit(X_train_split, y_train_split).decision_function(X_test_split)

                y_test_all = np.concatenate([y_test_all, y_test_split], axis=0) if len(y_test_all) else y_test_split
                y_pred_score_all = np.concatenate([y_pred_score_all, y_pred_score], axis=0) if len(y_pred_score_all) else y_pred_score

                if i % 5 == 0:
                    auc = sklearn.metrics.roc_auc_score(y_true=y_test_all, y_score=y_pred_score_all)
                    print(f"Shape of y_test: {y_test_all.shape}")
                    print(f"Shape of y_pred: {y_pred_score_all.shape}")

                    res.append([cp, gp, auc])
                    print("AUC: " + str(auc))

                    j += 1
                    y_test_all = []
                    y_pred_score_all = []

                i += 1
                print("fold " + str(i))
        Main.GS_temp_res(res, key)
    return res


def OptimizedPred(y_test, y_pred):
    intersect = (y_test == 1) & (y_pred == 1)
    y = y_test == 1
    z = y_pred == 1
    abs_true = 0
    abs_false = 0
    accuracy = 0
    cov = 0
    aim = 0
    sample = y_test.shape[0]
    label = 6
    for k in range(sample):
        intersection = np.sum(intersect[k])
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

        abs_false += (joint - intersection)

        if joint == intersection:
            abs_true += 1

        accuracy += weird_division(intersection, joint)

        if np.sum(y[k] != 0):
            cov += normal_division(intersection, np.sum(y[k]))
        elif (np.sum(y[k]) == 0 and joint == 0):
            cov += weird_division(intersection, np.sum(y[k]))
        else:
            cov += 0

        if np.sum(z[k] != 0):
            aim += normal_division(intersection, np.sum(z[k]))
        elif (np.sum(z[k]) == 0 and joint == 0):
            aim += weird_division(intersection, np.sum(z[k]))
        else:
            aim += 0

    abs_true /= sample
    abs_false = abs_false / (sample * label)
    accuracy /= sample
    cov /= sample
    aim /= sample

    curr_res = [aim, cov, accuracy, abs_true, abs_false]
    result = [curr_res]

    print("aim: " + str(aim))
    print("cov: " + str(cov))
    print("acc: " + str(accuracy))
    print("abs-true: " + str(abs_true))
    print("abs-false: " + str(abs_false))

    return result

def Train_Full(X, key, value, clf_type):
    """
    Train a single SVC on ALL data (no cross-validation) and print training results.
    Uses same label wiring and C/gamma logic as Optimized_CLF.
    Returns the fitted classifier.
    """
    import math
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    # --- paths/labels mirroring Optimized_CLF ---
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

    # class weights
    wp = X_train.shape[0] / (2 * pos_ind.shape[0])
    wn = X_train.shape[0] / (2 * neg_ind.shape[0])
    weight = {0: wn, 1: wp}

    # ---- choose C/gamma exactly like Optimized_CLF ----
    if clf_type in {"yes", "y"}:
        if key == "Ace":
            C = np.array([math.pow(2, 1)] * 5)
            gamma = np.array([math.pow(2, -8)] * 5)
        elif key == "Cro":
            C = np.array([math.pow(2, 9), math.pow(2, 3), math.pow(2, 2), math.pow(2, 5), math.pow(2, 4)])
            gamma = np.array([math.pow(2, -10)] * 5)
        elif key == "Met":
            C = np.array([math.pow(2, 3), math.pow(2, 3), math.pow(2, 1), math.pow(2, 2), math.pow(2, 1)])
            gamma = np.array([math.pow(2, -8), math.pow(2, -7), math.pow(2, -7), math.pow(2, -8), math.pow(2, -8)])
        elif key == "Suc":
            C = np.array([math.pow(2, 3)] * 5)
            gamma = np.array([math.pow(2, -10)] * 5)
        elif key == "Glut":
            C = np.array([math.pow(2, 5), math.pow(2, 3), math.pow(2, 2), math.pow(2, 1), math.pow(2, 1)])
            gamma = np.array([math.pow(2, -8)] * 5)
    else:  # "no"/"n"
        C = np.array([1, 1, 1, 1, 1])
        g = 1 / X_train.shape[1]
        gamma = np.array([g] * 5)

    # ensure scalars
    C = np.asarray(C, dtype=float).ravel()
    gamma = np.asarray(gamma, dtype=float).ravel()
    # pick the first set (you can choose another index if you prefer)
    C_val = C[0].item()
    gamma_val = gamma[0].item()

    # ---- fit on ALL data ----
    clf = SVC(C=C_val, kernel='rbf', gamma=gamma_val, class_weight=weight,
              probability=True, cache_size=500, random_state=0)
    clf.fit(X_train, y_train)

    # ---- training predictions & report ----
    y_pred_train = clf.predict(X_train)
    acc = accuracy_score(y_train, y_pred_train)
    labels = [0, 1]
    cm = confusion_matrix(y_train, y_pred_train, labels=labels)
    report = classification_report(
        y_train, y_pred_train, labels=labels,
        target_names=["neg(0)", "pos(1)"], zero_division=0
    )

    print("\n=== Training results (ALL data) ===")
    print(f"Key: {key} | clf_type: {clf_type} | C: {C_val} | gamma: {gamma_val}")
    print(f"Training accuracy: {acc:.4f}")
    print("Confusion matrix [rows=true, cols=pred]:\n", cm)
    print("Classification report:\n", report)

    return clf
