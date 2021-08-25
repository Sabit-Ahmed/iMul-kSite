import numpy as np
import pandas as pd
import integrate_select_unmapped_features
import fun_biased_libsvm
import os

absolute_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..\\..\\'))
result_path = absolute_path+'\\performance\\'
feature_path = 'IFS\\'


def result(K,acc,af):
    K=np.array(K)
    K=K.reshape(K.shape[0],-1)
    
    acc=np.array(acc)
    acc=acc.reshape(acc.shape[0],-1)
    
    af=np.array(af)
    af=af.reshape(af.shape[0],-1)
    
    result = np.concatenate([K,acc,af],axis=1)
    result=pd.DataFrame(result)
    result.to_csv(result_path+feature_path+'IFS_3100_step50.csv',header=['K', 'Accuracy', 'Absolute-False'], index=None)


"""Integrated"""
def Backup(y_test, y_pred, feature_name, K):    
    y_test = pd.concat(y_test, axis=1)
    y_pred = pd.concat(y_pred, axis=1)
    y_test.columns = y_test.columns.droplevel(1)
    y_pred.columns = y_pred.columns.droplevel(1)
    y_test.to_csv(result_path+feature_path+'Y_Test'+str(K)+'.csv', index=None)
    y_pred.to_csv(result_path+feature_path+'Y_Pred'+str(K)+'.csv', index=None)

   
    
def Optimized(K):
    due = []
    ptm_type = {'Ace':4154, 'Cro':208, 'Met':325, 'Suc':1253, 'Glut':236 }
    """Integrated"""
    if os.path.isfile(result_path+feature_path+'Y_Test'+str(K)+'.csv') and os.path.isfile(result_path+feature_path+'Y_Pred'+str(K)+'.csv'):
        y_pred_saved = pd.read_csv(result_path+feature_path+'Y_Pred'+str(K)+'.csv')
        y_test_saved = pd.read_csv(result_path+feature_path+'Y_Test'+str(K)+'.csv')
        cols = set(y_pred_saved.columns)
        ptm_type_keys = set(ptm_type.keys())
        due = list(ptm_type_keys - cols)
        if not due:
            acc, af = fun_biased_libsvm.OptimizedPred(y_test_saved.values, y_pred_saved.values)
            return acc, af
        ptm_type = { key:value for (key,value) in ptm_type.items() if key in due }
    
    y_test_all = {}
    y_pred_all = {}
    for key,value in ptm_type.items():
        
        feature_name = ['aaFeature','ProbabilityFeature','binaryFeature', 'C5SAAP']
        X,y = integrate_select_unmapped_features.OptimizedReadFeatures(absolute_path, key, value, feature_name)
        if K>=3527:
            X = integrate_select_unmapped_features.int_and_sel('all',X,y)
        else:
            X = integrate_select_unmapped_features.int_and_sel(K,X,y)
        print(f"Shape of X_train: {X.shape[0]}")
        feature_name = 'IFS'
        y_test, y_pred = fun_biased_libsvm.Optimized_CLF(X, key, value)
        print(f"Shape of Optimized_CLF OP: {y_pred.shape[0]}")
        y_test_all[key], y_pred_all[key] = pd.DataFrame(y_test), pd.DataFrame(y_pred) 
        print(f"Shape of Dict: {y_pred_all[key].shape[0]}")
        Backup(y_test_all, y_pred_all, feature_name, K) #Integrated
    
    y_test_all = pd.concat(y_test_all, axis=1, ignore_index=True)
    y_pred_all = pd.concat(y_pred_all, axis=1, ignore_index=True)
    print(f"Shape of y_pred after concatenating: {y_pred_all.shape[0]}")

    """Integrated"""
    if due:
        print(y_test_all.shape())
        print(y_pred_all.shape())
        y_test_all = np.concatenate([y_test_saved.values, y_test_all.values], axis=1)
        y_pred_all = np.concatenate([y_pred_saved.values, y_pred_all.values], axis=1)
        print(f"Shape of y_pred after concatenating(IF due): {y_pred_all.shape[0]}")
        acc, af = fun_biased_libsvm.OptimizedPred(y_test_all, y_pred_all)
        return acc, af
    
    acc, af = fun_biased_libsvm.OptimizedPred(y_test_all.values, y_pred_all.values)
    return acc, af
        
        
        
def main(): 
    accuracy = []
    abs_false=[]    
    nft = []
    K=3100 ## Not DONE
    for i in range(1):
        acc, af = Optimized(K)
        print(str(K)+"   "+ str(acc)+"   "+str(af))
        accuracy.append(acc)
        abs_false.append(af)
        nft.append(K)
        result(nft,accuracy,abs_false)
        os.remove(result_path + feature_path+ 'Y_Test'+str(K)+'.csv')
        os.remove(result_path + feature_path + 'Y_Pred'+str(K)+'.csv')  # Integrated
        K+=50
        print(K)
    result(nft,accuracy,abs_false)

    
if __name__=="__main__": 
    main()
