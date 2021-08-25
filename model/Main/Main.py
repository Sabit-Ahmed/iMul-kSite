import pandas as pd
import integrate_select_unmapped_features
import fun_biased_libsvm
from datetime import datetime
import os

features = ['aaFeature', 'binaryFeature','ProbabilityFeature','C1SAAP','C2SAAP','C3SAAP', 'C4SAAP', 'C5SAAP']
absolute_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..\\..\\'))
result_path = absolute_path + '\\performance\\'

def result(res_m, feature_name, clf_type):
    
    res_m=pd.DataFrame(res_m)
    if clf_type == "yes" or clf_type == "y":
        res_m.to_csv(result_path+feature_name+'\\res_Optimized.csv', index=None, header=['Aim','Cov','Acc','Abs-true','Abs_false'])
    elif clf_type == "no" or clf_type == "n":
        res_m.to_csv(result_path+feature_name+'\\res_LibSVM.csv', index=None, header=['Aim','Cov','Acc','Abs-true','Abs_false'])


def Backup(y_test, y_pred, feature_name, clf_type):    
    y_test = pd.concat(y_test, axis=1)
    y_pred = pd.concat(y_pred, axis=1)
    y_test.columns = y_test.columns.droplevel(1)
    y_pred.columns = y_pred.columns.droplevel(1)
    if clf_type == "yes" or clf_type == "y":
        y_test.to_csv(result_path+feature_name+'\\Y_Test_Optimized.csv', index=None)
        y_pred.to_csv(result_path+feature_name+'\\Y_Pred_Optimized.csv', index=None)
    elif clf_type == "no" or clf_type == "n":
        y_test.to_csv(result_path+feature_name+'\\Y_Test_LibSVM.csv', index=None)
        y_pred.to_csv(result_path+feature_name+'\\Y_Pred_LibSVM.csv', index=None)
    
    

def save_indices(indices, feature_name):
    
    indices = pd.concat(indices, axis=1)
    indices.columns = indices.columns.droplevel(1)
    indices.to_csv(result_path+feature_name+'\\FeatureIndices.csv', index=None)
    
    
def get_indices():
    feature_name = 'Selection'
    indices = pd.read_csv(result_path+feature_name+'\\FeatureIndices.csv')
    return indices
    
    
def Optimized():
    no = int(input("0:aaFeature\n1:binaryFeature\n2:ProbabilityFeature\n3:C1SAAP\n4:C2SAAP\n5:C3SAAP\n6:C4SAAP\n7:C5SAAP\n8:all\n9:selection\n\nChoose an option: "))
    
    ptm_type = {'Ace':4154, 'Cro':208, 'Met':325, 'Suc':1253, 'Glut':236 }
            
    
    y_test_all = {}
    y_pred_all = {}
    indices_all = {}
    for key,value in ptm_type.items():
        if no==8:
            feature_name = ['aaFeature','ProbabilityFeature','binaryFeature','C5SAAP']
            X,y = integrate_select_unmapped_features.OptimizedReadFeatures(absolute_path, key, value, feature_name)
            feature_name = 'Integrated_all'
            clf_type = 'no'
            
        elif no==9:
            feature_name = ['aaFeature','ProbabilityFeature','binaryFeature','C5SAAP']
            X,y = integrate_select_unmapped_features.OptimizedReadFeatures(absolute_path, key, value, feature_name)
            print(X.shape)
            K=100
            if K>=3527:
                X, indices = integrate_select_unmapped_features.int_and_sel('all',X,y)
            else:
                X, indices = integrate_select_unmapped_features.int_and_sel(K,X,y)
            indices_all[key] = pd.DataFrame(indices)
            feature_name = 'Selection'
            # print("Enter --> yes <-- for optimized classification and --> no <-- for LibSVM.")
            # clf_type = input('Y/N? ').lower()
            clf_type = "no"
            if clf_type == "yes" or clf_type == "y":
                print("Model with Optimized SVM")
            elif clf_type == "no" or clf_type == "n":
                print("Model with LibSVM")
            
        else:
            feature_name = features[no]
            X,y = integrate_select_unmapped_features.OptimizedReadFeatures(absolute_path, key, value, feature_name)
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

        

def GS_temp_res(result, ptm_type):
    now = datetime.now() # current date and time
    time = now.strftime("%H_%M_%S")
    result = pd.DataFrame(result)
    result.to_csv(result_path+'GridSearch\\6_res_'+ptm_type+'_'+time+'.csv', index=None, header=['C','gamma','AUC']) 



# def GS():
#     now = datetime.now() # current date and time
#     time = now.strftime("%d_%m_%Y")
#     ptm_type = input("Enter PTM type for grid-search: ").lower()
#     if ptm_type == "acetylation" or ptm_type == "ace":
#         ptm_type="Ace"
#         value = 3725
#     elif ptm_type == "crotonylation" or ptm_type == "cro":
#         ptm_type="Cro"
#         value = 183
#     elif ptm_type == "methylation" or ptm_type == "met":
#         ptm_type="Met"
#         value = 309
#     elif ptm_type == "succinylation" or ptm_type == "suc":
#         ptm_type="Suc"
#         value = 1143
#     elif ptm_type == "glutarylation" or ptm_type == "glut":
#         ptm_type="Glut"
#         value = 215
#     else:
#         print("Wrong keywords!")
    
    
#     feature_name = ['aaFeature','ProbabilityFeature','binaryFeature','C5SAAP']
#     X,y = integrate_select_unmapped_features.OptimizedReadFeatures(absolute_path, ptm_type, value, feature_name)
#     indices = get_indices()
#     X = X[:,indices[ptm_type]] 
#     result = fun_biased_libsvm.GS_CLF(X, ptm_type, value)
#     result = pd.DataFrame(result)
#     result.to_csv(result_path+'GridSearch\\6_res_'+ptm_type+'_'+time+'.csv', index=None, header=['C','gamma','AUC'])


     
        
def main(): 
    Optimized()
    # GS()    
        
if __name__=="__main__": 
    main()


