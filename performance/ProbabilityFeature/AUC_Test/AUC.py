import pandas as pd
import sklearn.metrics as sm

# path = r'C:\Users\User\Dropbox\Bioinformatics(sabit)\Protein\PTM\Model\Combined_New\performance\Selection' ###laptop
path = r'C:\Users\Admin\Dropbox\Bioinformatics(sabit)\Protein\PTM\Model\Combined_New\performance\Selection' ####PC
y_test = pd.read_csv(path+'\\Y_Test_iML-LysPTM.csv')
y_pred = pd.read_csv(path+'\\Y_Pred_iML-LysPTM.csv')

n_times = 5
ptm_type = ['Ace', 'Cro', 'Met', 'Suc', 'Glut'] 
auc=[]
for i in ptm_type:
    y_test_all = y_test[[i]]
    y_pred_all = y_pred[[i]]
    result=[]
    f_range=0
    limit = int(y_test_all.shape[0]/n_times)
    for j in range(n_times):
        y_test_time = y_test_all.iloc[f_range:f_range+limit, :].values
        y_pred_time = y_pred_all.iloc[f_range:f_range+limit, :].values
        print(f_range)
        print(f_range+limit)
        print("\n")
        mcc = sm.matthews_corrcoef(y_true = y_test_time, y_pred = y_pred_time)
        print(i+" MCC: "+str(mcc))
    
