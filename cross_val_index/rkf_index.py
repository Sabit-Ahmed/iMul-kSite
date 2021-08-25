from sklearn.model_selection import RepeatedKFold
import numpy as np
import pandas as pd

#ptm_type = {'Ace':3659, 'Cro':201, 'Met':294, 'Suc':1112, 'Glut':189}
#dataset_dim = 8980
#
#for key, value in ptm_type.items():
#    y_ace = []
#    for i in range(value):
#        y_ace.append(1);
#    for i in range(dataset_dim-value):
#        y_ace.append(-1);
#    
#    y_ace = np.array(y_ace, dtype=np.int64)
#    y_train_ace = y_ace
#    
#    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
#    train_index=[]
#    test_index=[]
#    train_df=pd.DataFrame()   
#    test_df=pd.DataFrame()
#    
#    for train_i, test_i in rkf.split(y_train_ace):
#        #train_index.append(train_i), test_index.append(test_i) 
#        train_df1=pd.DataFrame(train_i)
#        test_df1=pd.DataFrame(test_i)
#        
#        train_df=pd.concat([train_df,train_df1],axis=1, ignore_index=True)
#        test_df=pd.concat([test_df,test_df1],axis=1, ignore_index=True)
#        
#    train_df.to_csv('train_index'+key+'.csv', index=None, header=None)
#    test_df.to_csv('test_index'+key+'.csv', index=None, header=None)


y_ace = []
for i in range(5059):
    y_ace.append(1);
for i in range(5059):
    y_ace.append(-1);

y_ace = np.array(y_ace, dtype=np.int64)
y_train_ace = y_ace

rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
train_index=[]
test_index=[]
train_df=pd.DataFrame()   
test_df=pd.DataFrame()

for train_i, test_i in rkf.split(y_train_ace):
    #train_index.append(train_i), test_index.append(test_i) 
    train_df1=pd.DataFrame(train_i)
    test_df1=pd.DataFrame(test_i)
    
    train_df=pd.concat([train_df,train_df1],axis=1, ignore_index=True)
    test_df=pd.concat([test_df,test_df1],axis=1, ignore_index=True)
    
train_df.to_csv('train_index.csv', index=None, header=None)
test_df.to_csv('test_index.csv', index=None, header=None)