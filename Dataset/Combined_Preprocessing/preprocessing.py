import pandas as pd 


ptm_dataset = pd.read_csv(r"S13_PTM_Dataset.csv",index_col=[0])

##### Acetylation ####
ace_pos = ptm_dataset[ptm_dataset['Ace'].isin([1])]    ### POSITIVE
ace_neg = ptm_dataset[ptm_dataset['Ace'].isin([0])]    ### NEGATIVE

pos_df1 = pd.DataFrame([ace_pos['Sequence']])
neg_df1 = pd.DataFrame([ace_neg['Sequence']])

pos_df1 = pos_df1.transpose()
neg_df1 = neg_df1.transpose()

ace_pos_ind = pd.DataFrame(pos_df1.index.values)
ace_neg_ind = pd.DataFrame(neg_df1.index.values)

pos_df1.to_csv(r'ace_pos.csv',index=None,header=None)
neg_df1.to_csv(r'ace_neg.csv',index=None,header=None)
ace_pos_ind.to_csv(r'ace_pos_ind.csv',index=None,header=None)
ace_neg_ind.to_csv(r'ace_neg_ind.csv',index=None,header=None)


##### Crotonylation ####
cro_pos = ptm_dataset[ptm_dataset['Cro'].isin([1])]    ### POSITIVE
cro_neg = ptm_dataset[ptm_dataset['Cro'].isin([0])]    ### NEGATIVE

pos_df2 = pd.DataFrame([cro_pos['Sequence']])
neg_df2 = pd.DataFrame([cro_neg['Sequence']])

pos_df2 = pos_df2.transpose()
neg_df2 = neg_df2.transpose()

pos_df2.to_csv(r'cro_pos.csv',index=None,header=None)
neg_df2.to_csv(r'cro_neg.csv',index=None,header=None)
cro_pos_ind = pd.DataFrame(pos_df2.index.values)
cro_neg_ind = pd.DataFrame(neg_df2.index.values)
cro_pos_ind.to_csv(r'cro_pos_ind.csv',index=None,header=None)
cro_neg_ind.to_csv(r'cro_neg_ind.csv',index=None,header=None)


##### Methylation ####
met_pos = ptm_dataset[ptm_dataset['Met'].isin([1])]    ### POSITIVE
met_neg = ptm_dataset[ptm_dataset['Met'].isin([0])]    ### NEGATIVE

pos_df3 = pd.DataFrame([met_pos['Sequence']])
neg_df3 = pd.DataFrame([met_neg['Sequence']])

pos_df3 = pos_df3.transpose()
neg_df3 = neg_df3.transpose()

pos_df3.to_csv(r'met_pos.csv',index=None,header=None)
neg_df3.to_csv(r'met_neg.csv',index=None,header=None)
met_pos_ind = pd.DataFrame(pos_df3.index.values)
met_neg_ind = pd.DataFrame(neg_df3.index.values)
met_pos_ind.to_csv(r'met_pos_ind.csv',index=None,header=None)
met_neg_ind.to_csv(r'met_neg_ind.csv',index=None,header=None)


##### Succinylation ####
suc_pos = ptm_dataset[ptm_dataset['Suc'].isin([1])]    ### POSITIVE
suc_neg = ptm_dataset[ptm_dataset['Suc'].isin([0])]    ### NEGATIVE

pos_df4 = pd.DataFrame([suc_pos['Sequence']])
neg_df4 = pd.DataFrame([suc_neg['Sequence']])

pos_df4 = pos_df4.transpose()
neg_df4 = neg_df4.transpose()

pos_df4.to_csv(r'suc_pos.csv',index=None,header=None)
neg_df4.to_csv(r'suc_neg.csv',index=None,header=None)
suc_pos_ind = pd.DataFrame(pos_df4.index.values)
suc_neg_ind = pd.DataFrame(neg_df4.index.values)
suc_pos_ind.to_csv(r'suc_pos_ind.csv',index=None,header=None)
suc_neg_ind.to_csv(r'suc_neg_ind.csv',index=None,header=None)


##### Glutarylation ####
glut_pos = ptm_dataset[ptm_dataset['Glut'].isin([1])]    ### POSITIVE
glut_neg = ptm_dataset[ptm_dataset['Glut'].isin([0])]    ### NEGATIVE

pos_df6 = pd.DataFrame([glut_pos['Sequence']])
neg_df6 = pd.DataFrame([glut_neg['Sequence']])

pos_df6 = pos_df6.transpose()
neg_df6 = neg_df6.transpose()

pos_df6.to_csv(r'glut_pos.csv',index=None,header=None)
neg_df6.to_csv(r'glut_neg.csv',index=None,header=None)
glut_pos_ind = pd.DataFrame(pos_df6.index.values)
glut_neg_ind = pd.DataFrame(neg_df6.index.values)
glut_pos_ind.to_csv(r'glut_pos_ind.csv',index=None,header=None)
glut_neg_ind.to_csv(r'glut_neg_ind.csv',index=None,header=None)