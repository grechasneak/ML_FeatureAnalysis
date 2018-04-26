
'''
This module will prepare the data for machine learning by scaling it so it has a mean of
0 and a variance of zero.

'''

import pandas as pd
import pickle
from sklearn.model_selection import StratifiedShuffleSplit


summed_sen_df = pickle.load( open( "/sen_data_summed.p", "rb" ) )  

#This creates a category column so stratified sampling can be done
category = []
for i in summed_sen_df.index:
    category.append(i[:6])
summed_sen_df['category'] = category	
summed_sen_df.drop('ieu-comp-therm-002-003', inplace = True)
	
#This function stratifies the data based on the categories in the problem, 
#so that every type of problem  is sampled representatively. 
def strat_test_train_split(dataframe):
    stratifier = StratifiedShuffleSplit(n_splits=4, test_size=0.2, random_state=13)
    for train_i, test_i in stratifier.split(dataframe, dataframe['category']):
        
        test_set = dataframe.iloc[test_i]
        train_set = dataframe.iloc[train_i]
    
    return train_set, test_set
	
train, test = strat_test_train_split(summed_sen_df)