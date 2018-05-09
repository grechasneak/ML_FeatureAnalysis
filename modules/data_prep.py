
'''
This module will prepare the data for machine learning by scaling it so it has a mean of
0 and a variance of zero.

'''

import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler


summed_sen_df = pickle.load( open( "../data/sen_data_summed.p", "rb" ) )  
summed_sen_df.drop('ieu-comp-therm-002-003', inplace = True)

#This creates a category column so stratified sampling can be done
category = []
for i in summed_sen_df.index:
	category.append(i[:6])
summed_sen_df['category'] = category	

	

def strat_test_train_split(dataframe, category_column):
	'''
	This function stratifies the data based on the categories in the problem, 
	so that every type of problem is sampled representatively. 
	'''

	stratifier = StratifiedShuffleSplit(n_splits=4, test_size=0.2, random_state=13)
	for train_i, test_i in stratifier.split(dataframe, dataframe[category_column]):
		
		test_set = dataframe.iloc[test_i]
		train_set = dataframe.iloc[train_i]
	
	return (train_set, test_set)
	
	

def generate_training_data(data, feature, predicting, normalize = True):
	'''
	Takes in a tuple of pandas dataframes for the training and testing data
	and returns vectors for the training and testing data
	'''
	X_train = np.vstack(data[0][feature].values)
	X_test = np.vstack(data[1][feature].values)

	y_train = np.asarray(data[0][predicting].values).reshape(-1,1)
	y_test = np.asarray(data[1][predicting].values).reshape(-1,1)
	
	if normalize == True:
		X_train, X_test, y_train, y_test = normalize_data(X_train, X_test, y_train, y_test)
		
	return X_train, X_test, y_train, y_test


def normalize_data(X_train, X_test, y_train, y_test):
    '''The data fed to a neural network must be normalized unlike the decision trees.
       Centers the data with a mean of 0 and a standard deviation of one.
    '''
    scaler = Normalizer()
    X_train_nn = scaler.fit_transform(X_train)
    X_test_nn = scaler.fit_transform(X_test)
    y_train_nn = scaler.fit_transform(y_train)
    y_test_nn = scaler.fit_transform(y_test)
    return X_train_nn, X_test_nn, y_train_nn, y_test_nn
	
stratified_data = strat_test_train_split(summed_sen_df, 'category')
X_train, X_test, y_train, y_test= generate_training_data(stratified_data, 's', 'bias', True)