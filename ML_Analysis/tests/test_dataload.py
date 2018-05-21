
'''
This module will test the data loader.

'''



import sys, os
import pandas
# testdir = os.path.dirname(__file__)
# srcdir = '../'
# datadir = '../data/'

# sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
#from data_prep import *

import data_prep
	
exclude_list = ['ieu-comp-therm-002-003']	
	
	

	
def test_loading_dataframe():
	dataframe = data_prep.load_dataframe('sen_data_summed.p', exclude_list)	# datadir + 
	assert type(dataframe) == pandas.core.frame.DataFrame
	return dataframe


def assert_trainingDim():
	assert X_train.shape[1] == X_test.shape[1]

	




#def test_adding_category()


dataframe = test_loading_dataframe()
data_prep.add_category_column(dataframe)	
stratified_data = data_prep.strat_test_train_split(dataframe, 'category')
X_train, X_test, y_train, y_test = data_prep.generate_training_data(stratified_data, 's', 'bias', True)
assert_trainingDim()


