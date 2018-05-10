
'''
This module will test the data loader.

'''


import sys, os
testdir = os.path.dirname(__file__)
srcdir = '../'
datadir = '../data/'

sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
from data_prep import *


sys.path.insert(0, os.path.abspath(os.path.join(testdir, datadir)))	
	
	
	
	
# import sys, os
# import pkg_resources


# testdir = os.path.dirname(__file__)
# # datadir = '../data/'

# sys.path.insert(0, os.path.abspath(os.path.join(testdir, '/../')))



# file_path = os.path.join('../data/')
# filename = pkg_resources.resource_filename(__name__, file_path)
# sys.path.insert(0, os.path.abspath(filename))
	
	
	
	
	
	
	
	

exclude_list = ['ieu-comp-therm-002-003']

dataframe = load_dataframe(datadir + 'sen_data_summed.p', exclude_list)	

add_category_column(dataframe)	

stratified_data = strat_test_train_split(dataframe, 'category')

X_train, X_test, y_train, y_test = generate_training_data(stratified_data, 's', 'bias', True)


def assert_trainingDim():
	assert X_train.shape[1] == X_test.shape[1]
	
assert_trainingDim()
