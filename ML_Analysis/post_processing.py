'''
This module will contain the code that analyzes the outputs and returns the important features in the problem.

'''


def generate_sorted_error(regressor_error, index):
	'''
	Function takes a list of errors on each case pairs it with the index
	and sorts it by greatest error.
	'''
    rel_errors = tuple(zip(index, regressor_error))
    rel_errors_sorted = sorted(rel_errors, reverse = True, key = lambda v:v[1])  
    return rel_errors_sorted
	
def generate_important_features(regressor, feature_index):
	'''
	Function returns the most important features on predicting the qoi.
	'''
	importances = regressor.feature_importances_
	return list(zip(feature_index, importances))