'''
This module will contain the code that analyzes the outputs and returns the important features in the problem.

'''

from collections import Counter

def generate_statistics(error):
    absolute_errors = abs(error)
    squared_errors = error**2
    summed_errors = squared_errors.sum()
    mse = summed_errors/float(len(error))
    rmse = np.sqrt(mse)
    mae = absolute_errors.sum()/len(error)
    print(mse, rmse, mae)
    return mse, rmse, mae

	
def gen_sorted_name_error(sorted_err_tuple):
    sorted_err_names = []
    sorted_errors = []
    for i in range(len(sorted_err_tuple)):
        sorted_err_names.append(sorted_err_tuple[i][0])
        sorted_errors.append(sorted_err_tuple[i][1])
    return sorted_err_names, sorted_errors

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
	
	

def count_case_types(index)
	short_names = []
	for i in index:
		short_names.append(i[:7])
	short_data = Counter(short_names).most_common()
	return short_data
	
def generate_type_errors(index, reg_err, short_data):
    rel_errors = tuple(zip(index, reg_err))
    case_errors = []
    for i, type_case in enumerate(short_data):
        case_type = type_case[0]
        errors = []
        for i, case_error in enumerate(rel_errors):
            short_name = case_error[0][:7]
            error = float(case_error[1])

            if short_name == case_type:
                errors.append(error)
        case_errors.append(errors)
        
        case_errors_array = np.array([np.array(x) for x in case_errors])
    return case_errors_array
	
	
def gen_type_MSE_MAE_data(reg_err):
    type_mse = []
    type_mae = []
    for i in generate_type_errors(reg_err):
        mse, rmse, mae = generate_statistics(i)
        type_mse.append(mse)
        type_mae.append(mae)    
    
    return type_mse, type_mae