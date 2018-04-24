
'''
This module will prepare the data for machine learning by scaling it so it has a mean of
0 and a variance of zero.

'''

def generate_training_data(dataframe):
	'''
	Parameters
	----------
		dataframe : Pandas DataFrame
		Should contain all of the cases that will be trained and tested upon.

	Returns
	-----------
	X_train, X_test, X_all, matrix of shape = [n_samples, n_features]
	y_train, y_test, y_all, array, shape = [n_samples, 1]
	'''
	train_set, test_set = train_test_split(dataframe, test_size=0.2, random_state=42)

	X_all = np.vstack(dataframe['s'].values)
	X_all = np.insert(X_all, 0, dataframe['keig_siml'], axis =1)

	y_all = np.asarray(dataframe['bias'].values).reshape(-1,1)

	X_train = np.vstack(train_set['s'].values)
	X_test = np.vstack(test_set['s'].values)

	X_train = np.insert(X_train, 0, train_set['keig_siml'], axis =1)
	X_test = np.insert(X_test, 0, test_set['keig_siml'], axis =1)

	y_train = np.asarray(train_set['bias'].values).reshape(-1,1)
	y_test = np.asarray(test_set['bias'].values).reshape(-1,1)
	
	return X_train, X_test, y_train, y_test, X_all, y_all

X_train, X_test, y_train, y_test, X_all, y_all = generate_training_data(summed_sensitivity_df)