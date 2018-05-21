'''
This module will contain the models: random forest, adaboost, and convolutional neural network.

'''
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_val_predict


forest_reg = RandomForestRegressor(max_features=200, n_estimators=1000, random_state = 42)

ADA = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 35, max_features=300))
	
def train_model(model, X_train, y_train):
	'''
	Trains a model with X features predicting y.
	'''
	model.fit(X_train, y_train.ravel())
	return model
	

def evaluate_model(model, X_test, y_test):
	'''Generates statistics using the train and test data, much faster than cross validation, but not as accurate.
	'''
	y_pred = model.predict(X_test)
	mse = mean_squared_error(y_test, y_pred)
	mae = mean_absolute_error(y_test, y_pred)
	rmse = np.sqrt(mse)	   
	print('mse {}, rmse {}'.format(mse, rmse))
	print('Coefficient of Determination R^2:', model.score(X_test, y_test))
	print('Mean Absolute Error:', mae)
	
def cross_val(regressor, X_train, y_train):
	''' Generates ten fold cross validation MSE and RMSE values
	Parameters
	------------
	regressor : any scikit-learn regressor
		Regressor object.

	X_train, y_train : ndarrays
		Arrays generated by the generate_training_data function

	Returns
	-------
	Mean squared error, standard deviation of MSE, and the root mean squared error
	'''
	scores = cross_val_score(classifier, X_train, y_train.ravel(),
						 scoring="neg_mean_squared_error", cv=10)
	print("Mean MSE:", abs(scores.mean()))
	print("Standard deviation of MSE:", scores.std())
	print('Mean RMSE:', np.sqrt(abs(scores.mean())))
	


def generate_predictions(regressor, X_all, y_all):
	'''
	Function uses all of the data to make predictions using cross validation
	'''
	y_predicted = cross_val_predict(regressor, X_all, y_all.ravel(), cv=10)
	y_predicted = y_predicted.reshape(-1,1)
	return y_predicted