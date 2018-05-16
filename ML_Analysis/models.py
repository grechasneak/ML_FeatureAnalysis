'''
This module will contain the models: random forest, adaboost, and convolutional neural network.

'''
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import cross_val_score

forest_reg = RandomForestRegressor(max_features=200, n_estimators=1000, random_state = 42)

#ADA = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 35, max_features=300, 
	#splitter ='best', n_estimators=1000, learning_rate=1.3)
	
def train_model(model, X_train, y_train):
	'''
	Trains a model with X features predicting y.
	'''
	model.fit(X_train, y_train.ravel())
	

def evaluate_model(model, Xt, yt):
	y_pred = model.predict(Xt)
	mse = mean_squared_error(yt, y_pred)
	mae = mean_absolute_error(yt, y_pred)
	rmse = np.sqrt(mse)	   
	print('mse {}, rmse {}'.format(mse, rmse))
	print('Coefficient of Determination R^2:', model.score(Xt, yt))
	print('Mean Absolute Error:', mae)
	
def cross_val(model, X_train, y_train):
    scores = cross_val_score(classifier, X_train, y_train.ravel(),
                         scoring="neg_mean_squared_error", cv=10)
    print("MSE Values:", scores)
    print("Mean MSE:", scores.mean())
    print("Standard deviation of MSE:", scores.std())
    print('Mean RMSE:', np.sqrt(abs(scores.mean())))