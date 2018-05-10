'''
This module will contain the models: random forest, adaboost, and convolutional neural network.

'''
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

from data_prep import *

# forest_reg = RandomForestRegressor(max_features=200, n_estimators=1000, random_state = 42) 
# ADA = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 35, max_features=300, splitter ='best', n_estimators=1000, learning_rate=1.3)

