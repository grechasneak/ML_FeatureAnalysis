'''
This module will contain the code that analyzes the outputs and returns the important features in the problem.

'''


def generate_sorted_error(regressor_error, index):
    rel_errors = tuple(zip(index, regressor_error))
    rel_errors_sorted = sorted(rel_errors, reverse = True, key = lambda v:v[1])  
    return rel_errors_sorted