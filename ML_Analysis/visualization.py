'''
This module will contain the models: random forest, adaboost, and convolutional neural network.

'''
from matplotlib.pyplot import mlab
import matplotlib.pyplot as plt


def plot_MSE_hist(error): 
	'''
	Function takes a list of the errors on each benchmark and 
	plots a histogram.
	'''
    fig, (ax2) = plt.subplots(1, dpi = 100)

    ax2.set_title('Distribution of Error')
    ax2.set_xlabel('Prediction Error (Predicted - Actual Bias)')
    ax2.set_ylabel('Relative Frequency')

    n, bins, patches = plt.hist(error, 80, normed=1, color = '#800080')
    mu = np.mean(error)
    sigma = np.std(error)
    ax2.text(-.033, 100, '$\mu = {:.4e}, \sigma = {:.4f}$'.format(mu, sigma))

    plt.show()
