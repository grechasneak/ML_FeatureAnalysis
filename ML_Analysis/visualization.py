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

	
def plotError_among_benchType(err_by_type):
    
    x_pos = np.arange(len(err_by_type))
    fig, (ax) = plt.subplots(1, dpi = 160)
    x = ax.get_xaxis()
    x.set_ticks(x_pos)
    x.set_ticklabels([])
    x.set_ticklabels(names,rotation=80)
    ax.bar(x_pos[:-1], err_by_type[:-1])

    ax.set_ylabel('MSE')
    ax.set_title('MSE Among Benchmark Types')
    plt.show()
	
def gen_error_plot(reg_err, regressor, labels):
    
    median = np.median(reg_err)
    mean = np.mean(reg_err)

    case_errors_array = post_processing.generate_type_errors(reg_err)
    
    fig, (ax) = plt.subplots(1, dpi = 160)
    last_point = 0
    for i, case in enumerate(case_errors_array):
        ax.scatter(range(last_point, last_point + len(case)), case, label = labels[i])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        last_point = last_point +len(case)

    xs = np.linspace(1,1100,100)
    median_data = np.array([median for i in range(len(xs))])
    mean_data = np.array([mean for i in range(len(xs))])

    ax.plot(xs, median_data, 'r--') 
    ax.plot(xs, mean_data, 'b-') 

    ax.set_xlabel('Benchmarks')
    ax.set_ylabel('Error (Prediced - Actual Bias)')
    ax.set_title('Errors of '+ regressor +' on Predicting Bias')
    ax.set_ylim([-0.03, 0.03])
    plt.show()