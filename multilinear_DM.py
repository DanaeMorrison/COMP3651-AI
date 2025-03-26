import numpy as np
import numpy.linalg as npla
import random
from matplotlib import pyplot as plt

# Function: simple_f
# Parameters:
#   x: Dependent variable
# Returns:
#   4.2x - 6
# Purpose:
#   Function for generating simple test data
def simple_f(x):
    return 4.2 * x - 6.0

# Function: make_data
# Parameters:
#   true_fn:     Function of one floating point value that returns a floating
#                point value
#   domain:      Tuple representing the minimum and maximum values of the
#                independent variable
#   noise_sigma: Standard deviation for Gaussian noise
#   n_pts:       Number of points to generate
# Returns:
#   data_x: Independent variable values
#   data_y: Dependent variable values (what we want to predict)
def make_data(true_fn, domain, noise_sigma, n_pts):
    # Create Numpy matrices
    data_x = np.zeros((n_pts, 1))
    data_y = np.zeros((n_pts, 1))
    # Generate each data point
    for i in range(n_pts):
        data_x[i, 0] = random.uniform(domain[0], domain[1])
        data_y[i, 0] = true_fn(data_x[i, 0]) + random.normalvariate(0.0, noise_sigma)
    return data_x, data_y

def make_example_plot(x_data, y_data, x_min, x_max, x_step, true_fn, learned_fn):
    # Sort data points by X value
    orig_pts = np.hstack([x_data, y_data])
    orig_pts = orig_pts[orig_pts[:,0].argsort(axis=0)]
    x_data = orig_pts[:, 0]
    y_data = orig_pts[:, 1]
    # Generate plot of original points
    plt.scatter(x_data, y_data, marker='x', color='red')
    # Generate dense list of X points
    x_dense = np.arange(x_min, x_max, x_step)
    # Generate plot of true function
    y_true = true_fn(x_dense)
    plt.plot(x_dense, y_true, color='green')
    # Generate plot of estimated function
    y_learned = learned_fn(x_dense)
    plt.plot(x_dense, y_learned, color='blue')
    # Generate legend
    plt.legend(['Red = Original Points', 'Green = True Function', 'Blue = Learned Function'])
    # Display plot
    plt.show()

# Assumes that w has the shape (n_feat, 1); may need to tweak for your code!
def evaluate_polynomial(x_array, w):
    # Get degree of polynomial from the shape of the weight vector
    degree = w.shape[0] - 1
    # Build output as weighted sum
    y_array = np.zeros(x_array.shape)
    for i in range(degree + 1):
        y_array = y_array + w[i, 0] * np.power(x_array, i)
    # Return array of y-values
    return y_array

def make_final_plot(x_data, y_data, x_min, x_max, x_step, true_fn, learned_fn1, learned_fn2):
    # Sort data points by X value
    orig_pts = np.hstack([x_data, y_data])
    orig_pts = orig_pts[orig_pts[:,0].argsort(axis=0)]
    x_data = orig_pts[:, 0]
    y_data = orig_pts[:, 1]
    # Generate plot of original points
    plt.scatter(x_data, y_data, marker='x', color='red')
    # Generate dense list of X points
    x_dense = np.arange(x_min, x_max, x_step)
    # Generate plot of true function
    y_true = true_fn(x_dense)
    plt.plot(x_dense, y_true, color='green')

    # Generate plot of 1st estimated function
    y_learned1 = learned_fn1(x_dense)
    plt.plot(x_dense, y_learned1, color='blue')
    # Generate plot of 2nd estimated function
    y_learned2 = learned_fn2(x_dense)
    plt.plot(x_dense, y_learned2, color='Cyan')
    # Generate legend
    plt.legend(['Red = Original Points', 'Green = True Function', 'Blue = Good Fit Function', 'Cyan = Overfit Function'])
    # Display plot
    plt.show()

# Parameters:
#   complexity: weight of how much complexity is punished
#   degree: degree for the hypothesis space   

def multi_var_reg(true_fn, domain, noise_sigma, n_points, complexity, degree):
    data_x, data_y = make_data(simple_f, domain, noise_sigma, n_points)
    x = np.zeros((n_points, (degree + 1)))
    for i in range(n_points):
        x[i,0] = 1
        for j in range(degree):
            x[i, (j + 1)] = data_x[i, 0]**(j+1)

    trans_mult_reg = np.matmul(np.transpose(x), x)
    complex_matrix = np.identity(trans_mult_reg.shape[0]) * complexity
           
    weight_vec = np.matmul(np.matmul(npla.inv(np.add(trans_mult_reg, complex_matrix)), np.transpose(x)), data_y)
    new_y = np.transpose(np.matmul(np.transpose(weight_vec), np.transpose(x)))
    avg_squared_loss = ((new_y - data_y)**2).sum()/n_points
     
    print("The average squared loss for", n_points, "examples in a domain of ", domain,
          ", noise of", noise_sigma, ", complexity of", complexity, "and degree of", degree, ":")
    print(avg_squared_loss)
    
    plt.title("Graph of True Function, Original Points, and Learned Function")
    
    make_example_plot(data_x,
                  data_y,
                  domain[0],
                  domain[1],
                  (domain[1] - domain[0])/100,
                  lambda x: evaluate_polynomial(x, np.array([[-6.0], [4.2]])),
                  lambda x: evaluate_polynomial(x, weight_vec))



#multi_var_reg(simple_f, (0, 10), 0, 25, 0, 1)

def multi_var_reg_two(true_fn, domain, noise_sigma, n_points, complexity, degree1, degree2):
    data_x, data_y = make_data(simple_f, domain, noise_sigma, n_points)
    
    x1 = np.zeros((n_points, (degree1 + 1)))
    for i in range(n_points):
        x1[i,0] = 1
        for j in range(degree1):
            x1[i, (j + 1)] = data_x[i, 0]**(j+1)

    trans_mult_reg1 = np.matmul(np.transpose(x1), x1)
    complex_matrix1 = np.identity(trans_mult_reg1.shape[0]) * complexity
           
    weight_vec1 = np.matmul(np.matmul(npla.inv(np.add(trans_mult_reg1, complex_matrix1)), np.transpose(x1)), data_y)
    #new_y1 = np.transpose(np.matmul(np.transpose(weight_vec1), np.transpose(x1)))

    x2 = np.zeros((n_points, (degree2 + 1)))
    for i in range(n_points):
        x2[i,0] = 1
        for j in range(degree2):
            x2[i, (j + 1)] = data_x[i, 0]**(j+1)

    trans_mult_reg2 = np.matmul(np.transpose(x2), x2)
    complex_matrix2 = np.identity(trans_mult_reg2.shape[0]) * complexity
           
    weight_vec2 = np.matmul(np.matmul(npla.inv(np.add(trans_mult_reg2, complex_matrix2)), np.transpose(x2)), data_y)
    
    
    plt.title("Graph of True Function, Original Points, Good Fit and Overfit Functions")
    

    make_final_plot(data_x,
                  data_y,
                  domain[0],
                  domain[1],
                  (domain[1] - domain[0])/100,
                  lambda x: evaluate_polynomial(x, np.array([[-6.0], [4.2]])),
                  lambda x: evaluate_polynomial(x, weight_vec1), lambda x: evaluate_polynomial(x, weight_vec2))

multi_var_reg_two(simple_f, (0, 10), 2, 25, 0, 1, 10)
