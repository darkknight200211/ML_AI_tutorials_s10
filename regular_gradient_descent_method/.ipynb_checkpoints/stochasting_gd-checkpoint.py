import numpy as np
import matplotlib.pyplot as plt
import math

########### Auxullary functions ######################################
def line_data_generator(
        intercept   :   float,              # the
        slope       :   float,
        x_min       :   float   =   -5.0,
        x_max       :   float   =   5.0,
        num         :   float   =   100,
        noise       :   bool    =   True
):
    """
    use     :   create noisy line data
    params  :   intercept   :   the intercept of the line y = mx + c
                slope       :   the slope of the line y = mx + c
                x_min       :   start of the domain of the line
                x_max       :   end of the domain of the line
                num         :   number of data points
    """
    x_list          =   np.linspace(x_min,x_max,num=num)
    y_list          =   np.zeros(num)
    if (noise==True):
        for index, x_pt in enumerate(x_list):
            y_list[index]   =   slope*x_pt + intercept + np.random.normal(loc=0,scale=1)
    else:
        for index, x_pt in enumerate(x_list):
            y_list[index]   =   slope*x_pt + intercept
    return  x_list, y_list

########### Auxullary functions ######################################

########### eg code to test gradient descent method ##################
print("Eg.1\n")
np.random.seed(40)
# producing the noisy and non-noisy data using the line_data_generator and ploting
intercept   =   5.0
slope       =   2.5
num_data_pt =   100     # number of data points
noisy_x_data, noisy_y_data      =       line_data_generator(intercept=intercept, slope=slope, num=num_data_pt)
nonis_x_data, nonis_y_data      =       line_data_generator(intercept=intercept, slope=slope, noise=False, num=num_data_pt)
figsize =   (5,3)
# data_set        =   np.zeros((num_data_pt,2))
# data_set[:,0]   =   noisy_x_data[:]
# data_set[:,1]   =   noisy_y_data[:]
# plt.figure(1, figsize=figsize)
# plt.scatter(noisy_x_data, noisy_y_data, s=2.5, color="red")
# plt.plot(noisy_x_data, nonis_y_data, lw=2)
# plt.tight_layout()
# plt.show()

# guessing the initial parameters m and c that will go into the model function y_hat=m*x_data + c
beta        =   np.zeros(2)
y_predict   =   np.zeros(num_data_pt)
gradient    =   np.zeros(2)

# initial guess of the parameters
beta[0]     =   0.0
beta[1]     =   0.0
learn_rate  =   0.01
beta_new    =   beta
tol         =   1e-7
# iteration loop for the rgd method
for i in range(1000):
    beta        =   beta_new
    # randomly shuffle the dataset
    rand_int    =   np.random.randint(0,num_data_pt)
    # np.random.shuffle(data_set)
    # predicted value of function at x data points using the model line
    y_predict   =   beta[0] + beta[1]*noisy_x_data[rand_int]
    loss        =   abs(noisy_y_data[rand_int] - y_predict)
    # gradient of the loss function
    gradient[0] =   -2.0 * (noisy_y_data[rand_int]-y_predict)
    gradient[1] =   -2.0 * (noisy_x_data[rand_int]*(noisy_y_data[rand_int]-y_predict))
    beta_new    =   beta - (learn_rate*gradient)
    print(i, np.sqrt(np.sum(np.square(beta_new-beta))))
    if (np.sqrt(np.sum(np.square(beta_new-beta))) < tol):
        break
print(beta, i)
########### eg code to test gradient descent method ##################