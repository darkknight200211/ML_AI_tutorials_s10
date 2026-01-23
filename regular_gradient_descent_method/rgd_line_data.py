import numpy as np
import matplotlib.pyplot as plt

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
                num         :   number of 
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
# producing the noisy and non-noisy data using the line_data_generator and ploting
np.random.seed(40)
intercept   =   5.0
slope       =   2.5
num_data_pt =   100     # number of data points
noisy_x_data, noisy_y_data      =       line_data_generator(intercept=intercept, slope=slope, num=num_data_pt)
nonis_x_data, nonis_y_data      =       line_data_generator(intercept=intercept, slope=slope, noise=False, num=num_data_pt)
figsize =   (5,3)
# plt.figure(1, figsize=figsize)
# plt.scatter(noisy_x_data, noisy_y_data, s=2.5, color="red")
# plt.plot(noisy_x_data, nonis_y_data, lw=2)
# plt.tight_layout()
# plt.show()

# guessing the initial parameters m and c that will go into the model function y_hat=m*x_data + c
beta        =   np.zeros(2)
# initial guess of the parameters
beta[0]     =   0.0
beta[1]     =   0.0
beta_new    =   beta

# construnct the X, and Y matrices and their transposes
x_mat       =   np.zeros((num_data_pt,2))
x_mat[:,0]  =   1.0
x_mat[:,1]  =   noisy_x_data[:]
y_mat       =   np.zeros(num_data_pt)
y_mat[:]    =   noisy_y_data[:]

# doing the iterative gradient descent method for epoch number of iterations
epoch       =   10000
learn_rate  =   0.01
predict_y   =   np.zeros(num_data_pt)
loss_fn     =   []
tol         =   1e-10

for i in range(epoch):
    beta        =   beta_new
    xty         =   x_mat.T @ y_mat
    xtx         =   x_mat.T @ x_mat
    xtxb        =   xtx @ beta
    yty         =   noisy_y_data.T @ noisy_y_data
    gradient    =   2.0*(xtxb - xty)
    beta_new    =   beta - (learn_rate * gradient)
    loss        =   (beta.T @ (xtx @ beta) - 2.0 * (beta @ xty) + yty)/num_data_pt
    print(i, np.sqrt(np.sum(np.square(beta-beta_new))))
    loss_fn.append(loss)
    if (np.sqrt(np.sum(np.square(beta-beta_new))) < tol):
        break

print(beta, beta_new)
# evaluating the values evaluated due to fitted m and c values
predict_y   =   beta[0] + beta[1] * noisy_x_data

plt.figure(2, figsize=figsize)
plt.scatter(noisy_x_data, noisy_y_data, s=3.5, color="green")
plt.plot(noisy_x_data, predict_y, lw=2, label="fitted")
plt.plot(noisy_x_data, nonis_y_data, lw=2, ls="--", label="original")
plt.legend(fontsize=15)

plt.figure(3, figsize=figsize)
plt.plot(np.linspace(0,i,num=i+1), loss_fn, lw=2)

plt.tight_layout()
plt.show()
########### eg code to test gradient descent method ##################