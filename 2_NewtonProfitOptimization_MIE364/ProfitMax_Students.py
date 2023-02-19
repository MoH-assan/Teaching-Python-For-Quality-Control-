import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.stats import norm
import numpy as np
from math import factorial as fact
def negative_profit_function(x,const):
    # x should be a np.array of size(2,)
    # x[0] is mu and x[1] is lambda
    # const should be a np.array of size(6,)
    #l_1=const[0]
    #l_2=const[1]
    #a_1=const[2]
    #a_2=const[3]
    #c=const[4]
    #c_1=const[5]
    #c_2=const[6]
    """
    This function defines the cost function that we are trying to minimize.
    In this example, the cost function is the negative of the expected profit function
    """
    l_1=const[0]
    l_2=const[1]
    a_1=const[2]
    a_2=const[3]
    c=const[4]
    c_1=const[5]
    c_2=const[6]
    
    
    mu=x[0]
    lmda=x[1]
    
    profit=(a_2+
            (a_1-a_2)*(1-norm.cdf(l_1-mu))*  \
            np.sum([(lmda**j)*np.exp(-lmda)/fact(j) for j in np.arange(0,l_2+1)]))-  \
            (c+c_1*mu+c_2*np.exp(l_2-lmda+lmda/np.exp(1)))    
    
    return -profit



# %%

# Define the gradient and hessian of the cost function
# And also the analytical solution
def gradient_function(x,const):
    l_1=const[0]
    l_2=const[1]
    a_1=const[2]
    a_2=const[3]
    c=const[4]
    c_1=const[5]
    c_2=const[6]
    
    
    mu=x[0]
    lmda=x[1]
    
    profit_mu=###############Fill this part#######################

    profit_lmbda=###############Fill this part#######################
          
    grad_vector=np.array([profit_mu,profit_lmbda])
    
    globals().update(locals())
    return -grad_vector

const=[52.5,3,20,5,4,0.05,0.1]
x=[54,1]
profit=-negative_profit_function(x,const)
gradient=gradient_function(x,const)

print(profit,gradient) #should be 10.964624625164738 [-1.85587443  0.18348437]
# %%

def hessian(x):
    l_1=const[0]
    l_2=const[1]
    a_1=const[2]
    a_2=const[3]
    c=const[4]
    c_1=const[5]
    c_2=const[6]
    
    
    mu=x[0]
    lmda=x[1]
    profit_mu2=###############Fill this part#######################
    profit_lmbda=###############Fill this part#######################
    profit_lmbda_2=###############Fill this part#######################
    
    hess_matrix=np.array([
                           [profit_mu2,profit_lmbda],
                           [profit_lmbda,profit_lmbda_2]
                           ])
    return -hess_matrix 
# %%
const=[52.5,3,20,5,4,0.05,0.1]
x=[54,1]
profit=-negative_profit_function(??,??)
gradient=gradient_function(x,???)
hessian=hessian_function(???,const)
print(profit) #should be 10.964624625164738
print(gradient) #should be 10.964624625164738 [-1.85587443  0.18348437]
print(hessian) #should be[[2.85881164 6.50357612]
                         #[6.50357612 2.14304932]]
analytical_solution = None # There is no analytical solution for this non-convex problem
# %%

# Initialize starting point
x_0 = np.array([58.0,0])

# Set tolerance and maximum number of iterations
step_size=0.2
tolerance = 1e-6
max_iterations = 1000
# Create an empty list to store the solution path
solution_path = []
solution_path_cost = []
solution_path.append(list(x_0))
solution_path_cost.append(-negative_profit_function(x_0,const))
# Newton's method loop
for i in range(max_iterations):
    # Compute gradient and Hessian at current point
    grad = gradient_function(x_0,const)
    hess = hessian(x_0)
    # Compute Newton step
    step = -np.linalg.solve(hess, grad)
    #step = -grad
    # Update current point
    x_0 += step*step_size
    # Add current point to solution path
    solution_path.append(list(x_0))    
    solution_path_cost.append(-negative_profit_function(x_0,const))
    print(solution_path)
    #print("Current solution:", x_0)
    # Check for convergence
    if np.linalg.norm(step) < tolerance:
        break

