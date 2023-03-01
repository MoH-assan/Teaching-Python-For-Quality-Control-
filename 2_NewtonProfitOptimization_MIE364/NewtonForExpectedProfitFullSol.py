import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.stats import norm
import numpy as np
from math import factorial as fact
import pdb

def revenue_function(x,const):
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
    sigma=const[7]
    
    
    mu=x[0]
    lmda=x[1]
    
    revenuce=(a_2+
            (a_1-a_2)*(1-norm.cdf((l_1-mu)/sigma))*  \
            np.sum([(lmda**j)*np.exp(-lmda)/fact(j) for j in np.arange(0,l_2+1)]))
    
    return revenuce

def cost_function(x,const):

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
    sigma=const[7]
    
    
    mu=x[0]
    lmda=x[1]
    cost=(c+c_1*mu+c_2*np.exp(l_2-lmda+lmda/np.exp(1)))
    
    return cost
# %%
def negative_profit_function(x,const):

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
    sigma=const[7]
    
    
    mu=x[0]
    lmda=x[1]
    profit= revenue_function(x,const)- cost_function(x,const)    
    
    return -profit
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
    sigma=const[7]
    
    
    mu=x[0]
    lmda=x[1]
    profit_mu=(((a_1-a_2)/sigma)*norm.pdf((l_1-mu)/sigma)* 
            np.sum([(lmda**j)*np.exp(-lmda)/fact(j) for j in np.arange(0,l_2+1)])-   #This is the same term from the cost function
            c_1)
    profit_lmbda=-(a_1-a_2)*(1-norm.cdf((l_1-mu)/sigma))*  \
            (lmda**l_2)*np.exp(-lmda)/fact(l_2) -  \
            (c_2*(np.exp(-1)-1)*np.exp(l_2-lmda+lmda*np.exp(-1)))            
    grad_vector=np.array([profit_mu,profit_lmbda])
    
    
    return -grad_vector


# %%

def hessian_function(x,const):
    l_1=const[0]
    l_2=const[1]
    a_1=const[2]
    a_2=const[3]
    c=const[4]
    c_1=const[5]
    c_2=const[6]
    sigma=const[7]
    
    
    mu=x[0]
    lmda=x[1]
    profit_mu2=(a_1-a_2)*((l_1-mu)/(sigma**3))*norm.pdf((l_1-mu)/sigma)*np.sum([(lmda**j)*np.exp(-lmda)/fact(j) for j in np.arange(0,l_2+1)])
    #profit_mu2=(a_1-a_2)*((l_1-mu)/(sigma**1))*norm.pdf((l_1-mu)/sigma)*np.sum([(lmda**j)*np.exp(-lmda)/fact(j) for j in np.arange(0,l_2+1)])
    
    profit_lmbda_mu=-(a_1-a_2)*(sigma**-1)*norm.pdf((l_1-mu)/sigma)*np.exp(-lmda)*(lmda**l_2)/fact(l_2)      
    
    profit_lmbda_2=-(a_1-a_2)*(1-norm.cdf((l_1-mu)/sigma))*np.exp(-lmda) \
            *((lmda**(l_2-1))/fact(l_2-1)-(lmda**(l_2))/fact(l_2))-  \
            c_2*((np.exp(-1)-1)**2)*np.exp(l_2-lmda+lmda/np.exp(1)) 
    
    hess_matrix=np.array([
                           [profit_mu2,profit_lmbda_mu],
                           [profit_lmbda_mu,profit_lmbda_2]
                           ])
    globals().update(locals())
    return -hess_matrix
# %%
const=[52.5,3,20,5,4,0.05,0.1,2] #sigma is last value in this list
x=[54.74185012,0.89004805]
profit=-negative_profit_function(x,const)
gradient=gradient_function(x,const)
hessian=hessian_function(x,const)
print(profit) #should be 10.964624625164738
print(gradient) #should be 10.964624625164738 [-1.85587443  0.18348437]
print(hessian) #should be[[2.85881164 6.50357612]
                         #[6.50357612 2.14304932]]
 
# %%
# Initialize starting point
# %%
# Initialize starting point
print('Solution is starting')

#x_0 = np.array([55.58927173, 0.89015551]) #Opt

#x_0 = np.array([58.1,0.1]) #Newton: Singular Matrix

#x_0 = np.array([55.0, 0.5]) #Newton: Converge, GD: Converge 

#x_0 = np.array([52.0, 2.0]) #Newton: Diverge, GD: Converge 

#x_0 = np.array([53.1, 2.0]) #Newton: Converge Intresting, GD: Converge 
x_0 = np.array([53.5, 2.0]) #Newton: Converge, GD: Converge 
#x_0 = np.array([51.0, 2.0]) #Newton: Diverge, GD: Converge 
'''
x_0 = np.array([58.0, 1.0]) #Newton: Singular Matrix, GD: Converge 
'''
x_0 = np.array([54.74185012, 0.89004805]) 
step_size=1 # You migh need to lower this value if using Gradient Descent
tolerance = 1e-6
max_iterations = 100
# Create an empty list to store the solution path
solution_path = []
solution_path_cost = []
solution_path.append(list(x_0))
solution_path_cost.append(-negative_profit_function(x_0,const))
# Newton's method loop
for i in range(max_iterations):
    
    # Compute gradient and Hessian at current point
    grad = gradient_function(x_0,const)
    hess = hessian_function(x_0,const)
    # Compute Newton step
    step = -np.linalg.solve(hess, grad)
    '''
    calculates the step direction using the Hessian matrix and the gradient of the objective function. Specifically, it solves the linear system H_k * step = -g_k for the step direction step, where H_k is the Hessian matrix at the current point and g_k is the gradient of the objective function at the current point. The negative sign in front of the np.linalg.solve function indicates that we want to move in the direction of steepest descent.

    Note that this code assumes that the Hessian matrix is invertible. If the Hessian matrix is not invertible, then np.linalg.solve will raise an error. In such cases, a modified version of the algorithm, such as a quasi-Newton method, may be needed to compute the step direction.
    
    If the Hessian matrix is not invertible, it can tell us several things about the objective function we are trying to optimize. Here are a few possible interpretations:

    The objective function has a flat region: If the Hessian matrix has a zero eigenvalue at a point, it indicates that the objective function has a flat region around that point. In other words, the function does not change much in any direction at that point, so any direction could be a valid direction of descent. This can make it difficult for an optimization algorithm to find a valid step direction and may require using a quasi-Newton or other regularization method.

    The objective function has a saddle point: If the Hessian matrix has both positive and negative eigenvalues at a point, it indicates that the objective function has a saddle point at that point. A saddle point is a point where the function has a critical point (gradient equals zero), but it is not a local minimum or maximum. This can make it difficult for an optimization algorithm to find a valid direction of descent.

    The objective function is not twice differentiable: If the Hessian matrix is not defined or not continuous at a point, it indicates that the objective function is not twice differentiable at that point. This means that the function may have discontinuities or sharp corners that prevent the second-order partial derivatives from being defined or continuous. In such cases, optimization algorithms that rely on the Hessian matrix may not be applicable or may require modifications to handle these irregularities.

    These are just a few possible interpretations, and the specific meaning of a non-invertible Hessian matrix can depend on the specific optimization problem and the characteristics of the objective function.

    '''
    #step =-np.linalg.pinv(hess)@grad
    # Compute Gradient Descent step
    #step = -grad; step_size=step_size*0.9

    # Update current point
    x_0 += step*step_size
    profit=-negative_profit_function(x_0,const)
    print(f'Current Paramaters are{x_0}')
    print(f'Current profit is {profit}')
    
    # Add current point to solution path
    solution_path.append(list(x_0))    
    solution_path_cost.append(profit)
    #print(solution_path)
    
    #print("Current solution:", x_0)
    # Check for convergence
    if np.linalg.norm(step) < tolerance:
        break

# Print final solution
print("Final solution:", x_0)
print(f"sigma:", const[-1])
# Check if the final solution is a local minimum or a local maximum
# If you don't define the hessian function the following check can not be done 
# and the code will give you an error 
eigenvalues = np.linalg.eigvals(hessian_function(x_0,const))
if (eigenvalues > 0).all():
    print("The final solution is a local minimum.")
elif (eigenvalues < 0).all():
    print("The final solution is a local maximum.")
else:
    print("The final solution is neither a local minimum nor a local maximum.")

# %%
solution_path = np.array(solution_path)
# Create a grid of points 
x = np.linspace(50, 60, 500)
y = np.linspace(-2, 7, 100)
X, Y = np.meshgrid(x, y)
Z_Profit=X*0
Z_rev=X*0
Z_cost=X*0

for counter_x in np.arange(X.shape[0]):
   for counter_y in np.arange(X.shape[1]): 
       Z_Profit[counter_x,counter_y] = -negative_profit_function(np.array([X[counter_x,counter_y], Y[counter_x,counter_y]]),const)
       Z_rev[counter_x,counter_y] = revenue_function(np.array([X[counter_x,counter_y], Y[counter_x,counter_y]]),const)
       Z_cost[counter_x,counter_y] = cost_function(np.array([X[counter_x,counter_y], Y[counter_x,counter_y]]),const)

# %%
# Convert solution path to a numpy array



fig, ax = plt.subplots(dpi=300)
cbar = fig.colorbar(ax.contourf(X, Y, Z_Profit))
ax.plot(solution_path[:, 0], solution_path[:, 1], 'o-')
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\lambda$')
ax.plot(solution_path[0, 0], solution_path[0, 1], 'k-o')
ax.plot(solution_path[-1, 0], solution_path[-1, 1], 'r-o')
plt.title(f'intial in black ({solution_path[0,0]}, {solution_path[0,1]}),\n and final solution in red, method Newton\n sigma={const[7]}')

cbar.set_label("Profit")
plt.show()

exit(1)
# %%





# %%
fig = plt.figure(dpi=300)
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z_Profit, cmap='coolwarm',alpha=0.5 )
ax.scatter(solution_path[:, 0], solution_path[:, 1],np.array(solution_path_cost),
           'k-o')
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\lambda$')
plt.title(f'intial in black ({solution_path[0,0]}, {solution_path[0,1]}),\n and final solution in red, method Newton\n sigma={const[7]}'
)

ax.set_zlabel('profit_function')
ax.view_init(elev=45, azim=120)
plt.show()

# %%
# Create a grid of points

fig = plt.figure(dpi=300)
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z_rev, cmap='coolwarm')
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\lambda$')
ax.set_zlabel('revnue_function')
ax.view_init(elev=45, azim=120)
plt.show()

# %%
# Create a grid of points



fig = plt.figure(dpi=300)
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z_cost, cmap='coolwarm')
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\lambda$')
ax.set_zlabel('cost_function')
ax.view_init(elev=45, azim=120)
plt.show()


# %%


# %%
Z = cost_function(np.array([X, Y]),const)
#Z = revenue_function(np.array([X, Y]),const)
#Z = -negative_profit_function(np.array([X, Y]),const)
# Plot the profit function and the solution path
fig, ax = plt.subplots(dpi=300)
cbar = fig.colorbar(ax.contourf(X, Y, Z_cost))
ax.plot(solution_path[:, 0], solution_path[:, 1], 'o-')
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\lambda$')
ax.plot(solution_path[0, 0], solution_path[0, 1], 'k-o')
ax.plot(solution_path[-1, 0], solution_path[-1, 1], 'r-o')
plt.title(f'intial in black ({solution_path[0,0]}, {solution_path[0,1]}),\n and final solution in red, method Newton\n sigma={const[7]}')

cbar.set_label("Cost")
plt.show()

# %%
#Z = cost_function(np.array([X, Y]),const)

#Z = -negative_profit_function(np.array([X, Y]),const)
# Plot the profit function and the solution path
fig, ax = plt.subplots(dpi=300)
cbar = fig.colorbar(ax.contourf(X, Y, Z_rev))
ax.plot(solution_path[:, 0], solution_path[:, 1], 'o-')
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\lambda$')
ax.plot(solution_path[0, 0], solution_path[0, 1], 'k-o')
ax.plot(solution_path[-1, 0], solution_path[-1, 1], 'r-o')
cbar.set_label("Revenue")
plt.show()


# %%
# %%
fig, ax = plt.subplots(dpi=300)
cbar = fig.colorbar(ax.contourf(X, Y, Z_Profit))
ax.plot(solution_path[:, 0], solution_path[:, 1], 'o-')
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\lambda$')
ax.plot(solution_path[0, 0], solution_path[0, 1], 'k-o')
ax.plot(solution_path[-1, 0], solution_path[-1, 1], 'r-o')
plt.title(f'intial in black ({solution_path[0,0]}, {solution_path[0,1]}),\n and final solution in red, method Newton\n sigma={const[7]}')

cbar.set_label("Profit")
plt.show()




# %%
# Create a grid of points 
exit(0)
x = np.linspace(55, 57, 500)
y = np.linspace(0.75, 1.25, 100)
X, Y = np.meshgrid(x, y)
Z_Profit=X*0
Z_rev=X*0
Z_cost=X*0

for counter_x in np.arange(X.shape[0]):
   for counter_y in np.arange(X.shape[1]): 
       Z_Profit[counter_x,counter_y] = -negative_profit_function(np.array([X[counter_x,counter_y], Y[counter_x,counter_y]]),const)
       Z_rev[counter_x,counter_y] = revenue_function(np.array([X[counter_x,counter_y], Y[counter_x,counter_y]]),const)
       Z_cost[counter_x,counter_y] = cost_function(np.array([X[counter_x,counter_y], Y[counter_x,counter_y]]),const)

# %%
# Convert solution path to a numpy array
solution_path = np.array(solution_path)

# %%
fig = plt.figure(dpi=300)
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z_Profit, cmap='coolwarm',alpha=0.5 )
ax.scatter(solution_path[:, 0], solution_path[:, 1],np.array(solution_path_cost),
           'k-o')
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\lambda$')
ax.set_zlabel('profit_function')
plt.title(f'intial in black ({solution_path[0,0]}, {solution_path[0,1]}),\n and final solution in red, method Newton\n sigma={const[7]}')

ax.view_init(elev=45, azim=120)
plt.show()

# %%
# Create a grid of points

fig = plt.figure(dpi=300)
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z_rev, cmap='coolwarm')
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\lambda$')
ax.set_zlabel('revnue_function')
ax.view_init(elev=45, azim=120)
plt.show()

# %%
# Create a grid of points



fig = plt.figure(dpi=300)
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z_cost, cmap='coolwarm')
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\lambda$')
ax.set_zlabel('cost_function')
ax.view_init(elev=45, azim=120)
plt.show()


# %%

# %%
fig, ax = plt.subplots(dpi=300)
cbar = fig.colorbar(ax.contourf(X, Y, Z_Profit))
ax.plot(solution_path[:, 0], solution_path[:, 1], 'o-')
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\lambda$')
ax.plot(solution_path[0, 0], solution_path[0, 1], 'k-o')
ax.plot(solution_path[-1, 0], solution_path[-1, 1], 'r-o')
plt.title(f'intial in black ({solution_path[0,0]}, {solution_path[0,1]}),\n and final solution in red, method Newton\n sigma={const[7]}')

cbar.set_label("Profit")
plt.show()
# %%
Z = cost_function(np.array([X, Y]),const)
#Z = revenue_function(np.array([X, Y]),const)
#Z = -negative_profit_function(np.array([X, Y]),const)
# Plot the profit function and the solution path
fig, ax = plt.subplots(dpi=300)
cbar = fig.colorbar(ax.contourf(X, Y, Z_cost))
ax.plot(solution_path[:, 0], solution_path[:, 1], 'o-')
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\lambda$')
ax.plot(solution_path[0, 0], solution_path[0, 1], 'k-o')
ax.plot(solution_path[-1, 0], solution_path[-1, 1], 'r-o')

cbar.set_label("Cost")
plt.show()

# %%
#Z = cost_function(np.array([X, Y]),const)

#Z = -negative_profit_function(np.array([X, Y]),const)
# Plot the profit function and the solution path
fig, ax = plt.subplots(dpi=300)
cbar = fig.colorbar(ax.contourf(X, Y, Z_rev))
ax.plot(solution_path[:, 0], solution_path[:, 1], 'o-')
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\lambda$')
ax.plot(solution_path[0, 0], solution_path[0, 1], 'k-o')
ax.plot(solution_path[-1, 0], solution_path[-1, 1], 'r-o')
cbar.set_label("Revenue")
plt.show()


# %%
# %%
fig, ax = plt.subplots(dpi=300)
cbar = fig.colorbar(ax.contourf(X, Y, Z_Profit))
ax.plot(solution_path[:, 0], solution_path[:, 1], 'o-')
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\lambda$')
ax.plot(solution_path[0, 0], solution_path[0, 1], 'k-o')
ax.plot(solution_path[-1, 0], solution_path[-1, 1], 'r-o')

cbar.set_label("Profit")
plt.title(f'intial in black ({solution_path[0,0]}, {solution_path[0,1]}),\n and final solution in red, method Newton\n sigma={const[7]}')
plt.show()



# %%
# Create a grid of points
''' 
x = np.linspace(np.min([solution_path[:, 0],solution_path[:, 0]])-5,
                np.max([solution_path[:, 0],solution_path[:, 0]])+5, 500)
y = np.linspace(np.min([solution_path[:, 1],solution_path[:, 1]]),
                np.max([solution_path[:, 1],solution_path[:, 1]])+1, 500)
'''
xrange=[51.8,52.2]
yrange=[1.9,2.1]
x = np.linspace(xrange[0],xrange[1], 500)
y = np.linspace(yrange[0],yrange[1], 500)

X, Y = np.meshgrid(x, y)
Z_Profit=X*0
Z_rev=X*0
Z_cost=X*0

for counter_x in np.arange(X.shape[0]):
   for counter_y in np.arange(X.shape[1]): 
       Z_Profit[counter_x,counter_y] = -negative_profit_function(np.array([X[counter_x,counter_y], Y[counter_x,counter_y]]),const)
       Z_rev[counter_x,counter_y] = revenue_function(np.array([X[counter_x,counter_y], Y[counter_x,counter_y]]),const)
       Z_cost[counter_x,counter_y] = cost_function(np.array([X[counter_x,counter_y], Y[counter_x,counter_y]]),const)

# %%
# Convert solution path to a numpy array
solution_path = np.array(solution_path)

# %%
fig = plt.figure(dpi=300)
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z_Profit, cmap='coolwarm',alpha=0.5 )
ax.scatter(solution_path[:, 0], solution_path[:, 1],np.array(solution_path_cost),
           'k-o')
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\lambda$')
ax.set_zlabel('profit_function')
plt.title(f'intial in black ({solution_path[0,0]}, {solution_path[0,1]}),\n and final solution in red, method Newton\n sigma={const[7]}')
ax.set_xlim(xrange)
ax.set_ylim(yrange)

ax.view_init(elev=45, azim=120)
plt.show()

# %%
# Create a grid of points

fig = plt.figure(dpi=300)
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z_rev, cmap='coolwarm')
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\lambda$')
ax.set_zlabel('revnue_function')
ax.view_init(elev=45, azim=120)
ax.set_xlim(xrange)
ax.set_ylim(yrange)
plt.show()

# %%
# Create a grid of points



fig = plt.figure(dpi=300)
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z_cost, cmap='coolwarm')
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\lambda$')
ax.set_zlabel('cost_function')
ax.view_init(elev=45, azim=120)
ax.set_xlim(xrange)
ax.set_ylim(yrange)
plt.show()


# %%

# %%
fig, ax = plt.subplots(dpi=300)
cbar = fig.colorbar(ax.contourf(X, Y, Z_Profit))
ax.plot(solution_path[:, 0], solution_path[:, 1], 'o-')
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\lambda$')
ax.plot(solution_path[0, 0], solution_path[0, 1], 'k-o')
ax.plot(solution_path[-1, 0], solution_path[-1, 1], 'r-o')
plt.title(f'intial in black ({solution_path[0,0]}, {solution_path[0,1]}),\n and final solution in red, method Newton\n sigma={const[7]}')
ax.set_xlim(xrange)
ax.set_ylim(yrange)
cbar.set_label("Profit")
plt.show()
# %%
Z = cost_function(np.array([X, Y]),const)
#Z = revenue_function(np.array([X, Y]),const)
#Z = -negative_profit_function(np.array([X, Y]),const)
# Plot the profit function and the solution path
fig, ax = plt.subplots(dpi=300)
cbar = fig.colorbar(ax.contourf(X, Y, Z_cost))
ax.plot(solution_path[:, 0], solution_path[:, 1], 'o-')
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\lambda$')
ax.plot(solution_path[0, 0], solution_path[0, 1], 'k-o')
ax.plot(solution_path[-1, 0], solution_path[-1, 1], 'r-o')

cbar.set_label("Cost")
plt.show()

# %%
#Z = cost_function(np.array([X, Y]),const)

#Z = -negative_profit_function(np.array([X, Y]),const)
# Plot the profit function and the solution path
fig, ax = plt.subplots(dpi=300)
cbar = fig.colorbar(ax.contourf(X, Y, Z_rev))
ax.plot(solution_path[:, 0], solution_path[:, 1], 'o-')
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\lambda$')
ax.plot(solution_path[0, 0], solution_path[0, 1], 'k-o')
ax.plot(solution_path[-1, 0], solution_path[-1, 1], 'r-o')
cbar.set_label("Revenue")
plt.show()


# %%
# %%
fig, ax = plt.subplots(dpi=300)
cbar = fig.colorbar(ax.contourf(X, Y, Z_Profit))
ax.plot(solution_path[:, 0], solution_path[:, 1], 'o-')
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\lambda$')
ax.plot(solution_path[0, 0], solution_path[0, 1], 'k-o')
ax.plot(solution_path[-1, 0], solution_path[-1, 1], 'r-o')

cbar.set_label("Profit")
plt.title(f'intial in black ({solution_path[0,0]}, {solution_path[0,1]}),\n and final solution in red, method Newton\n sigma={const[7]}')
plt.show()

# %%