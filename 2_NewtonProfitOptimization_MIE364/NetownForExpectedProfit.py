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


def revenue_function(x,const):
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
    
    revenuce=(a_2+
            (a_1-a_2)*(1-norm.cdf(l_1-mu))*  \
            np.sum([(lmda**j)*np.exp(-lmda)/fact(j) for j in np.arange(0,l_2+1)]))
    
    return revenuce

def cost_function(x,const):
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
    cost=(c+c_1*mu+c_2*np.exp(l_2-lmda+lmda/np.exp(1)))
    
    return cost
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
    profit_mu=((a_1-a_2)*norm.pdf(l_1-mu)* 
            np.sum([(lmda**j)*np.exp(-lmda)/fact(j) for j in np.arange(0,l_2+1)])-   #This is the same term from the cost function
            c_1)
    profit_lmbda=-(a_1-a_2)*(1-norm.cdf(l_1-mu))*  \
            (lmda**l_2)*np.exp(-lmda)/fact(l_2) -  \
            (c_2*(np.exp(-1)-1)*np.exp(l_2-lmda+lmda/np.exp(1)))            
    grad_vector=np.array([profit_mu,profit_lmbda])
    
    globals().update(locals())
    return -grad_vector

const=[52.5,3,20,5,4,0.05,0.1]
x=[54,1]
cost=negative_profit_function(x,const)
gradient=gradient_function(x,const)
# %%
'''
def hessian(x):
    hess_matrix=###############Fill this part#######################
    return hess_matrix

analytical_solution = None # There is no analytical solution for this non-convex problem
'''
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
    #hess = hessian(x_0)
    # Compute Newton step
    #step = -np.linalg.solve(hess, grad)
    step = -grad
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

# Print final solution
print("Final solution:", x_0)
'''
# Check if the final solution is a local minimum or a local maximum
eigenvalues = np.linalg.eigvals(hessian(x_0))
if (eigenvalues > 0).all():
    print("The final solution is a local minimum.")
elif (eigenvalues < 0).all():
    print("The final solution is a local maximum.")
else:
    print("The final solution is neither a local minimum nor a local maximum.")
'''
# %%
# Create a grid of points 
x = np.linspace(52, 58, 500)
y = np.linspace(0, 3, 100)
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
ax.plot(solution_path[-1, 0], solution_path[-1, 1], 'g-o')

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
ax.plot(solution_path[-1, 0], solution_path[-1, 1], 'g-o')

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
ax.plot(solution_path[-1, 0], solution_path[-1, 1], 'g-o')
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
ax.plot(solution_path[-1, 0], solution_path[-1, 1], 'g-o')

cbar.set_label("Profit")
plt.show()



# %%
# Create a grid of points 
x = np.linspace(54, 56, 500)
y = np.linspace(1, 2, 100)
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
ax.plot(solution_path[-1, 0], solution_path[-1, 1], 'g-o')

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
ax.plot(solution_path[-1, 0], solution_path[-1, 1], 'g-o')

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
ax.plot(solution_path[-1, 0], solution_path[-1, 1], 'g-o')
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
ax.plot(solution_path[-1, 0], solution_path[-1, 1], 'g-o')

cbar.set_label("Profit")
plt.show()