import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.stats import norm
import numpy as np
from math import factorial as fact
import pdb
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
    
    
    mu=x[0]
    lmda=x[1]
    profit_mu2=(a_1-a_2)*(l_1-mu)*norm.pdf(l_1-mu)*np.sum([(lmda**j)*np.exp(-lmda)/fact(j) for j in np.arange(0,l_2+1)])
    
    profit_lmbda_mu=-(a_1-a_2)*norm.pdf(l_1-mu)*np.exp(-lmda)*(lmda**l_2)/fact(l_2)      
    
    profit_lmbda_2=-(a_1-a_2)*(1-norm.cdf(l_1-mu))*np.exp(-lmda) \
            *((lmda**(l_2-1))/fact(l_2-1)-(lmda**(l_2))/fact(l_2))-  \
            c_2*((np.exp(-1)-1)**2)*np.exp(l_2-lmda+lmda/np.exp(1)) 
    
    hess_matrix=np.array([
                           [profit_mu2,profit_lmbda_mu],
                           [profit_lmbda_mu,profit_lmbda_2]
                           ])
    globals().update(locals())
    return -hess_matrix
# %%
const=[52.5,3,20,5,4,0.05,0.1]
x=[54,1]
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

x_0 = np.array([53.1, 2.0]) #Newton: Converge Intresting, GD: Converge 
#x_0 = np.array([53.5, 2.0]) #Newton: Converge, GD: Converge 
#x_0 = np.array([51.0, 2.0]) #Newton: Diverge, GD: Converge 
'''
x_0 = np.array([58.0, 1.0]) #Newton: Singular Matrix, GD: Converge 
'''

step_size=1 # You migh need to lower this value if using Gradient Descent
tolerance = 1e-6
max_iterations = 20
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
# Create a grid of points 
x = np.linspace(50, 58, 500)
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
solution_path = np.array(solution_path)



# %%
fig = plt.figure(dpi=300)
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z_Profit, cmap='coolwarm',alpha=0.5 )
ax.scatter(solution_path[:, 0], solution_path[:, 1],np.array(solution_path_cost),
           'k-o')
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\lambda$')
plt.title(f'intial in black {[solution_path[0, 0], solution_path[0, 1]]}, and final solution in red, method Newton')

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
ax.plot(solution_path[-1, 0], solution_path[-1, 1], 'r-o')
plt.title(f'intial in black {[solution_path[0, 0], solution_path[0, 1]]}, and final solution in red, method Newton')

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
plt.title(f'intial in black {[solution_path[0, 0], solution_path[0, 1]]}, and final solution in red, method Newton')

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
plt.title(f'intial in black {[solution_path[0, 0], solution_path[0, 1]]}, and final solution in red, method Newton')

cbar.set_label("Profit")
plt.show()



# %%
# Create a grid of points 
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
plt.title(f'intial in black {[solution_path[0, 0], solution_path[0, 1]]}, and final solution in red, method Newton')

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
plt.title(f'intial in black {[solution_path[0, 0], solution_path[0, 1]]}, and final solution in red, method Newton')

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
plt.title(f'intial in black {[solution_path[0, 0], solution_path[0, 1]]}, and final solution in red, method Newton')
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
plt.title(f'intial in black {[solution_path[0, 0], solution_path[0, 1]]}, and final solution in red, method Newton')
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
plt.title(f'intial in black {[solution_path[0, 0], solution_path[0, 1]]}, and final solution in red, method Newton')
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
plt.title(f'intial in black {[solution_path[0, 0], solution_path[0, 1]]}, and final solution in red, method Newton')
plt.show()