import numpy as np
import matplotlib.pyplot as plt

import numpy as np
def cost_function(x):
    """
    This function defines the cost function that we are trying to minimize.
    In this example, the cost function is x[0]**4 + x[1]**4
    """
    return x[0]**4 + x[1]**4

# Define the gradient and hessian of the cost function
# And also the analytical solution
def gradient(x):
    return np.array([4*x[0]**3, 4*x[1]**3])

def hessian(x):
    return np.array([[12*x[0]**2, 0],[0, 12*x[1]**2]])

analytical_solution = None # There is no analytical solution for this non-convex problem

# Initialize starting point
x_0 = np.array([2.0,2.0])

# Set tolerance and maximum number of iterations
step_size=1
tolerance = 1e-6
max_iterations = 5
# Create an empty list to store the solution path
solution_path = []
solution_path.append(list(x_0))
# Newton's method loop
for i in range(max_iterations):
    # Compute gradient and Hessian at current point
    grad = gradient(x_0)
    hess = hessian(x_0)
    # Compute Newton step
    step = -np.linalg.solve(hess, grad)
    #step = -grad
    # Update current point
    x_0 += step*step_size
    # Add current point to solution path
    solution_path.append(list(x_0))    
    print(solution_path)
    #print("Current solution:", x_0)
    # Check for convergence
    if np.linalg.norm(step) < tolerance:
        break

# Print final solution
print("Final solution:", x_0)

# Check if the final solution is a local minimum or a local maximum
eigenvalues = np.linalg.eigvals(hessian(x_0))
if (eigenvalues > 0).all():
    print("The final solution is a local minimum.")
elif (eigenvalues < 0).all():
    print("The final solution is a local maximum.")
else:
    print("The final solution is neither a local minimum nor a local maximum.")


# Create a grid of points
x = np.linspace(-6, 6, 100)
y = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x, y)
Z = cost_function(np.array([X, Y]))

# Convert solution path to a numpy array
solution_path = np.array(solution_path)

# Plot the cost function and the solution path
fig, ax = plt.subplots(dpi=300)
ax.contour(X, Y, Z, cmap='Reds')
ax.plot(solution_path[:, 0], solution_path[:, 1], 'o-')
ax.set_xlabel('x[0]')
ax.set_ylabel('x[1]')
ax.plot(solution_path[0, 0], solution_path[0, 1], 'k-o')
ax.plot(solution_path[-1, 0], solution_path[-1, 1], 'g-o')
plt.show()


fig = plt.figure(dpi=300)
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z, cmap='coolwarm')
ax.set_xlabel('x[0]')
ax.set_ylabel('x[1]')
ax.set_zlabel('cost_function')
ax.view_init(elev=45, azim=120)
plt.show()


