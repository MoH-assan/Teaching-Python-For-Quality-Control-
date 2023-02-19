# Introduction:
# In this code, we will simulate the distribution of sample means by generating random samples from a normal distribution and calculating the mean of each sample. We will then use both analytical and simulation methods to calculate the 95% confidence interval and standard deviation of the sample means and compare the results. 

import numpy as np # NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
from scipy.stats import norm # SciPy is a library for the Python programming language, adding support for optimization, signal processing, statistics, and more. The `norm` module of the `scipy.stats` library is used to work with the normal distribution and calculate critical values.

# Define the population mean and standard deviation
pop_mean = 100
pop_std = 20

# Define the sample size
sample_size = 30

# Define the number of simulations
num_sims = 1000

# Create an empty list to store the sample means
sample_means = []

# Run the simulation
for i in range(num_sims):
    # Draw a random sample from the population
    sample = np.random.normal(pop_mean, pop_std, sample_size) # np.random.normal is a function that generates random numbers from a normal distribution with a specified mean and standard deviation.
    # Calculate the mean of the sample
    sample_mean = np.mean(sample) # np.mean is a function that calculates the mean of a given array or list of numbers.
    # Append the sample mean to the list
    sample_means.append(sample_mean)

# Analytical Method for Confidence Interval
# The Central Limit Theorem states that the distribution of sample means approaches a normal distribution as the sample size increases. Therefore, we can use the normal distribution to approximate the distribution of sample means and calculate the standard error of the mean (sem).
# The standard error of the mean is the standard deviation of the distribution of sample means and is equal to the population standard deviation divided by the square root of the sample size.
sem = pop_std/np.sqrt(sample_size)

# To calculate the 95% confidence interval, we use the critical value from the standard normal distribution (z-value) for a given level of confidence. This z-value corresponds to the number of standard deviations a value is from the mean, in this case we want the value that corresponds to the 97.5th percentile or the 2.5th percentile of the standard normal distribution which are the critical values for a 95% confidence interval
alpha = 0.05
z_critical = norm.ppf(1-alpha/2)

# Then we can use this z-value to calculate the lower and upper bounds of the 95% confidence interval for the population mean.
CI_lower_analytical = pop_mean - z_critical * sem
CI_upper_analytical = pop_mean + z_critical * sem

print("95% Confidence Interval (Analytical Method): (", CI_lower_analytical, ", ", CI_upper_analytical, ")")

# Simulation Method for Confidence Interval
# Sort the sample means
sample_means.sort()

# Find the lower and upper bounds of the 95% confidence interval
CI_lower_simulation = np.percentile(sample_means, alpha/2*100)
CI_upper_simulation = np.percentile(sample_means, (1-alpha/2)*100)

print("95% Confidence Interval (Simulation Method): (", CI_lower_simulation, ", ", CI_upper_simulation, ")")

# Compare the two methods for Confidence Interval
print("Difference between lower bound: ",CI_lower_analytical - CI_lower_simulation)
print("Difference between upper bound: ",CI_upper_analytical - CI_upper_simulation)

# Analytical Method for Standard deviation
# According to the central limit theorem the standard deviation of the sample means is equal to the population standard deviation divided by the square root of the sample size.
analytical_std = pop_std/np.sqrt(sample_size)
print("Standard deviation (Analytical Method): ", analytical_std)

# Simulation Method for Standard deviation
# The standard deviation of the sample means can also be calculated using simulation by finding the standard deviation of the list of sample means.
simulation_std = np.std(sample_means)
print("Standard deviation (Simulation Method): ", simulation_std)

# Compare the two methods for Standard deviation
print("Difference between Analytical and Simulation Method: ", analytical_std - simulation_std)


