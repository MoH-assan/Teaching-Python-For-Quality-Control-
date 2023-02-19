'''
In this code, we are performing a hypothesis test to determine whether the true population mean,
 mu, is equal to a specified value, mu0. The null hypothesis for this test is that mu = mu0,
 and the alternative hypothesis is that mu does not equal mu0.

The analytical method uses the sample mean, x_bar_sample_observed, and the known population standard deviation,
 sigma, to calculate the test statistic, z_bar_sample_observed.
 The test statistic is calculated by subtracting the null hypothesis mean, mu0, from the sample mean,
 x_bar_sample_observed, and dividing by the standard error of the mean, (sigma / sqrt(n)).
 The p-value is calculated by finding the probability that a standard normal random variable is
 greater than or equal to the absolute value of the test statistic,
 using the cumulative distribution function (cdf) from the scipy.stats library.

The simulation method is a way to estimate the p-value of the null hypothesis by generating random samples from the null hypothesis.
Here, we simulate m sets of data of size n, assuming the null hypothesis is true.
In each iteration, we generate random normal data using the numpy function np.random.normal(mu0, sigma, n).
This function returns a sample of random numbers with mean mu0 and standard deviation sigma, of size n.
Then we normalize the test statistic using the population standard deviation by dividing the absolute difference of the mean of the generated random data and mu0 by (sigma/np.sqrt(n)).
Finally, we calculate the empirical p-value by dividing the number of test statistics greater than the absolute difference between the observed sample and mu0 by the number of simulations.
The result of this method will be a number between 0 and 1, representing the probability of observing
a test statistic as extreme or more extreme than the one observed under the assumption that the null hypothesis is true.

It's worth noting that even though the results obtained by the simulation method may slightly vary with each run,
 it should be close to the analytical method if the number of simulations is large enough.

And also, both methods are testing the same hypothesis and should give similar results 
if the assumptions of the test are met.
'''


import numpy as np
from scipy import stats

# Specify test statistic and null value
x_bar_sample_observed = 8.6 # Observed mean
n = 6 # Sample size
mu0 = 9.2 # Null hypothesis mean
sigma = 0.8 # Known population standard deviation

# Analytical method
# Compute the observed sample test statistic
z_bar_sample_observed = (x_bar_sample_observed - mu0) / (sigma / np.sqrt(n))
# Compute the p-value
p_value_analytical = 2 * (1 - stats.norm.cdf(abs(z_bar_sample_observed)))

# Simulation method
# 
# Simulate N sets of data of size n, assuming the null is true, we sample from the Null Hypothesis
m = 200000 # Number of simulations, also total number of samples we draw
z_bar_sim_list = np.zeros(m) # Array to store the test statistics

for j in range(m):
    # Generate random normal data with mean mu0 and standard deviation sigma
    z_bar_sim = np.random.normal(mu0, sigma, n) 
    # Compute the test statistic for this simulated sample
    z_bar_sim_list[j] = abs(np.mean(z_bar_sim) - mu0)/(sigma/np.sqrt(n))

# Calculate empirical p-value
p_value_simulation = sum(z_bar_sim_list > abs(z_bar_sample_observed)) / m

print("p-value (analytical): ", p_value_analytical)
print("p-value (simulation): ", p_value_simulation)