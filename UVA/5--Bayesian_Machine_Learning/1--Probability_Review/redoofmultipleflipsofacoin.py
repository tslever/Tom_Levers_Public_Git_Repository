message = '''
Redo of \"Example 6 From Basener and Brown 2022: Bayesian Machine Learning: Probabilistic Reasoning and Programming for Machine Learning with Applications in Python\" with number of coin flips N = 1000 and number of heads x = 680 and calculation of 95 percent credible interval

You find a coin and believe that it is either fair (Type A with p = 0.5) or unfair (Type B with p = 0.75) and that out of every 1000 coins one is Type B and the rest are Type A. You flip the coin N = 1000 times and get x = 680 heads and 320 tails. What is the probability that the coin is Type A and the probability that the coin is Type B?

We want to use Bayes's Theorem to compute the probability that the coin is Type A. The data is the number of heads x = 680 in N = 1000 flips of the coin, which we model using a binomial distribution.

The binomial distribution involves N = 1000 independent Bernoulli trials, each of which has a probability p_1 = 0.5 or p_2 = 0.75 of success (heads). The probability that exactly x of the N trials produce success given that the number of Bernoulli trials is equal to N and the probability of success in each trial is equal to p P(x │ N, p) = (N \ choose \ x) p^x (1 - p)^(N - x).

Using the binomial distribution, we find the likelihood P(x = 680 | N = 1000, p = 0.5) that the number of heads x = 680 given that we flipped the coin N = 1000 times and the probability of heads for each flip p = 0.5. We find the likelihood P(x = 680 | N = 1000, p = 0.75) that the number of heads x = 680 given that we flipped the coin N = 1000 times and the probability of heads for each flip p = 0.75.
'''
print(message)

x = 680
N = 1000
p1 = 0.5
p2 = 0.75
from scipy.stats import binom
likelihood_that_x_is_680_given_that_N_is_1000_and_p_is_one_half = binom.pmf(k = x, n = N, p = p1)
likelihood_that_x_is_680_given_that_N_is_1000_and_p_is_three_quarters = binom.pmf(k = x, n = N, p = p2)
print(f'Likelihood that x = 680 given that N = 1000 and p = 0.5 is {likelihood_that_x_is_680_given_that_N_is_1000_and_p_is_one_half}.')
print(f'Likelihood that x = 680 given that N = 1000 and p = 0.75 is {likelihood_that_x_is_680_given_that_N_is_1000_and_p_is_three_quarters}.')

message = '''
We define the prior probability that the probability of heads when flipping this coin is 0.5 as 0.999.
We define the prior probability that the probability of heads when flipping this coin is 0.75 as 0.001.
'''
print(message)
prior_probability_that_p_is_one_half = 0.999
prior_probability_that_p_is_three_quarters = 0.001
print(f'Prior probability that p is 0.5 is {prior_probability_that_p_is_one_half}.')
print(f'Prior probability that p is 0.75 is {prior_probability_that_p_is_three_quarters}.')

message = r'''
By the Law Of Total Probability, the total probability that the number of heads x = 680 given that we flipped the coin N = 1000 times
P(x = 680 | N = 1000) = \sum_{i=1}^2 \left[ P \left( x = 680 | N = 1000 and p = p_i \right) \right]
P(x = 680 | N = 1000) = P \left( x = 680 | N = 1000 \ and \ p = p_1 \right) + P\left( x = 680 | N = 1000 \ and \ p = p_2 \right)
P(x = 680 | N = 1000) = P \left( x = 680 | N = 1000, p = 0.5 \right) P(p = 0.5) + P\left( x = 680 | N = 1000, p = 0.75 \right) P(p = 0.75)
'''
print(message)
total_probability_that_x_is_680_given_that_N_is_1000 = (
    likelihood_that_x_is_680_given_that_N_is_1000_and_p_is_one_half * prior_probability_that_p_is_one_half
    + likelihood_that_x_is_680_given_that_N_is_1000_and_p_is_three_quarters * prior_probability_that_p_is_three_quarters
)
print(f'Total probability that x = 680 given that N = 1000 is {total_probability_that_x_is_680_given_that_N_is_1000}.')

message = '''
By Bayes's Theorem, the posterior probability that the probability of heads when flipping the coin p = 0.5 given that the total number of flips N = 1000 and the number of heads is x = 680 P(p = 0.5 | N = 1000, x = 680) = P(x = 680 | N = 1000, p = 0.5) P(p = 0.5) / p(x = 680 | N = 1000).
The posterior probability that the probability of heads when flipping the coin p = 0.75 given that the total number of flips N = 1000 and the number of heads is x = 680 P(p = 0.75 | N = 1000, x = 680) = P(x = 680 | N = 1000, p = 0.75) P(p = 0.75) / p(x = 680 | N = 1000).
'''
print(message)
posterior_probability_that_p_is_one_half_given_that_N_is_1000_and_x_is_680 = likelihood_that_x_is_680_given_that_N_is_1000_and_p_is_one_half * prior_probability_that_p_is_one_half / total_probability_that_x_is_680_given_that_N_is_1000
posterior_probability_that_p_is_three_quarters_given_that_N_is_1000_and_x_is_680 = likelihood_that_x_is_680_given_that_N_is_1000_and_p_is_three_quarters * prior_probability_that_p_is_three_quarters / total_probability_that_x_is_680_given_that_N_is_1000
print(f'Posterior probability that p = 0.5 given that N = 1000 and x = 680 is {posterior_probability_that_p_is_one_half_given_that_N_is_1000_and_x_is_680}')
print(f'Posterior probability that p = 0.75 given that N = 1000 and x = 680 is {posterior_probability_that_p_is_three_quarters_given_that_N_is_1000_and_x_is_680}')

message = '''
We create distributions of likelihoods, prior probabilities, and posterior probabilities.
'''
print(message)
import numpy as np
# We choose sample values of p for our distribution.
# We select 1001 values of p evenly spaced from 0 to 1.
# We choose 1001 so the step size is exactly 0.001, which is important for numeric integration.
number_of_values_of_p = 1001
array_of_values_of_p = np.linspace(0, 1, number_of_values_of_p)
distance_dp_between_two_adjacent_values_of_p = array_of_values_of_p[1] - array_of_values_of_p[0]
# We compute terms used with Bayes's Theorem.
#from scipy.stats import uniform
from scipy.stats import norm
#array_of_prior_probabilities_that_p_is_value = uniform.pdf(x = array_of_values_of_p, scale = 1 / 0.999)
array_of_prior_probabilities_that_p_is_value = norm.pdf(array_of_values_of_p, loc = 0.5, scale = 0.01)
array_of_likelihoods_that_x_is_680_given_that_N_is_1000_and_p_is_value = binom.pmf(x, N, array_of_values_of_p)
total_probability = np.sum(array_of_prior_probabilities_that_p_is_value * array_of_likelihoods_that_x_is_680_given_that_N_is_1000_and_p_is_value * distance_dp_between_two_adjacent_values_of_p)
# We create an array of posterior probabilities.
array_of_posterior_probabilities_that_p_is_value_given_that_N_is_1000_and_x_is_680 = array_of_likelihoods_that_x_is_680_given_that_N_is_1000_and_p_is_value * array_of_prior_probabilities_that_p_is_value / total_probability
# We confirm that our discrete distribution is a true probability distribution by checking that its total area is equal to 1.
#print("Total area:")
#print(np.sum(array_of_posterior_probabilities_that_p_is_value_given_that_N_is_1000_and_x_is_680 * distance_dp_between_two_adjacent_values_of_p))

# We plot distributions.
import matplotlib.pyplot as plt
# Create the first subplot in a figure with 1 row and 3 columns of plots.
plt.subplot(1, 3, 1)
plt.plot(array_of_values_of_p, array_of_prior_probabilities_that_p_is_value, color = 'blue')
plt.xlabel('$p\'$')
plt.ylabel('Prior Probability $P(p = p\')$')
plt.title('Prior Probability Distribution\n$P(p = p\')$ vs. $p\'$')

plt.subplot(1, 3, 2)
plt.plot(array_of_values_of_p, array_of_likelihoods_that_x_is_680_given_that_N_is_1000_and_p_is_value, color = "orange")
plt.xlabel('$p$')
plt.ylabel('Likelihood $P(p | k = 680, N = 1000)$')
plt.title('Likelihood Distribution\n$P(p | k = 680, N = 1000)$ vs. $p$')

plt.subplot(1, 3, 3)
plt.plot(array_of_values_of_p, array_of_posterior_probabilities_that_p_is_value_given_that_N_is_1000_and_x_is_680, color = "green")
plt.xlabel('$p$')
plt.ylabel('Posterior Probability $P(p = 0.5 | N = 1000, x = 680)$')
plt.title('Posterior Probability Distribution\n$P(p = 0.5 | N = 1000, x = 680)$ vs. $p$')

plt.show()

plt.plot(array_of_values_of_p, array_of_prior_probabilities_that_p_is_value, color = 'blue', label = 'prior')
plt.plot(array_of_values_of_p, array_of_likelihoods_that_x_is_680_given_that_N_is_1000_and_p_is_value * 100, color = "orange", label = "likelihood * 100")
plt.plot(array_of_values_of_p, array_of_posterior_probabilities_that_p_is_value_given_that_N_is_1000_and_x_is_680, color = "green", label = 'posterior')
plt.title('Prior Probability, Likelihood, and Posterior Probability Distributions')
plt.xlabel('$p$')
plt.ylabel('Probability$')
plt.legend()
plt.show()


array_of_CDF_values = np.zeros(len(array_of_posterior_probabilities_that_p_is_value_given_that_N_is_1000_and_x_is_680))
for i in range(0, number_of_values_of_p):
    array_of_CDF_values[i] = np.sum(array_of_posterior_probabilities_that_p_is_value_given_that_N_is_1000_and_x_is_680[0:i]) * distance_dp_between_two_adjacent_values_of_p

'''
We find the x value L where the Cumulative Distribution Function (CDF) is closest to 0.025.
We find the x value U where the Cumulative Distribution Function (CDF) is closest to 0.0975.
We construct a 95 percent credible interval [L, U].
'''
a = array_of_values_of_p[np.argmin(np.abs(array_of_CDF_values - 0.025))]
b = array_of_values_of_p[np.argmin(np.abs(array_of_CDF_values - 0.975))]
print(f'The probability is 0.95 that the true value for p is in the interval [{a}, {b}].')