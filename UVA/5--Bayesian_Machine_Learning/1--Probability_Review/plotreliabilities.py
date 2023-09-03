'''
Reliability is defined as the probability that a given item will perform its intended function for a given period of time under a given set of conditions.

We simulate a duo core processor. We consider thresholds 1, 2, ..., 100. For each threshold, we perform 100,000 iterations. For each iteration, we consider each core. For each core, we generate a random integer between 1 and 100 inclusive. If the random integer is less than or equal to our threshold, we consider the core to have worked. For each iteration, if at least one core is working, the duo core processor is working. For each threshold, we count the number of iterations for which the processor is working.

We simulate a quad core processor. We consider thresholds 1, 2, ..., 100. For each threshold, we perform 100,000 iterations. For each iteration, we consider each core. For each core, we generate a random integer between 1 and 100 inclusive. If the random integer is less than or equal to our threshold, we consider the core to have worked. For each iteration, if at least two cores are working, the quad core processor is working. For each threshold, we count the number of iterations for which the processor is working.

For thresholds and probabilities of a core working in the range [67, 100], the number of iterations for which the quad core processor is working is greater than or equal to the number of iterations for which the duo core processor is working.
'''

import matplotlib.pyplot as plt
import numpy as np
import random

maximum_threshold = 100
range_of_thresholds = range(1, maximum_threshold + 1)
number_of_thresholds = len(range_of_thresholds)
frequencies_that_duo_core_processor_works = np.zeros(number_of_thresholds)
number_of_iterations = 100000
for i in range_of_thresholds:
    for j in range(0, number_of_iterations):
        first_core_of_duo_core_processor_is_working = False
        second_core_of_duo_core_processor_is_working = False
        random_integer_between_1_and_100_inclusive = random.randint(a = 1, b = 100)
        if random_integer_between_1_and_100_inclusive <= i:
            first_core_of_duo_core_processor_is_working = True
        random_integer_between_1_and_100_inclusive = random.randint(a = 1, b = 100)
        if random_integer_between_1_and_100_inclusive <= i:
            second_core_of_duo_core_processor_is_working = True
        if first_core_of_duo_core_processor_is_working or second_core_of_duo_core_processor_is_working:
            frequencies_that_duo_core_processor_works[i - 1] += 1

frequencies_that_quad_core_processor_works = np.zeros(number_of_thresholds)
number_of_cores_in_quad_core_processor = 4
for i in range_of_thresholds:
    for j in range(0, number_of_iterations):
        statuses_of_cores_in_quad_core_processor = np.zeros(number_of_cores_in_quad_core_processor)
        for k in range(0, number_of_cores_in_quad_core_processor):
            random_integer_between_1_and_100_inclusive = random.randint(a = 1, b = 100)
            if random_integer_between_1_and_100_inclusive <= i:
                statuses_of_cores_in_quad_core_processor[k] = 1
        if sum(statuses_of_cores_in_quad_core_processor) >= 2:
            frequencies_that_quad_core_processor_works[i - 1] += 1

print(frequencies_that_duo_core_processor_works)
print(frequencies_that_quad_core_processor_works)

from matplotlib import ticker
fig, ax = plt.subplots()
plt.title("Frequencies That Duo Core And Quad Core Processors Work")
plt.xlabel("Threshold")
plt.ylabel("Frequency That Processor Works")
plt.legend(["duo", "quad"])
ax.plot(range_of_thresholds, frequencies_that_duo_core_processor_works)
ax.plot(range_of_thresholds, frequencies_that_quad_core_processor_works)
ax.set_xlim(0, maximum_threshold)
ax.set_ylim(0, number_of_iterations)
ax.grid(which = "major")
ax.grid(which = "minor", alpha = 0.2)
ax.xaxis.set_major_locator(ticker.LinearLocator(11))
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
ax.yaxis.set_major_locator(ticker.LinearLocator(11))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))
plt.show()