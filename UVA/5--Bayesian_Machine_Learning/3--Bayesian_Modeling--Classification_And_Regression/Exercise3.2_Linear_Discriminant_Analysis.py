# This is code for Linear Discriminant Analysis
# Written by William F Basener
# University of Virginia, School of Data Science
# For use in teaching Bayesian Machine Learning
#
# The code currently computes the maximum likelihood classification
# Student is to add method to compute posterior probabilities and maximum probability classification

import pandas as pd
import numpy as np


def multivariate_gaussian_pdf(X, MU, SIGMA):
    """Code from Data Blog https://xavierbourretsicotte.github.io/MLE_Multivariate_Gaussian.html
    Maximum Likelihood Estimator: Multivariate Gaussian Distribution
        by Xavier Bourret Sicotte, Fri 22 June 2018
    Returns the pdf of a multivariate Gaussian distribution
     - X, MU are p x 1 vectors
     - SIGMA is a p x p matrix"""
    # Initialize and reshape
    X = X.reshape(-1, 1)
    MU = MU.reshape(-1, 1)
    p, _ = SIGMA.shape

    # Compute values
    SIGMA_inv = np.linalg.inv(SIGMA)
    denominator = np.sqrt((2 * np.pi) ** p * np.linalg.det(SIGMA))
    exponent = -(1 / 2) * ((X - MU).T @ SIGMA_inv @ (X - MU))

    # Return result
    return float((1. / denominator) * np.exp(exponent))


class LDA:
    """Creates a class for Linear Discriminant Analysis
    Input:
        fname = file name for a csv file, must have one column labeled "class" and the rest numeric data
    Methods:
        compute_probabilities = given an input observation computes the likelihood for each class and the GML class
        compute_probabilities: given an input observation and prior probabilities,
            computes the posterior probabilities for each class and most probable class"""

    def __init__(self, fname):
        # reads the data and computes the statistics needed for classification

        # read the iris data as a Pandas data frame
        df = pd.read_csv(fname)

        # separate the class labels from the rest of the data
        # we are assuming the column name with class labels is 'Class'
        # and all other columns are numeric
        self.data_labels = df.loc[:]['Class']
        self.data = np.asarray(df.drop('Class', axis=1, inplace=False))

        # get information about the dimensions the data
        self.num_rows, self.num_cols = self.data.shape

        # get the class names as an array of strings
        self.class_names = np.unique(self.data_labels)

        # determine number of observations in each class
        self.num_obs = dict()
        for name in self.class_names:
            self.num_obs[name] = sum(self.data_labels == name)

        # compute the mean of each class
        self.means = dict()
        for name in self.class_names:
            self.means[name] = np.mean(self.data[self.data_labels == name, :], 0)

        # compute the mean covariance matrix
        self.cov = np.zeros([self.num_cols, self.num_cols])
        for name in self.class_names:
            self.cov = self.cov + self.num_obs[name] * np.cov(np.transpose(self.data[self.data_labels == name, :]))
        self.cov = self.cov / self.num_rows

    def compute_likelihoods(self, x):
        # compute and output the likelihood of each class and the maximum likelihood class

        # check that the input data x has the correct number of rows
        if not (len(x) == self.num_cols):
            print('Data vector has wrong number of values.')
            return -1

        # reformat x as a numpy array, incase the user input a list
        x = np.asarray(x)

        # compute the likelihood of each class
        likelihoods = np.zeros(len(self.class_names))
        idx = 0
        for name in self.class_names:
            likelihoods[idx] = multivariate_gaussian_pdf(x, self.means[name], self.cov)
            idx = idx + 1

        # get the indices for sorting the likelihoods (in descending order)
        indices_sorted = np.argsort(likelihoods)[::-1]

        # print the predicted class and all class likelihoods
        print('LDA Predicted Class: ' + self.class_names[indices_sorted[0]])
        print('LDA Class Likelihoods:')
        for idx in range(len(indices_sorted)):
            print(self.class_names[indices_sorted[idx]] + ': ' + str(likelihoods[indices_sorted[idx]]))

        # return the likelihoods
        return likelihoods

    def compute_probabilities(self, x, priors):
        likelihoods = self.compute_likelihoods(x)
        number_of_classes = len(priors)
        array_of_prior_probabilities = np.zeros(number_of_classes)
        index = 0
        for class_name in self.class_names:
            array_of_prior_probabilities[index] = priors[class_name]
            index += 1
        joint_probabilities = np.multiply(likelihoods, array_of_prior_probabilities)
        total_and_marginal_probability = np.sum(joint_probabilities)
        posterior_probabilities = joint_probabilities / total_and_marginal_probability
        indices_that_sort_posterior_probabilities = np.argsort(posterior_probabilities)[::-1]
        print('LDA Predicted Class:' + self.class_names[indices_that_sort_posterior_probabilities[0]])
        print('LDA Class Posterior Probabilities:')
        for index in range(0, len(indices_that_sort_posterior_probabilities)):
            print(self.class_names[indices_that_sort_posterior_probabilities[index]] + ': ' + str(posterior_probabilities[indices_that_sort_posterior_probabilities[index]]))
        return posterior_probabilities

model_lda = LDA('Exercise3.2_iris_data.csv')

Iris_setosa_observation = [5.1, 3.5, 1.4, 0.2]
model_lda.compute_likelihoods(Iris_setosa_observation)

uninformative_priors = {
    "Iris-setosa": 1 / 3,
    "Iris-versicolor": 1 / 3,
    "Iris-virginica": 1 / 3
}
model_lda.compute_probabilities(Iris_setosa_observation, uninformative_priors)
print(model_lda)
