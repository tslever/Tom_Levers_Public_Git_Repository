'''
This neural network has the advantage of being generalizable to any number of layers with any number of neurons.

To generate this neural network, I asked GPT 4 the following on 01/27/2024.

Write Python code that trains and tests a neural network from scratch. The neural network system will receive a training data set of m 28x28 images. The first fully connected hidden layer will have 25 neurons. The second fully connected hidden layer will have 20 neurons. The third fully connected hidden layer will have 10 neurons. Each neuron in a hidden layer will have a ReLU activation function. The neural network will have a softmax output layer. The neural network will have cross-entropy loss. The neural network will learn via minibatch gradient descent. The learning rate will be 0.1.

Based on GPT 4's response, I copied most of the following, massaged training and testing images and labels, implement minibatch training, asked twice for GPT 4 to correct calculating dZ, trained the neural network, reduced learning rate, and calculated accuracy.
'''

import numpy as np

def initialize_parameters(layers_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def compute_cost(A, Y):
    m = Y.shape[1]
    logprobs = np.multiply(-np.log(A), Y)
    cost = np.sum(logprobs) / m
    return cost

def forward_propagation(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A 
        Z = np.dot(parameters['W' + str(l)], A_prev) + parameters['b' + str(l)]
        A = relu(Z)
        caches.append((A_prev, Z))

    ZL = np.dot(parameters['W' + str(L)], A) + parameters['b' + str(L)]
    AL = softmax(ZL)
    caches.append((A, ZL))

    return AL, caches

def backward_propagation(AL, Y, caches, parameters):
    grads = {}
    L = len(caches)  # Number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # Gradient for the output layer
    dZL = AL - Y
    grads["dW" + str(L)] = 1. / m * np.dot(dZL, caches[L-1][0].T)
    grads["db" + str(L)] = 1. / m * np.sum(dZL, axis=1, keepdims=True)

    dZ = dZL  # Initialize dZ to the gradient of the last layer

    # Iterate through layers in reverse order starting from L-1
    for l in reversed(range(L-1)):
        A_prev, Z = caches[l]  # A_prev is the activation from previous layer, Z is the linear component
        dZ = np.dot(parameters["W" + str(l + 2)].T, dZ) * relu_derivative(Z)  # Update dZ for the current layer
        grads["dW" + str(l + 1)] = 1. / m * np.dot(dZ, A_prev.T)
        grads["db" + str(l + 1)] = 1. / m * np.sum(dZ, axis=1, keepdims=True)

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]

    return parameters

def predict(X, parameters):
    AL, _ = forward_propagation(X, parameters)
    predictions = np.argmax(AL, axis=0)
    return predictions

def model(X_train, Y_train, layers_dims, learning_rate=0.1, num_iterations=3000):
    parameters = initialize_parameters(layers_dims)

    for i in range(0, num_iterations):
        m = X_train.shape[1]
        random_indices = np.random.choice(m, 20, replace=False)
        images_in_minibatch = X_train[:, random_indices]
        labels_in_minibatch = Y_train[:, random_indices]
        AL, caches = forward_propagation(images_in_minibatch, parameters)
        cost = compute_cost(AL, labels_in_minibatch)
        grads = backward_propagation(AL, labels_in_minibatch, caches, parameters)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

import mnist
train_images = mnist.train_images()
m = train_images.shape[0]  # Number of examples
X_train = train_images.reshape(m, -1).T
train_labels = mnist.train_labels()
Y_train = np.zeros((10, m))
for i in range(0, m):
    Y_train[train_labels[i], i] = 1
test_images = mnist.test_images()
m = test_images.shape[0]
X_test = test_images.reshape(m, -1).T
Y_test = mnist.test_labels()

# Example usage
# X_train is (784, m) and Y_train is (10, m) with one-hot encoded labels
layers_dims = [784, 250, 200, 100, 10] # 10 output classes
# Learning Rate: https://ni.cmu.edu/~plaut/Lens/Manual/thumb.html
parameters = model(X_train, Y_train, layers_dims, learning_rate=0.0001, num_iterations = 5000)

# Test the model
predictions = predict(X_test, parameters)
number_of_predictions = len(predictions)
number_of_correct_predictions = 0
for i in range(0, number_of_predictions):
    if predictions[i] == Y_test[i]:
        number_of_correct_predictions += 1
print(number_of_correct_predictions / number_of_predictions)