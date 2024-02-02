import cifar10
import logging
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(filename = 'Neural_Network_Evaluation.log', level = logging.INFO)
logger = logging.getLogger(__name__)

def ReLU(x):
  return np.maximum(0, x)

def softmax(x):
    exponentiated_values = np.exp(x - np.max(x, axis = 1, keepdims = True))
    return exponentiated_values / np.sum(exponentiated_values, axis = 1, keepdims = True)

def combined_loss(Yhat, Y, W1, W2, lambda_for_L2_regularization):
    Yhat_clipped = np.clip(Yhat, 1e-9, 1 - 1e-9)
    cross_entropy_loss = -np.sum(Y * np.log(Yhat_clipped)) / Y.shape[0]
    L2_regularization_loss = lambda_for_L2_regularization * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return cross_entropy_loss + L2_regularization_loss

class TwoLayerNeuralNetwork(object):

    def __init__(self, input_size, hidden_size, output_size, scale):
        self._output_size = output_size
        self.params = {}
        self.params['W1'] = scale * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = scale * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y = None, reg = 0.0):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, _ = X.shape
        Z1 = np.matmul(X, W1) + b1
        A1 = ReLU(Z1)
        scores = np.matmul(A1, W2) + b2
        if y is None:
            return scores
        scores = softmax(scores)
        matrix_Y = np.zeros((N, self._output_size))
        for i in range(0, N):
          y_i = y[i]
          matrix_Y[i, y_i] = 1
        loss = combined_loss(scores, matrix_Y, W1, W2, reg)
        grads = {}
        delta_Y = scores - matrix_Y
        dW2 = np.matmul(A1.T, delta_Y) / N
        db2 = np.sum(delta_Y, axis = 0) / N
        dA1 = np.matmul(delta_Y, W2.T)
        dZ1 = dA1 * (Z1 > 0)
        dW1 = np.matmul(X.T, dZ1) / N
        db1 = np.sum(dZ1, axis = 0) / N
        dW2 += 2 * W2 * reg
        dW1 += 2 * W1 * reg
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W1'] = dW1
        grads['b1'] = db1
        return loss, grads

    def train(
        self,
        X,
        y,
        X_val,
        y_val,
        learning_rate = 1e-3,
        learning_rate_decay = 0.95,
        reg = 5e-6,
        num_iters = 100,
        batch_size = 200,
        verbose = False
    ):
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        for it in range(num_iters):
            array_of_random_indices = np.random.randint(0, X.shape[0], batch_size)
            X_batch = X[array_of_random_indices, :]
            y_batch = y[array_of_random_indices]
            loss, grads = self.loss(X = X_batch, y = y_batch, reg = reg)
            loss_history.append(loss)
            self.params['W2'] = self.params['W2'] - learning_rate * grads['W2']
            self.params['b2'] = self.params['b2'] - learning_rate * grads['b2']
            self.params['W1'] = self.params['W1'] - learning_rate * grads['W1']
            self.params['b1'] = self.params['b1'] - learning_rate * grads['b1']
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
            if it % 1000 == 0:
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
            if it % iterations_per_epoch == 0:
                learning_rate *= learning_rate_decay
        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        return np.argmax(self.loss(X), axis = 1)
    
    def save(self):
        np.save('W1.npy', self.params['W1'])
        np.save('b1.npy', self.params['b1'])
        np.save('W2.npy', self.params['W2'])
        np.save('b2.npy', self.params['b2'])

def load_CIFAR10():
    images = []
    labels = []
    for image, label in cifar10.data_batch_generator():
        images.append(image.astype('float32'))
        labels.append(label)
    images = np.array(images)
    images = images.reshape(images.shape[0], -1)
    labels = np.array(labels)
    array_of_random_indices = np.random.randint(0, images.shape[0], images.shape[0])
    images = images[array_of_random_indices, :]
    labels = labels[array_of_random_indices]
    return (images, labels)

def plot_loss_and_training_and_validation_accuracies(list_of_stats, description_of_neural_network):
    plt.subplot(2, 1, 1)
    for stats in list_of_stats:
        plt.plot(stats['loss_history'], 'b-')
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.subplot(2, 1, 2)
    for stats in list_of_stats:
        plt.plot(stats['train_acc_history'], 'b-', label='train')
        plt.plot(stats['val_acc_history'], 'g-', label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')
    plt.legend()
    filename = description_of_neural_network.replace(', ', '_')
    plt.savefig(fname = f'./plots/{filename}.png')
    plt.close()

from math import sqrt, ceil

def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid

def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.savefig(fname = f'./plots/Network_Weights.png')
    plt.close()

if __name__ == '__main__':

    best_network = None
    best_average_validation_accuracy = -1.0
    description_of_best_neural_network = ''

    images, labels = load_CIFAR10()
    number_of_training_and_validation_images = int(0.8 * images.shape[0])
    number_of_testing_images = int(0.2 * images.shape[0])
    training_and_validation_images = images[0 : number_of_training_and_validation_images, :]
    training_and_validation_labels = labels[0 : number_of_training_and_validation_images]
    test_images = images[-number_of_testing_images:, :]
    test_means = np.mean(test_images, axis = 0)
    test_standard_deviations = np.std(test_images, axis = 0)
    standardized_test_images = (test_images - test_means) / test_standard_deviations
    test_labels = labels[-number_of_testing_images:]
    average_validation_accuracy = 0
    number_of_folds = 3
    number_of_validation_images = int(number_of_training_and_validation_images / number_of_folds)
    number_of_training_images = number_of_training_and_validation_images - number_of_validation_images
    output_size = 10
    network = None
    for batch_size in [10, 100, 1000, 10_000, 20_000]:
        for hidden_size in [100, 600, 1100, 1600, 2100]:
            for L2_regularization_strength in [0, 1e-4, 1e-3, 1e-2, 1e-1]:
                for learning_rate in [1e-2, 1e-3, 1e-1, 5e-2, 5e-3]:
                    for learning_rate_decay in [1, 0.95, 0.9, 0.85, 0.8]:
                        for number_of_iterations in [10_000, 30_000, 50_000, 70_000, 90_000]:
                            for scale in [1e-4, 1e-3, 1e-2, 1e-1, 1]:
                                list_of_stats = []
                                description_of_neural_network = f'{hidden_size}, {scale}, {number_of_iterations}, {batch_size}, {learning_rate}, {learning_rate_decay}, {L2_regularization_strength}'
                                message = f'Neural Network: {description_of_neural_network}'
                                print(message)
                                logging.info(message)
                                for i in range(0, number_of_folds):
                                    indices_of_validation_objects = range((i - 1) * number_of_validation_images, i * number_of_validation_images)
                                    validation_images = images[indices_of_validation_objects, :]
                                    validation_means = np.mean(validation_images, axis = 0)
                                    validation_standard_deviations = np.std(validation_images, axis = 0)
                                    standardized_validation_images = (validation_images - validation_means) / validation_standard_deviations
                                    validation_labels = labels[indices_of_validation_objects]
                                    mask = np.ones(len(training_and_validation_images), dtype = bool)
                                    mask[indices_of_validation_objects] = False
                                    training_images = training_and_validation_images[mask]
                                    training_means = np.mean(training_images, axis = 0)
                                    training_standard_deviations = np.std(training_images, axis = 0)
                                    standardized_training_images = (training_images - training_means) / training_standard_deviations
                                    training_labels = training_and_validation_labels[mask]
                                    network = TwoLayerNeuralNetwork(input_size = training_images.shape[1], hidden_size = hidden_size, output_size = output_size, scale = scale)
                                    stats = network.train(
                                        standardized_training_images,
                                        training_labels,
                                        standardized_validation_images,
                                        validation_labels,
                                        num_iters = number_of_iterations,
                                        batch_size = batch_size,
                                        learning_rate = learning_rate,
                                        learning_rate_decay = learning_rate_decay,
                                        reg = L2_regularization_strength,
                                        verbose = True
                                    )
                                    validation_accuracy = (network.predict(standardized_validation_images) == validation_labels).mean()
                                    message = f'Validation Accuracy {i}: {validation_accuracy}'
                                    print(message)
                                    logging.info(message)
                                    average_validation_accuracy += validation_accuracy
                                    list_of_stats.append(stats)
                                average_validation_accuracy /= number_of_folds
                                message = f'Average Validation Accuracy: {average_validation_accuracy}'
                                print(message)
                                logging.info(message)
                                plot_loss_and_training_and_validation_accuracies(list_of_stats, description_of_neural_network)
                                if average_validation_accuracy > best_average_validation_accuracy:
                                    best_average_validation_accuracy = average_validation_accuracy
                                    best_network = network
                                    description_of_best_neural_network = description_of_neural_network
                                    message = f'Best Neural Network: {description_of_best_neural_network}'
                                    print(message)
                                    logging.info(message)
                                    message = f'Best Average Validation Accuracy: {best_average_validation_accuracy}'
                                    print(message)
                                    logging.info(message)
                                    test_accuracy = (best_network.predict(standardized_test_images) == test_labels).mean()
                                    message = f'Test Accuracy: {test_accuracy}'
                                    print(message)
                                    logging.info(message)
                                    show_net_weights(best_network)