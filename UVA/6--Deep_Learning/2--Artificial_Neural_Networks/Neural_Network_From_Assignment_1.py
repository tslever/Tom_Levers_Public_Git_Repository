import cifar10
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

    def __init__(self, input_size, hidden_size, output_size, std = 1e-4):
        self._output_size = output_size
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
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
            if it % iterations_per_epoch == 0:
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                learning_rate *= learning_rate_decay
        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        return np.argmax(self.loss(X), axis = 1)

def load_CIFAR10():
    images = []
    labels = []
    for image, label in cifar10.data_batch_generator():
        images.append(image.astype('float32'))
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.33, random_state = 42)
    return X_train, X_test, y_train, y_test

def get_CIFAR10_data(num_training = 49000, num_validation = 1000, num_test = 1000):
    try:
       del X_train, y_train
       del X_test, y_test
       print('Cleared previously loaded data.')
    except:
       pass
    X_train, X_test, y_train, y_test = load_CIFAR10()
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    mean_image = np.mean(X_train, axis = 0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
    input_size = 32 * 32 * 3
    hidden_size = 50
    num_classes = 10
    network = TwoLayerNeuralNetwork(input_size, hidden_size, num_classes)
    stats = network.train(
        X_train,
        y_train,
        X_val,
        y_val,
        num_iters = 1000,
        batch_size = 200,
        learning_rate = 1e-4,
        learning_rate_decay = 0.95,
        reg = 0.25,
        verbose = True
    )
    val_acc = (network.predict(X_val) == y_val).mean()
    print('Validation accuracy: ', val_acc)