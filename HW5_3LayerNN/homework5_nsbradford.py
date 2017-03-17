"""
    homework5_nsbradford.py
    Nicholas S. Bradford
    Wed 03/17/2017

    In this problem you will train a 3-layer neural network to classify images of 
        hand-written digits from the MNIST dataset. Similarly to Homework 4, 
        the input to the network will be a 28 x 28-pixel image 
        (converted into a 784-dim vector);
        the output will be a vector of 10 probabilities (one for each digit).

    Hyperparam tuning:
        # of units in hidden layer {30, 40, 50}
        Learning rate {0.001, 0.005, 0.01, 0.05, 0.1, 0.5}
        Minibatch size {16, 32, 64, 128, 256}
        # of training epochs
        Regularization strength

    Specifically, implement f: R784 -> R10, where:
        z1 = p1(x) = W1 * x + b1
        z2 = p2(x) = W2 * x + b2
        h1 = f1(x) = a1(p1(x))
        yhat = f2(h1) = a2(p2(h1))
        f(x) = f2(f1(x)) = a2(p2(a1(p1(x))))
    For activation functions a1, a2 in network, use:
        a1(z1) = relu(z1)
        a2(z2) = softmax(z2)

    Task
        A) Implement stochastic gradient descent (SGD; see Section 5.9 and Algorithm 6.4 
            in the Deep Learning textbook) for the 3-layer neural network shown above.
        B) Optimize hyperparams by creating a findBestHyperparameters() that tries at least
            10 different settings (Remember, optimize hyperparams on ONLY the validation set)
        C) Report cross-entropy and accuracy on test set, along with screenshot of these values
            with final 20 epochs of SGD. Cost should be < 0.16, accuracy should be > 95%.

    Unregularized test accuracy after 30 epochs: 96.2%
    Regularized with Alpha=1/1e-3: 96.42%

"""

import numpy as np
import math

import scipy
from sklearn.metrics import accuracy_score

DEBUG = False
N_INPUT = 784
N_OUTPUT = 10

def load_data():
    """ Load data.
        In ipython, use "run -i homework2_template.py" to avoid re-loading of data.
        Each row is an example, each column is a feature.
        Returns:
            train_data      (nd.array): (55000, 784)
            train_labels    (nd.array): (55000, 10)
            val_data        (nd.array): (5000, 784)
            val_labels      (nd.array): (5000, 10)
            test_data       (nd.array): (10000, 784)
            test_labels     (nd.array): (10000, 10)
    """
    prefix = 'mnist_data/'
    train_data = np.load(prefix + 'mnist_train_images.npy')
    train_labels = np.load(prefix + 'mnist_train_labels.npy')
    val_data = np.load(prefix + 'mnist_validation_images.npy')
    val_labels = np.load(prefix + 'mnist_validation_labels.npy')
    test_data = np.load(prefix + 'mnist_test_images.npy')
    test_labels = np.load(prefix + 'mnist_test_labels.npy')
    assert train_data.shape == (55000, 784) and train_labels.shape == (55000, 10)
    assert val_data.shape == (5000, 784) and val_labels.shape == (5000, 10)
    assert test_data.shape == (10000, 784) and test_labels.shape == (10000, 10)
    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def crossEntropy(y, yhat):
    m = y.shape[0]
    cost = np.sum(y * np.log(yhat)) # sum element-wise product
    return (-1.0 / m) * cost


def relu(z):
    return (z >= 0) * z

def grad_relu(z):
    return (z >= 0) * 1


def softmax(z):
    bottom_vec = np.exp(z)
    bottom = np.sum(bottom_vec, axis=1, keepdims=True) # sum of each row = sum for each dimension
    top = np.exp(z)
    yhat = np.divide(top, bottom) # could alternatively use bottom[:, None] to keep rows
    return yhat

#==================================================================================================


def flattenW(W1, b1, W2, b2):
    """ Flattens all weight and bias matrices into a weight vector.
            Can be used just as well for the gradients.
        Args:
            W1 (np.array): n_hidden x 784
            W2 (np.array): 10 x n_hidden
            b1 (np.array): n_hidden x 1
            b2 (np.array): 10 x 1
            x (np.array): m x 784
        Returns:
            w (np.array): length * 1
    """
    n_hidden_units = W1.shape[0]
    flattened = (W1.flatten(), b1.flatten(), W2.flatten(), b2.flatten())
    w = np.concatenate(flattened)
    length = (N_INPUT * n_hidden_units) + (n_hidden_units * N_OUTPUT) + n_hidden_units + N_OUTPUT
    assert w.shape == (length,), (w.shape, (length,))
    return w


def expandW(w, n_hidden_units):
    """ Unpacks the weight and bias matrices from a weight vector.
            Can be used just as well for the gradients.
    """
    i1 = 784 * n_hidden_units
    i2 = i1 + n_hidden_units
    i3 = i2 + n_hidden_units * 10
    i4 = i3 + 10
    assert i4 == w.size, str(i4) + ' ' + str(w.size)
    W1 = w[0:i1].reshape((n_hidden_units, 784))
    b1 = w[i1:i2]
    W2 = w[i2:i3].reshape((10, n_hidden_units))
    b2 = w[i3:i4]
    return W1, b1, W2, b2


def forwardPropagate(W1, b1, W2, b2, x):
    """ Produce output from the neural network.
        Args:
            W1 (np.array): n_hidden x 784
            W2 (np.array): 10 x n_hidden
            b1 (np.array): n_hidden x 1
            b2 (np.array): 10 x 1
            x (np.array): m x 784
        Returns:
            yhat (np.array): m x 10
    """
    z1 = x.dot(W1.T) + b1 
    h1 = relu(z1)
    z2 = h1.dot(W2.T) + b2
    yhat = softmax(z2)
    assert yhat.shape == (x.shape[0], W2.shape[0]), yhat.shape
    return yhat


def J(W1, b1, W2, b2, x, y):
    """ Computes cross-entropy loss function.
        J(w1, ..., w10) = -1/m SUM(j=1 to m) { SUM(k=1 to 10) { y } }
        Args:
            W1 (np.array): n_hidden x 784
            W2 (np.array): 10 x n_hidden
            b1 (np.array): n_hidden x 1
            b2 (np.array): 10 x 1
            x (np.array): m x 784
            y (np.array): m x 10
        Returns:
            J (float): cost of w given x and y

        Use cross-entropy cost functino:
        J(W1, b1, W2, b2) = -1/m * SUM(y * log(yhat))
    """
    yhat = forwardPropagate(W1, b1, W2, b2, x) # OLD: yhat = softmax(x.dot(w))
    return crossEntropy(y, yhat)


def JWrapper(w, x, y, n_hidden_units):
    W1, b1, W2, b2 = expandW(w, n_hidden_units)
    return J(W1, b1, W2, b2, x, y)


def gradJ(W1, b1, W2, b2, x, y):
    """
        Args:
            W1 (np.array): n_hidden x 784
            W2 (np.array): 10 x n_hidden
            b1 (np.array): n_hidden x 1
            b2 (np.array): 10 x 1
            x (np.array): m x 784
            y (np.array): m x 10
    """
    m = x.shape[0]
    n_hidden_units = W1.shape[0]

    if DEBUG: print('\t\tgradJ() backprop...')
    z1 = x.dot(W1.T) + b1
    h1 = relu(z1)
    yhat = forwardPropagate(W1, b1, W2, b2, x)
    
    if DEBUG: print('\t\t\tBegin Layer 2...')
    g2 = yhat - y
    dJ_dW2 = g2.T.dot(h1) #np.outer(g2, h1)
    dJ_db2 = np.copy(g2)
    dJ_db2 = np.sum(dJ_db2, axis=0) # sum of each column

    if DEBUG: print('\t\t\tBegin Layer 1...')
    g1 = g2.dot(W2)
    g1 = np.multiply(g1, grad_relu(z1))
    if DEBUG: print('\t\t\tdJ_dW1...')
    dJ_dW1 = g1.T.dot(x) #np.outer(g1, x.T)
    dJ_db1 = np.copy(g1)
    dJ_db1 = np.sum(dJ_db1, axis=0) # sum of each column

    assert yhat.shape == (m, N_OUTPUT), yhat.shape
    assert z1.shape == h1.shape == (m, n_hidden_units), (z1.shape, h1.shape, (m, n_hidden_units))
    assert g2.shape == (m, N_OUTPUT), g2.shape
    assert dJ_dW2.shape == W2.shape, dJ_dW2.shape
    assert g1.shape == h1.shape == (m, n_hidden_units), (g1.shape, h1.shape)
    assert dJ_dW1.shape == W1.shape, dJ_dW1.shape
    assert dJ_db2.shape == b2.shape, dJ_db2.shape
    assert dJ_db1.shape == b1.shape, dJ_db1.shape
    return flattenW(dJ_dW1, dJ_db1, dJ_dW2, dJ_db2) / m


def gradJWrapper(w, x, y, n_hidden_units):
    """    """
    length = (N_INPUT * n_hidden_units) + (n_hidden_units * N_OUTPUT) + n_hidden_units + N_OUTPUT
    assert w.shape == (length,), (w.shape, (length,))
    W1, b1, W2, b2 = expandW(w, n_hidden_units)
    return gradJ(W1, b1, W2, b2, x, y)


def backprop(w, x, y, n_hidden_units, learning_rate, alpha):
    dJ_dW1, dJ_db1, dJ_dW2, dJ_db2 = expandW(gradJWrapper(w, x, y, n_hidden_units), n_hidden_units)
    W1, b1, W2, b2 = expandW(w, n_hidden_units)
    decay = (1 - learning_rate * alpha)
    newW1 = decay * W1 - (dJ_dW1 * learning_rate)
    newb1 = decay * b1 - (dJ_db1 * learning_rate)
    newW2 = decay * W2 - (dJ_dW2 * learning_rate)
    newb2 = decay * b2 - (dJ_db2 * learning_rate)
    return flattenW(newW1, newb1, newW2, newb2)


#==================================================================================================


def shuffleArraysInUnison(x, y):
    assert x.shape[0] == y.shape[0]
    p = np.random.permutation(x.shape[0])
    return x[p, :], y[p, :]


def getMiniBatches(data, labels, minibatch_size):
    x, y = shuffleArraysInUnison(data, labels)
    n_batches = int(math.ceil(x.shape[0] / minibatch_size))
    batch_x = np.array_split(x, n_batches, axis=0)
    batch_y = np.array_split(y, n_batches, axis=0)
    return batch_x, batch_y


def gradient_descent(x, y, learning_rate, minibatch_size, n_hidden_units, alpha, n_epochs):
    """ 
    """
    W1, b1, W2, b2 = initializeWeights(n_hidden_units, n_inputs=N_INPUT, n_outputs=N_OUTPUT)
    w = flattenW(W1, b1, W2, b2)
    prevJ = JWrapper(w, x, y, n_hidden_units)
    epochJ = prevJ

    print ('Initial Cost:', prevJ)
    batch_x, batch_y = getMiniBatches(x, y, minibatch_size) # a list of individual batches
    print(len(batch_x))
    for i in range(n_epochs):
        for x, y in zip(batch_x, batch_y):
            w = backprop(w, x, y, n_hidden_units, learning_rate, alpha)
            if DEBUG: print('\tUpdated weights.')
            newJ = JWrapper(w, x, y, n_hidden_units)
            diff = prevJ - newJ
            prevJ = newJ
            # print('\t\t{} \tCost: {} \t Diff: {}'.format(i+1, newJ, diff))
        epochDiff = epochJ - prevJ
        epochJ = prevJ
        print('\tEnd Epoch {} \tCost: {} \t EpochDiff: {}'.format(i+1, newJ, epochDiff))
    W1, b1, W2, b2 = expandW(w, n_hidden_units)
    return W1, b1, W2, b2


def train_model(x, y, params):
    print('\nTrain 3-layer ANN with learning {}, batch_size {}, n_hidden_units {}, alpha {}'.format(
                    params.learning_rate, params.minibatch_size, params.n_hidden_units, params.alpha))
    return gradient_descent(x, y, 
                            params.learning_rate, 
                            params.minibatch_size, 
                            params.n_hidden_units, 
                            params.alpha,
                            n_epochs=3)

#==================================================================================================

class HyperParams():

    range_learning_rate = {0.1, 0.5} #{0.001, 0.005, 0.01, 0.05, 0.1, 0.5}
    range_minibatch_size = {256} # {16, 32, 64, 128, 256}
    range_n_hidden_units = {30, 40, 50} #{30, 40, 50}
    range_alpha = {1/1e2, 1/1e3} #{1e3, 1e4, 1e5}

    def __init__(self, learning_rate, minibatch_size, n_hidden_units, alpha):
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.n_hidden_units = n_hidden_units
        self.alpha = alpha

    @staticmethod
    def getHyperParamList():
        answer = []
        for rate in HyperParams.range_learning_rate:
            for size in HyperParams.range_minibatch_size:
                for units in HyperParams.range_n_hidden_units:
                    for alpha in HyperParams.range_alpha:
                        answer.append(HyperParams(rate, size, units, alpha))
        # return answer
        return [HyperParams(0.5, 256, 50, 1/1e3)] # best

    def toStr(self):
        return ('learning_rate :' + str(self.learning_rate), 
                'minibatch_size :' + str(self.minibatch_size), 
                'n_hidden_units :' + str(self.n_hidden_units), 
                'alpha :' + str(self.alpha))


def getLossAndAccuracy(W1, b1, W2, b2, data, labels):
    predictions = forwardPropagate(W1, b1, W2, b2, data)
    predict_labels = predictions.argmax(axis=1)
    true_labels = labels.argmax(axis=1)
    accuracy = accuracy_score(y_true=true_labels, y_pred=predict_labels)
    loss = J(W1, b1, W2, b2, data, labels)
    return loss, accuracy


def findBestHyperparameters(train_data, train_labels, val_data, val_labels):
    best_params = None
    best_accuracy = 0.0
    print('\nBegin findBestHyperparameters(): checking {} sets'.format(len(HyperParams.getHyperParamList())))
    for params in HyperParams.getHyperParamList():

        print('\nTesting pararms: {', params.toStr(), '}')
        W1, b1, W2, b2 = train_model(train_data, train_labels, params)
        loss, accuracy = getLossAndAccuracy(W1, b1, W2, b2, val_data, val_labels)
        reportResults(loss, accuracy, 'Validation')
        if accuracy > best_accuracy:
            best_params = params
            best_accuracy = accuracy
    print('\nBest params found: {', best_params.toStr(), '}\n')
    return best_params

#==================================================================================================

def initializeWeights(n_hidden_units, n_inputs, n_outputs):
    """ Normally we use Xavier initialization, where weights are randomly initialized to 
            a normal distribution with mean 0 and Var(W) = 1/N_input.
    """
    W1 = np.random.randn(n_hidden_units, n_inputs) * np.sqrt(1/(n_inputs*n_hidden_units))
    W2 = np.random.randn(n_outputs, n_hidden_units) * np.sqrt(1/(n_hidden_units*n_outputs))
    b1 = np.random.randn(n_hidden_units, 1) * np.sqrt(1/n_hidden_units)
    b2 = np.random.randn(n_outputs, 1) * np.sqrt(1/n_outputs)
    return W1, b1, W2, b2

def testBackpropGradient(x, y, n_hidden_units):
    """ Use check_grad() to ensure correctness of gradient expression. """
    assert x.shape[1] == 784 and y.shape[1] == 10
    print('testBackpropGradient...')
    W1, b1, W2, b2 = initializeWeights(n_hidden_units, n_inputs=784, n_outputs=10)
    w = flattenW(W1, b1, W2, b2)
    point_to_check = w
    gradient_check = scipy.optimize.check_grad(JWrapper, gradJWrapper, point_to_check, 
                        x, y, n_hidden_units)
    print('check_grad() value: {}'.format(gradient_check))
    print('Gradient is good!' if gradient_check < 1e-4 else 'WARNING: bad gradient!')


def reportResults(loss, accuracy, text='Test'):
    print()
    print(text + ' Loss:     {}'.format(loss))
    print(text + ' Accuracy: {}'.format(accuracy))


def main():
    np.random.seed(7)
    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_data()
    testBackpropGradient(x=val_data[:5, :], y=val_labels[:5, :], n_hidden_units=30)
    params = findBestHyperparameters(train_data, train_labels, val_data, val_labels)
    W1, b1, W2, b2 = train_model(train_data, train_labels, params)
    loss, accuracy = getLossAndAccuracy(W1, b1, W2, b2, test_data, test_labels)
    reportResults(loss, accuracy)


if __name__ == '__main__':
    main()
