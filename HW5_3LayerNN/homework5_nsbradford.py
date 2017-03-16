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

    Task
        A) Implement stochastic gradient descent (SGD; see Section 5.9 and Algorithm 6.4 
            in the Deep Learning textbook) for the 3-layer neural network shown above.
        B) Optimize hyperparams by creating a findBestHyperparameters() that tries at least
            10 different settings (Remember, optimize hyperparams on ONLY the validation set)
        C) Report cross-entropy and accuracy on test set, along with screenshot of these values
            with final 20 epochs of SGD. Cost should be < 0.16, accuracy should be > 95%.

"""

import numpy as np
from sklearn.metrics import accuracy_score

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
    return (z >= 1) * z


def softmax(z):
    bottom_vec = np.exp(z)
    bottom = np.sum(bottom_vec, axis=1, keepdims=True) # sum of each row = sum for each dimension
    top = np.exp(z)
    yhat = np.divide(top, bottom) # could alternatively use bottom[:, None] to keep rows
    return yhat


def propagateLayer(W, x, b):
    return W.dot(x) + b

def forwardPropagate(x, W1, b1, W2, b2):
    """
    Specifically, implement f: R784 -> R10, where:
        z1 = p1(x) = W1 * x + b1
        z2 = p2(x) = W2 * x + b2
        h1 = f1(x) = a1(p1(x))
        yhat = f2(h1) = a2(p2(h1))
        f(x) = f2(f1(x)) = a2(p2(a1(p1(x))))
    For activation functions a1, a2 in network, use:
        a1(z1) = relu(z1)
        a2(z2) = softmax(z2)
    """
    z1 = propagateLayer(W1, x, b1)
    h1 = relu(z1)
    z2 = propagateLayer(W2, h1, b2)
    yhat = softmax(z2)
    return yhat


def J(w, x, y, alpha=0):
    """ Computes cross-entropy loss function.
        J(w1, ..., w10) = -1/m SUM(j=1 to m) { SUM(k=1 to 10) { y } }
        Args:
            w       (np.array): 784 x 10
            x    (np.array): m x 784
            y  (np.array): m x 10
        Returns:
            J (float): cost of w given x and y

        Use cross-entropy cost functino:
        J(W1, b1, W2, b2) = -1/m * SUM(y * log(yhat))
    """
    yhat = softmax(x.dot(w))
    return crossEntropy(y, yhat)


def gradJ(w, x, y, alpha=0.0):
    """ Compute gradient of cross-entropy loss function. 
        For one training example: dJ/dw = (yhat - yi)x = SUM(1 to m) { yhat_i^(j) - y_i^(j)}
        Args:
            w    (np.array): 784 x 10
            x    (np.array): m x 784
            y    (np.array): m x 10
        Returns:
            grad (np.array): 784 x 10, gradients for each weight in w
    """
    m = float(x.shape[0])
    yhat = softmax(x.dot(w))
    answer = (yhat - y).T.dot(x).T / m
    assert answer.shape == (784, 10)
    return answer 
    

def gradient_descent(x, y, learning_rate, minibatch_size, n_hidden_units, alpha):
    """ Normally we use Xavier initialization, where weights are randomly initialized to 
            a normal distribution with mean 0 and Var(W) = 1/N_input.
        In this case for SoftMax, we can start with all 0s for initialization.
        Gradient descent is then applied until gain is < epsilon.
        learning_rate =  epsilon 
        threshold = delta
    """
    w  = np.zeros((x.shape[1], y.shape[1]))
    n_iterations = 5
    prevJ = J(w, x, y, alpha)

    print('\nTrain 3-layer ANN with regularization alpha: ', alpha)
    print ('Initial Cost:', prevJ)
    for i in range(n_iterations):
        update = learning_rate * gradJ(w, x, y, alpha)
        w = w - update
        newJ = J(w, x, y, alpha)
        diff = prevJ - newJ
        prevJ = newJ
        print('\t{} \tCost: {} \t Diff: {}'.format(i+1, newJ, diff))
    assert w.shape == (784, 10)
    return w


def train_model(x, y, params):
    return gradient_descent(x, y, 
                            params.learning_rate, 
                            params.minibatch_size, 
                            params.n_hidden_units, 
                            params.alpha)

#==================================================================================================

class HyperParams():

    range_learning_rate = {0.001, 0.005, 0.01, 0.05, 0.1, 0.5}
    range_minibatch_size = {16, 32, 64, 128, 256}
    range_n_hidden_units = {30, 40, 50}
    range_alpha = {1e3, 1e4, 1e5}

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
        return [HyperParams(0.5, 16, 0.0, 0.0)]

    def toStr(self):
        return ('learning_rate :' + str(self.learning_rate), 
                'minibatch_size :' + str(self.minibatch_size), 
                'n_hidden_units :' + str(self.n_hidden_units), 
                'alpha :' + str(self.alpha))


def getLossAndAccuracy(w, data, labels):
    predictions = data.dot(w)
    predict_labels = predictions.argmax(axis=1)
    true_labels = labels.argmax(axis=1)
    accuracy = accuracy_score(y_true=true_labels, y_pred=predict_labels)
    loss = J(w, data, labels)
    return loss, accuracy


def findBestHyperparameters(train_data, train_labels, val_data, val_labels):
    best_params = None
    best_accuracy = 0.0
    for params in HyperParams.getHyperParamList():
        print('\nTesting pararms: {', params.toStr(), '}')
        w = train_model(train_data, train_labels, params)
        loss, accuracy = getLossAndAccuracy(w, val_data, val_labels)
        if accuracy > best_accuracy:
            best_params = params
            best_accuracy = accuracy
    print('\nBest params found: {', best_params.toStr(), '}\n')
    return best_params


def reportResults(loss, accuracy):
    print()
    print('Test Loss:     {}'.format(loss))
    print('Test Accuracy: {}'.format(accuracy))


def main():
    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_data()
    params = findBestHyperparameters(train_data, train_labels, val_data, val_labels)
    w = train_model(train_data, train_labels, params)
    loss, accuracy = getLossAndAccuracy(w, test_data, test_labels)
    reportResults(loss, accuracy)


if __name__ == '__main__':
    main()