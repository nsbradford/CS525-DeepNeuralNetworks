"""
    homework4_nsbradford.py
    Nicholas S. Bradford
    Wed 02/22/2017

    Train a 2-layer neural network to classify images of hand-written digits from MNIST dataset.
    Implement gradient descent to minimize the cross-entropy loss function.
    Because there are 10 different outputs, there are 10 weights vectors;
        thus, the weight matrix is 784 x 10.
    After optimizing on the training set, compute both 1) loss, and 2) accurcy on test set.
    Submit screenshot.

"""

import numpy as np


def load_data():
    """ Load data.
        In ipython, use "run -i homework2_template.py" to avoid re-loading of data.
        Each row is an example, each column is a feature.
        Returns:
            train_data      (nd.array): (55000, 784)
            train_labels    (nd.array): (55000, 10)
            test_data       (nd.array): (10000, 784)
            test_labels     (nd.array): (10000, 10)
    """
    prefix = 'mnist_data/'
    train_data = np.load(prefix + "mnist_train_images.npy")
    train_labels = np.load(prefix + "mnist_train_labels.npy")
    test_data = np.load(prefix + "mnist_test_images.npy")
    test_labels = np.load(prefix + "mnist_test_labels.npy")
    assert train_data.shape == (55000, 784) and train_labels.shape == (55000, 10)
    assert train_data.shape == (10000, 784) and test_labels.shape == (10000, 10)
    return train_data, train_labels, test_data, test_labels


def softmax(x, wk, w):
    """ Softmax: exp(x.T.dot(w_k)) / SUM { exp(x.T.dot(w_k))} """
    top = np.exp(x.T.dot(wk))
    i = 0
    w_i = w[:, i]
    bottom = np.exp(x.T.dot(w_i))
    for i in range(1, 10):
        w_i = w[:, i]
        bottom += np.exp(x.T.dot(w_i))
    return top / bottom


def J(w, data, labels):
    """ Computes cross-entropy loss function.
        J(w1, ..., w10) = -1/m SUM(j=1 to m) { SUM(k=1 to 10) { y } }
        Args:
            w       (np.array): 784 x 10
            data    (np.array): m x 784
            labels  (np.array): m x 10
        Returns:
            J (float)
    """
    d = data.shape[1]
    m = labels.shape[1]
    assert d == 10
    scale = -1.0 / m
    cost = 0
    # TODO how to vectorize inner softmax() calculation?
    for j in range(m):
        for k in d:
            wk = w[:, k]
            cost += labels[j, k] * np.log(softmax(x, wk, w))
    return cost


def gradJ(w, data, labels):
    """ Compute gradient of cross-entropy loss function. 
        For one training example: dJ/dw = (yhat - yi)x = SUM(1 to m) { yhat_i^(j) - y_i^(j)}
    """
    d = data.shape[1]
    m = labels.shape[1]
    i = 0
    wk = w[:, i]
    grad += (softmax(x, wk, w) - labels[j, i]) 
    # TODO how to vectorize inner softmax() calculation?
    for j in range(1, m):
        for i in d:
            wk = w[:, i]
            grad += (softmax(x, wk, w) - labels[j, i])
    return grad
    


def gradient_descent(train_data, train_labels, alpha=0.0):
    """ Use Xavier initialization, where weights are randomly initialized to 
            a normal distribution with mean 0 and Var(W) = 1/N_input.
        Gradient descent is then applied until gain is < epsilon.
        learning_rate =  epsilon 
        threshold = delta
    """
    return np.ones(784, 10)

    print('METHOD 2: Train 2-layer ANN with regularization alpha: ', alpha)
    sigma = np.sqrt(1.0 / trainingFaces.size) # = 1/24
    w  = np.random.randn(trainingFaces.shape[1]) * sigma
    learning_rate = 3e-5
    threshold = 1e-3
    prevJ = J(w, trainingFaces, trainingLabels, alpha)
    diff = 1000
    count = 0
    while diff > threshold:
        update = learning_rate * gradJ(w, trainingFaces, trainingLabels, alpha)
        w = w - update
        newJ = J(w, trainingFaces, trainingLabels, alpha)
        diff = prevJ - newJ
        prevJ = newJ
        count += 1
        print('\t{} \tCost: {} \t Diff: {}'.format(count, newJ, diff))
    return w


def main():
    train_data, train_labels, test_data, test_labels = load_data()
    w = gradient_descent(train_data, train_labels)
    assert w.shape == (784, 10)


if __name__ == '__main__':
    main()