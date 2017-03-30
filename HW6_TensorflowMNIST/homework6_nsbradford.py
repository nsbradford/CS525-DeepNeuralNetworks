"""
    homework6_nsbradford.py
    Nicholas S. Bradford
    Fri 03/31/2017

    

    Based on the tutorial above, use TensorFlow to create a multi-layer fully-connected 
        neural network to recognize hand-written digits from the MNIST dataset; you should 
        attain a test accuracy of at least 97.5%. You are free to optimize the number of 
        layers and the number of units per layer, along with all other hyperparameters 
        (learning rate schedule, momentum, minibatch size, etc.), using only the training 
        and/or validation data. Create a screenshot showing the last 20 SGD iterations on 
        the training set (after performing whatever hyperparameter optimization was necessary 
        on the validation set), along with a screenshot of your final accuracy and cost on 
        the test set. 

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def accuracy(sess, x, y, y_, test_data, test_labels):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: test_data,
                                        y_: test_labels}))    

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def main(_):
    """
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .learningRate(0.006)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .regularization(true).l2(1e-4)
    """
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W1 = init_weights([784, 200])
    b1 = init_weights([200])
    h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    W2 = init_weights([200, 40])
    b2 = init_weights([40])
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
    W3 = init_weights([40, 10])
    b3 = init_weights([10])
    y = tf.matmul(h2, W3) + b3

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(.1).minimize(cross_entropy)

    sess = tf.InteractiveSession()# with tf.InteractiveSession() as sess:
    tf.global_variables_initializer().run()
    # Train
    N_BATCHES = 10000
    for i in range(N_BATCHES):
        batch_xs, batch_ys = mnist.train.next_batch(256)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i % 200 == 0:
            accuracy(sess, x, y, y_, test_data=mnist.test.images, test_labels=mnist.test.labels)

    # Test trained model
    accuracy(sess, x, y, y_, test_data=mnist.test.images, test_labels=mnist.test.labels)
    print('Reset TF graph...')
    sess.close()
    tf.reset_default_graph()
    print('Successful exit')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                                            help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True) # Import data
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

