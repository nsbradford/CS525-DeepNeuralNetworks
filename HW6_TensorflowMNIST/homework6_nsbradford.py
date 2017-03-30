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

def accuracy(text, sess, x, y, y_, keep_prob, test_data, test_labels):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(text, ':', sess.run(accuracy, feed_dict={x: test_data,  
                                        y_: test_labels,
                                        keep_prob: 1.0 }))

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

    # ============================================
    # model with 98.2% accuracy
    # # Create the model
    # SIZE_H1 = 300
    # SIZE_H2 = 40
    # SIZE_H3 = 20
    # SIZE_OUTPUT = 10
    # SIZE_INPUT = 784
    # LEARNING_RATE = 0.5
    # N_BATCHES = 10000

    # keep_prob = tf.placeholder(tf.float32)
    # x = tf.placeholder(tf.float32, [None, SIZE_INPUT])
    # W1 = init_weights([SIZE_INPUT, SIZE_H1])
    # b1 = init_weights([SIZE_H1])
    # W2 = init_weights([SIZE_H1, SIZE_H2])
    # b2 = init_weights([SIZE_H2])
    # W3 = init_weights([SIZE_H2, SIZE_OUTPUT])
    # b3 = init_weights([SIZE_OUTPUT])

    # h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    # h1_drop = tf.nn.dropout(h1, keep_prob)
    # h2 = tf.nn.relu(tf.matmul(h1_drop, W2) + b2)
    # h2_drop = tf.nn.dropout(h2, keep_prob)
    # y = tf.matmul(h2_drop, W3) + b3


    # ============================================
    # model with 98.32% accuracy
    # SIZE_H1 = 300
    # SIZE_H2 = 40
    # SIZE_H3 = 20
    # SIZE_OUTPUT = 10
    # SIZE_INPUT = 784
    # LEARNING_RATE = 0.4
    # N_BATCHES = 10000
    # BATCH_SIZE = 124

    # keep_prob = tf.placeholder(tf.float32)
    # x = tf.placeholder(tf.float32, [None, SIZE_INPUT])
    # W1 = init_weights([SIZE_INPUT, SIZE_H1])
    # b1 = init_weights([SIZE_H1])
    # W2 = init_weights([SIZE_H1, SIZE_H2])
    # b2 = init_weights([SIZE_H2])
    # W3 = init_weights([SIZE_H2, SIZE_OUTPUT])
    # b3 = init_weights([SIZE_OUTPUT])

    # h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    # # h1_drop = tf.nn.dropout(h1, keep_prob)
    # h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
    # # h2_drop = tf.nn.dropout(h2, keep_prob)
    # y = tf.matmul(h2, W3) + b3

    # ============================================
    # Create the model
    # SIZE_H1 = 300
    # SIZE_H2 = 100
    # SIZE_H3 = 40
    # SIZE_OUTPUT = 10
    # SIZE_INPUT = 784
    # LEARNING_RATE = 0.4
    # N_BATCHES = 10000
    # BATCH_SIZE = 124

    # keep_prob = tf.placeholder(tf.float32)
    # x = tf.placeholder(tf.float32, [None, SIZE_INPUT])
    # W1 = init_weights([SIZE_INPUT, SIZE_H1])
    # b1 = init_weights([SIZE_H1])
    # W2 = init_weights([SIZE_H1, SIZE_H2])
    # b2 = init_weights([SIZE_H2])
    # W3 = init_weights([SIZE_H2, SIZE_H3])
    # b3 = init_weights([SIZE_H3])
    # W4 = init_weights([SIZE_H3, SIZE_OUTPUT])
    # b4 = init_weights([SIZE_OUTPUT])

    # h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    # h1_drop = tf.nn.dropout(h1, keep_prob)
    # h2 = tf.nn.relu(tf.matmul(h1_drop, W2) + b2)
    # h2_drop = tf.nn.dropout(h2, keep_prob)
    # h3 = tf.nn.relu(tf.matmul(h2_drop, W3) + b3)
    # y = tf.matmul(h3, W4) + b4

def main(_):
    """
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .learningRate(0.006)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .regularization(true).l2(1e-4)
    """
    # model with 98.32% accuracy
    SIZE_H1 = 300
    SIZE_H2 = 40
    SIZE_H3 = 20
    SIZE_OUTPUT = 10
    SIZE_INPUT = 784
    LEARNING_RATE = 0.4
    N_BATCHES = 7000
    BATCH_SIZE = 124

    keep_prob = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32, [None, SIZE_INPUT])
    W1 = init_weights([SIZE_INPUT, SIZE_H1])
    b1 = init_weights([SIZE_H1])
    W2 = init_weights([SIZE_H1, SIZE_H2])
    b2 = init_weights([SIZE_H2])
    W3 = init_weights([SIZE_H2, SIZE_OUTPUT])
    b3 = init_weights([SIZE_OUTPUT])

    h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    # h1_drop = tf.nn.dropout(h1, keep_prob)
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
    # h2_drop = tf.nn.dropout(h2, keep_prob)
    y = tf.matmul(h2, W3) + b3

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

    sess = tf.InteractiveSession()# with tf.InteractiveSession() as sess:
    tf.global_variables_initializer().run()
    
    # Train
    for i in range(N_BATCHES):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        text = str(i+1) + ' Train'
        if i > N_BATCHES - 21 or i % 200 == 0:
            accuracy(text, sess, x, y, y_, keep_prob, test_data=mnist.train.images, 
                                test_labels=mnist.train.labels)

    # Test trained model
    text = 'Final accuracy'
    accuracy(text, sess, x, y, y_, keep_prob, test_data=mnist.test.images, test_labels=mnist.test.labels)
    # print('Reset TF graph...')
    # sess.close()
    # tf.reset_default_graph()
    print('Successful exit')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                                            help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True) # Import data
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

