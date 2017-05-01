import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn
from tensorflow.contrib import rnn
import numpy as np

#Import MINST data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
#Parameters
learning_rate = 0.001
batch_size = 128
display_step = 10
hm_epochs = 100000
# Network Parameters
chunk_size = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)
#Input
x = tf.placeholder('float32', [None, n_steps,chunk_size])
y = tf.placeholder('float32', [None, n_classes])
# Define weights
weights = {'w': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
biases  = {'b': tf.Variable(tf.random_normal([n_classes]))}
def recurrent_neural_network(x, weights, biases):   
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, chunk_size)
    x = tf.unstack(x, n_steps, 1)
    lstm_cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['w']) + biases['b']

def train_neural_network(x):
    prediction = recurrent_neural_network(x, weights, biases)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < hm_epochs:
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            epoch_x = epoch_x.reshape((batch_size, n_steps, chunk_size))
            sess.run(optimizer, feed_dict={x: epoch_x, y: epoch_y})
            if step % display_step == 0:
                acc = sess.run(accuracy, feed_dict={x: epoch_x, y: epoch_y})
                loss = sess.run(cost, feed_dict={x: epoch_x, y: epoch_y})
                print("Epoch " + str(step) + ", Epoch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
        print ("Optimization Finished!")
        # Calculate accuracy for 128 mnist test images
        # test_len = 256
        # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, chunk_size))
        # test_label = mnist.test.labels[:test_len]
        # print ("Partial Testing Accuracy:", \
            # sess.run(accuracy, feed_dict={x: test_data, y: test_label})) 
        print('Global Testing Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_steps, chunk_size)), y:mnist.test.labels}))

train_neural_network(x)    