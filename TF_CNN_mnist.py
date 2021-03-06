import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn
from tensorflow.contrib import rnn
import numpy as np

#Import MINST data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
# Parameters
learning_rate = 0.001
hm_epochs = 200000
batch_size = 128
display_step = 10
log_path = './exampleCNN/'

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units
n_steps = 28 # timesteps
# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),# 5x5 conv, 1 input, 32 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),# 5x5 conv, 32 inputs, 64 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])), # fully connected, 7*7*64 inputs, 1024 outputs
    'out': tf.Variable(tf.random_normal([1024, n_classes]))# 1024 inputs, 10 outputs (class prediction)
}
biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def conv2d(x, W, b, strides=1):    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# Create model
def convolutional_neural_network(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])    # Convolution Layer
    conv1 = maxpool2d(conv1, k=2)    # Max Pooling (down-sampling)
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)    # Apply Dropout
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])    # Class prediction
    return out

def train_neural_network(x):
    prediction = convolutional_neural_network(x, weights, biases, keep_prob)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar("loss", cost)         # Create a summary to monitor cost tensor
    tf.summary.scalar("accuracy", accuracy) # Create a summary to monitor accuracy tensor
    for var in tf.trainable_variables():    # Create summaries to visualize weights
        tf.summary.histogram(var.name, var)
    merged_summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        step = 1
        #Write log to Tensorboard
        summary_writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())
        # Keep training until reach max iterations
        while step * batch_size < hm_epochs:
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            #epoch_x = epoch_x.reshape((batch_size, n_steps, chunk_size))
            sess.run(optimizer, feed_dict={x: epoch_x, y: epoch_y,keep_prob: dropout})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc, summary  = sess.run([cost, accuracy, merged_summary_op], feed_dict={x: epoch_x, y: epoch_y, keep_prob: 1.})
                print("Epoch " + str(step * batch_size) + ", Epoch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
                # Write log entry per iteration
                summary_writer.add_summary(summary, step * batch_size)
            step += 1
        print ("Optimization Finished!")
        # Calculate accuracy for 128 mnist test images
        # test_len = 256
        # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, chunk_size))
        # test_label = mnist.test.labels[:test_len]
        # print ("Partial Testing Accuracy:", \
            # sess.run(accuracy, feed_dict={x: test_data, y: test_label, keep_prob: 1.})) 
        print('Global Testing Accuracy:',accuracy.eval({x:mnist.test.images[:256], y:mnist.test.labels[:256], keep_prob: 1.}))
        print ('Run the command line:\n --> tensorboard --logdir=./exampleCNN \nThen open http://0.0.0.0:6006/ into your web browser')
train_neural_network(x)    