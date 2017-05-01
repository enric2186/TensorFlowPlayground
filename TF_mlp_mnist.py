import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#Neural Network methods
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
def init_neural_layer(weights,biases):
    return {'weights':init_weights(weights),'biases':init_weights(biases)}
def neural_network_model(layer_weights,data):
    for i,lw in enumerate(layer_weights):
        print('layer',i,lw)
        layer=init_neural_layer(lw,[nodes])
        if(i==0):
            input=data
        else:
            input=li
        li = tf.add(tf.matmul(input,layer['weights']), layer['biases'])
        li = tf.nn.relu(li)
    layer_output =  init_neural_layer([nodes, n_classes],[n_classes])
    output = tf.matmul(li,layer_output['weights']) + layer_output['biases']         
    return output

def train_neural_network(layer_weights,x,hm_epochs):
    prediction = neural_network_model(layer_weights,x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
        
        
#Load the Mnist repository
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

#Shape definitions for the neural net
nodes=500
n_classes = 10
batch_size = 100
hm_epochs = 10

#HeightxWeight -> As images are 28x28, we are going to process 784 values internally
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')        

layer_weights=[[784, nodes],[nodes, nodes],[nodes, nodes]]
train_neural_network(layer_weights,x,hm_epochs)
