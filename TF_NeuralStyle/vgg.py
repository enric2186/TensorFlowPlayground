#################################################################################################
### Load VGGNet model and extract pretrained weights

import numpy as np
import tensorflow as tf
import scipy.io


def _weights(vgg_layers, layer, expected_layer_name):
    ### Return the weights and biases already trained by VGG
    W = vgg_layers[0][layer][0][0][2][0][0]
    b = vgg_layers[0][layer][0][0][2][0][1]
    layer_name = vgg_layers[0][layer][0][0][0][0]
    assert layer_name == expected_layer_name
    return W, b.reshape(b.size)

def _conv2dr(vgg_layers, prev_layer, layer, layer_name):
    with tf.variable_scope(layer_name) as scope:
        W, b = _weights(vgg_layers, layer, layer_name)
        W = tf.constant(W, name='weights')
        b = tf.constant(b, name='bias')
        conv2d = tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv2d+b)
    
def _avgpool(prev_layer):
    return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                          padding='SAME', name='avg_pool_')
                          
def _maxpool(prev_layer):
    return tf.nn.max_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                          padding='SAME', name='max_pool_')                          
def _pool(prev_layer,opt):
    if opt=='avg':
        return _avgpool(prev_layer)
    elif opt=='max':
        return _maxpool(prev_layer)
        
def load_vgg(path, input_image,opt):
    vgg = scipy.io.loadmat(path)
    vgg_layers = vgg['layers']

    graph = {} 
    graph['conv1_1']  = _conv2dr(vgg_layers, input_image, 0, 'conv1_1')
    graph['conv1_2']  = _conv2dr(vgg_layers, graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _pool(graph['conv1_2'],opt)
    graph['conv2_1']  = _conv2dr(vgg_layers, graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = _conv2dr(vgg_layers, graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _pool(graph['conv2_2'],opt)
    graph['conv3_1']  = _conv2dr(vgg_layers, graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = _conv2dr(vgg_layers, graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = _conv2dr(vgg_layers, graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = _conv2dr(vgg_layers, graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _pool(graph['conv3_4'],opt)
    graph['conv4_1']  = _conv2dr(vgg_layers, graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = _conv2dr(vgg_layers, graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = _conv2dr(vgg_layers, graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = _conv2dr(vgg_layers, graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _pool(graph['conv4_4'],opt)
    graph['conv5_1']  = _conv2dr(vgg_layers, graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = _conv2dr(vgg_layers, graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = _conv2dr(vgg_layers, graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = _conv2dr(vgg_layers, graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _pool(graph['conv5_4'],opt)
       
    return graph