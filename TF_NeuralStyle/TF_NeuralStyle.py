#################################################################################################
###An implementation of "A Neural Algorithm of Artistic Style" by Leon A et al. using TensorFlow.
###Author: Enrique Bermejo
###Based on assignment examples of Stanford - CS20SI
#################################################################################################
from __future__ import print_function
import os
import time
import numpy as np
import tensorflow as tf
import vgg
import utils

MEAN_PIXELS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
""" MEAN_PIXELS is defined according to description on their github:
https://gist.github.com/ksimonyan/211839e770f7b538e2d8
'In the paper, the model is denoted as the configuration D trained with scale jittering. 
The input images should be zero-centered by mean pixel (rather than mean image) subtraction. 
Namely, the following BGR values should be subtracted: [103.939, 116.779, 123.68].'
""" 

def _create_content_loss(p, f):
    return tf.reduce_sum((f - p) ** 2) / (4.0 * p.size)

def _gram_matrix(F, N, M):
    F = tf.reshape(F, (M, N))
    return tf.matmul(tf.transpose(F), F)

def _single_style_loss(a, g):
    N = a.shape[3] # number of filters
    M = a.shape[1] * a.shape[2] # height times width of the feature map
    A = _gram_matrix(a, N, M)
    G = _gram_matrix(g, N, M)
    return tf.reduce_sum((G - A) ** 2 / ((2 * N * M) ** 2))

def _create_style_loss(A, model):
    n_layers = len(argoptions.style_layers)
    E = [_single_style_loss(A[i], model[argoptions.style_layers[i]]) for i in range(n_layers)]
    W= argoptions.style_layer_weights
    return sum([W[i] * E[i] for i in range(n_layers)])

def _create_losses(model, input_image, content_image, style_image):
    with tf.variable_scope('loss') as scope:
        with tf.Session() as sess:
            sess.run(input_image.assign(content_image)) # assign content image to the input variable
            p = sess.run(model[argoptions.content_layers])
        content_loss = _create_content_loss(p, model[argoptions.content_layers])
        with tf.Session() as sess:
            sess.run(input_image.assign(style_image))
            A = sess.run([model[layer_name] for layer_name in argoptions.style_layers])                              
        style_loss = _create_style_loss(A, model)

        total_loss = argoptions.content_weight * content_loss + argoptions.style_weight * style_loss

    return content_loss, style_loss, total_loss

def _create_summary(model):
    with tf.name_scope('summaries'):
        tf.summary.scalar('content loss', model['content_loss'])
        tf.summary.scalar('style loss', model['style_loss'])
        tf.summary.scalar('total loss', model['total_loss'])
        tf.summary.histogram('histogram content loss', model['content_loss'])
        tf.summary.histogram('histogram style loss', model['style_loss'])
        tf.summary.histogram('histogram total loss', model['total_loss'])
        return tf.summary.merge_all()

def train(model, generated_image, initial_image):
    skip_step = 1
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(argoptions.img_output_dir + '/graphs', sess.graph)
        
        sess.run(generated_image.assign(initial_image))
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        initial_step = model['global_step'].eval()
        
        start_time = time.time()
        for index in range(initial_step, argoptions.max_iter):
            if index >= 5 and index < 20:
                skip_step = 10
            elif index >= 20:
                skip_step = 20
            
            sess.run(model['optimizer'])
            if (index + 1) % skip_step == 0:
                gen_image, total_loss, summary = sess.run([generated_image, model['total_loss'], model['summary_op']])

                gen_image = gen_image + MEAN_PIXELS
                writer.add_summary(summary, global_step=index)
                print('Step {}\n   Sum: {:5.1f}'.format(index + 1, np.sum(gen_image)))
                print('   Loss: {:5.1f}'.format(total_loss))
                print('   Time: {}'.format(time.time() - start_time))
                start_time = time.time()

                filename = argoptions.img_output_dir+'\\outputs\\'+'%d.png' % (index)
                utils.save_image(filename, gen_image)

                if (index + 1) % 20 == 0:
                    saver.save(sess, argoptions.img_output_dir+'\\checkpoints\\style_transfer', index)

def main():
    global argoptions
    argoptions = utils.parse_args()
    content_img = os.path.join(argoptions.content_img_dir, argoptions.content_img)
    style_img = os.path.join(argoptions.style_img_dir, argoptions.style_img)

    with tf.variable_scope('input') as scope:
        # use variable instead of placeholder because we're training the intial image to make it
        # look like both the content image and the style image
        input_image = tf.Variable(np.zeros([1, argoptions.out_height, argoptions.out_width, 3]), dtype=tf.float32)
    
    utils.maybe_download(argoptions.model_link, argoptions.model_weights, argoptions.model_size)
    model = vgg.load_vgg(argoptions.model_weights, input_image,argoptions.pooling_type)
    model['global_step'] = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    content_image = utils.get_resized_image(content_img, argoptions.out_height, argoptions.out_width)
    content_image = content_image - MEAN_PIXELS
    style_image = utils.get_resized_image(style_img,argoptions.out_height, argoptions.out_width)
    style_image = style_image - MEAN_PIXELS

    model['content_loss'], model['style_loss'], model['total_loss'] = _create_losses(model, input_image, content_image, style_image)
    if(argoptions.optimizer=='adam'):
        model['optimizer'] = tf.train.AdamOptimizer(argoptions.lr).minimize(model['total_loss'],global_step=model['global_step'])   
    else:
        model['optimizer'] = tf.contrib.opt.ScipyOptimizerInterface(model['total_loss'], method='L-BFGS-B',
          options={'maxiter': argoptions.max_iterations,'disp': argoptions.print_iterations})        
    model['summary_op'] = _create_summary(model)

    initial_image = utils.generate_noise_image(content_image, argoptions.out_height, argoptions.out_width, argoptions.noise_ratio)
    train(model, input_image, initial_image)

if __name__ == '__main__':
    main()
