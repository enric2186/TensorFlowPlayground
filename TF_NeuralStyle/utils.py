""" Utils needed for the implementation of the paper "A Neural Algorithm of Artistic Style"
by Gatys et al. in TensorFlow.

Author: Chip Huyen (huyenn@stanford.edu)
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
For more details, please read the assignment handout:
http://web.stanford.edu/class/cs20si/assignments/a2.pdf
"""
#from __future__ import print_function
from PIL import Image, ImageOps
import os
import numpy as np
import scipy.misc
import argparse 
import progressbar
from six.moves import urllib

widgets = [
    'Test: ', progressbar.Percentage(),
    ' ', progressbar.Bar(marker=progressbar.RotatingMarker()),
    ' ', progressbar.ETA(),
    ' ', progressbar.FileTransferSpeed(),
]

def parse_args():

  desc = "TensorFlow implementation of 'A Neural Algorithm for Artisitc Style'"  
  parser = argparse.ArgumentParser(description=desc)

  parser.add_argument('--img_name', type=str, 
    default='result',
    help='Filename of the output image.')

  parser.add_argument('--style_img', type=str,
    help='Filename of the style image (example: starry-night.jpg)')#,required=True)
  
  parser.add_argument('--style_img_weights', nargs='+', type=float,
    default=[1.0],
    help='Interpolation weights of the style image. (example: 0.5 0.5)')
  
  parser.add_argument('--content_img', type=str,
    help='Filename of the content image (example: lion.jpg)')#,required=True) 

  parser.add_argument('--style_img_dir', type=str,
    default='.\styles',
    help='Directory path to the style image. (default: %(default)s)')

  parser.add_argument('--content_img_dir', type=str,
    default='.\content',
    help='Directory path to the content image. (default: %(default)s)')
  
  parser.add_argument('--init_img_type', type=str, 
    default='content',
    choices=['random', 'content', 'style'], 
    help='Image used to initialize the network. (default: %(default)s)')
  
  parser.add_argument('--out_height', type=int, 
    default=300,
    help='Maximum height of the output images. (default: %(default)s)')
   
  parser.add_argument('--out_width', type=int, 
    default=400,
    help='Maximum width  of the output images. (default: %(default)s)')
   
  parser.add_argument('--content_weight', type=float, 
    default=5e0,
    help='Weight for the content loss function. (default: %(default)s)')
  
  parser.add_argument('--style_weight', type=float, 
    default=1e4,
    help='Weight for the style loss function. (default: %(default)s)')
  
  parser.add_argument('--tv_weight', type=float, 
    default=1e-3,
    help='Weight for the total variational loss function. Set small (e.g. 1e-3). (default: %(default)s)')

  parser.add_argument('--temporal_weight', type=float, 
    default=2e2,
    help='Weight for the temporal loss function. (default: %(default)s)')

  parser.add_argument('--content_loss_function', type=int,
    default=1,
    choices=[1, 2, 3],
    help='Different constants for the content layer loss function. (default: %(default)s)')
  
  parser.add_argument('--content_layers', nargs='+', type=str, 
    default='conv4_2',
    help='VGG19 layers used for the content image. (default: %(default)s)')
    
  parser.add_argument('--content_layer_weights', nargs='+', type=float, 
    default=[1.0], 
    help='Contributions (weights) of each content layer to loss. (default: %(default)s)')
    
  parser.add_argument('--style_layers', nargs='+', type=str,
   # default=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'],
    default = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
    help='VGG19 layers used for the style image. (default: %(default)s)')
    
  parser.add_argument('--style_layer_weights', nargs='+', type=float, 
    default=[0.5, 1.0, 1.5, 3.0, 4.0],
    help='Contributions (weights) of each style layer to loss. (default: %(default)s)')
    
  parser.add_argument('--original_colors', action='store_true',
    help='Transfer the style but not the colors.')

  parser.add_argument('--color_convert_type', type=str,
    default='yuv',
    choices=['yuv', 'ycrcb', 'luv', 'lab'],
    help='Color space for conversion to original colors (default: %(default)s)')

  parser.add_argument('--color_convert_time', type=str,
    default='after',
    choices=['after', 'before'],
    help='Time (before or after) to convert to original colors (default: %(default)s)')
    
  parser.add_argument('--noise_ratio', type=float, 
    default=0.6, 
    help="Interpolation value between the content image and noise image if the network is initialized with 'random'.")
    
  parser.add_argument('--style_mask', action='store_true',
    help='Transfer the style to masked regions.')

  parser.add_argument('--style_mask_img', type=str, 
    default=None,
    help='Filenames of the style mask image (example: face_mask.png) (default: %(default)s)')
  
  parser.add_argument('--seed', type=int, 
    default=0,
    help='Seed for the random number generator. (default: %(default)s)')
  
  parser.add_argument('--model_link', type=str, 
    default='http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat',
    help='Download link for VGG-19 network.')
  
  parser.add_argument('--model_size', type=str, 
    default='534904783',
    help='Download bytes size for VGG-19 network link.')
    
  parser.add_argument('--model_weights', type=str, 
    default='imagenet-vgg-verydeep-19.mat',
    help='Weights and biases of the VGG-19 network.')
  
  parser.add_argument('--pooling_type', type=str,
    default='max',
    choices=['avg', 'max'],
    help='Type of pooling in convolutional neural network. (default: %(default)s)')
  
  parser.add_argument('--img_output_dir', type=str, 
    default='.\image_output',
    help='Relative or absolute directory path to output image and data.')
  
  # optimizations
  parser.add_argument('--optimizer', type=str, 
    default='adam',
    choices=['lbfgs', 'adam'],
    help='Loss minimization optimizer.  L-BFGS gives better results.  Adam uses less memory. (default|recommended: %(default)s)')
  
  parser.add_argument('--lr', type=float, 
    default=2e0, 
    help='Learning rate parameter for the Adam optimizer. (default: %(default)s)')
  
  parser.add_argument('--max_iter', type=int, 
    default=1000,
    help='Max number of iterations for the Adam or L-BFGS optimizer. (default: %(default)s)')
    
  parser.add_argument('--print_iterations', type=int, 
    default=50,
    help='Number of iterations between optimizer print statements. (default: %(default)s)')
    
  args = parser.parse_args()

  # normalize weights
  # args.style_layer_weights   = normalize(args.style_layer_weights)
  # args.content_layer_weights = normalize(args.content_layer_weights)
  # args.style_img_weights    = normalize(args.style_img_weights)

  return args

def normalize(weights):
  denom = sum(weights)
  if denom > 0.:
    return [float(i) / denom for i in weights]
  else: return [0.] * len(weights)
     
def get_resized_image(img_path, height, width, save=True):
    image = Image.open(img_path)
    image = ImageOps.fit(image, (width, height), Image.ANTIALIAS)
    if save:
        image_dirs = img_path.split('\\')
        image_dirs[-1] = 'resized_' + image_dirs[-1]
        out_path = '\\'.join(image_dirs)
        if not os.path.exists(out_path):
            image.save(out_path)
    image = np.asarray(image, np.float32)
    return np.expand_dims(image, 0)

def generate_noise_image(content_image, height, width, noise_ratio=0.6):
    noise_image = np.random.uniform(-20, 20, 
                                    (1, height, width, 3)).astype(np.float32)
    return noise_image * noise_ratio + content_image * (1 - noise_ratio)

def save_image(path, image):
    # Output should add back the mean pixels we subtracted at the beginning
    image = image[0] # the image
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)
    
def dlProgress(count, blockSize, totalSize):
    if pbar.max_value is None:
        pbar.max_value = totalSize
        pbar.start()

    pbar.update(min(count*blockSize, totalSize))
    
def maybe_download(download_link, file_name, expected_bytes):
    ### Download the pretrained VGG-19 model if it's not already downloaded
    if os.path.exists(file_name):
        print("VGG-19 model already exists")
        return
    print("Downloading the VGG pre-trained model. Please wait ...")
    global pbar
    pbar = progressbar.ProgressBar(widgets=widgets, max_value=expected_bytes)
    file_name, _ = urllib.request.urlretrieve(download_link, file_name,reporthook=dlProgress)
    pbar.finish()
    file_stat = os.stat(file_name)
    if file_stat.st_size == expected_bytes:
        print('Download complete', file_name)
    else:
        raise Exception('File '+file_name+'might be corrupted. Try wget http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat.')
    