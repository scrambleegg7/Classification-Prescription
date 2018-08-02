#
import sys
import numpy as np
from PIL import Image

import cv2
import os
import logging
import tensorflow as tf
from glob import glob

import matplotlib.pyplot as plt

from scipy.misc import imread
from scipy.misc import imresize

import shutil

#import fcn8_vgg as fcn8_vgg

from vgg19_finetune_classify import Vgg19


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

def build_vgg(batch_images,train_mode):

    vgg = Vgg19(vgg19_npy_path="./vgg19.npy", trainable=False, dropout=0.5)
    with tf.name_scope("content_vgg"):
        vgg.build(batch_images,NUM_CATEGORY=2, train_mode=train_mode, SKIP_LAYER="fc8_tune" )

    return vgg

def proc():

    num_classes = 2
    BATCH_SIZE = 8
    EPOCHS = 2
    image_shape = (224, 224)

    data_folder="/Users/donchan/Documents/mydata/miyuki/camera"
    data_folder = "/volumes/m1124/FTP/072316"
    move_folder = "/volumes/m1124/DeskImage/prescription"

    prescript_image_paths = glob(os.path.join(data_folder, '*.jpg'))   
    #other_image_paths = glob(os.path.join(data_folder,"None", '*.jpg'))    
    logging.info("total images under " + data_folder + " " + str( len(prescript_image_paths)  ) )

    # placeholder for input images
    #batch_images = tf.placeholder("float")
    batch_images = tf.placeholder("float", shape=[None, 224, 224, 3], name="inputBatchImages")    
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    train_mode = tf.placeholder( "bool", name="train_mode"  )
    vgg_model = build_vgg(batch_images, train_mode)

    probs = vgg_model.prob

    image_ops = lambda x:imread(x)
    resize_ops = lambda x:imresize(x, image_shape ,  interp="lanczos")
    crop_img_ops = lambda x:x[:,200:1200,:]


    #with tf.Session() as sess:
    sess = tf.Session()

    saver = tf.train.Saver( max_to_keep = 3)
    ckpt = tf.train.get_checkpoint_state('./save_models/')
    if ckpt: # if checkpoint found
        last_model = ckpt.model_checkpoint_path # path for last saved model
        logging.info("load model file.." + last_model)
        saver.restore(sess, last_model) # restore all parameters


        for batch_i in range(0, len( prescript_image_paths ), BATCH_SIZE ):

            image_path = np.array( prescript_image_paths )[batch_i:batch_i+BATCH_SIZE]
            prescript_images = list( map( image_ops,image_path )   )
            prescript_images = list( map(   crop_img_ops,  prescript_images )   )
            prescript_images = list( map( resize_ops,prescript_images )  )

            image = np.array(prescript_images)

            feed_dict = {
                train_mode: False,
                batch_images: image
            }

            vgg_probs = sess.run( probs,  feed_dict=feed_dict )
            result = np.argmax( vgg_probs, axis=1  )
            print(result)

            idx = np.array( range( len(image_path) )  )
            file_idx = idx[ result == 1 ]
            prescript_file_list = image_path[file_idx]

            logging.info("prescription image %s" , (prescript_file_list,)  )

            for f in prescript_file_list:

                baseFilename = os.path.basename(f)
                copy_file = os.path.join(move_folder,baseFilename)
                shutil.copy( f, copy_file  )


def main():
    proc()


if __name__ == "__main__":
    main()