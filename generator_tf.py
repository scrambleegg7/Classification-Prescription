
from __future__ import print_function

import os, sys
import logging

import tensorflow as tf
import numpy as np

from scipy.misc import imresize

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)





def gen_batch_function_v2( lyftdataSet,  data_folder, image_shape=(224,224), batch_size=4, ftype="Camera"):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """

    # X - image filename with strastified splitter
    # train_idx - train_idx with split
    # test_idx - test_idx with split

    def get_batches_fn(X,  data_idx, batch_size,mode="train"):

        #lyftdataSet = LyftDataSet(data_folder=data_folder , ftype=ftype,  batch_size=batch_size)
        #if mode == "train":
        #    logging.info("- " * 40)
        #    logging.info("data shuffle for %s" % mode )
            # shuffled data is stored on image_paths and valid_paths respectively of LyftDataSet
        #    lyftdataSet.shuffleLyftData()

        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths_length = len(data_idx)
            
        for batch_i in range(0, image_paths_length, batch_size):

            #print("generator batch index ", batch_i, image_paths_length )
            select_idx = data_idx[ batch_i:batch_i+batch_size]

            images, gt_images = lyftdataSet.batch_next_v2( X[ select_idx ], image_shape)

            yield np.array(images), np.array( gt_images )

    return get_batches_fn
