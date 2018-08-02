import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2

from scipy.misc import imresize
from scipy.misc import imread

import logging, sys

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


class DeskImgDataSet(object):

    def __init__(self, data_folder="/Users/donchan/Documents/mydata/miyuki/camera", batch_size=32, num_class=2):

        self.num_class = num_class

        self.data_folder = data_folder
        self.prescript_image_paths = glob(os.path.join(data_folder,"prescription", '*.jpg'))   
        self.other_image_paths = glob(os.path.join(data_folder,"None", '*.jpg'))    
        
        self.images = []
        self.labels = []

        prescrpt_length = len( self.prescript_image_paths )
        # convert int for transfering into one hot code
        y_pres = np.ones( prescrpt_length ).astype(int)

        none_length = len( self.other_image_paths )
        # convert int for tranfering into one hot code
        y_none = np.zeros( none_length ).astype(int)

        # build data index for prescription and others        
        total_data_length = prescrpt_length + none_length
        logging.info("total data length: %d" %  (  total_data_length  ,) )

        idx = range(total_data_length)
        idx = shuffle(idx)

        training_len = int( total_data_length * 0.8 )
        validation_len = total_data_length - training_len
        logging.info("traing length: %d" %  (  training_len  ,) )
        logging.info("validation length: %d" %  (  validation_len  ,) )
        logging.info("index length: %d" %  (  len(idx)  ,) )
        
        training_idx = idx[:training_len]
        validation_idx = idx[training_len:]

        self.train_size = training_len
        self.valid_size = validation_len
        
        self.images.extend(  self.prescript_image_paths  )
        self.images.extend(  self.other_image_paths  )

        self.labels.extend( list( y_pres ) )
        self.labels.extend( list( y_none ) )

        # convert numpy array to utilize index search
        self.images = np.array( self.images  )
        self.labels = np.array( self.labels )

        #self.labels = self.one_hot(self.labels)

        #self.traing_image = np.array( self.images )[training_idx]
        #self.traing_label = np.array( self.labels )[training_idx]

        #self.valid_image = np.array( self.images )[validation_idx]
        #self.valid_label = np.array( self.labels )[validation_idx]

    def one_hot(self,label):

        onehot = np.eye(self.num_class)[label]

        return onehot
    
    def equalizeHist(self, image):

        equ = np.zeros_like(image.copy() , dtype=np.uint8)
        equ[:,:,0] = cv2.equalizeHist(image[:,:,0])
        equ[:,:,1] = cv2.equalizeHist(image[:,:,1])
        equ[:,:,2] = cv2.equalizeHist(image[:,:,2])
    
        return equ
    
    def batch_next(self,X,image_shape):

        # 
        # lambda ops
        #
        image_read_op = lambda x:imread(x)
        image_equ_op = lambda x:self.equalizeHist(x)
        image_resize_op = lambda x:imresize(x, image_shape, interp="lanczos")
        crop_img_ops = lambda x:x[:,200:1200,:]


        orig_image_path_list = X

        # read images & crop & resizes
        image_list = list( map(  image_read_op,  orig_image_path_list   ) )
        image_list = list( map(   crop_img_ops,  image_list )   )
        #image_equlizers = list( map( image_equ_op, image_list ) )
        images = list( map( image_resize_op, image_list ) )



        return images