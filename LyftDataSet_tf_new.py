import re
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

from ImageAugmentationClass import ImageAugmentationClass

class LyftDataSet(object):

    def __init__(self, data_folder="../data/Train", ftype="Camera", batch_size=32):

        _rgb = ftype + "RGB"
        _seg = ftype + "Seg"

        self.orig_image_paths = glob(os.path.join(data_folder, _rgb, '*.png'))
        #self.orig_image_paths = glob(os.path.join(data_folder, _rgb, '*.png'))[:100]
        self.label_paths = { os.path.basename(path): path
            for path in glob(os.path.join(data_folder, _seg, '*.png'))}
        
        # shuffle image and label paths
        self.image_paths = shuffle(self.orig_image_paths)

        # splitting 85% train / 15% test
        self.train_size = int( len(self.orig_image_paths) * .85 )
        self.valid_size = int( len(self.orig_image_paths) * .15 )
        
        self.valid_paths = self.image_paths[self.train_size:]
        self.image_paths = self.image_paths[:self.train_size]
        #self.valid_label_paths = [ self.label_paths[ os.path.basename(path) ] for path in self.valid_paths   ]

        # set default batch size
        self.batch_size = batch_size

        # singleton class
        self.imageAugmentationClass = ImageAugmentationClass()

        #self.additional_ReadData()

    def additional_ReadData(self, data_folder="../data/lyft_cars"):

        _rgb = "CameraRGB"
        _seg = "CameraSeg"
        print(" read additional data under ", data_folder)

        self.add_image_paths = glob(os.path.join(data_folder, _rgb, '*.png'))
        
        for path in glob(os.path.join(data_folder, _seg, '*.png')):
            self.label_paths[ os.path.basename(path) ] =  path

        print(" length of original paths (base) .. ",   len( self.orig_image_paths ))
        print(" additional length image paths .. ",   len( self.add_image_paths ))

        self.orig_image_paths.extend( self.add_image_paths  )

        print(" length of all image paths (base+additional) .. ",   len( self.orig_image_paths ))

        label_length = len( list(  self.label_paths.values() ) )

        print(" length of all label paths (base+additional) .. ",   label_length)

        self.train_size = int( len(self.orig_image_paths) * .85 )
        self.valid_size = int( len(self.orig_image_paths) * .15 )


    def carNumLabeling(self):

        car_numbers = []
        car_number_label_dict = {}
        label_image_ops = lambda x:imread(x)

        for idx, img in enumerate(self.orig_image_paths):
            if idx % 100 == 0:
                print("%d %s processing.." % (idx,os.path.basename( img ))  )
                
            label_file = self.label_paths[ os.path.basename( img )]
            gt_image = label_image_ops(label_file)

            gt_sky = gt_image[:490,:,0].copy()
            gt_hood = gt_image[490:,:,0].copy()
            gt_hood[ gt_image[490:,:,0] == 10 ] = 0       
            gt_composit = np.vstack( [ gt_sky, gt_hood ])
            gt_image[:,:,0] = gt_composit[:,:]


            # car binary
            binary_car = np.zeros_like( gt_image[:490,:,0] , dtype = np.uint8  )
            binary_hood = np.zeros_like( gt_image[490:,:,0] , dtype = np.uint8  )

            binary_car[ gt_image[:490,:,0] == 10 ] = 1
            binary = np.vstack( [ binary_car, binary_hood ])

            car_number = np.sum(binary)
            
            if car_number < 1:
                #print( "ZERO car", os.path.basename(img) )
                continue
            
            if car_number < 2000 and car_number > 0:
                car_number_label_dict[img] = 1        
            if car_number >= 2000 and car_number < 10000:
                car_number_label_dict[img] = 2
            if car_number >= 10000 :
                car_number_label_dict[img] = 3
            
            car_numbers.append( car_number)        

        return car_number_label_dict
                
    def preprocess_labels(self, label_image):

        # 
        # road ----> 1
        #
        labels_new = np.zeros_like(label_image, dtype=np.uint8)
        # Identify lane marking pixels (label is 6)
        lane_marking_pixels = (label_image[:,:,0] == 6).nonzero()
        # Set lane marking pixels to road (label is 7)
        labels_new[lane_marking_pixels] = 1

        road_marking_pixels = (label_image[:,:,0] == 7).nonzero()
        labels_new[ road_marking_pixels  ] = 1

        #
        # 
        #         
        # Identify all vehicle pixels
        #
        # vehicle -----> 2
        #
        vehicle_pixels = (label_image[:,:,0] == 10).nonzero()
        labels_new[vehicle_pixels] = 2
        
        # Isolate vehicle pixels associated with the hood (y-position > 496)
        hood_indices = (vehicle_pixels[0] >= 496).nonzero()[0]
        hood_pixels = (vehicle_pixels[0][hood_indices], \
                    vehicle_pixels[1][hood_indices])
        # Set hood pixel labels to 0
        labels_new[hood_pixels] = 0
        # Return the preprocessed label image 
        labels_new = labels_new[:,:,0]
        #labels_new = labels_new[:,:,np.newaxis]

        # output is single frame (m x n x 1)
        return labels_new

    def equalizeHist(self, image):

        equ = np.zeros_like(image.copy() , dtype=np.uint8)

        equ[:,:,0] = cv2.equalizeHist(image[:,:,0])
        equ[:,:,1] = cv2.equalizeHist(image[:,:,1])
        equ[:,:,2] = cv2.equalizeHist(image[:,:,2])
    
        return equ

    def batch_next_X(self, X, image_shape):
        

        # 
        # lambda ops
        #
        image_read_op = lambda x:imread(x)
        image_equ_op = lambda x:self.equalizeHist(x)
        image_resize_op = lambda x:imresize(x, image_shape, interp="lanczos")

        lable_resize_op = lambda x:imresize(x, image_shape, interp="lanczos")
        preprocess_labels_ops = lambda x:self.preprocess_labels(x)

        image_augmentation_ops = lambda im,lb:self.imageAugmentation(im,lb) 

        # set filenames

        orig_image_path_list = X
        label_path_list = [self.label_paths[ os.path.basename(p) ] for p in orig_image_path_list]


        # read images
        image_list = list( map(  image_read_op,  orig_image_path_list   ) )
        # read label file path        
        label_list = list( map(  image_read_op, label_path_list   ) )

        #
        # image augmentation ....
        #
        q = list( map(  image_augmentation_ops, image_list, label_list      ))
        image_list = np.array(q)[:,0]
        label_list = np.array(q)[:,1]


        # equalize and resize 
        image_equlizers = list( map( image_equ_op, image_list ) )
        images = list( map( image_resize_op, image_equlizers ) )

        # preprocess for label / defining 0 1 2 for segmentation image label
        gt_images = list( map( preprocess_labels_ops, label_list    ))
        # resize same as image 
        gt_images = list( map( lable_resize_op, gt_images ))


        return images, gt_images

    def imageAugmentation(self,img,lbl):

        #if np.random.random_sample() > 0.5:
        image ,label = self.imageAugmentationClass.choiceAugmentation(img,lbl)
        return image,label
        #else:
        #    return img,lbl

    def vehicleZeroFiles(self):

        print("- " * 40)
        print(" count vehicle binary from Segmentation file..")
        print(" if ZERO vehicles found, filenames listup.")
        label_files = list(self.label_paths.values())
        print( len( label_files ))
        print( label_files[:5])

        image_ops = lambda x:imread(x)[:,:,0]
        label_bins = list(map(image_ops,label_files))
        label_bins = np.array(label_bins)
        print("total binary label shape..",label_bins.shape)

        # convert m x n x 3 (3 channel - none, road, vehicle)
        label_bins = self.one_hot_labels(label_bins,13)
        sum_car_bins = np.zeros_like(label_bins[0] , dtype = np.uint8 )
        
        zeroVehicles = []
        for l in range( len( label_bins ) ):
            label = label_bins[l]    
            car_bins = label[:,:,2] #vehicle
            
            if np.sum(car_bins) < 1:
                print("car not found filename", label_files[l])

                zeroVehicles.append( label_files[l] )
        
        return zeroVehicles


def main():

    lyftDataSet = LyftDataSet()

if __name__ == "__main__":
    main()