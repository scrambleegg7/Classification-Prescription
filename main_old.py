
import os
import scipy as scp
import scipy.misc

import numpy as np
import logging
import tensorflow as tf
import sys


import fcn8_vgg_imp as fcn8_vgg

import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.utils import shuffle


# data augmentation 
from PIL import Image


from sklearn.model_selection import StratifiedShuffleSplit

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

num_classes = 4


def softmax_logits_loss(logits, labels, num_classes, vehicle_recall, head=None):
    """Calculate the loss from the logits and the labels.

    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
          Use vgg_fcn.upscore as logits.
      labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
          The ground truth of your data.
      head: numpy array - [num_classes]
          Weighting the loss of each class
          Optional: Prioritize some classes

    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, num_classes))
        epsilon = tf.constant(value=1e-4)
        labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))

        softmax = tf.nn.softmax(logits) + epsilon

        if head is not None:
            cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax),
                                           head), reduction_indices=[1])
        else:
            cross_entropy = -tf.reduce_sum(
                labels * tf.log(softmax), reduction_indices=[1])

        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='xentropy_mean')

        #
        # add vehicle recall penalty on loss 
        #
        #cross_entropy_mean = tf.add( (vehicle_recall * 2. ) , cross_entropy_mean )
        
        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss



def build_vgg(batchSize, batch_images, keep_prob, num_classes = 3):

    #vgg_fcn = fcn16_vgg.FCN16VGG()
    vgg_fcn = fcn8_vgg.FCN2VGG(batchSize, statsFile=False, enableTensorboardVisualization=False, vgg16_npy_path=None)  # initialize FCN8
    
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(batch_images, keep_prob,  num_classes= num_classes, random_init_fc8=True ,  debug=True )

    return vgg_fcn

def optimize_weights(nn_last_layer, correct_label, learning_rate, num_classes=4):

    #
    # correct_label  -  batch x m x n 
    #
    # nn_last_layer -> orig_size_logits 
    # 

    # batch x m x 
    #correct_label = tf.expand_dims(  correct_label, -1   )

    # build one hot code from  0 1 2 single flat label..
    with tf.name_scope('label_encode'):
        correct_label = tf.one_hot(correct_label, depth=num_classes)
    


    logits_flat = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label_flat = tf.reshape(correct_label, (-1,num_classes))
   
    #
    # please setup class weights after running classWeights.py
    # result is displayed on the screen
    #

    class_weights = tf.constant( [ [ 0.36128865 , 1.     ,     1.14426048 ] ] )
    #class_weights = tf.constant( [ [ 0.7629334  , 1.232497,  1.438815,  0.8412] ] ) # VehicleRGB
    #class_weights = tf.constant( [ [  0.79306502 , 1.22315188 , 1.39748502 , 0.84570885 ] ] ) # RotVehicleRGB
    
    weights = tf.reduce_sum(class_weights *  tf.cast(  correct_label_flat , tf.float32   )  , axis=1)

    # create accuracy
    pred_up = tf.argmax(nn_last_layer, axis=3)
    correct_label_up = tf.argmax(correct_label,axis=3)
    #correct_pred = tf.equal( pred_up, correct_label_up )
    #accuracy = tf.reduce_mean( tf.cast(  correct_pred, tf.float32  )  )


    with tf.name_scope('custom_score'):
        gt_label_road =   tf.equal( correct_label_up, 1  )
        logits_road =     tf.equal( pred_up, 1  )

        gt_label_vehicle =   tf.equal( correct_label_up, 2  )
        logits_vehicle =     tf.equal( pred_up, 2  )


        #vehicle_fn, update_fn_vehicle = tf.metrics.false_negatives( predictions=logits_vehicle, labels=gt_label_vehicle )


        P_road, update_p_road = tf.metrics.precision(predictions = logits_road, labels = gt_label_road)
        R_road, update_r_road = tf.metrics.recall(predictions = logits_road, labels = gt_label_road)

        P_vehicle, update_p_vehicle = tf.metrics.precision(predictions = logits_vehicle, labels = gt_label_vehicle)
        R_vehicle, update_r_vehicle = tf.metrics.recall(predictions = logits_vehicle, labels = gt_label_vehicle)

        #
        # if none of cars found on image, 
        P_vehicle = tf.cond( tf.equal( P_vehicle , 0)  ,lambda:0., lambda:P_vehicle   )
        R_vehicle = tf.cond( tf.equal( P_vehicle , 0)  ,lambda:0., lambda:R_vehicle   )
        
        road_alpha = 0.5 ** 2
        score_road = (1. + road_alpha) * (P_road * R_road)/( road_alpha * P_road + R_road)

        vehicle_alpha = 2. ** 2
        score_vehicle = (1. + vehicle_alpha) * (P_vehicle * R_vehicle)/( vehicle_alpha * P_vehicle + R_vehicle)

        average_score = ( score_road + score_vehicle) / 2. 
        loss_average_score = 1. - average_score


    #var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"content_vgg")
    #cost = tf.reduce_mean(loss_average_score)
    #grads = tf.gradients(cost, var)
    #gradients = list(zip(grads, var))

    #opt = tf.train.GradientDescentOptimizer(learning_rate= learning_rate)
    #g_and_v = opt.compute_gradients(cost, var)

    #p = 1.
    #eta = opt._learning_rate
    #my_grads_and_vars = [(g-(1/eta)*p, v) for g, v in grads_and_vars]
    #train_op = opt.apply_gradients(grads_and_vars=g_and_v)


    #print(gradients)
    # create loss function.
    #cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits_flat, labels= correct_label_flat))
    # calculates unweighted_losses
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits= logits_flat, labels= correct_label_flat)
    #softmax_loss = softmax_logits_loss(nn_last_layer, correct_label, num_classes, (1. - R_vehicle) ,  head=  class_weights )

    # apply the weights, relying on broadcasting of the multiplication
    #weighted_losses = unweighted_losses * weights
    cross_entropy_loss = tf.reduce_mean(unweighted_losses + (1. - score_vehicle) )
    
    # Define optimizer. Adam in this case to have variable learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
    # Apply optimizer to the loss function.
    #train_op = optimizer.minimize(cross_entropy_loss)    
    train_op = optimizer.minimize( cross_entropy_loss )

    return  train_op, cross_entropy_loss, score_road , score_vehicle, \
            tf.group( update_p_road,update_p_vehicle, update_r_road, update_r_vehicle  )

def optimize(nn_last_layer, correct_label, learning_rate, num_classes=4):

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label_flat = tf.reshape(correct_label, (-1,num_classes))

    # create accuracy
    pred_up = tf.argmax(nn_last_layer, axis=3)
    correct_label_up = tf.argmax(correct_label,axis=3)
    correct_pred = tf.equal( pred_up, correct_label_up )
    accuracy = tf.reduce_mean( tf.cast(  correct_pred, tf.float32  )  )

    # create loss function.
    #cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= correct_label_flat))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= correct_label_flat))
    # Define optimizer. Adam in this case to have variable learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
    # Apply optimizer to the loss function.
    train_op = optimizer.minimize(cross_entropy_loss)    

    return logits, train_op, cross_entropy_loss, accuracy

def train_nn(lyftdataSet , sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, score_road, score_vehicle, update_ops, 
             training_steps_per_epch, valid_steps_per_epch):
    """
    Train neural network and logging.info out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    logging.info("- " * 40)
    logging.info("training steps per epoch: : %d " % training_steps_per_epch )
    logging.info("validation steps per epoch : %d " % valid_steps_per_epch )



    saver = tf.train.Saver( max_to_keep = 3)
    model_filename = "./save_models_2/fcn_vgg8-imp"
    ckpt = tf.train.get_checkpoint_state('./save_models_2/')
    if ckpt: # if checkpoint found
        last_model = ckpt.model_checkpoint_path # path for last saved model

        load_step = int(os.path.basename(last_model).split('-')[2])
        load_step += 1

        logging.info("load model file.." + last_model + "step : "  + str( load_step ))
        saver.restore(sess, last_model) # restore all parameters

        init_local = tf.local_variables_initializer()
        sess.run( [init_local])

    else: # if model NOT found 
        logging.info("- " * 40)
        logging.info(" model NOT found, all variables are initialized.")

        # initialize for global and local    
        init = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()

        sess.run( [init, init_local])
        load_step = 0

    logging.info('Starting training with cross_entropy... for {} epochs'.format(epochs))
    
    #logging.info()

    # use stratifiedshuffleSplit for making balanced data of car binary number 
    car_number_label_dict = lyftdataSet.carNumLabeling()

    # X -- image file names 
    # y = label for car binary numbers ( 1 : bin < 2000 , 2 : 2000 <= bin < 10000 , 3: bin >= 10000  )
    X  = np.array( [ k for k in car_number_label_dict.keys()    ] )
    y  = np.array( [ v for v in car_number_label_dict.values()    ] )
    #
    logging.info('total length of image train + image test  {} '.format( len(X)))

    # split data with number of epochs
    #
    sss = StratifiedShuffleSplit(n_splits= epochs, test_size= 0.15, random_state=0)

    for epoch, ( train_idx, test_idx) in enumerate( sss.split(X,y) ):

        epoch = epoch + load_step

        logging.info("- " * 40)
        logging.info('* Epoch : {}'.format(epoch + 1))
        loss_log = []
        accs_log = []

        road_log = []
        vehicle_log = []

        for idx,( image, label) in enumerate(get_batches_fn(X, train_idx,  batch_size) ):



            #logging.info( " image shape %r  label shape %r " % ( ( image.shape,  ) , ( label.shape,  )  )  )
            feed_dict = { 
                            input_image: image,
                            correct_label: label,
                            keep_prob: 0.4,
                            learning_rate: 1e-5 # learning rate should be checked 0.0001 IS NOT right.
                            #learning_rate: 1e-6 # learning rate should be checked 0.0001 IS NOT right.
                        }

            # road score 
            _ = sess.run(  [update_ops] , feed_dict = feed_dict)
            score_road_, score_vehicle_ = sess.run( [score_road, score_vehicle ], feed_dict=feed_dict)
            _, loss = sess.run([train_op, cross_entropy_loss], 
                                feed_dict=feed_dict)

            # for test purpose , accuracy result is out
            #logging.info("")
            #logging.info(" ---- accuracy detailed by step ------")
            #logging.info("TRAIN loss: %.4f steps:%d" % ( loss, idx )  )
            #logging.info(" f1 score / road :%.4f vehicle : %.4f "   % (score_road_ , score_vehicle_ ) )

            #logging.info("")
            #logging.info("vehicle false-negative : %.4f " %  sess.run( vehicle_fn, feed_dict = feed_dict ) )

            if idx % 100 == 0 and idx != 0:
                logging.info("TRAIN loss: %.4f  steps:%d" % ( loss, idx )  )
                logging.info(" f1 score / road :%.4f vehicle : %.4f "   % (score_road_ , score_vehicle_ ) )
                logging.info(" Average f1 score : %.4f "   %   ( (score_road_ + score_vehicle_ ) / 2. ) )
 
            loss_log.append(loss)
            #accs_log.append(acc)
            road_log.append(score_road_)
            vehicle_log.append(score_vehicle_)

        #print(loss_log)
        logging.info(" -- training --")
        logging.info("* average loss : %.4f  per %d steps" % ( np.mean( loss_log  ) , idx )   )
        logging.info("* average road score : %.4f  average vehicle score : %.4f per %d steps" % ( np.mean( road_log ) , np.mean(vehicle_log) , idx )   )
        logging.info(" Average f1 score : %.4f "   % ( (  np.mean(road_log) + np.mean(vehicle_log)   )/ 2. )  )


        loss_log = []
        accs_log = []
        road_log = []
        vehicle_log = []

        for idx,( image, label) in enumerate( get_batches_fn( X, test_idx,  batch_size, mode="test") ):

            feed_dict = { 
                            input_image: image,
                            correct_label: label,
                            keep_prob: 1.,
                            learning_rate: 1e-6
                            #learning_rate: 1e-6
                        }

            _ = sess.run(  [update_ops] , feed_dict = feed_dict)
            score_road_, score_vehicle_ = sess.run( [score_road, score_vehicle ], feed_dict=feed_dict)
            _, loss = sess.run([train_op, cross_entropy_loss], 
                                feed_dict=feed_dict)
            loss_log.append(loss)
            #accs_log.append(acc)
            road_log.append(score_road_)
            vehicle_log.append(score_vehicle_)
        #print(loss_log)
        logging.info(" -- Validation --")
        logging.info("* average loss : %.4f  per %d steps" % ( np.mean( loss_log  ) , idx )   )
        logging.info("* average road score : %.4f  average vehicle score : %.4f per %d steps" % ( np.mean( road_log ) , np.mean(vehicle_log) , idx )   )
        #print()
        logging.info(" model saved..")
        saver.save(sess, model_filename, global_step=epoch)

def proc1():

    num_classes = 2
    BATCH_SIZE = 1
    EPOCHS = 10
    image_shape = (224, 224)
    orig_shape = (600, 800)
    data_folder = "../data/myTrain/"

    # placeholder for input images
    #batch_images = tf.placeholder("float")
    batch_images = tf.placeholder("float", shape=[None, 224, 224, 3], name="inputBatchImages")    
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    vgg_fcn = build_vgg(BATCH_SIZE, batch_images, keep_prob, num_classes )



def proc2():

    num_classes = 2
    BATCH_SIZE = 1
    EPOCHS = 48
    image_shape = (224, 224)
    orig_shape = (600, 800)
    data_folder = "../data/myTrain/"
    #data_folder = "../data/Carla_data/LowCarF/"
    
    ftype = "Camera"

    # placeholder for input images
    #batch_images = tf.placeholder("float")
    batch_images = tf.placeholder("float", shape=[None, 224, 224, 3], name="inputBatchImages")
    
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    vgg_fcn = build_vgg(BATCH_SIZE, batch_images, keep_prob, num_classes )



    # call LyftData Model
    lyftdataSet = LyftDataSet(data_folder=data_folder , ftype=ftype,  batch_size=BATCH_SIZE)

    training_steps_per_epch = lyftdataSet.train_size // BATCH_SIZE
    valid_steps_per_epch = lyftdataSet.valid_size // BATCH_SIZE


    with tf.Session() as sess:

        # Other placeholders
        nn_last_layer = vgg_fcn.upscore32_pred
        logging.info("model last layer shape: %s" %  (  nn_last_layer.get_shape().as_list()  ,) )

        
        #correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        correct_label = tf.placeholder(tf.int32,  name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')


        #print( sess.run( [  nn_last_layer ]  ) )
        #logits, train_op, cross_entropy_loss, accuracy = optimize(nn_last_layer,correct_label,learning_rate,num_classes)
        train_op, cross_entropy_loss, score_road, score_vehicle, update_ops = optimize_weights(nn_last_layer,correct_label,learning_rate,num_classes)
        
        logging.info("- " * 40)
        logging.info(" training start..")
        get_batches_fn = gen_batch_function_v2(lyftdataSet,  data_folder,image_shape, BATCH_SIZE ,  ftype)    

        train_nn( lyftdataSet,  sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, batch_images,
             correct_label, keep_prob, learning_rate, score_road, score_vehicle, update_ops, 
             training_steps_per_epch, valid_steps_per_epch)
        
        
def main():
    proc1()

if __name__ == "__main__":
    main()
    
