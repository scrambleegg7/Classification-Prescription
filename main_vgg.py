import os
import numpy as np
import logging
import tensorflow as tf
import sys

#from vgg16 import Vgg16

from vgg19_finetune_classify import Vgg19
from DeskImgDataSet import DeskImgDataSet
from glob import glob
import matplotlib.pyplot as plt 

from sklearn.utils import shuffle
from scipy.misc import imread

# data augmentation 
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

num_classes = 2


def gen_batch_function_v2( deskImgdataSet,  data_folder, image_shape=(224,224), batch_size=4):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """

    # X - image filename with strastified splitter
    # train_idx - train_idx with split
    # test_idx - test_idx with split

    def get_batches_fn(X, y,  data_idx, batch_size , mode="train"):


        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths_length = len(data_idx)
        logging.info("image path length: %s" %  (  image_paths_length  ,) )        
            
        for batch_i in range(0, image_paths_length, batch_size):

            #print("generator batch index ", batch_i, image_paths_length )
            select_idx = data_idx[ batch_i:batch_i+batch_size]
            #logging.info("selected idx from image paths  %s", (select_idx,)   )

            images = deskImgdataSet.batch_next( X[ select_idx ], image_shape)
            labels = y[select_idx]

            yield np.array(images), np.array( labels )

    return get_batches_fn

def optimize_weights(nn_last_layer,correct_label,learning_rate,num_classes):

    with tf.name_scope('label_encode'):
        correct_label = tf.one_hot(correct_label, depth=num_classes)

    
    logits_flat = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label_flat = tf.reshape(correct_label, (-1,num_classes))

    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits= logits_flat, labels= correct_label_flat)
    cross_entropy_loss = tf.reduce_mean(unweighted_losses )


    pred_up = tf.argmax(nn_last_layer, axis=1)
    correct_label_up = tf.argmax(correct_label,axis=1)
    correct_pred = tf.equal( pred_up, correct_label_up )
    accuracy = tf.reduce_mean( tf.cast(  correct_pred, tf.float32  )  )

    
    # Define optimizer. Adam in this case to have variable learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
    # Apply optimizer to the loss function.
    train_op = optimizer.minimize( cross_entropy_loss )

    return  train_op, cross_entropy_loss, accuracy

def train_nn(deskImageDataSet , sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, accuracy , 
             input_image,
             correct_label, keep_prob, learning_rate, train_mode,
             training_steps_per_epch, valid_steps_per_epch):


    logging.info("- " * 40)
    logging.info("training steps per epoch: : %d " % training_steps_per_epch )
    logging.info("validation steps per epoch : %d " % valid_steps_per_epch )


    saver = tf.train.Saver( max_to_keep = 3)
    model_filename = "./save_models/vgg16"
    ckpt = tf.train.get_checkpoint_state('./save_models/')
    if ckpt: # if checkpoint found
        last_model = ckpt.model_checkpoint_path # path for last saved model

        #load_step = int(os.path.basename(last_model).split('-')[2])
        load_step = 1

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


    X  = deskImageDataSet.images
    y  = deskImageDataSet.labels

    logging.info('top 10 X : %s ' % X[:10] )
    logging.info('top 10 y : %s ' % y[:10] )

    sss = StratifiedShuffleSplit(n_splits= epochs, test_size= 0.15, random_state=0)


    for epoch, ( train_idx, test_idx) in enumerate( sss.split(X,y) ):

        logging.info("- " * 40)
        logging.info('* Epoch : {}'.format(epoch + 1))
        loss_log = []
        accs_log = []

        for idx,( image, label) in enumerate(get_batches_fn(X, y, train_idx,  batch_size) ):

            #logging.info("idx : %d" % idx)
            #logging.info("image shape :  %s" % (image.shape,) ) 
            #logging.info("label shape :  %s" % (label.shape,) )        

            feed_dict = {
                train_mode: True,
                input_image: image,
                correct_label: label,
                keep_prob: 0.5,
                learning_rate: 1e-4 # learning rate should be checked 0.0001 IS NOT right.
                #learning_rate: 1e-6 # learning rate should be checked 0.0001 IS NOT right.
            }

            # road score 
            _, loss, acc = sess.run([train_op, cross_entropy_loss, accuracy  ], 
                                feed_dict=feed_dict)
            
            loss_log.append( loss )
            accs_log.append( acc ) 

            #logging.info("loss :  %.4f" % (loss,) )        
            #logging.info("accuracy :  %.4f" % (acc,) )
            if idx % 10 == 0 and idx > 0:
                logging.info("loss : %.4f on idx : %d" % (  np.mean( loss_log ) , idx   ) )
                logging.info("acc : %.4f on idx : %d" % (  np.mean( accs_log ) , idx   ) )

        #    
        # for validation
        #
        loss_log = []
        accs_log = []
        logging.info("- " * 40)
        logging.info("validation start .... " )
            
        for idx,( image, label) in enumerate(get_batches_fn(X, y, test_idx,  batch_size) ):

            feed_dict = {
                train_mode: False,
                input_image: image,
                correct_label: label,
                keep_prob: 0.5,
                learning_rate: 1e-4 # learning rate should be checked 0.0001 IS NOT right.
                #learning_rate: 1e-6 # learning rate should be checked 0.0001 IS NOT right.
            }

            # road score 
            _, loss, acc = sess.run([train_op, cross_entropy_loss, accuracy  ], 
                                feed_dict=feed_dict)

            loss_log.append( loss )
            accs_log.append( acc )

        logging.info("loss : %.4f on idx : %d" % (  np.mean( loss_log ) , idx   ) )
        logging.info("acc : %.4f on idx : %d" % (  np.mean( accs_log ) , idx   ) )


    logging.info(" model saved..")
    saver.save(sess, model_filename, global_step=epoch)


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
    # placeholder for input images
    #batch_images = tf.placeholder("float")
    batch_images = tf.placeholder("float", shape=[None, 224, 224, 3], name="inputBatchImages")    
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    train_mode = tf.placeholder( "bool", name="train_mode"  )

    vgg_model = build_vgg(batch_images, train_mode)

    deskImageDataSet = DeskImgDataSet(num_class=num_classes)
    training_steps_per_epch = deskImageDataSet.train_size // BATCH_SIZE
    valid_steps_per_epch = deskImageDataSet.valid_size // BATCH_SIZE

    with tf.Session() as sess:

        # Other placeholders
        nn_last_layer = vgg_model.fc8
        logging.info("model last layer shape: %s" %  (  nn_last_layer.get_shape().as_list()  ,) )

        #correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        correct_label = tf.placeholder(tf.int32,  name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        #print( sess.run( [  nn_last_layer ]  ) )
        #logits, train_op, cross_entropy_loss, accuracy = optimize(nn_last_layer,correct_label,learning_rate,num_classes)
        train_op, cross_entropy_loss, accuracy = optimize_weights(nn_last_layer,correct_label,learning_rate,num_classes)
        
        logging.info("- " * 40)
        logging.info(" training start..")
        get_batches_fn = gen_batch_function_v2(deskImageDataSet,  data_folder,image_shape, BATCH_SIZE)    

        train_nn( deskImageDataSet,  sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, accuracy , batch_images,
             correct_label, keep_prob, learning_rate, train_mode,
             training_steps_per_epch, valid_steps_per_epch)


def load_image():

    data_folder = "/Users/donchan/Documents/mydata/miyuki/camera"
    
    orig_image_paths = glob(os.path.join(data_folder,"prescription", '*.jpg'))    
    print( orig_image_paths[:10] )
    print( len(orig_image_paths))

    img = imread( orig_image_paths[0] )
    crop_img = img[:,200:1200,:]
    
def main():
    proc()
    
if __name__ == "__main__":
    main()