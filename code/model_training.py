import os
import tensorflow as tf

from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import layers, models

from utils.separable_conv2d import SeparableConv2DKeras, BilinearUpSampling2D
from utils import data_iterator
from utils import plotting_tools 
from utils import model_tools





def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer

def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, 
                      padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer



def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer



def encoder_block(input_layer, filters, strides):
    
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer

def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    small_ip_layer_up = bilinear_upsample(small_ip_layer)
    
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    output_layer = layers.concatenate([small_ip_layer_up,large_ip_layer])
    
    # TODO Add some number of separable convolution layers
    
    # What?
    
    
    return output_layer


# This function saves miles of code
# inputs - your tensor
# num - how many layers you want in your encode/decode 
#       (each encode & decode, not in total)
# depthmult - base depth is 16 it will be multiplied by depthmult with each layer
#
# return - tensor with original shape processed with encode/decode
#
# all rights are free, copy, use, change...
def encode_decode(inputs,num, depthmult=2, depth_=32, keepProb=0.6):
    arr_encode=[]
    arr_decode=[]
    

    arr_encode.append(encoder_block(inputs, depth_, 2))
    
    depth_ *= depthmult
    
    # Encoder
    for i in range(num):
        arr_encode.append(encoder_block(arr_encode[i], depth_, 2))
        depth_ *= depthmult
    
    # Fully connected convoluiton
    arr_encode[i] = conv2d_batchnorm(arr_encode[i], depth_, kernel_size=1, strides=1)
    arr_decode.append(arr_encode[i])
    
    arr_encode[i] = tf.layers.dropout(arr_encode[i],keepProb)
    
    depth_ /= depthmult
    ii = 0
    
    # Decoder
    while i > 0:
        arr_decode.append(decoder_block(arr_decode[ii], arr_encode[i-1], depth_))
        i -= 1
        ii +=1
        depth_ /= depthmult
    
    arr_encode[-1] = tf.layers.dropout(arr_encode[-1],keepProb)
    
    return decoder_block(arr_decode[-1], inputs, depth_)




# The model
def fcn_model(inputs, num_classes, depth_, keepProb):
    
    x = encode_decode(inputs, 3, depth_=depth_, keepProb=keepProb)
    x = encode_decode(x, 3, depth_=depth_, keepProb=keepProb)
    x = encode_decode(x, 3, depth_=depth_, keepProb=keepProb)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)




if __name__ == "__main__":
    
    ### Hyperparameters ###
    
    # Input size
    image_hw = 160
    image_shape = (image_hw, image_hw, 3)
    inputs = layers.Input(image_shape)
    
    # Classes
    num_classes = 3
    
    # Starting depth of encode/decode
    depth_ = 32
    # Dropout rate
    keepProb = 0.6
    
    # Standard model parameters
    learning_rate = 0.001
    batch_size = 8
    num_epochs = 200
    steps_per_epoch = len([n for n in os.listdir("../data/train/images")]) // batch_size
    validation_steps = len([n for n in os.listdir("../data/train/images")]) // batch_size
    workers = 1
    
    
    # Call fcn_model()
    output_layer = fcn_model(inputs, num_classes, depth_, keepProb)
    
    # Define the Keras model and compile it for training
    model = models.Model(inputs=inputs, outputs=output_layer)
    
    #model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy')
    
    # adagrad is good for benchmark optimizer as it is hyperparameter agnostic
    model.compile(optimizer=keras.optimizers.Adagrad(), loss='categorical_crossentropy')
    
    # Data iterators for loading the training and validation data
    train_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                                   data_folder=os.path.join('..', 'data', 'train'),
                                                   image_shape=image_shape,
                                                   shift_aug=True)
    
    val_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                                 data_folder=os.path.join('..', 'data', 'validation'),
                                                 image_shape=image_shape)
    
    logger_cb = plotting_tools.LoggerPlotter()
    callbacks = [logger_cb]
    
    model.fit_generator(train_iter,
                        steps_per_epoch = steps_per_epoch, # the number of batches per epoch,
                        epochs = num_epochs, # the number of epochs to train for,
                        validation_data = val_iter, # validation iterator
                        validation_steps = validation_steps, # the number of batches to validate on
                        callbacks=callbacks,
                        workers = workers)
    
    
    
    # Save your trained model weights
    from datetime import datetime
    weight_file_name = datetime.now().strftime('model_%H_%M_%d_%m_%Y')
    model_tools.save_network(your_model=model, your_weight_filename=weight_file_name)
    