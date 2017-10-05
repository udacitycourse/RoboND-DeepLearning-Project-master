import os
import glob

from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import layers, models

from utils.separable_conv2d import SeparableConv2DKeras, BilinearUpSampling2D
from utils import data_iterator
from utils import plotting_tools 
from utils import model_tools

from datetime import datetime
from utils import scoring_utils


# convolution + batch norm (required for decoder to work)
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


# upsample
def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer


# just a function to call separable_conv2d_batchnorm, here for scalability
def encoder_block(input_layer, filters, strides):
    
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer

# decoder takes small layer -> upsamples -> concatenate (or conc depending from tensorflow version)
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


### Example
    

# layers encode

# arr_encode[0] = input

# for
    # arr_encode[1] = encode(arr_encode[0])
    # arr_encode[2] = encode(arr_encode[1])
    # ...
    # arr_encode[n] = encode(arr_encode[n-1])

# fully connected
# arr_encode[n] = conv2d_batchnorm(arr_encode[n])

# arr_decode[0] = arr_encode[n]

# layers decode
#for
    # arr_decode[1] = decode(arr_decode[0],arr_encode[n-1])
    # ...
    # arr_decode[nn] = decode(arr_decode[nn-1],arr_encode[n-1])

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
    
    
    depth_ /= depthmult
    ii = 0
    
    # Decoder
    while i > 0:
        arr_decode.append(decoder_block(arr_decode[ii], arr_encode[i-1], depth_))
        i -= 1
        ii +=1
        depth_ /= depthmult
    
    return decoder_block(arr_decode[-1], inputs, depth_)




# The model
def fcn_model(inputs, num_classes, depth_=32, keepProb=0.6):
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    '''
    Basic encode/decode
    
    x0 = encoder_block(inputs, depth_, 2)
    x1 = encoder_block(x0, depth_*2, 2)
    
    #x1_2 = conv2d_batchnorm(x1, depth_*4, kernel_size=1, strides=1)
    
    x2 = encoder_block(x1, depth_*4, 2)
    #x3 = encoder_block(x2, 256, 2)
    
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    x3 = conv2d_batchnorm(x2, depth_*8, kernel_size=1, strides=1)
    
    #x3=layers.Dropout(keepProb)(x3)
    
    
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    #x3 = decoder_block(x4, x2, 256)
    x2 = decoder_block(x3, x1, depth_*4)
    x1 = decoder_block(x2, x0, depth_*2)
    x = decoder_block(x1, inputs, depth_)
    
    #x=layers.Dropout(keepProb)(x)
    inputs=x
    '''
    
    # Encode 1
    x0 = encoder_block(inputs, depth_, 2)
    # Encode 2
    x1 = encoder_block(x0, depth_*2, 2)
    
    # Encode 2.1
    x1_2 = conv2d_batchnorm(x1, depth_*4, kernel_size=1, strides=1)
    
    # Encode 3
    x2 = encoder_block(x1_2, depth_*4, 2)
    
    
    x3 = encoder_block(x2, depth_*8, 2)
    
    # Fully connected
    x4 = conv2d_batchnorm(x3, depth_*16, kernel_size=1, strides=1)
    
    # Decode 1
    x3 = decoder_block(x4, x2, depth_*8)
    # Decode 2
    x2 = decoder_block(x3, x1, depth_*4)
    # Decode 3
    x1 = decoder_block(x2, x0, depth_*2)
    # Decode 4
    x = decoder_block(x1, inputs, depth_)
    
    
    # Decode 4 - update variable
    inputs=x
    
    # Encode 1
    x0 = encoder_block(inputs, depth_, 2)
    # Encode 2
    x1 = encoder_block(x0, depth_*2, 2)
    # Encode 3
    x2 = encoder_block(x1, depth_*4, 2)
    

    # Fully connected
    x3 = conv2d_batchnorm(x2, depth_*8, kernel_size=1, strides=1)
    
    # Decode 1
    x2 = decoder_block(x3, x1_2, depth_*4)
    # Decode 2
    x1 = decoder_block(x2, x0, depth_*2)
    # Decode 3
    x = decoder_block(x1, inputs, depth_)
    
    
    
    ## How to use encode/decode
    
    #x = encode_decode(inputs, 3, depth_=depth_, keepProb=keepProb)
    #x = encode_decode(x, 3, depth_=depth_, keepProb=keepProb)
    #x = encode_decode(x, 3, depth_=depth_, keepProb=keepProb)
    
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
    depth_ = 64
    # Dropout rate
    keepProb = 0.6
    
    # Standard model parameters
    learning_rate = 0.0005
    batch_size = 4
    num_epochs = 20
    steps_per_epoch = len([n for n in os.listdir("../data/train/images")]) // batch_size
    validation_steps = len([n for n in os.listdir("../data/train/images")]) // batch_size
    workers = 1
    
    
    # Call fcn_model()
    output_layer = fcn_model(inputs, num_classes, depth_, keepProb)
    
    # Define the Keras model and compile it for training
    model = models.Model(inputs=inputs, outputs=output_layer)
    
    
    
    
    # Load latest model for retraining
    list_of_files = glob.glob('../data/weights/*') # * means all if need specific format then *.csv
    weight_file_name =  max(list_of_files, key=os.path.getctime)
    weight_file_name = weight_file_name.split("\\")
    model = model_tools.load_network(weight_file_name[-1])
    
    # Compile
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy')
    
    # Current best score
    s = weight_file_name[-1].split('_')
    fg = float(s[0]) 
    arr=[]
    
    # To change evaluation pipeline
    # that way we test after each epoch and save the best
    for i in range(num_epochs):
        
        # Train pics
        train_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                                       data_folder=os.path.join('..', 'data', 'train'),
                                                       image_shape=image_shape,
                                                       shift_aug=True)
            
        # Validation pics
        val_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                                     data_folder=os.path.join('..', 'data', 'validation'),
                                                     image_shape=image_shape)
        # Logs
        logger_cb = plotting_tools.LoggerPlotter()
        callbacks = [logger_cb]
        
        # Fit into the model
        model.fit_generator(train_iter,
                            steps_per_epoch = steps_per_epoch, # the number of batches per epoch,
                            epochs = 1, # the number of epochs to train for,
                            validation_data = val_iter, # validation iterator
                            validation_steps = validation_steps, # the number of batches to validate on
                            callbacks=callbacks,
                            workers = workers)
    
        #### Evaluate ####
        
        # Run provided tests
        
        run_num = 'run_1'
        
        
        val_with_targ, pred_with_targ = model_tools.write_predictions_grade_set(model,
                                                run_num,'patrol_with_targ', 'sample_evaluation_data') 
        
        val_no_targ, pred_no_targ = model_tools.write_predictions_grade_set(model, 
                                                run_num,'patrol_non_targ', 'sample_evaluation_data') 
        
        val_following, pred_following = model_tools.write_predictions_grade_set(model,
                                                run_num,'following_images', 'sample_evaluation_data')
    
        
        print("Quad behind the target")
        true_pos1, false_pos1, false_neg1, iou1 = scoring_utils.score_run_iou(val_following, pred_following)
        
        print("Target not visible")
        true_pos2, false_pos2, false_neg2, iou2 = scoring_utils.score_run_iou(val_no_targ, pred_no_targ)
        
        print("Target far away")
        true_pos3, false_pos3, false_neg3, iou3 = scoring_utils.score_run_iou(val_with_targ, pred_with_targ)
        
        # Sum all the true positives, etc from the three datasets to get a weight for the score
        true_pos = true_pos1 + true_pos2 + true_pos3
        false_pos = false_pos1 + false_pos2 + false_pos3
        false_neg = false_neg1 + false_neg2 + false_neg3
        
        weight = true_pos/(true_pos+false_neg+false_pos)
        #print(weight)
        
        
        # The IoU for the dataset that never includes the hero is excluded from grading
        final_IoU = (iou1 + iou3)/2
        print("IoU no hero - ", final_IoU)
        
        
        # And the final grade score is 
        final_score = final_IoU * weight
        print("Final Grade - ", final_score)
        
        if final_score > fg:
            # Save model with best score
            weight_file_name = str(final_score) + '_' + datetime.now().strftime('model_%H_%M_%d_%m_%Y') + '.h5'
            model_tools.save_network(your_model=model, your_weight_filename=weight_file_name)
            
            
            fg = final_score
            arr = [fg,final_IoU,weight,weight_file_name]
            
        print('Best --- ', fg)
        print('Data --- ', arr)
    
    