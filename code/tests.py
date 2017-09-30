import os
import glob

from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import layers, models

from utils import scoring_utils
from utils import plotting_tools 
from utils import model_tools


from model_training import fcn_model


#### Build model ####

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

model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy')


#### Test on Pics ####
list_of_files = glob.glob('../data/weights/*') # * means all if need specific format then *.csv
# Last created file
weight_file_name = max(list_of_files, key=os.path.getctime)

model_tools.load_network(weight_file_name)

run_num = 'run_1'


val_with_targ, pred_with_targ = model_tools.write_predictions_grade_set(model,
                                        run_num,'patrol_with_targ', 'sample_evaluation_data') 

val_no_targ, pred_no_targ = model_tools.write_predictions_grade_set(model, 
                                        run_num,'patrol_non_targ', 'sample_evaluation_data') 

val_following, pred_following = model_tools.write_predictions_grade_set(model,
                                        run_num,'following_images', 'sample_evaluation_data')


print("Images while following the target")
im_files = plotting_tools.get_im_file_sample('sample_evaluation_data','following_images', run_num) 
for i in range(3):
    im_tuple = plotting_tools.load_images(im_files[i])
    plotting_tools.show_images(im_tuple)
    
    
print("Images while at patrol without target")
im_files = plotting_tools.get_im_file_sample('sample_evaluation_data','patrol_non_targ', run_num) 
for i in range(3):
    im_tuple = plotting_tools.load_images(im_files[i])
    plotting_tools.show_images(im_tuple)
 
    
print("Images while at patrol with target")
im_files = plotting_tools.get_im_file_sample('sample_evaluation_data','patrol_with_targ', run_num) 
for i in range(3):
    im_tuple = plotting_tools.load_images(im_files[i])
    plotting_tools.show_images(im_tuple)



#### Evaluate ####

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