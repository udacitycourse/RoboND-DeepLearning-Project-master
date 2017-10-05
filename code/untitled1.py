# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 12:31:10 2017

@author: moroz
"""

'''
###################
from keras.preprocessing.image import ImageDataGenerator
# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     horizontal_flip=True)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
from PIL import Image
import glob
import numpy as np
images = []

for filename in glob.glob('../data/processed_sim_data/train/images/*.jpeg'):
    im=Image.open(filename).transpose(Image.FLIP_LEFT_RIGHT)
    s= filename.split("\\")
    im.save('../data/processed_sim_data/train/images/flipped_' + s[-1])

masks = []

for filename in glob.glob('../data/processed_sim_data/train/masks/*.png'):
    im=Image.open(filename).transpose(Image.FLIP_LEFT_RIGHT)
    s= filename.split("\\")
    im.save('../data/processed_sim_data/train/masks/flipped_' + s[-1])


#image_datagen.fit(images, augment=True, seed=seed)
#mask_datagen.fit(masks, augment=True, seed=seed)


from PIL import Image
import glob
import os

arr=[]
for filename in glob.glob('../data/processed_sim_data/train/images/*.jpeg'): 
    s = filename.split("\\")
    arr.append(s[-1])

i=0
arr_new=[]
for filename in glob.glob('../data/processed_sim_data/train/masks/*.png'): 
    s = filename.split("\\")
    s2 = arr[i].split(".")
    s3 = s2[-2].split("_")
    os.rename('../data/processed_sim_data/train/masks/' + s[-1],'../data/processed_sim_data/train/masks/' + s3[0] + '_' + s3[1] + '_mask_' + s3[2] + '.png')
    i+=1






'''


import glob
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from utils import model_tools
import os

from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import layers, models

data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     horizontal_flip=True)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)


trainset = {}
features = []

for filename in glob.glob('../data/train/images/*.jpeg'):
    s = filename.split('\\')
    
    features.append(cv2.imread('../data/train/images/' + s[-1]))

labels = []

for filename in glob.glob('../data/train/masks/*.png'):
    s = filename.split('\\')
    labels.append(cv2.imread('../data/train/masks/' + s[-1]))


seed = 1
image_datagen.fit(features, augment=True, seed=seed)
mask_datagen.fit(labels, augment=True, seed=seed)

ind=0


image_generator = image_datagen.flow_from_directory('../data/train/images/',
                                                    save_to_dir='../data/processed_sim_data/images/',
                                                    save_prefix='created_' + str(ind), 
                                                    save_format='jpeg',
                                                    seed=seed)

mask_generator = mask_datagen.flow_from_directory('../data/train/masks/',
                                                  save_to_dir='../data/processed_sim_data/masks/',
                                                  save_prefix='created_' + str(ind),
                                                  save_format='png')

train_iter = zip(image_generator, mask_generator)

learning_rate = 0.0005

list_of_files = glob.glob('../data/weights/*') # * means all if need specific format then *.csv
weight_file_name =  max(list_of_files, key=os.path.getctime)
weight_file_name = weight_file_name.split("\\")
model = model_tools.load_network(weight_file_name[-1])

#model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy')

# Current best score
s = weight_file_name[-1].split('_')
fg = float(s[0]) 
arr=[]
num_epochs=1


    
model.fit_generator(train_iter,
                    steps_per_epoch = 200, # the number of batches per epoch,
                    epochs = 3, # the number of epochs to train for,
                    workers = 1)


'''
for i,j in zip(image_datagen.flow_from_directory('../data/train/images/',save_to_dir='../data/processed_sim_data/images/',save_prefix='created_' + str(ind), save_format='jpeg'),
                mask_datagen.flow_from_directory('../data/train/masks/' ,save_to_dir='../data/processed_sim_data/masks/' ,save_prefix='created_' + str(ind), save_format='png')):

#for i,j in zip(image_datagen.flow(np.array(features),save_to_dir='../data/processed_sim_data/images/',save_prefix='created_' + str(ind), save_format='jpeg'),
#               mask_datagen.flow(np.array(labels)),save_to_dir='../data/processed_sim_data/masks/',save_prefix='created_' + str(ind), save_format='png'):
    
        ind+=1
        #print('../data/processed_sim_data/images/' + 'created_' + str(ind) + '.jpeg')
        #cv2.imwrite('../data/processed_sim_data/images/' + 'created_' + str(ind) + '.jpeg',i)
        #cv2.imwrite('../data/processed_sim_data/masks/' + 'created_' + str(ind) + '.png',j)
        if ind > 3:
            break

''' 