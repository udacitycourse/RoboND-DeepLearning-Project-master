# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 12:31:10 2017

@author: moroz
"""


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
images = []

for filename in glob.glob('../data/train/images/*'): #assuming gif
    print(filename)
    im=Image.open(filename)
    images.append(im)


masks = []

for filename in glob.glob('../data/train/images/*.jpeg'): #assuming gif
    im=Image.open(filename)
    masks.append(im)

image_datagen.fit(images, augment=True, seed=seed)
mask_datagen.fit(masks, augment=True, seed=seed)


