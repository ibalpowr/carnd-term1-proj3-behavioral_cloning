import numpy as np
import cv2
import tensorflow as tf
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import random

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import MaxPooling2D, Lambda, Cropping2D
from keras.layers import Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam

# load udacity and recovery datasets

# change directory if necessary
#cd /items/machine_learning-datasets/carnd-term1-project3

log_udacity = pd.read_csv('driving_log.csv')

log_recovery = pd.read_csv('recovery_new/driving_log.csv')

left0000_udacity = mpimg.imread(log_udacity["left"][0].strip())
height, width, depth = left0000_udacity.shape

# creat a new dataset
left, center, right = [], [], []
# go thru each time stamps in udacity dataset
for i in range(len(log_udacity)):
    left_img = log_udacity["left"][i]
    center_img = log_udacity["center"][i]
    right_img = log_udacity["right"][i]
    steering = log_udacity["steering"][i]
    
    # positive means right steering
    if (steering > 0.0):
        right.append([center_img, left_img, right_img, steering])
            
    # negative means left steering
    elif (steering < 0.0):
        left.append([center_img, left_img, right_img, steering])
            
    else:
        if (steering == 0.0):
            center.append([center_img, left_img, right_img, steering])

# go thru each time stamps in recovery dataset
for i in range(len(log_recovery)):
    left_img = log_recovery["left"][i]
    center_img = log_recovery["center"][i]
    right_img = log_recovery["right"][i]
    steering = log_recovery["steering"][i]
    
    # positive means right steering
    if (steering > 0.0):
        right.append([center_img, left_img, right_img, steering])
            
    # negative means left steering
    elif (steering < 0.0):
        left.append([center_img, left_img, right_img, steering])
            
    else:
        # drop all 0.0 steerings
        if (steering == 0.0):
            None

# convert to data frame
df_all = pd.DataFrame(left+center+right, columns=['center','left','right','steering'])
X_data = df_all[["center","left","right","steering"]]
y_data = df_all["steering"]

X_train_data, X_valid_data, y_train_data, y_valid_data = train_test_split(X_data, y_data, test_size=0.2)

# reset the index
X_train_data = X_train_data.reset_index(drop=True)
X_valid_data = X_valid_data.reset_index(drop=True)

# preprocessing

# value aka brightness ... [0,1]
def set_darker(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * random.uniform(0.7, 1)
    #hsv[:,:,2] = hsv[:,:,2] * 0.7

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# 1 mean along y axis ... vertically
# 0 mean along x axis ... horizontally
def flip_along_y_axis(image,steering):       
    return cv2.flip(image,1), -steering

nvidia_height = 66
nvidia_width = 220

def crop_image(image):

    start = int(image.shape[0] * 0.35)
    end = int(image.shape[0] * 0.875)   
    
    # removes the sky on the top
    # and the hood on the bottom
    new_image = image[start:end, :]
    
    # resizes to 66 x 220 for nvidia model
    new_image = cv2.resize(new_image, (nvidia_width,nvidia_height), interpolation=cv2.INTER_AREA)
    
    return new_image

def preprocess_image_train(data_row_df):

    image = mpimg.imread(data_row_df["center"][0])
    steering = data_row_df["steering"][0]
    
    image = set_darker(image)
    
    # flip half of the images
    if np.random.randint(2) == 1:
        image, steering = flip_along_y_axis(image,steering)
    
    image = crop_image(image)

    return np.array(image), steering

def preprocess_image_valid(data_row_df):

    image = mpimg.imread(data_row_df["center"][0])   
    steering = data_row_df['steering'][0]
    
    image = crop_image(image)
    
    return np.array(image), steering

# define generators

def generate_batch_train_from_dataframe(data_df, batch_size):
    
    batch_images = np.zeros((batch_size, nvidia_height, nvidia_width, depth))
    batch_steerings = np.zeros(batch_size)
    
    while True:
        for i in range (batch_size):

            idx = np.random.randint(len(data_df))
            data_row = data_df.iloc[[idx]].reset_index()

            # preprocess image and steering
            img, steering = preprocess_image_train(data_row)

            batch_images[i] = img
            batch_steerings[i] = steering

        yield batch_images, batch_steerings

def generate_valid_from_dataframe(data_df):
    while True:
        for idx in range(len(data_df)):
            
            data_row = data_df.iloc[[idx]].reset_index()
            
            # preprocess image and steering
            img, steering = preprocess_image_valid(data_row)

            # reshape to (1, height, width, channel)
            # before feeding to the model
            img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
            steering = np.array([[steering]])
            yield img, steering

# define convnet model
input_shape = (nvidia_height, nvidia_width, depth)

def get_model():
    
    model = Sequential()
    
    # normalize to [-0.5, 0.5]
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = input_shape))
    
    # convolution + maxpooling
        # filter size 3x3
        # subsample is stride
        # valid means 0 padding
        # pool size 2
    #1
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #2
    model.add(Convolution2D(36, 3, 3, subsample=(1, 1), border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    #3
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    
    # fully-connected layers
    model.add(Dense(512, activation='relu'))
       
    model.add(Dense(128, activation='relu'))
    
    model.add(Dense(16, activation='relu'))
    
    model.add(Dense(1))

    # learning rate 1e-4
    # value function: mse
    # optimizer: adam
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    
    return model

# training
train_samples = len(X_train_data)
validation_samples = len(X_valid_data)
epoch = 4

batch = 256

for idx in range(3):
    print("iteration: ", idx)    
    train_data_generator = generate_batch_train_from_dataframe(X_train_data, batch)

    # initialize generator
    valid_data_generator = generate_valid_from_dataframe(X_valid_data)
    
    model = get_model()
    model.fit_generator(train_data_generator,samples_per_epoch=train_samples,nb_epoch=epoch,validation_data=valid_data_generator, nb_val_samples=validation_samples)
       
    fileWeightsNew = 'model-udacity_recovery-' + str(idx) + '.h5'
    model.save(fileWeightsNew)

