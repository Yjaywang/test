import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import datetime


data_dir = r'data folder path'

batch_size =32
img_height =148
img_width =148


#import training materials
train_ds =tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split= 0.2, 
    subset='training', 
    color_mode='grayscale', #my ML model can only use gray scale
    shuffle =True, 
    seed=123, 
    image_size=(img_height, img_width), 
    batch_size=batch_size
)
val_ds =tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split= 0.2, 
    subset='validation', 
    color_mode='grayscale', #my ML model can only use gray scale
    shuffle =True, 
    seed=123, 
    image_size=(img_height, img_width), 
    batch_size=batch_size
)



calss_names = train_ds.class_names  #check the label for dataset

#use buffered prefetching so you can tield datafrom dist withouthaving I/O become blocking. these are 2 important methods you should use when loading data.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(4000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", 
                        input_shape=(img_height,img_width,1)), #grayscale=1
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),

    ]
)

num_classes = 3 # 3 catagoties

def get_model():
    model = keras.Sequential([
        layers.Input((148, 148, 1)), #grayscale
        layers.Rescaling(1./255), #normalization
        layers.Conv2D(16, kernel_size= 3),
        layers.BatchNormalization(), #prevent overfitting
        layers.ReLU(), 
        layers.MaxPooling2D(), 
        layers.Conv2D(16, kernel_size= 3),
        layers.BatchNormalization(), #prevent overfitting
        layers.ReLU(), 
        layers.MaxPooling2D(), 
        layers.Conv2D(16, kernel_size= 3),
        layers.BatchNormalization(), #prevent overfitting
        layers.ReLU(), 
        layers.MaxPooling2D(), 
        layers.Flatten(),
        layers.Dense(32),
        layers.BatchNormalization(), #prevent overfitting
        layers.ReLU(), 
        layers.Dense(num_classes), 
        layers.BatchNormalization(),
        layers.Softmax()
    ])
    return model

model = get_model()
model.summary() #yield parameters table
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

tensorboard_callback = keras.callbacks.TensorBoard(   #output tensorboard data to visuallize results
    log_dir='tb_callback_dir', 
    histogram_freq=1
    write_graph=True
)

model.fit( 
    train_ds, 
    epochs=200, 
    validation_data=val_ds, 
    callbacks=[tensorboard_callback], 
    verbose=1
)


#save model
model.save('tensorflow_trained_model.h5') #save model as H5 format
model.save_weights('tensorflow_trained_model_checkpoint', save_format='tf')  # save model weights

#load check point, model code need to exist
#model.load_weights('tensorflow_trained_model_checkpoint') #load model weights
#weights =model.get_weights() #retrieves the status of model
#model.set_weights(weights) #set model status
