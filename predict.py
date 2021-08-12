import tensorflow as tf
import discord
import os
import urllib
from io import BytesIO
from PIL import Image
import requests
import numpy as np
from keras.preprocessing import image

def newCNN():
    CNN = tf.keras.models.Sequential()
    CNN.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
    CNN.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    CNN.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    CNN.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    CNN.add(tf.keras.layers.Flatten())
    CNN.add(tf.keras.layers.Dense(units=128, activation='relu'))
    CNN.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    CNN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return CNN
