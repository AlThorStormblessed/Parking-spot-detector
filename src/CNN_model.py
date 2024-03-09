from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import keras
import glob, random
from keras import optimizers, Model, layers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.layers import Input, Lambda, Dense, Flatten, Dropout,BatchNormalization, GlobalAveragePooling2D, Conv2D, MaxPool2D, Activation
from keras.models import Sequential
from keras.applications import MobileNet, ResNet50, xception
from keras.utils import to_categorical
from livelossplot import PlotLossesKeras
from PIL import Image

img_height = 128
img_width = 128
batch_size = 4

training_ds = keras.preprocessing.image_dataset_from_directory(
    'Data',
    seed=42,
    image_size= (img_height, img_width),
    batch_size=batch_size
)

training_ds, testing_ds = keras.utils.split_dataset(training_ds, left_size = 0.8, shuffle = True)
# validation_ds, testing_ds = keras.utils.split_dataset(testing_ds, left_size = 0.5)
def get_model():
    model=Sequential(
        [ layers.BatchNormalization(),
          layers.Conv2D(32, 3, activation='relu'),
          layers.MaxPooling2D(),
          layers.Conv2D(64, 3, activation='relu'),
          layers.MaxPooling2D(),
          layers.Conv2D(128, 3, activation='relu'),
          layers.MaxPooling2D(),
          layers.Flatten(),
          layers.Dense(256, activation='relu'),
          layers.Dense(2, activation= 'softmax')]
    )
    return model

model = get_model()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(training_ds, validation_data = testing_ds, epochs = 5)

model.save("Saved_model/model2")