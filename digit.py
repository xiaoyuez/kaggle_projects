# code for Kaggle Digit Recognizer
# multiclass classification with CNN and image augmentation

# Mount Google Drive 
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

root_path = '/content/gdrive/My Drive/google_colab/'
import os
os.chdir(root_path)

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!kaggle competitions download -c digit-recognizer --force

# libraries
import os
import sys
import pdb
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Input
from keras.layers import Lambda, Dense, Dropout, MaxPooling2D, Conv2D, BatchNormalization, Flatten

# load data
train_df = pd.read_csv("train.csv.zip")
test_df = pd.read_csv("test.csv.zip")

# hyperparameters
NUM_CLASSES = len(train_df.label.unique())
BATCH_SIZE = 64
EPOCH = 20

# visualize one random example per class
def one_per_class(train_df):
  num_classes = len(train_df.label.unique())
  fig, axes = plt.subplots(1, num_classes, figsize = (num_classes,3))
  for i, ax in enumerate(axes.flatten()):
    X = train_df[train_df.label == i].sample()
    X = np.array(X.drop('label', axis = 1))
    X = X.reshape(28,28)
    ax.imshow(X, cmap = "gray")
    plt.setp(axes, xticks = [], yticks = [], frame_on = False)
    plt.tight_layout(h_pad = 0.5, w_pad = 0.01)

one_per_class(train_df)

# reshape data
X_train = train_df.iloc[:,1:].values
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
y_train = train_df.iloc[:,0].values
#y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)

# train validation split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.15, random_state = 0)

# define keras data generator 
datagen = ImageDataGenerator(rotation_range = 8,
                             width_shift_range = 0.08,
                             shear_range = 0.3,
                             height_shift_range = 0.08,
                             zoom_range = 0.08)
batches = datagen.flow(X_train, y_train, batch_size=64)
datagen.fit(X_train)

# feature standardization
def standardize(x): 
  mean_px = X_train.mean().astype(np.float32)
  std_px = X_train.std().astype(np.float32)
  return (x - mean_px) / std_px

# build a simple convolution model
model = Sequential()

# first convolution block
model.add((Lambda(standardize,input_shape = (28,28,1))))
model.add(Conv2D(64, (5, 5), activation = 'relu'))
model.add(MaxPooling2D())
model.add(BatchNormalization())
model.add(Dropout(0.25))

# second convolution block
model.add(Conv2D(64, (2, 2), activation = 'relu'))
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(Conv2D(256, (5, 5), activation = 'relu'))
model.add(MaxPooling2D())
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(NUM_CLASSES, activation = 'softmax'))

# compile model
model.compile(
    optimizer = 'adam', 
    loss = 'sparse_categorical_crossentropy', 
    metrics = ['acc'])

# fits the model on batches with real-time data augmentation if any
model.fit_generator(datagen.flow(X_train, y_train, batch_size = BATCH_SIZE),
                    validation_data = datagen.flow(X_val, y_val, batch_size = BATCH_SIZE),
                    steps_per_epoch = len(X_train) / BATCH_SIZE, epochs = EPOCH)

# prepare test data
X_test = test_df.values
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# model predict
y_pred = model.predict_classes(X_test) # 99.4% accuracy on test set 

# prepare for submission
submission = pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)),
                         "Label": y_pred})
submission.to_csv("submission.csv", index = False)