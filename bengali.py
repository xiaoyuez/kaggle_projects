# Code for Bengali.AI Handwritten Grapheme Classification challenge
# Multi-output multiclass classification on images
# some code are adapted from https://www.kaggle.com/kaushal2896/bengali-graphemes-starter-eda-multi-output-cnn by Kaushal Shah

# Download data and unzip
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!kaggle competitions download -c bengaliai-cv19
#!unzip train_image_data_0.parquet.zip

# libraries
import pdb 
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
import PIL.Image as Image
import cv2

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Input, Model
from keras.layers import add, Activation, Lambda, Dense, Dropout
from keras.layers import AveragePooling2D, ZeroPadding2D, MaxPooling2D, Conv2D, BatchNormalization, Flatten

# load data
train_df = pd.read_csv('train.csv.zip')
test_df = pd.read_csv('test.csv')
class_df = pd.read_csv('class_map.csv')

train_images = ['train_image_data_0.parquet',
                'train_image_data_1.parquet',
                'train_image_data_2.parquet',
                'train_image_data_3.parquet']

test_images = ['test_image_data_0.parquet',
                'test_image_data_1.parquet',
                'test_image_data_2.parquet',
                'test_image_data_3.parquet']

# exploratory data analysis
def get_n(df, col_name, n = 10):
  if n > 0: # most
    counts = df.groupby(col_name).size().sort_values(ascending = False)[0:n]
  elif n < 0: # least
    counts = df.groupby(col_name).size().sort_values(ascending = False)[n:]
  out_df = class_df[class_df['component_type'] == col_name].iloc[counts.index.values, ]
  out_df['counts'] = counts[counts.index].values
  return out_df

# get top 10 graphmemes
get_n(train_df, 'grapheme_root', n = 10)

# get last 10 grapheme_root
get_n(train_df, 'grapheme_root', n = -10)

# get top 3 vowel
get_n(train_df, 'vowel_diacritic', n = 3)

# get top 3 consonant
get_n(train_df, 'consonant_diacritic', n = 3)

# minimize disk usage
train_df_ = train_df.drop(['grapheme'], axis = 1)
train_df_[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']] = train_df_[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].astype('uint8')

# hyperparameters
HEIGHT = 137
WIDTH = 236
IMG_SIZE = 128
BATCH_SIZE = 64

ROOT_NUM = 168
VOWEL_NUM = 11
CONSONANT_NUM = 7

# crop and center the images
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size = IMG_SIZE, pad=16):
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin, ymax, xmin, xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove low intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax - xmin, ymax - ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))

"""Simple CNN"""
inputs = Input(shape = (IMG_SIZE, IMG_SIZE, 1))

# first convolution block
model = Conv2D(32, (3, 3), activation = 'relu')(inputs)
model = Conv2D(32, (3, 3), activation = 'relu')(model)
model = Conv2D(32, (3, 3), activation = 'relu')(model)
model = Conv2D(32, (3, 3), activation = 'relu')(model)
model = BatchNormalization()(model)
model = MaxPooling2D(2)(model)
model = Conv2D(32, (5, 5), activation='relu')(model)
model = Dropout(0.25)(model)

# second convolution block
model = Conv2D(64, (3, 3), activation = 'relu')(model)
model = Conv2D(64, (3, 3), activation = 'relu')(model)
model = Conv2D(64, (3, 3), activation = 'relu')(model)
model = Conv2D(64, (3, 3), activation = 'relu')(model)
model = BatchNormalization()(model)
model = MaxPooling2D(2)(model)
model = Conv2D(64, (5, 5), activation='relu')(model)
model = Dropout(0.25)(model)

# third convolution block
model = Conv2D(128, (3, 3), activation = 'relu')(model)
model = Conv2D(128, (3, 3), activation = 'relu')(model)
model = BatchNormalization()(model)
model = MaxPooling2D(2)(model)
model = Dropout(0.25)(model)

# fourth convolution block
model = Conv2D(256, (3, 3), activation = 'relu')(model)
model = Conv2D(256, (3, 3), activation = 'relu')(model)
model = BatchNormalization()(model)
model = Dropout(0.25)(model)

# multiple output
model = Flatten()(model)
model = Dense(1024, activation = 'relu')(model)
model = Dense(512, activation = 'relu')(model)
head_root = Dense(168, activation = 'softmax', name = "root")(model)
head_vowel = Dense(11, activation = 'softmax', name = "vowel")(model)
head_consonant = Dense(7, activation = 'softmax', name = "consonant")(model)

model = Model(inputs = inputs, outputs = [head_root, head_vowel, head_consonant])
#from keras.utils import plot_model
#plot_model(model)
model.summary()

# compile model
model.compile(optimizer = 'adam', 
              loss = {"root": "categorical_crossentropy",
                      "vowel": "categorical_crossentropy",
                      "consonant": "categorical_crossentropy"},  
              metrics = ["accuracy"])

"""ResNet"""
# build ResNet from scratch as the input size of ResNet in Keras applications must be of dim 3
# code adapted from https://towardsdatascience.com/implementing-a-resnet-model-from-scratch-971be7193718 by Gracelyn Shi
class ResNet:
    @staticmethod
    def residual_module(data, K, stride, chanDim, reduce = False, reg = 0.0001, bnEps = 2e-5, bnMom = 0.9):
      shortcut = data
      # the first block of the ResNet module are the 1x1 CONVs
      bn1 = BatchNormalization(axis = chanDim, epsilon = bnEps, momentum = bnMom)(data)
      act1 = Activation("relu")(bn1)
      conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False)(act1)
              
      # the second block of the ResNet module are the 3x3 CONVs
      bn2 = BatchNormalization(axis = chanDim, epsilon = bnEps, momentum = bnMom)(conv1)
      act2 = Activation("relu")(bn2)
      conv2 = Conv2D(int(K * 0.25), (3, 3), strides = stride, padding = "same", use_bias = False)(act2)

      # the third block of the ResNet module is another set of 1x1 CONVs
      bn3 = BatchNormalization(axis = chanDim, epsilon = bnEps, momentum = bnMom)(conv2)
      act3 = Activation("relu")(bn3)
      conv3 = Conv2D(K, (1, 1), use_bias = False)(act3)

      # if we are to reduce the spatial size, apply a CONV layer to the shortcut
      if reduce:
        shortcut = Conv2D(K, (1, 1), strides = stride, use_bias = False)(act1)

      # add together the shortcut and the final CONV
      x = add([conv3, shortcut])
      return x
    
    @staticmethod
    def build(width, height, depth, classes, stages, filters, reg = 0.0001, bnEps = 2e-5, bnMom = 0.9):
      inputShape = (height, width, depth)
      chanDim = -1
      # set the input and apply BN
      inputs = Input(shape = inputShape)
      x = BatchNormalization(axis = chanDim, epsilon = bnEps, momentum = bnMom)(inputs)
      
      # apply CONV => BN => ACT => POOL to reduce spatial size
      x = Conv2D(filters[0], (5, 5), use_bias = False, padding = "same")(x)
      x = BatchNormalization(axis = chanDim, epsilon = bnEps, momentum = bnMom)(x)
      x = Activation("relu")(x)
      x = ZeroPadding2D((1, 1))(x)
      x = MaxPooling2D((3, 3), strides = (2, 2))(x)
      
      # loop over the number of stage
      for i in range(0, len(stages)):
          stride = (1, 1) if i == 0 else (2, 2)
          x = ResNet.residual_module(x, filters[i + 1], stride, chanDim, reduce = True, bnEps = bnEps, bnMom = bnMom)
          # loop over the number of layers in the stage
          for j in range(0, stages[i] - 1):
              # apply a ResNet module
              x = ResNet.residual_module(x, filters[i + 1], (1, 1), chanDim, bnEps = bnEps, bnMom = bnMom)
      # apply BN => ACT => POOL
      x = BatchNormalization(axis = chanDim, epsilon = bnEps, momentum = bnMom)(x)
      x = Activation("relu")(x)
      x = AveragePooling2D((8, 8))(x)
      
      # multi-output softmax classifier
      x = Flatten()(x)
      x = Dense(1024, activation = 'relu')(x)
      x = Dense(512, activation = 'relu')(x)
      head_root = Dense(168, activation = 'softmax', name = "root")(x)
      head_vowel = Dense(11, activation = 'softmax', name = "vowel")(x)
      head_consonant = Dense(7, activation = 'softmax', name = "consonant")(x)
      
      model = Model(inputs = inputs, outputs = [head_root, head_vowel, head_consonant], name = 'resnet')
      
      # compile model
      model.compile(optimizer = 'adam', 
              loss = {"root": "categorical_crossentropy",
                      "vowel": "categorical_crossentropy",
                      "consonant": "categorical_crossentropy"},  
              metrics = ["accuracy"])
      return model

model = ResNet.build(IMG_SIZE, IMG_SIZE, 1, classes = 10, stages = (3,4,6), filters = (32, 64, 128, 256))

# training time
for k in range(5): # kinda like k-fold
  for i in range(4):
    train_df = pd.merge(pd.read_parquet(train_images[i]), train_df_, on='image_id')
    # prepare X train, y train
    y_train_root = train_df.grapheme_root
    y_train_vowel = train_df.vowel_diacritic
    y_train_consonant = train_df.consonant_diacritic
    X_train = train_df.drop(['image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1)
    X_train = 255 - X_train.values.reshape(X_train.shape[0], HEIGHT, WIDTH).astype(np.uint8)
    del train_df

    # reshape and crop_resize X train
    X_train_cropped = np.zeros((X_train.shape[0],IMG_SIZE, IMG_SIZE))
    for i, image in enumerate(X_train):
      X_train_cropped[i] = crop_resize(image)
    X_train_cropped = np.expand_dims(X_train_cropped, axis = 3)

    # one-hot encoding of y_train
    y_train_root = to_categorical(y_train_root, num_classes = ROOT_NUM)
    y_train_vowel = to_categorical(y_train_vowel, num_classes = VOWEL_NUM)
    y_train_consonant = to_categorical(y_train_consonant, num_classes = CONSONANT_NUM)

    # train validation split
    X_train, X_val, y_train_root, y_val_root, y_train_vowel, y_val_vowel, y_train_consonant, y_val_consonant  = train_test_split(X_train_cropped, y_train_root, y_train_vowel, y_train_consonant, test_size = 0.15, random_state = 0)
    print("X_train has shape {}, X_val has shape {}.\n y_train_root has shape {}, y_val_root has shape {}.".format(
        X_train.shape, X_val.shape, y_train_root.shape, y_val_root.shape))
    del X_train_cropped

    # fit model
    es = EarlyStopping(monitor = 'val_root_loss', patience = 2) # stops if val_root_loss not improving for 2 epochs
    mc = ModelCheckpoint('bengali_model.h5', monitor = 'val_root_loss') # autosaves the best model

    model.fit(X_train, {"root": y_train_root,
                        "vowel": y_train_vowel,
                        "consonant": y_train_consonant},
              validation_data = (X_val, 
                                {"root": y_val_root,
                                  "vowel": y_val_vowel,
                                  "consonant": y_val_consonant}),
              batch_size = BATCH_SIZE, 
              epochs = 10,
              verbose = 1,
              callbacks = [es, mc])

    del X_train, X_val

# generate predictions using test data
preds_dict = {'grapheme_root': [],
            'vowel_diacritic': [],
            'consonant_diacritic': []}
components = ['grapheme_root','vowel_diacritic','consonant_diacritic']
row_id = []
target = []
test_df_ = pd.read_csv('test.csv')
for i in range(4):
  test_df = pd.read_parquet(test_images[i])
  test_df.set_index('image_id', inplace=True)
  X_test = test_df
  X_test = 255 - X_test.values.reshape(X_test.shape[0], HEIGHT, WIDTH).astype(np.uint8)
  X_test_cropped = np.zeros((X_test.shape[0],IMG_SIZE, IMG_SIZE))
  for i, image in enumerate(X_test):
    X_test_cropped[i] = crop_resize(image)
  X_test_cropped = np.expand_dims(X_test_cropped, axis = 3)

  preds = model.predict(X_test_cropped)

  for i, p in enumerate(preds_dict):
        preds_dict[p] = np.argmax(preds[i], axis = 1)

  for k, id in enumerate(test_df.index.values):
    for i, comp in enumerate(components):
      id_sample = '{}_{}'.format(id, comp)
      row_id.append(id_sample)
      target.append(preds_dict[comp][k])
  del test_df
  del X_test, X_test_cropped

# ready for submission
sub_df = pd.DataFrame({'row_id': row_id,'target':target})
sub_df.to_csv("bengali_sub1.csv")

