# code for Bengali.AI Handwritten Grapheme Classification
# multiple output multiclass classification on images

# libraries
import pdb 
import tqdm
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras

import PIL.Image as Image
import cv2

from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Input, Model
from keras.layers import add, Activation, Lambda, Dense, Dropout
from keras.layers import AveragePooling2D, GlobalAveragePooling2D, ZeroPadding2D, MaxPooling2D, Conv2D, BatchNormalization, Flatten

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

K = 3 # the number of repeats of the training loop

# some utility functions
def bbox(img):
  # crop and center the images
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

def load_training_data(i):
  # loads training data from train parquet
  train_df = pd.merge(pd.read_parquet(train_images[i]), train_df_, on = 'image_id')
  
  # prepare X train, y train
  y_train_root = train_df.grapheme_root
  y_train_vowel = train_df.vowel_diacritic
  y_train_consonant = train_df.consonant_diacritic
  X_train = train_df.drop(['image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1)
  X_train = 255 - X_train.values.reshape(X_train.shape[0], HEIGHT, WIDTH).astype(np.uint8)
  
  # compute class weights
  root_weight = class_weight.compute_class_weight('balanced', np.unique(y_train_root), y_train_root)
  vowel_weight = class_weight.compute_class_weight('balanced', np.unique(y_train_vowel), y_train_vowel)
  consonant_weight = class_weight.compute_class_weight('balanced', np.unique(y_train_consonant), y_train_consonant)
  del train_df

  # reshape and crop_resize X train
  X_train_cropped = np.zeros((X_train.shape[0],IMG_SIZE, IMG_SIZE))
  for i, image in enumerate(X_train):
    X_train_cropped[i] = crop_resize(image)
  X_train_cropped = np.expand_dims(X_train_cropped, axis = 3)
  #X_val = np.stack((X_val,)*3, axis=-1)

  # one-hot encoding y_train
  y_train_root = to_categorical(y_train_root, num_classes = ROOT_NUM)
  y_train_vowel = to_categorical(y_train_vowel, num_classes = VOWEL_NUM)
  y_train_consonant = to_categorical(y_train_consonant, num_classes = CONSONANT_NUM)

  # train validation split
  X_train, X_val, y_train_root, y_val_root, y_train_vowel, y_val_vowel, y_train_consonant, y_val_consonant  = train_test_split(X_train_cropped, y_train_root, y_train_vowel, y_train_consonant, test_size = 0.15, random_state = 0)
  print("X_train has shape {}, X_val has shape {}.\n y_train_root has shape {}, y_val_root has shape {}.".format(
      X_train.shape, X_val.shape, y_train_root.shape, y_val_root.shape))
  del X_train_cropped
  return X_train, X_val, y_train_root, y_val_root, y_train_vowel, y_val_vowel, y_train_consonant, y_val_consonant

# build model
def build_model(base = 'resnet', trainable = False):
  # choose a base model
  if base == 'resnet':
    # Resnet
    base_model = keras.applications.ResNet50(weights='imagenet', include_top = False)
  elif base == 'efficientnet':
    # EfficientNet
    !pip install -U git+https://github.com/qubvel/efficientnet
    import efficientnet.keras as efn 
    base_model = efn.EfficientNetB7(weights='imagenet', include_top = False)  
  base_model.trainable = trainable

  # build on top of the base model
  inputs = Input(shape = (IMG_SIZE, IMG_SIZE, 1))
  x = Conv2D(3, (3, 3), padding = 'same')(inputs)
  x = base_model(x)
  x = GlobalAveragePooling2D()(x)
  x = BatchNormalization()(x)
  x = Dropout(0.5)(x)
  x = Dense(1024, activation = 'relu')(x)
  x = Dense(512, activation = 'relu')(x)
  x = Dense(256, activation='relu')(x)
  x = BatchNormalization()(x)
  x = Dropout(0.5)(x)
  head_root = Dense(168, activation = 'softmax', name = "root")(x)
  head_vowel = Dense(11, activation = 'softmax', name = "vowel")(x)
  head_consonant = Dense(7, activation = 'softmax', name = "consonant")(x)

  model = Model(inputs = inputs, outputs = [head_root, head_vowel, head_consonant])
  model.summary()

  # compile model
  model.compile(optimizer = 'adam', 
                loss = {"root": "categorical_crossentropy",
                        "vowel": "categorical_crossentropy",
                        "consonant": "categorical_crossentropy"},  
                metrics = ["accuracy"])
  return model

model = build_model(base = 'resnet', trainable = False)

# train time 
for k in range(K): # kinda like k-fold
  for i in range(4):
    X_train, X_val, y_train_root, y_val_root, y_train_vowel, y_val_vowel, y_train_consonant, y_val_consonant = load_training_data(i)

    # fit model 
    es = EarlyStopping(monitor = 'val_root_loss', patience = 2)
    mc = ModelCheckpoint('bengali_model.h5', monitor = 'val_root_loss')

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
              callbacks = [es, mc],
              class_weight = {"root": root_weight,
                              "vowel": vowel_weight,
                              "consonant": consonant_weight})

    del X_train, X_val

# predict using the test data
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

sub_df = pd.DataFrame({'row_id': row_id,'target':target})
sub_df.to_csv("submission.csv", index = False)

