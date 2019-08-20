#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os, sys
import tensorflow as tf
import random
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import UpSampling2D
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience = 3)


# ### Data Preprocess



# load data
filelist = os.listdir("@FILEPATH@")
filelist = filelist[1:]
filename_dic = {}


for x, a_file in enumerate(filelist):
    filename_dic[x] = pd.read_csv("@FILEPATH@" + a_file, index_col=0)

# make 24*24 image data
training_data = np.array([]).reshape(0, 24, 24)
validation_data = np.array([]).reshape(0, 24, 24)
test_data = np.array([]).reshape(0, 24, 24)
training_label = np.array([]).reshape(0)
validation_label = np.array([]).reshape(0)
test_label = np.array([]).reshape(0)
for week_idx in range(35, 61):
    answer = filename_dic[week_idx]
    for i in range(week_idx - 23, week_idx)[::-1]:
        answer = pd.merge(filename_dic[i], answer, how='right', left_index=True, right_index=True)
    answer.fillna(0, inplace=True)
    answer = np.fmin(1, answer.div(answer.quantile(0.9, 1) + 1, axis=0))
    tri_di_ans = answer.to_numpy().reshape(answer.shape[0], 24, 24)
    ans_label = np.repeat(0, len(tri_di_ans))
    if week_idx % 5 == 1:
        validation_data = np.append(validation_data, tri_di_ans, axis = 0)
        validation_label = np.append(validation_label, ans_label)
    elif week_idx % 5 == 4:
        test_data = np.append(test_data, tri_di_ans, axis = 0)
        test_label = np.append(test_label , ans_label)
    else:
        training_data = np.append(training_data, tri_di_ans, axis = 0)
        training_label = np.append(training_label, ans_label)
for week_idx in range(61, 87):
    answer = filename_dic[week_idx]
    for i in range(week_idx - 23, week_idx)[::-1]:
        answer = pd.merge(filename_dic[i], answer, how='right', left_index=True, right_index=True)
    answer.fillna(0, inplace=True)
    answer = np.fmin(1, answer.div(answer.quantile(0.9, 1) + 1, axis=0))
    tri_di_ans = answer.to_numpy().reshape(answer.shape[0], 24, 24)
    ans_label = np.repeat(1, len(tri_di_ans))
    if week_idx % 5 == 1:
        validation_data = np.append(validation_data, tri_di_ans, axis = 0)
        validation_label = np.append(validation_label, ans_label)
    elif week_idx % 5 == 4:
        test_data = np.append(test_data, tri_di_ans, axis = 0)
        test_label = np.append(test_label , ans_label)
    else:
        training_data = np.append(training_data, tri_di_ans, axis = 0)
        training_label = np.append(training_label, ans_label)


# make label, flag
training_label = training_label.reshape(len(training_label), 1)
validation_label = validation_label.reshape(len(validation_label), 1)
test_label = test_label.reshape(len(test_label), 1)
tr_flag = np.repeat(3, len(training_label)).reshape(len(training_label), 1)
vl_flag = np.repeat(4, len(validation_label)).reshape(len(validation_label), 1)
te_flag = np.repeat(5, len(test_label)).reshape(len(test_label), 1)

# make shuffled data
perm1 = np.arange(576)
random.shuffle(perm1)
s_training_data = training_data.reshape(len(training_data), 576)
s_training_data = s_training_data[:, perm1]
s_training_data = s_training_data.reshape(len(training_data), 24, 24)
s_validation_data = validation_data.reshape(len(validation_data), 576)
s_validation_data = s_validation_data[:, perm1]
s_validation_data = s_validation_data.reshape(len(validation_data), 24, 24)
s_test_data = test_data.reshape(len(test_data), 576)
s_test_data = s_test_data[:, perm1]
s_test_data = s_test_data.reshape(len(test_data), 24, 24)

# make data for CNN
training_data_cnn = training_data[:, :, :, np.newaxis]
validation_data_cnn = validation_data[:, :, :, np.newaxis]
test_data_cnn = test_data[:, :, :, np.newaxis]
s_training_data_cnn = s_training_data[:, :, :, np.newaxis]
s_validation_data_cnn = s_validation_data[:, :, :, np.newaxis]
s_test_data_cnn = s_test_data[:, :, :, np.newaxis]


# PCA model
input_layer = Input(shape=(24, 24))
flatten_layer = Flatten()(input_layer)
encoding_layer = Dense(9, use_bias = False)(flatten_layer)
embedding_layer = Dense(576, use_bias = False)(encoding_layer)
decoding_layer = Reshape((24, 24))(embedding_layer)

encoder = Model(input_layer, encoding_layer)
autoencoder = Model(input_layer, decoding_layer)
autoencoder.compile(optimizer='rmsprop', loss='mean_squared_error')

autoencoder.summary()
autoencoder.fit(training_data, training_data, epochs=10, validation_data=(validation_data, validation_data))

autoencoder.evaluate(test_data, test_data)



# PCA model use shuffled data
s_input_layer = Input(shape=(24, 24))
s_flatten_layer = Flatten()(s_input_layer)
s_encoding_layer = Dense(9, use_bias = False)(s_flatten_layer)
s_embedding_layer = Dense(576, use_bias = False)(s_encoding_layer)
s_decoding_layer = Reshape((24, 24))(s_embedding_layer)

s_encoder = Model(s_input_layer, s_encoding_layer)
s_autoencoder = Model(s_input_layer, s_decoding_layer)
s_autoencoder.compile(optimizer='rmsprop', loss='mean_squared_error')

s_autoencoder.summary()

s_autoencoder.fit(s_training_data, s_training_data, epochs=10, validation_data=(s_validation_data, s_validation_data))

s_autoencoder.evaluate(s_test_data, s_test_data)


# CNN model, low params
input_layer_cnn = Input(shape=(24, 24, 1))
conv1 = Conv2D(2, (4, 4), padding="same", strides = 2)(input_layer_cnn)
mxp1 = MaxPooling2D((2, 2))(conv1)
encoding_layer_cnn = Conv2D(1, (4, 4), padding='same', strides = 2 )(mxp1)
conv3 = Conv2DTranspose(2, (4, 4),  padding='same', strides = 2)(encoding_layer_cnn)
us1 = UpSampling2D((2, 2))(conv3)
decoding_layer_cnn = Conv2DTranspose(1, (4, 4), padding='same', strides = 2)(us1)

encoder_cnn = Model(input_layer_cnn, encoding_layer_cnn)
autoencoder_cnn = Model(input_layer_cnn, decoding_layer_cnn)
autoencoder_cnn.compile(optimizer='rmsprop', loss='mean_squared_error')

autoencoder_cnn.summary()

autoencoder_cnn.fit(training_data_cnn, training_data_cnn, epochs=100, validation_data=(validation_data_cnn, validation_data_cnn), callbacks=[early_stopping])

autoencoder_cnn.evaluate(test_data_cnn, test_data_cnn)


# CNN model, use shuffled data, low params
s_input_layer_cnn = Input(shape=(24, 24, 1))
s_conv1 = Conv2D(2, (4, 4), padding="same", strides = 2)(s_input_layer_cnn)
s_mxp1 = MaxPooling2D((2, 2))(s_conv1)
s_encoding_layer_cnn = Conv2D(1, (4, 4), padding='same', strides = 2 )(s_mxp1)
s_conv3 = Conv2DTranspose(2, (4, 4),  padding='same', strides = 2)(s_encoding_layer_cnn)
s_us1 = UpSampling2D((2, 2))(s_conv3)
s_decoding_layer_cnn = Conv2DTranspose(1, (4, 4), padding='same', strides = 2)(s_us1)

s_encoder_cnn = Model(s_input_layer_cnn, s_encoding_layer_cnn)
s_autoencoder_cnn = Model(s_input_layer_cnn, s_decoding_layer_cnn)
s_autoencoder_cnn.compile(optimizer='rmsprop', loss='mean_squared_error')

s_autoencoder_cnn.summary()

s_autoencoder_cnn.fit(s_training_data_cnn, s_training_data_cnn, epochs=100, validation_data=(s_validation_data_cnn, s_validation_data_cnn), callbacks=[early_stopping])

s_autoencoder_cnn.evaluate(s_test_data_cnn, s_test_data_cnn)

# ## Save data
training_cnn = encoder_cnn.predict(training_data_cnn).reshape(len(training_data_cnn), 9)
validation_cnn = encoder_cnn.predict(validation_data_cnn).reshape(len(validation_data_cnn), 9)
test_cnn = encoder_cnn.predict(test_data_cnn).reshape(len(test_data_cnn), 9)

tr_data_cnn = np.concatenate((training_cnn, training_label, tr_flag), axis = 1)
vl_data_cnn = np.concatenate((validation_cnn, validation_label, vl_flag), axis = 1)
te_data_cnn = np.concatenate((test_cnn, test_label, te_flag), axis = 1)
data_cnn = pd.DataFrame(np.concatenate((tr_data_cnn, vl_data_cnn, te_data_cnn), axis = 0))


data_cnn.columns = ['F_' + str(x) for x in range(9)] + ['Label', 'Flag']
pd.DataFrame(data_cnn).to_csv("data_9_cnn_samepad_2str.csv", index = False)
