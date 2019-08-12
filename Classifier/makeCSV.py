#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os, sys
import tensorflow as tf
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
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience = 3)


# In[ ]:


filelist = os.listdir("../target_csv/")
filelist = filelist[1:]
target = {}

for x, a_file in enumerate(filelist):
    target[x] = pd.read_csv("../target_csv/" + a_file, index_col=0)


# ### pp ML TSNE data

# In[ ]:


training_data = np.array([])
validation_data = np.array([])
test_data = np.array([])
training_label = np.array([])
validation_label = np.array([])
test_label = np.array([])
training_data = training_data.reshape(0, 24, 24)
validation_data = validation_data.reshape(0, 24, 24)
test_data = test_data.reshape(0, 24, 24)
training_label = training_label.reshape(0)
validation_label = validation_label.reshape(0)
test_label = test_label.reshape(0)
for week_idx in range(35, 61):
    answer = target[week_idx]
    for i in range(week_idx - 23, week_idx)[::-1]:
        answer = pd.merge(target[i], answer, how='right', left_index=True, right_index=True)
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
    answer = target[week_idx]
    for i in range(week_idx - 23, week_idx)[::-1]:
        answer = pd.merge(target[i], answer, how='right', left_index=True, right_index=True)
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


# In[ ]:


training_label = training_label.reshape(len(training_label), 1)
validation_label = validation_label.reshape(len(validation_label), 1)
test_label = test_label.reshape(len(test_label), 1)
tr_flag = np.repeat(3, len(training_label)).reshape(len(training_label), 1)
vl_flag = np.repeat(4, len(validation_label)).reshape(len(validation_label), 1)
te_flag = np.repeat(5, len(test_label)).reshape(len(test_label), 1)


# ### PCA

# In[ ]:


training_data_pca = training_data.reshape(len(training_data), 576)
validation_data_pca = validation_data.reshape(len(validation_data), 576)
test_data_pca = test_data.reshape(len(test_data), 576)


# In[ ]:


fitting_data = np.concatenate((training_data_pca, validation_data_pca), axis = 0) # 이거 쓰면 MemoryError


# In[ ]:


pca = PCA(9)
pca.fit(fitting_data)


# In[ ]:


training_pca = pca.transform(training_data_pca)
validation_pca = pca.transform(validation_data_pca)
test_pca = pca.transform(test_data_pca)


# In[ ]:


tr_data_pca = np.concatenate((training_pca, training_label, tr_flag), axis = 1)
vl_data_pca = np.concatenate((validation_pca, validation_label, vl_flag), axis = 1)
te_data_pca = np.concatenate((test_pca, test_label, te_flag), axis = 1)
data_pca = pd.DataFrame(np.concatenate((tr_data_pca, vl_data_pca, te_data_pca), axis = 0))


# In[ ]:


data_pca.columns = ['F_' + str(x) for x in range(9)] + ['Label', 'Flag']
pd.DataFrame(data_pca).to_csv("target_9_pca.csv", index=False)


# ### basic autoencoder

# In[ ]:


input_target = Input(shape=(24, 24))
flatten_layer = Flatten()(input_target)
encoding_layer = Dense(9)(flatten_layer)
embedding_layer = Dense(576, activation = 'sigmoid')(encoding_layer)
decoding_layer = Reshape((24, 24))(embedding_layer)


encoder = Model(input_target, encoding_layer)
autoencoder = Model(input_target, decoding_layer)
autoencoder.compile(optimizer='rmsprop', loss='mean_squared_error')


# In[ ]:


autoencoder.summary()


# In[ ]:


autoencoder.fit(training_data, training_data, epochs=100, validation_data=(validation_data, validation_data), callbacks=[early_stopping])


# In[ ]:


autoencoder.evaluate(test_data, test_data)


# In[ ]:


training = encoder.predict(training_data)
validation = encoder.predict(validation_data)
test = encoder.predict(test_data)


# In[ ]:


tr_data_BE = np.concatenate((training, training_label, tr_flag), axis = 1)
vl_data_BE = np.concatenate((validation, validation_label, vl_flag), axis = 1)
te_data_BE = np.concatenate((test, test_label, te_flag), axis = 1)
data_BE = pd.DataFrame(np.concatenate((tr_data_BE, vl_data_BE, te_data_BE), axis = 0))


# In[ ]:


data_BE.columns = ['F_' + str(x) for x in range(9)] + ['Label', 'Flag']
pd.DataFrame(data_BE).to_csv("target_9_basic_autoencoder.csv", index = False)


# ### autoencoder

# In[ ]:


input_target_ae = Input(shape=(24, 24))
flatten_layer_ae = Flatten()(input_target_ae)
hidden_layer_ae_1 = Dense(48, activation = 'relu')(flatten_layer_ae)
encoding_layer_ae = Dense(9)(hidden_layer_ae_1)
hidden_layer_ae_2 = Dense(48, activation = 'relu')(encoding_layer_ae)
embedding_layer_ae = Dense(576, activation = 'sigmoid')(hidden_layer_ae_2)
decoding_layer_ae = Reshape((24, 24))(embedding_layer_ae)


encoder_ae = Model(input_target_ae, encoding_layer_ae)
autoencoder_ae = Model(input_target_ae, decoding_layer_ae)
autoencoder_ae.compile(optimizer='rmsprop', loss='mean_squared_error')


# In[ ]:


autoencoder_ae.summary()


# In[ ]:


autoencoder_ae.fit(training_data, training_data, epochs=100, validation_data=(validation_data, validation_data), callbacks=[early_stopping])


# In[ ]:


autoencoder_ae.evaluate(test_data, test_data)


# In[ ]:


training_ae = encoder_ae.predict(training_data)
validation_ae = encoder_ae.predict(validation_data)
test_ae = encoder_ae.predict(test_data)


# In[ ]:


tr_data_ae = np.concatenate((training_ae, training_label, tr_flag), axis = 1)
vl_data_ae = np.concatenate((validation_ae, validation_label, vl_flag), axis = 1)
te_data_ae = np.concatenate((test_ae, test_label, te_flag), axis = 1)
data_ae = pd.DataFrame(np.concatenate((tr_data_ae, vl_data_ae, te_data_ae), axis = 0))


# In[ ]:


data_ae.columns = ['F_' + str(x) for x in range(9)] + ['Label', 'Flag']
pd.DataFrame(data_ae).to_csv("target_9_autoencoder.csv", index = False)


# ### CNN

# In[ ]:


training_data_cnn = training_data[:, :, :, np.newaxis]
validation_data_cnn = validation_data[:, :, :, np.newaxis]
test_data_cnn = test_data[:, :, :, np.newaxis]


# In[ ]:


input_target_cnn = Input(shape=(24, 24, 1))
conv1 = Conv2D(2, (4, 4), padding="same", strides = 2)(input_target_cnn)
mxp1 = MaxPooling2D((2, 2))(conv1)
encoding_layer_cnn = Conv2D(1, (4, 4), padding='same', strides = 2 )(mxp1)
conv3 = Conv2DTranspose(1, (4, 4),  padding='same', strides = 2)(encoding_layer_cnn)
us1 = UpSampling2D((2, 2))(conv3)
decoding_layer_cnn = Conv2DTranspose(1, (4, 4), padding='same', strides = 2)(us1)





encoder_cnn = Model(input_target_cnn, encoding_layer_cnn)
autoencoder_cnn = Model(input_target_cnn, decoding_layer_cnn)
autoencoder_cnn.compile(optimizer='rmsprop', loss='mean_squared_error')


# In[ ]:


autoencoder_cnn.summary()


# In[ ]:


autoencoder_cnn.fit(training_data_cnn, training_data_cnn, epochs=100, validation_data=(validation_data_cnn, validation_data_cnn), callbacks=[early_stopping])


# In[ ]:


autoencoder_cnn.evaluate(test_data_cnn, test_data_cnn)


# In[ ]:


training_cnn = encoder_cnn.predict(training_data_cnn).reshape(len(training_data_cnn), 9)
validation_cnn = encoder_cnn.predict(validation_data_cnn).reshape(len(validation_data_cnn), 9)
test_cnn = encoder_cnn.predict(test_data_cnn).reshape(len(test_data_cnn), 9)


# In[ ]:


tr_data_cnn = np.concatenate((training_cnn, training_label, tr_flag), axis = 1)
vl_data_cnn = np.concatenate((validation_cnn, validation_label, vl_flag), axis = 1)
te_data_cnn = np.concatenate((test_cnn, test_label, te_flag), axis = 1)
data_cnn = pd.DataFrame(np.concatenate((tr_data_cnn, vl_data_cnn, te_data_cnn), axis = 0))


# In[ ]:


data_cnn.columns = ['F_' + str(x) for x in range(9)] + ['Label', 'Flag']
pd.DataFrame(data_cnn).to_csv("target_9_cnn_samepad_2str.csv", index = False)


# ### CNN2

# In[ ]:


training_data_cnn = training_data[:, :, :, np.newaxis]
validation_data_cnn = validation_data[:, :, :, np.newaxis]
test_data_cnn = test_data[:, :, :, np.newaxis]


# In[ ]:


input_target_cnn = Input(shape=(24, 24, 1))
conv1 = Conv2D(2, (4, 4), padding="valid", strides = 2)(input_target_cnn)
encoding_layer_cnn = Conv2D(1, (4, 4), padding='valid', strides = 3 )(conv1)
conv3 = Conv2DTranspose(2, (5, 5),  padding='valid', strides = 3)(encoding_layer_cnn)
decoding_layer_cnn = Conv2DTranspose(1, (4, 4), padding='valid', strides = 2)(conv3)





encoder_cnn = Model(input_target_cnn, encoding_layer_cnn)
autoencoder_cnn = Model(input_target_cnn, decoding_layer_cnn)
autoencoder_cnn.compile(optimizer='rmsprop', loss='mean_squared_error')


# In[ ]:


autoencoder_cnn.summary()


# In[ ]:


autoencoder_cnn.fit(training_data_cnn, training_data_cnn, epochs=100, validation_data=(validation_data_cnn, validation_data_cnn), callbacks=[early_stopping])


# In[ ]:


autoencoder_cnn.evaluate(test_data_cnn, test_data_cnn)


# In[ ]:


training_cnn = encoder_cnn.predict(training_data_cnn).reshape(len(training_data_cnn), 9)
validation_cnn = encoder_cnn.predict(validation_data_cnn).reshape(len(validation_data_cnn), 9)
test_cnn = encoder_cnn.predict(test_data_cnn).reshape(len(test_data_cnn), 9)


# In[ ]:


tr_data_cnn = np.concatenate((training_cnn, training_label, tr_flag), axis = 1)
vl_data_cnn = np.concatenate((validation_cnn, validation_label, vl_flag), axis = 1)
te_data_cnn = np.concatenate((test_cnn, test_label, te_flag), axis = 1)
data_cnn = pd.DataFrame(np.concatenate((tr_data_cnn, vl_data_cnn, te_data_cnn), axis = 0))


# In[ ]:


data_cnn.columns = ['F_' + str(x) for x in range(9)] + ['Label', 'Flag']
pd.DataFrame(data_cnn).to_csv("target_9_cnn_validpad_2a3str.csv", index = False)


# ### CNN deep

# In[ ]:


training_data_cnn = training_data[:, :, :, np.newaxis]
validation_data_cnn = validation_data[:, :, :, np.newaxis]
test_data_cnn = test_data[:, :, :, np.newaxis]


# In[ ]:


input_target_cnn_deep = Input(shape=(24, 24, 1))
conv1_deep = Conv2D(10, (3, 3), padding="same")(input_target_cnn_deep)
mxp1_deep = MaxPooling2D((2, 2), padding="same")(conv1_deep)
conv2_deep = Conv2D(20, (3, 3), padding="same")(mxp1_deep)
mxp2_deep = MaxPooling2D((2, 2), padding="same")(conv2_deep)
conv3_deep = Conv2D(50, (3, 3), padding="same")(mxp2_deep)
mxp3_deep = MaxPooling2D((2, 2), padding="same")(conv3_deep)
encoding_layer_cnn_deep = Conv2D(1, (3, 3), padding='same')(mxp3_deep)
conv4_deep = Conv2DTranspose(1, (3, 3), padding='same')(encoding_layer_cnn_deep)
us1_deep = UpSampling2D((2, 2))(conv4_deep)
conv5_deep = Conv2DTranspose(50, (3, 3), padding='same')(us1_deep)
us2_deep = UpSampling2D((2, 2))(conv5_deep)
conv6_deep = Conv2DTranspose(20, (3, 3), padding='same')(us2_deep)
us3_deep = UpSampling2D((2, 2))(conv6_deep)
conv7_deep = Conv2DTranspose(10, (3, 3), padding='same')(us3_deep)
decoding_layer_cnn_deep = Conv2DTranspose(1, (3, 3), padding='same')(conv7_deep)




encoder_cnn_deep = Model(input_target_cnn_deep, encoding_layer_cnn_deep)
autoencoder_cnn_deep = Model(input_target_cnn_deep, decoding_layer_cnn_deep)
autoencoder_cnn_deep.compile(optimizer='rmsprop', loss='mean_squared_error')


# In[ ]:


autoencoder_cnn_deep.summary()


# In[ ]:


autoencoder_cnn_deep.fit(training_data_cnn, training_data_cnn, epochs=100, validation_data=(validation_data_cnn, validation_data_cnn), callbacks=[early_stopping])


# In[ ]:


autoencoder_cnn_deep.evaluate(test_data_cnn, test_data_cnn)


# In[ ]:


training_cnn_deep = encoder_cnn_deep.predict(training_data_cnn).reshape(len(training_data_cnn), 9)
validation_cnn_deep = encoder_cnn_deep.predict(validation_data_cnn).reshape(len(validation_data_cnn), 9)
test_cnn_deep = encoder_cnn_deep.predict(test_data_cnn).reshape(len(test_data_cnn), 9)


# In[ ]:


tr_data_cnn_deep = np.concatenate((training_cnn_deep, training_label, tr_flag), axis = 1)
vl_data_cnn_deep = np.concatenate((validation_cnn_deep, validation_label, vl_flag), axis = 1)
te_data_cnn_deep = np.concatenate((test_cnn_deep, test_label, te_flag), axis = 1)
data_cnn_deep = pd.DataFrame(np.concatenate((tr_data_cnn_deep, vl_data_cnn_deep, te_data_cnn_deep), axis = 0))


# In[ ]:


data_cnn_deep.columns = ['F_' + str(x) for x in range(9)] + ['Label', 'Flag']
pd.DataFrame(data_cnn_deep).to_csv("target_9_cnn_deep.csv", index = False)
