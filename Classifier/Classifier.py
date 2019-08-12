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

import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


# ### PCA

# In[ ]:


target_pca = pd.read_csv("target_9_pca.csv")

training_pca = target_pca[target_pca['Flag'] == 3]
validation_pca = target_pca[target_pca['Flag'] == 4]
test_pca = target_pca[target_pca['Flag'] == 5]

training_data_pca, training_label_pca = training_pca.iloc[:, :9].to_numpy(), training_pca.iloc[:, 9:10].to_numpy()
validation_data_pca, validation_label_pca = validation_pca.iloc[:, :9].to_numpy(), validation_pca.iloc[:, 9:10].to_numpy()
test_data_pca, test_label_pca = test_pca.iloc[:, :9].to_numpy(), test_pca.iloc[:, 9:10].to_numpy()


# In[ ]:


input_target_pca= Input(shape = (9,))
hidden_layer_pca_1 = Dense(40, activation = 'relu')(input_target_pca)
hidden_layer_pca_2 = Dense(20, activation = 'relu')(hidden_layer_pca_1)
hidden_layer_pca_3 = Dense(10, activation = 'relu')(hidden_layer_pca_2)
hidden_layer_pca_4 = Dense(4, activation = 'relu')(hidden_layer_pca_3)
classified_layer_pca = Dense(1, activation = 'sigmoid')(hidden_layer_pca_4)

classifier_pca = Model(input_target_pca, classified_layer_pca)
classifier_pca.compile(optimizer='rmsprop', loss='binary_crossentropy')


classifier_pca.fit(training_data_pca, training_label_pca, epochs = 100, validation_data = (validation_data_pca, validation_label_pca), callbacks = [early_stopping])

classifier_pca.evaluate(test_data_pca, test_label_pca)

predict_pca = classifier_pca.predict(test_data_pca)

fpr_pca, tpr_pca, threshold_pca = roc_curve(test_label_pca, predict_pca)


# In[ ]:


classifier_pca.summary()


# ### basic Autoencoder

# In[ ]:


target_bae = pd.read_csv("target_9_basic_autoencoder.csv")

training_bae = target_bae[target_bae['Flag'] == 3]
validation_bae = target_bae[target_bae['Flag'] == 4]
test_bae = target_bae[target_bae['Flag'] == 5]

training_data_bae, training_label_bae = training_bae.iloc[:, :9].to_numpy(), training_bae.iloc[:, 9:10].to_numpy()
validation_data_bae, validation_label_bae = validation_bae.iloc[:, :9].to_numpy(), validation_bae.iloc[:, 9:10].to_numpy()
test_data_bae, test_label_bae = test_bae.iloc[:, :9].to_numpy(), test_bae.iloc[:, 9:10].to_numpy()


# In[ ]:


input_target_bae= Input(shape = (9,))
hidden_layer_bae_1 = Dense(20, activation = 'relu')(input_target_bae)
hidden_layer_bae_2 = Dense(10, activation = 'relu')(hidden_layer_bae_1)
hidden_layer_bae_3 = Dense(4, activation = 'relu')(hidden_layer_bae_2)
classified_layer_bae = Dense(1, activation = 'sigmoid')(hidden_layer_bae_3)

classifier_bae = Model(input_target_bae, classified_layer_bae)
classifier_bae.compile(optimizer='rmsprop', loss='binary_crossentropy')


classifier_bae.fit(training_data_bae, training_label_bae, epochs = 100, validation_data = (validation_data_bae, validation_label_bae), callbacks = [early_stopping])

classifier_bae.evaluate(test_data_bae, test_label_bae)

predict_bae = classifier_bae.predict(test_data_bae)

fpr_bae, tpr_bae, threshold_bae = roc_curve(test_label_bae, predict_bae)


# ### Autoencoder

# In[ ]:


target_ae = pd.read_csv("target_9_autoencoder.csv")

training_ae = target_ae[target_ae['Flag'] == 3]
validation_ae = target_ae[target_ae['Flag'] == 4]
test_ae = target_ae[target_ae['Flag'] == 5]

training_data_ae, training_label_ae = training_ae.iloc[:, :9].to_numpy(), training_ae.iloc[:, 9:10].to_numpy()
validation_data_ae, validation_label_ae = validation_ae.iloc[:, :9].to_numpy(), validation_ae.iloc[:, 9:10].to_numpy()
test_data_ae, test_label_ae = test_ae.iloc[:, :9].to_numpy(), test_ae.iloc[:, 9:10].to_numpy()


# In[ ]:


input_target_ae= Input(shape = (9,))
hidden_layer_ae_1 = Dense(100, activation = 'relu')(input_target_ae)
hidden_layer_ae_2 = Dense(50, activation = 'relu')(hidden_layer_ae_1)
hidden_layer_ae_3 = Dense(25, activation = 'relu')(hidden_layer_ae_2)
hidden_layer_ae_4 = Dense(10, activation = 'relu')(hidden_layer_ae_3)
hidden_layer_ae_5 = Dense(5, activation = 'relu')(hidden_layer_ae_4)
hidden_layer_ae_6 = Dense(2, activation = 'relu')(hidden_layer_ae_5)
classified_layer_ae = Dense(1, activation = 'sigmoid')(hidden_layer_ae_6)
#
classifier_ae = Model(input_target_ae, classified_layer_ae)
classifier_ae.compile(optimizer='rmsprop', loss='binary_crossentropy')


classifier_ae.fit(training_data_ae, training_label_ae, epochs = 100, validation_data = (validation_data_ae, validation_label_ae), callbacks = [early_stopping])

classifier_ae.evaluate(test_data_ae, test_label_ae)

predict_ae = classifier_ae.predict(test_data_ae)

fpr_ae, tpr_ae, threshold_ae = roc_curve(test_label_ae, predict_ae)


# ### CNN no stride, same padding

# In[ ]:


target_cnnNst = pd.read_csv("target_9_cnn_deep.csv")

training_cnnNst = target_cnnNst[target_cnnNst['Flag'] == 3]
validation_cnnNst = target_cnnNst[target_cnnNst['Flag'] == 4]
test_cnnNst = target_cnnNst[target_cnnNst['Flag'] == 5]

training_data_cnnNst, training_label_cnnNst = training_cnnNst.iloc[:, :9].to_numpy(), training_cnnNst.iloc[:, 9:10].to_numpy()
validation_data_cnnNst, validation_label_cnnNst = validation_cnnNst.iloc[:, :9].to_numpy(), validation_cnnNst.iloc[:, 9:10].to_numpy()
test_data_cnnNst, test_label_cnnNst = test_cnnNst.iloc[:, :9].to_numpy(), test_cnnNst.iloc[:, 9:10].to_numpy()


# In[ ]:


input_target_cnnNst= Input(shape = (9,))
hidden_layer_cnnNst_1 = Dense(20, activation = 'relu')(input_target_cnnNst)
hidden_layer_cnnNst_2 = Dense(10, activation = 'relu')(hidden_layer_cnnNst_1)
hidden_layer_cnnNst_3 = Dense(4, activation = 'relu')(hidden_layer_cnnNst_2)
classified_layer_cnnNst = Dense(1, activation = 'sigmoid')(hidden_layer_cnnNst_3)

classifier_cnnNst = Model(input_target_cnnNst, classified_layer_cnnNst)
classifier_cnnNst.compile(optimizer='rmsprop', loss='binary_crossentropy')


classifier_cnnNst.fit(training_data_cnnNst, training_label_cnnNst, epochs = 100, validation_data = (validation_data_cnnNst, validation_label_cnnNst), callbacks = [early_stopping])

classifier_cnnNst.evaluate(test_data_cnnNst, test_label_cnnNst)

predict_cnnNst = classifier_cnnNst.predict(test_data_cnnNst)

fpr_cnnNst, tpr_cnnNst, threshold_cnnNst = roc_curve(test_label_cnnNst, predict_cnnNst)


# ### CNN  2 stride, same padding

# In[ ]:


target_cnnsp2st = pd.read_csv("target_9_cnn_samepad_2str.csv")

training_cnnsp2st = target_cnnsp2st[target_cnnsp2st['Flag'] == 3]
validation_cnnsp2st = target_cnnsp2st[target_cnnsp2st['Flag'] == 4]
test_cnnsp2st = target_cnnsp2st[target_cnnsp2st['Flag'] == 5]

training_data_cnnsp2st, training_label_cnnsp2st = training_cnnsp2st.iloc[:, :9].to_numpy(), training_cnnsp2st.iloc[:, 9:10].to_numpy()
validation_data_cnnsp2st, validation_label_cnnsp2st = validation_cnnsp2st.iloc[:, :9].to_numpy(), validation_cnnsp2st.iloc[:, 9:10].to_numpy()
test_data_cnnsp2st, test_label_cnnsp2st = test_cnnsp2st.iloc[:, :9].to_numpy(), test_cnnsp2st.iloc[:, 9:10].to_numpy()


# In[ ]:


input_target_cnnsp2st= Input(shape = (9,))
hidden_layer_cnnsp2st_1 = Dense(64, activation = 'relu')(input_target_cnnsp2st)
hidden_layer_cnnsp2st_2 = Dense(32, activation = 'relu')(hidden_layer_cnnsp2st_1)
hidden_layer_cnnsp2st_3 = Dense(16, activation = 'relu')(hidden_layer_cnnsp2st_2)
hidden_layer_cnnsp2st_4 = Dense(8, activation = 'relu')(hidden_layer_cnnsp2st_3)
hidden_layer_cnnsp2st_5 = Dense(4, activation = 'relu')(hidden_layer_cnnsp2st_4)
classified_layer_cnnsp2st = Dense(1, activation = 'sigmoid')(hidden_layer_cnnsp2st_5)

classifier_cnnsp2st = Model(input_target_cnnsp2st, classified_layer_cnnsp2st)
classifier_cnnsp2st.compile(optimizer='rmsprop', loss='binary_crossentropy')


classifier_cnnsp2st.fit(training_data_cnnsp2st, training_label_cnnsp2st, epochs = 100, validation_data = (validation_data_cnnsp2st, validation_label_cnnsp2st), callbacks = [early_stopping])

classifier_cnnsp2st.evaluate(test_data_cnnsp2st, test_label_cnnsp2st)

predict_cnnsp2st = classifier_cnnsp2st.predict(test_data_cnnsp2st)

fpr_cnnsp2st, tpr_cnnsp2st, threshold_cnnsp2st = roc_curve(test_label_cnnsp2st, predict_cnnsp2st)


# ### CNN 2 stride, valid padding

# In[ ]:


target_cnnvp2st = pd.read_csv("target_9_cnn_validpad_2a3str.csv")

training_cnnvp2st = target_cnnvp2st[target_cnnvp2st['Flag'] == 3]
validation_cnnvp2st = target_cnnvp2st[target_cnnvp2st['Flag'] == 4]
test_cnnvp2st = target_cnnvp2st[target_cnnvp2st['Flag'] == 5]

training_data_cnnvp2st, training_label_cnnvp2st = training_cnnvp2st.iloc[:, :9].to_numpy(), training_cnnvp2st.iloc[:, 9:10].to_numpy()
validation_data_cnnvp2st, validation_label_cnnvp2st = validation_cnnvp2st.iloc[:, :9].to_numpy(), validation_cnnvp2st.iloc[:, 9:10].to_numpy()
test_data_cnnvp2st, test_label_cnnvp2st = test_cnnvp2st.iloc[:, :9].to_numpy(), test_cnnvp2st.iloc[:, 9:10].to_numpy()


# In[ ]:


input_target_cnnvp2st= Input(shape = (9,))
hidden_layer_cnnvp2st_1 = Dense(20, activation = 'relu')(input_target_cnnvp2st)
hidden_layer_cnnvp2st_2 = Dense(10, activation = 'relu')(hidden_layer_cnnvp2st_1)
hidden_layer_cnnvp2st_3 = Dense(4, activation = 'relu')(hidden_layer_cnnvp2st_2)
classified_layer_cnnvp2st = Dense(1, activation = 'sigmoid')(hidden_layer_cnnvp2st_3)

classifier_cnnvp2st = Model(input_target_cnnvp2st, classified_layer_cnnvp2st)
classifier_cnnvp2st.compile(optimizer='rmsprop', loss='binary_crossentropy')


classifier_cnnvp2st.fit(training_data_cnnvp2st, training_label_cnnvp2st, epochs = 100, validation_data = (validation_data_cnnvp2st, validation_label_cnnvp2st), callbacks = [early_stopping])

classifier_cnnvp2st.evaluate(test_data_cnnvp2st, test_label_cnnvp2st)

predict_cnnvp2st = classifier_cnnvp2st.predict(test_data_cnnvp2st)

fpr_cnnvp2st, tpr_cnnvp2st, threshold_cnnvp2st = roc_curve(test_label_cnnvp2st, predict_cnnvp2st)


# ## ROC curve

# In[ ]:


plt.plot(fpr_pca, tpr_pca, 'o-', label="Logistic Regression")
plt.plot([0, 1], [0, 1], 'k--', label="random guess")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('PCA')
plt.show()
print(auc(fpr_pca, tpr_pca))


# In[ ]:


plt.plot(fpr_bae, tpr_bae, 'o-', label="Logistic Regression")
plt.plot([0, 1], [0, 1], 'k--', label="random guess")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Basic Autoencoder')
plt.show()
print(auc(fpr_bae, tpr_bae))


# In[ ]:


plt.plot(fpr_ae, tpr_ae, 'o-', label="Logistic Regression")
plt.plot([0, 1], [0, 1], 'k--', label="random guess")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Autoencoder')
plt.show()
print(auc(fpr_ae, tpr_ae))


# In[ ]:


plt.plot(fpr_cnnNst, tpr_cnnNst, 'o-', label="Logistic Regression")
plt.plot([0, 1], [0, 1], 'k--', label="random guess")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('cnn, Deep, No stride')
plt.show()
print(auc(fpr_cnnNst, tpr_cnnNst))


# In[ ]:


plt.plot(fpr_cnnsp2st, tpr_cnnsp2st, 'o-', label="Logistic Regression")
plt.plot([0, 1], [0, 1], 'k--', label="random guess")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('CNN, same padding, 2 strides')
plt.show()
print(auc(fpr_cnnsp2st, tpr_cnnsp2st))


# In[ ]:


plt.plot(fpr_cnnvp2st, tpr_cnnvp2st, 'o-', label="Logistic Regression")
plt.plot([0, 1], [0, 1], 'k--', label="random guess")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('CNN, valid padding, 2 strides')
plt.show()
print(auc(fpr_cnnvp2st, tpr_cnnvp2st))
