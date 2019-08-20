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
from keras.layers import Dropout
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
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

from scipy.stats import binom
from scipy.stats import bernoulli
import scipy
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 10)





## Load Data

data_cnnsp2st = pd.read_csv("data_9_cnn_samepad_2str.csv")

training_cnnsp2st = data_cnnsp2st[data_cnnsp2st['Flag'] == 3]
validation_cnnsp2st = data_cnnsp2st[data_cnnsp2st['Flag'] == 4]
test_cnnsp2st = data_cnnsp2st[data_cnnsp2st['Flag'] == 5]

training_data_cnnsp2st, training_label_cnnsp2st = training_cnnsp2st.iloc[:, :9].to_numpy(), training_cnnsp2st.iloc[:, 9:10].to_numpy()
validation_data_cnnsp2st, validation_label_cnnsp2st = validation_cnnsp2st.iloc[:, :9].to_numpy(), validation_cnnsp2st.iloc[:, 9:10].to_numpy()
test_data_cnnsp2st, test_label_cnnsp2st = test_cnnsp2st.iloc[:, :9].to_numpy(), test_cnnsp2st.iloc[:, 9:10].to_numpy()





# tSNE

data_tSNE = np.append(random.sample(test_data_cnnsp2st[:40090].tolist(), 10000), random.sample(test_data_cnnsp2st[40090:len(test_data_cnnsp2st)].tolist(), 10000), axis=0)
tSNE_label = np.append(np.repeat(0, 10000), np.repeat(1, 10000), axis=0)
tSNE_label = tSNE_label.reshape(len(tSNE_label), 1)

t_model = TSNE()
tf_TSNE = t_model.fit_transform(data_tSNE)
tf_shuffle = np.append(tf_TSNE, tSNE_label, axis=1)
np.random.shuffle(tf_shuffle)

plt.rcParams["figure.figsize"] = (10, 10)


np.random.shuffle(tf_shuffle)
plt.scatter(tf_shuffle[:, 0], tf_shuffle[:, 1], c=tf_shuffle[:, 2], s = 10)
plt.show()


colors = np.random.choice([0, 1], 20000)
plt.scatter(tf_shuffle[:, 0], tf_shuffle[:, 1], c=colors, s = 10)
plt.show()





### CNN  2 stride, same padding
input_layer_cnnsp2st= Input(shape = (9,))
hidden_layer_cnnsp2st_1 = Dense(50, activation = 'relu')(input_layer_cnnsp2st)
hidden_layer_cnnsp2st_2 = Dense(25, activation = 'relu')(hidden_layer_cnnsp2st_1)
hidden_layer_cnnsp2st_3 = Dense(10, activation = 'relu')(hidden_layer_cnnsp2st_2)
hidden_layer_cnnsp2st_4 = Dense(4, activation = 'relu')(hidden_layer_cnnsp2st_3)
classified_layer_cnnsp2st = Dense(1, activation = 'sigmoid')(hidden_layer_cnnsp2st_4)
classifier_cnnsp2st = Model(input_layer_cnnsp2st, classified_layer_cnnsp2st)
classifier_cnnsp2st.compile(optimizer='rmsprop', loss='binary_crossentropy')
classifier_cnnsp2st.summary()
classifier_cnnsp2st.fit(training_data_cnnsp2st, training_label_cnnsp2st, epochs = 30, validation_data = (validation_data_cnnsp2st, validation_label_cnnsp2st))
classifier_cnnsp2st.evaluate(test_data_cnnsp2st, test_label_cnnsp2st)





## ROC curve, AUC
predict_cnnsp2st = classifier_cnnsp2st.predict(test_data_cnnsp2st)
fpr_cnnsp2st, tpr_cnnsp2st, threshold_cnnsp2st = roc_curve(test_label_cnnsp2st, predict_cnnsp2st)
plt.plot(fpr_cnnsp2st, tpr_cnnsp2st, 'o-', label="Logistic Regression")
plt.plot([0, 1], [0, 1], 'k--', label="random guess")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()
print(auc(fpr_cnnsp2st, tpr_cnnsp2st))





## F1 Score
f1_score(test_label_cnnsp2st, (predict_cnnsp2st>0.5).astype(int))





## Accuracy
accuracy_score(test_label_cnnsp2st, (predict_cnnsp2st>0.5).astype(int))





### Confusion Matrix
trans_predict = (predict_cnnsp2st>0.5).astype(int)

tp, fn, fp, tn = 0, 0, 0, 0
for x in range(0, len(trans_predict)):
    if trans_predict[x] == 1 and test_label_cnnsp2st[x] == 1:
        tp+=1
    elif trans_predict[x] == 1 and test_label_cnnsp2st[x] == 0:
        fp+=1
    elif trans_predict[x] == 0 and test_label_cnnsp2st[x] == 1:
        fn+=1
    elif trans_predict[x] == 0 and test_label_cnnsp2st[x] == 0:
        tn+=1
print(tp, fn, fp, tn)


accuracy = ( tp + tn ) / ( tp + fn + fp + tn)
precision = ( tp ) / (tp + fp)
recall = ( tp ) / (tp+ fn)
f_score = ( 2 * precision * recall ) / ( precision + recall )

print(accuracy, precision, recall, f_score)


### binom test

print((10**16)*binom.cdf(60000, 80550, 0.5))
print(scipy.stats.binom_test(60123, 80550, 1/2, alternative='less'))
