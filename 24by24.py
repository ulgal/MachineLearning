#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams["figure.figsize"] = (8, 7)

from keras.models import Model
from keras import layers


# In[ ]:


filelist = os.listdir("./@folderName/")
filelist = filelist[1:]
target = {}

for x, a_file in enumerate(filelist):
    target[x] = pd.read_csv("./@folderName/" + a_file, index_col=0)


# In[ ]:


[x[29:35] for x in filelist][87]


# ### pp ML TSNE data

# In[ ]:


training_data = np.array([])
training_data = training_data.reshape(0, 24, 24)
for week_idx in range(32, 61):
    answer = target[week_idx]
    for i in range(week_idx - 23, week_idx)[::-1]:
        answer = pd.merge(target[i], answer, how='right', left_index=True, right_index=True)
    answer.fillna(0, inplace=True)

    answer = np.fmin(1, answer.div(answer.quantile(0.9, 1) + 1, axis=0))
    tri_di_ans = answer.to_numpy().reshape(answer.shape[0], 24, 24)
    training_data = np.append(training_data, tri_di_ans, axis = 0)


# In[ ]:


test_data = np.array([])
test_data= test_data.reshape(0, 24, 24)
for week_idx in range(87, 90):
    answer = target[week_idx]
    for i in range(week_idx - 23, week_idx)[::-1]:
        answer = pd.merge(target[i], answer, how='right', left_index=True, right_index=True)
    answer.fillna(0, inplace=True)

    answer = np.fmin(1, answer.div(answer.quantile(0.9, 1) + 1, axis=0))
    tri_di_ans = answer.to_numpy().reshape(answer.shape[0], 24, 24)
    test_data = np.append(test_data, tri_di_ans, axis = 0)


# In[ ]:


test_data.shape


# ### 24x24 gray scale image

# In[ ]:


# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 7)


# In[ ]:


plt.imshow(training_data[12], cmap='gray_r')
plt.show()


# ### ML, 576D to 2D to 576D

# In[ ]:


input_target = layers.Input(shape=(24, 24))
flat_input = layers.Flatten()(input_target)
encoding_layer = layers.Dense(2)(flat_input)  # None (linear), Relu, sigmoid
embedding_layer = layers.Dense(576)(encoding_layer)
decoding_layer = layers.Reshape((24, 24))(embedding_layer)

model = Model(input=input_target, output=decoding_layer)
encoder = Model(input=input_target, output=encoding_layer)
model.compile(optimizer='rmsprop', loss='mean_squared_error')


# In[ ]:


model.summary()


# In[ ]:


# training case loss
model.fit(training_data, training_data, epochs=10)


# In[ ]:


model.evaluate(test_data, test_data)


# In[ ]:


target_recovered = model.predict(target)


# In[ ]:


fig = plt.figure()

X, Y = np.meshgrid(np.arange(24), np.arange(23, -1, -1))
Z = target[1500]

ax = fig.gca(projection='3d')
ax.view_init(elev=40, azim=280)
ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=False);


# In[ ]:


fig = plt.figure()

X, Y = np.meshgrid(np.arange(24), np.arange(23, -1, -1))
Z = target_recovered[1500]

ax = fig.gca(projection='3d')
ax.view_init(elev=40, azim=280)
ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=False);


# ### TSNE

# In[ ]:


t_model = TSNE()

encoded = encoder.predict(training_data)
tf_for_TSNE = t_model.fit_transform(encoded)

repeated = np.append(np.repeat(0, tri_di_ans_1.shape[0]), np.repeat(1, tri_di_ans_2.shape[0]))
repeated = repeated.reshape(repeated.size, 1)
tf_shuffle = np.append(tf_for_TSNE, repeated, axis=1)


# In[ ]:


np.random.shuffle(tf_shuffle)

plt.scatter(tf_shuffle[:, 0], tf_shuffle[:, 1], c=tf_shuffle[:, 2])
plt.show()
