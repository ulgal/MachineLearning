#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np


# In[4]:


a_list = [0, 10, 20, 30]
an_array = np.array(a_list)           # numpy = provides effective calculations of numbers


# In[5]:


# a_list + 1         # error
an_array + 1


# In[6]:


an_array[2:]


# In[7]:


an_array[:2]


# In[8]:


an_array[1:3]


# In[9]:


an_array[-1]


# In[10]:


two_d_array = np.array([[1, 2, 3], [20, 30, 40]])


# In[11]:


two_d_array.sum()


# In[12]:


two_d_array.sum(axis=0)


# In[13]:


two_d_array.sum(axis=1)


# In[14]:


# most of math functions can be applied directly
np.cos(two_d_array)


# In[15]:


for i in [0, 1, 2, 3]:
    print(i)


# In[16]:


for i in range(4):
    print(i)


# In[17]:


for i in an_array:
    print(i)


# In[18]:


for i in an_array:
    if i > 15:
        print(i)
    elif i > 5:
        print('abc')
    else:
        print('XYZ')


# In[19]:


# Pythonic way to apply a simple function.
[x + 30 for x in range(4)]


# In[20]:


bool_array = (an_array >= 3)
bool_array


# In[21]:


an_array[bool_array]

