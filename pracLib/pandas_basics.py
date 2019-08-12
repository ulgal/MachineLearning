#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.DataFrame({'AA': [10, 20, 30], 'BB': ['a', 'bcd', 'efg']}, index=[3, 2, 8])
df


# In[3]:


df['AA']           # contents of the column named 'AA'. return type = pd.Series
df.AA


# In[4]:


df[['AA']]        # contents of the column named 'AA'. return type = pd.DataFrame


# In[12]:


df.AA[8]      # the value corresponding to key=8. NOT 8th element.


# In[17]:


df.at['AA', 8] = 10
df.AA.loc[8]


# In[6]:


df.AA.iloc[2]         # integer location. 2nd element.


# In[7]:


df.loc[[8, 3], 'BB']


# In[8]:


df.iloc[:2, 1]


# In[9]:


df.AA.sum()


# In[10]:


df.AA.mean()


# In[11]:


[len(x) for x in df.BB]

