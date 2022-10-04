#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


x_train=np.array([1.0,2.0])
y_train=np.array([300.0,500.0])
print(f"x_train={x_train}")
print(f"y_train={y_train}")


# In[3]:


print(f"x_train.shape: {x_train.shape}")
m=x_train.shape[0]
print(f"Number of training examples is : {m}")


# In[4]:


i=1
x_i=x_train[i]
y_i=y_train[i]
print(f"(x^({i}),y^({i}))=({x_i},{y_i})")


# In[ ]:




