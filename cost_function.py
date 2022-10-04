#!/usr/bin/env python
# coding: utf-8

# # COST FUNCTION

# In[21]:


import numpy as np
import matplotlib.pyplot as plt


# In[4]:


x_train=np.array([1.0,2.0])
y_train=np.array([300.0,500.0])


# In[5]:


def compute_cost(x,y,w,b):
    m=x.shape[0]
    cost_sum=0
    for i in range(m):
        f_wb=w*x[i]+b
        cost=(f_wb - y[i])**2
        cost_sum=cost_sum+cost
    total_cost=(1/(2*m))*cost_sum
    
    return total_cost


# In[22]:


plt_intuition(x_train,y_train)


# In[23]:


x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])


# In[24]:


plt.close('all') 
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)


# In[25]:


soup_bowl()


# In[ ]:




