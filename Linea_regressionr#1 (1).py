#!/usr/bin/env python
# coding: utf-8

# # THE FIRST LEARNING

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


# In[9]:


plt.scatter(x_train,y_train,marker='x',c='g')
plt.title("Housing prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.show()


# In[13]:


w=200
b=100
print(f"w:{w}")
print(f"b:{b}")


# In[14]:


def compute_model_output(x,w,b):
    m=x.shape[0]
    f_wb=np.zeros(m)
    for i in range(m):
        f_wb[i]=w*x[i] + b
    return f_wb


# In[15]:


tmp_f_wb=compute_model_output(x_train,w,b)
    
plt.plot(x_train,tmp_f_wb,c='r',label='Our prediction')

plt.scatter(x_train,y_train,marker='x',c='b',label='Actual values')

plt.title("Housing prices")
plt.ylabel('Prices')
plt.xlabel("Size")
plt.legend()
plt.show()


# In[16]:


w=200
b=100
x_i=1.2
cost=w*x_i +b
print(f"{cost:.0f} dollars")

