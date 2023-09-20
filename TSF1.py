#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd

url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"

df = pd.read_csv(url)


# In[13]:


df.head()


# In[14]:


import matplotlib.pyplot as plt
df.plot.scatter(x='Hours', y='Scores', figsize=(8,6))
plt.show()


# In[15]:


X=df["Hours"]
y=df["Scores"]


# In[16]:


X.head()


# In[17]:


y.head()


# In[18]:


X = X.to_numpy()  # Convert to a NumPy array
# Reshape X into a 2D array with one column
X = X.reshape(-1, 1)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)


# In[19]:


import numpy as np
X_new=np.array([[9.25]])
predictions = lr.predict(X_new)

# Print the predictions
print(predictions)


# In[ ]:





# In[ ]:




