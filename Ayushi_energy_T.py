
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


data = pd.read_csv('D:/datasets/energy_dataset/train_dataset.csv')


# In[5]:


data.describe()


# In[6]:


x = data.drop(['Energy'], axis = 1)
y = data['Energy']


# In[7]:


from sklearn.preprocessing import Normalizer


# In[8]:


transformer = Normalizer().fit(x)
x = transformer.transform(x)


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2 , random_state = 42)


# In[11]:


corr_matrix= data.corr()
corr_matrix['Energy'].sort_values(ascending=False)


# In[170]:


from sklearn.neighbors import KNeighborsRegressor
# neigh = KNeighborsRegressor(n_neighbors= 250)

di = {}

for i in range(1, 500):
    neigh = KNeighborsRegressor(n_neighbors= i)
    neigh.fit(x_train, y_train)
    y_pred = neigh.predict(x_test)
    r2 = r2_score(y_test, y_pred)
#     print(r2)
    di[i]= r2
    
  


# In[177]:


import operator
max(di.items(), key=operator.itemgetter(1))[0]


# In[178]:


neigh = KNeighborsRegressor(n_neighbors= 131)
neigh.fit(x_train, y_train)


# In[179]:


y_pred = neigh.predict(x_test)


# In[180]:


from sklearn.metrics import r2_score


# In[181]:


r2_score(y_test, y_pred)


# In[142]:


from sklearn.linear_model import LinearRegression


# In[143]:


reg = LinearRegression().fit(x_train, y_train)


# In[144]:


reg.score(x_train, y_train)


# In[147]:


y_pred_reg = reg.predict(x_test)


# In[182]:


r2_score(y_test, y_pred_reg)


# In[183]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


regr = RandomForestRegressor(max_depth=100, random_state=10,
                             n_estimators=750)


# In[ ]:


regr.fit(x_train, y_train)


# In[ ]:


y_pred_rn = regr.predict(x_test)


# In[ ]:


r2_score(y_test, y_pred_rn)

