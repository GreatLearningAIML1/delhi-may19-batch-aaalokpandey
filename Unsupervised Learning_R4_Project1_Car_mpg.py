#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import os
import numpy as np
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


data = pd.read_csv('C:/Users/alok.1.pandey/Desktop/AiML/unsuperwised learning/cars-dataset.csv')


# In[ ]:


Q1. EDA & Pre-processing (Make sure to remove all non-numeric entries from numeric columns) – 2.5 points


# In[7]:


data.info()


# In[9]:


data.columns


# In[16]:


data.describe().transpose()


# In[17]:


data.sample(10)


# In[ ]:



From the analysis done above following can be observed
There are 6 numeric columns and 2 non numeric columns in the data.
numeric columns have all 398 entries (no missing values).
The column "hp" is non numeric but contains numeric data.
Hence column "hp" can be type converted to have numeric data type.


# In[20]:


try:
    data=data.astype({'hp':np.int64})
except Exception as error:
    print(error)
else:
    print(data['hp'].dtype)


# In[21]:



data[data['hp']=='?']


# In[23]:


data.drop(data[data['hp']=='?'].index,axis=0,inplace=True)


# In[24]:


data[data['hp']=='?']


# In[25]:


try:
    data=data.astype({'hp':int},inplace=True)
except Exception as error:
    print(error)
else:
    print(data['hp'].dtype)


# In[26]:



car_name_col = data["car name"]
data.drop("car name",inplace=True,axis=1)
data = data.apply(zscore)
data.insert(loc=0,column="car name",value=car_name_col)
data.sample(5)


# In[27]:


Q2. Use pair plot or scatter matrix to visualize how the different variables are related (Hint: The amount of Gaussian curves in the plot should give a visual identification of different clusters existing in the dataset) – 5 points


# In[32]:


sb.pairplot(data,diag_kind='kde',hue='cyl')


# In[ ]:



Q3. Use K Means or Hierarchical clustering to find out the optimal no of clusters in the data. Identify and separate the clusters (15 points)


# In[33]:


no_of_clusters = range(2,11)
cluster_error = []
for each_cluster in no_of_clusters:
    clusters = KMeans(each_cluster,n_init=5)
    clusters.fit(data.iloc[:,1:9])
    cluster_error.append(clusters.inertia_)

cluster_df = pd.DataFrame({"cluster_number":no_of_clusters,"cluster_error":cluster_error})
cluster_df 


# In[34]:


plt.plot( cluster_df.cluster_number, cluster_df.cluster_error, marker = "o" )


# In[ ]:


From the elbow plot it can be observed that number of clusters are 4


# In[35]:


cluster = KMeans(4,n_init=5)
cluster.fit(data.iloc[:,1:9])
cluster_centers = cluster.cluster_centers_
cluster_centers


# In[36]:


label_array=np.unique(cluster.labels_, return_counts=True)
plt.title("Bar chart of Cluster Vs Frequency")
plt.xticks(label_array[0])
plt.xlabel("Cluster Label")
plt.ylabel("Frequency")
plt.bar(x=label_array[0],height=label_array[1])


# In[37]:



cluster_df = pd.DataFrame(cluster_centers,columns=data.iloc[:,1:9].columns)
cluster_df.insert(column="Cluster_label",value=np.unique(cluster.labels_),loc=7)
cluster_df


# In[ ]:


Now 4 clusters have been found each with a centroid and label. Labels can be added to original dataset so that different clusters can be seperated based on their labels.


# In[38]:



data.insert(column="cluster_label",value=cluster.labels_,loc=8)
data.sample(10)


# In[ ]:



Q4. Use linear regression model on different clusters separately and print the coefficients of the models individually (7.5 points)


# In[39]:



#linear regression models
for each_label in label_array[0]:
    cluster_data = data[data['cluster_label']==each_label]
    linear_reg_model = LinearRegression()
    linear_reg_model.fit(X=cluster_data.iloc[:,2:8],y=cluster_data.iloc[:,1])
    print('coefficients for cluster label = {}'.format(each_label))
    print(linear_reg_model.coef_)
    print("")


# In[ ]:




