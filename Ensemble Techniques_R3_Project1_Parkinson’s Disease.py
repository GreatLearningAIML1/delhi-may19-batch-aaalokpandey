#!/usr/bin/env python
# coding: utf-8

# In[166]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # Data Set Information:
# 
# This dataset is composed of a range of biomedical voice measurements from 
# 31 people, 23 with Parkinson's disease (PD). Each column in the table is a 
# particular voice measure, and each row corresponds one of 195 voice 
# recording from these individuals ("name" column). The main aim of the data 
# is to discriminate healthy people from those with PD, according to "status" 
# column which is set to 0 for healthy and 1 for PD.
# 
# The data is in ASCII CSV format. The rows of the CSV file contain an 
# instance corresponding to one voice recording. There are around six 
# recordings per patient, the name of the patient is identified in the first 
# column.For further information or to pass on comments, please contact Max 
# Little (littlem '@' robots.ox.ac.uk).
# 
# Further details are contained in the following reference -- if you use this 
# dataset, please cite:
# Max A. Little, Patrick E. McSharry, Eric J. Hunter, Lorraine O. Ramig (2008), 
# 'Suitability of dysphonia measurements for telemonitoring of Parkinson's disease', 
# IEEE Transactions on Biomedical Engineering (to appear).

# Q.1. Load the dataset11. Load the dataset
# Q.2. It is always a good practice to eye-ball raw data to get a feel of the data in terms of
# number of structure of the file, number of attributes, types of attributes and a general idea of likely challenges in the dataset.

# In[167]:


df = pd.read_csv('C:/Users/alok.1.pandey/Downloads/parkinsons.csv')


# In[168]:


df.info()


# In[169]:


df.head(5)


**Attribute Information:

**Matrix column entries (attributes):

**name - ASCII subject name and recording 

**MDVP:Fo(Hz) - Average vocal fundamental frequency

**MDVP:Fhi(Hz) - Maximum vocal fundamental frequency

**MDVP:Flo(Hz) - Minimum vocal fundamental frequency

**MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP - Several 

**measures of variation in fundamental frequency

**MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA - Several measures of variation in amplitude

**NHR,HNR - Two measures of ratio of noise to tonal components in the voice

**status - Health status of the subject (one) - Parkinson's, (zero) - healthy

**RPDE,D2 - Two nonlinear dynamical complexity measures

**DFA - Signal fractal scaling exponent

**spread1,spread2,PPE - Three nonlinear measures of fundamental frequency variation **
# In[170]:


df['status'].value_counts()


# In[171]:


plt.figure(figsize=(14,14))

sns.heatmap(df.corr())


# Q.3. Using univariate &amp; bivariate analysis to check the individual attributes for their basic statistic such as central values, spread, tails etc. What are your observations?Q.3.

# In[172]:


df.describe()


# In[ ]:





# In[ ]:


Q.4. Split the dataset into training and test set in the ratio of 70:30.


# In[173]:


df.columns


# In[175]:


y = df['status']


# In[176]:


X = df[['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
       'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
       'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
       'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
       'spread1', 'spread2', 'D2', 'PPE']]


# In[177]:


from sklearn.model_selection import train_test_split

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.30)


# In[178]:


print(Xtest.shape)
print(Xtrain.shape)
print(ytrain.shape)
print(ytest.shape)


# In[ ]:


Q.5.Create the model using “entropy” method of reducing the entropy and fit it to training data.
Q.6.Test the model on test data and what is the accuracy achieved. Capture the predicted values and do a crosstab. 
Q.7.Use regularization parameters of max_depth, min_sample_leaf to recreate the model. What is the impact on the model accuracy? How does regularization help? 


# In[180]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score


# dt_model =DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
#                        max_features=None, max_leaf_nodes=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=1, min_samples_split=2,
#                        min_weight_fraction_leaf=0.0, presort=False,
#                        random_state=None, splitter='best')

# In[181]:


dt_model =DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=10,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')


# In[182]:


dt_model_fit = dt_model.fit(Xtrain,ytrain)


# In[183]:


dt_model_fit


# In[184]:


predected =dt_model_fit.predict(Xtest)


# In[185]:


confusion_matrix(ytest,predected)


# In[186]:


accuracy_score(ytest,predected)


# In[187]:


dt_model =DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')


# In[188]:


dt_model =DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=10,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')


# In[189]:


dt_model_fit = dt_model.fit(Xtrain,ytrain)


# In[190]:


confusion_matrix(ytest,predected)


# In[191]:


accuracy_score(ytest,predected)


# In[ ]:


get_ipython().set_next_input('Q.8.Next implement the decision tree using Random Forest. What is the optimal number of trees that gives the best result');get_ipython().run_line_magic('pinfo', 'result')


# In[209]:


# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 1000 decision trees
rf = RandomForestClassifier()
# Train the model on training data
rf.fit(Xtrain, ytrain);


# In[210]:


rf


# In[211]:


predict_random =rf.predict(Xtest)


# In[212]:


(predict_random)


# In[214]:


accuracy_score(ytest,predict_random)


# In[ ]:




