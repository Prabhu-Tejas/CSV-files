#!/usr/bin/env python
# coding: utf-8

# # Logistic regression on Iris dataset

# In[6]:


import os


# In[7]:


os.getcwd()


# In[8]:


os.chdir('C:\\Users\\prabh\\OneDrive\\Desktop\\Letsupgrade FS Datascience Jan-22')


# In[9]:


os.getcwd()


# In[10]:


import warnings
warnings.filterwarnings("ignore")


# In[33]:


import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns


# In[12]:


df=pd.read_csv('Iris.csv')


# In[13]:


df.head()


# In[14]:


df.drop(columns='Id', axis=1,inplace=True)


# In[15]:


df.info()


# # Performing basic Data analysis

# In[16]:


df.describe()


# In[17]:


df.shape


# In[18]:


df.describe().T


# # Performing EDA

# In[19]:


df.hist(figsize=(12,12))


# In[20]:


sns.boxplot(df["SepalLengthCm"])


# In[21]:


sns.boxplot(df["SepalWidthCm"])


# In[22]:


sns.boxplot(df["PetalLengthCm"])


# In[23]:


sns.boxplot(df["PetalWidthCm"])


# In[24]:


sns.distplot(df['SepalLengthCm'])


# In[25]:


sns.distplot(df['SepalWidthCm'])


# In[26]:


sns.distplot(df['PetalLengthCm'])


# In[27]:


sns.distplot(df['PetalWidthCm'])


# In[30]:


sns.pairplot(df, hue="Species")


# Observations:
# 
# Petal length and petal width are the most useful features,setosa is easily distingushble, but there is a huge overlap in virginica and versicolor.

# In[34]:


import matplotlib.pyplot as plt


# In[36]:


plt.figure(figsize=(16, 8))

for i, col in enumerate(df.drop("Species", axis=1).columns):
    
    plt.subplot(1, 4, i+1)
    counts, bins = np.histogram(df[col], bins=10, density = True)
    pdf = counts/(sum(counts))

    cdf = np.cumsum(pdf)
    plt.plot(bins[1:] ,pdf, label="PDF")
    plt.plot(bins[1:], cdf, label="CDF")
    plt.xlabel(col)
    plt.title(f"PDF Vs CDF for {col}")
    plt.legend(loc="best")
    
plt.show()


# Logistic Regression ML algorithm

# In[37]:


x=df.iloc[:,:-1]


# In[38]:


x


# In[39]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
y=df['Species']
y
y.value_counts()


# In[40]:


from sklearn.model_selection import train_test_split


# In[41]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2, random_state=3)


# In[42]:


print('xtrain size:', xtrain.shape)
print('xtest size:', xtest.shape)


# In[43]:


from sklearn.linear_model import LogisticRegression 
log= LogisticRegression()


# In[44]:


log.fit(xtrain,ytrain)
print("training completed")
ypred=log.predict(xtest)
print("testing completed")

ypred


# In[45]:


#Measure the performance of the model
from sklearn.metrics  import confusion_matrix

print('Performance Measures of Logestic regression Model')

print('Confusion Matrix:\n',confusion_matrix(ytest,ypred))


# In[46]:


from sklearn.metrics import classification_report
print(classification_report(ytest, ypred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




