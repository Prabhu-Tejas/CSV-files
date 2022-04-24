#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[4]:


os.getcwd()


# In[5]:


os.chdir("C:\\Users\\prabh\\OneDrive\\Desktop\\Letsupgrade FS Datascience Jan-22\\Projects")


# In[6]:


os.getcwd()


# In[7]:


import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns


# In[9]:


df=pd.read_csv("diabetes.csv")


# In[10]:


df.head()


# In[11]:


df.info()


# In[12]:


df.describe()


# In[14]:


df.describe().T


# In[17]:


df.shape


# In[ ]:


#EDA


# In[21]:


df.hist(figsize=(12,12))


# In[ ]:





# In[ ]:





# In[ ]:





# Train Test Split

# In[29]:


x=df.iloc[:,:-1]


# In[30]:


x.head()


# In[31]:


y=df.iloc[:,-1:]


# In[32]:


y.head()


# In[34]:


y.value_counts()


# In[43]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2, random_state=2)


# In[44]:


print('xtrain size:', xtrain.shape)
print('xtest size:', xtest.shape)


# # LogisticRegression

# In[57]:


from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(xtrain,ytrain)
print("training completed")
ypred=log.predict(xtest)
print("testing completed")

ypred


# In[46]:


#Measure the performance of the model
from sklearn.metrics  import confusion_matrix

print('Performance Measures of Logestic regression Model')

print('Confusion Matrix:\n',confusion_matrix(ytest,ypred))


# In[83]:


from sklearn.metrics import classification_report
Log_classification_report=classification_report(ytest, ypred)
print(Log_classification_report)


# In[55]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(ytest, ypred)
fpr, tpr, thresholds = roc_curve(ytest, log.predict(xtest))
plt.figure
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# # RandomForestClassifier

# In[86]:


from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(xtrain,ytrain)
y_pred=clf.predict(xtest)

from sklearn import metrics
RF_accuracy=metrics.accuracy_score(ytest, y_pred)
print("Accuracy of Random forest:", RF_accuracy)


# # Gaussion Naive_bayes algorithm

# In[58]:


from sklearn.naive_bayes import GaussianNB
gaus=GaussianNB()
#train the model
gaus.fit(xtrain,ytrain)
print('Training Completed\n')
#test the model
ypred=gaus.predict(xtest)
print('Test done,length of test set: ',len(ypred))


# In[90]:


#Measure the performance of the model
from sklearn.metrics  import confusion_matrix,accuracy_score,classification_report

print('Performance Measures of Gaussian Model')
GNB_CM=confusion_matrix(ytest,ypred)

print('Confusion Matrix of Gaussion Naive_bayes :\n',GNB_CM)

Accuracy_GNB=accuracy_score(ytest,ypred)
print('Accuracy Score Gaussion Naive_bayes: ',Accuracy_GNB)
Clf_rpt_GNB=classification_report(ytest,ypred)

print('Classification Report:\n',Clf_rpt_GNB)


# # Multinomial NB Algorithm

# In[62]:


from sklearn.naive_bayes import MultinomialNB
print('==========================Multinomial NB Algorithm======================\n')
mult=MultinomialNB()
#training
mult.fit(xtrain,ytrain)
print('Training Completed\n')
#testing
ypred=mult.predict(xtest)
print('Test done,length of test set: ',len(ypred))

from sklearn.metrics import confusion_matrix,multilabel_confusion_matrix,classification_report,accuracy_score

print('Accuracy Score: ',accuracy_score(ytest,ypred))

print('Multilabel confusion matrix:\n',multilabel_confusion_matrix(ytest,ypred))

print('Classification Report:\n',classification_report(ytest,ypred))

print('Confusion Matrix:\n',confusion_matrix(ytest,ypred))


# # Bernoulli naive_bayes

# In[63]:


from sklearn.naive_bayes import BernoulliNB
br=BernoulliNB()
#train
br.fit(xtrain,ytrain)
print('training Completed\n')
#test
ypred=br.predict(xtest)
print('Test done\n')
print('Performance Measures:\n')

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

print('Accuracy Score: ',accuracy_score(ytest,ypred))

print('Classification Report: \n',classification_report(ytest,ypred))

print('Confusion Matrix:\n',confusion_matrix(ytest,ypred))


# In[ ]:





# In[65]:


from sklearn.tree import DecisionTreeClassifier
dt_model=DecisionTreeClassifier(criterion='entropy')
dt_model.fit(xtrain,ytrain)


# In[66]:


ypred=dt_model.predict(xtest)


# In[67]:


print('Training score-dtree: ',dt_model.score(xtrain,ytrain))
print('Testing score-dtree: ',dt_model.score(xtest,ytest))


# In[68]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

print()
print('Accuracy score: ',accuracy_score(ytest,ypred))
print()
print('Confusion matrix:\n ',confusion_matrix(ytest,ypred))
print()
print('Classification Report:\n',classification_report(ytest,ypred))
print()


# # Results:

# In[91]:


print('Performance Measures of Logestic regression Model')
print('Confusion Matrix:\n',confusion_matrix(ytest,ypred))
print(Log_classification_report)
print()
print('*'*80)
print()

print("Accuracy of Random forest:", RF_accuracy)
print()
print('*'*80)
print()

print('Performance Measures of Gaussian Model')
GNB_CM=confusion_matrix(ytest,ypred)
print('Confusion Matrix of Gaussion Naive_bayes :\n',GNB_CM)
Accuracy_GNB=accuracy_score(ytest,ypred)
print('Accuracy Score Gaussion Naive_bayes: ',Accuracy_GNB)
Clf_rpt_GNB=classification_report(ytest,ypred)
print('Classification Report:\n',Clf_rpt_GNB)


# In[ ]:




