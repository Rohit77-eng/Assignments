#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[27]:


df = pd.read_csv('oil_spill.csv')


# In[6]:


df.head()


# In[7]:


df.shape


# In[29]:


df.isnull().sum()


# In[9]:


df.dtypes


# In[31]:


print(df.describe())


# In[36]:


r1 = df['target'].value_counts()
r1


# In[37]:


plt.bar(r1.index,r1,color='maroon')
plt.xlabel('target')
plt.ylabel('Count')
plt.show()


# In[51]:


from sklearn.model_selection import train_test_split, cross_val_score  ###importing the necessary libraries
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[45]:


X = df.drop('target',axis=1)  # x is independent variable
y = df['target']              # y is dependent variable
print(type(X))                  # x is dataframe
print(type(y))                  # y is series
print(X.shape)              
print(y.shape)


# In[46]:


print(X.shape)
print(937*0.2)


# In[47]:


# Split the dataset into features (X) and target variable (y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2022)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[48]:


# Initialize the classifiers
decision_tree = DecisionTreeClassifier(random_state=2022, max_depth=3, criterion='entropy', min_samples_split=8)
random_forest = RandomForestClassifier(random_state=2022)
ada_boost = AdaBoostClassifier(base_estimator=decision_tree, n_estimators=80, random_state=2022)


# In[50]:


# Train the classifiers
decision_tree.fit(X_train, y_train)
random_forest.fit(X_train, y_train)
ada_boost.fit(X_train, y_train)


# In[55]:


# Make predictions on the test set
dt_predictions = decision_tree.predict(X_test)
rf_predictions = random_forest.predict(X_test)
ab_predictions = ada_boost.predict(X_test)


# In[ ]:


# Evaluate the models


# In[57]:


print("Decision Tree:")
print("Accuracy:", accuracy_score(y_test, dt_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, dt_predictions))
print("Classification Report:\n", classification_report(y_test, dt_predictions))


# In[60]:


print("AdaBoost:")
print("Accuracy:", accuracy_score(y_test, ab_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, ab_predictions))
print("Classification Report:\n", classification_report(y_test, ab_predictions))



# In[61]:


print("Random Forest:")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_predictions))
print("Classification Report:\n", classification_report(y_test, rf_predictions))


# In[62]:


import pickle


# In[69]:


filename = 'ada_boost.sav'
pickle.dump(ada_boost, open('ada_boost.pkl', 'wb'))


# In[71]:


loaded_model = pickle.load(open('ada_boost.pkl', 'rb'))


# In[72]:


test_acc = loaded_model.score(X_test, y_test)
print(test_acc)


# In[85]:


random_dataset = df.sample(n=20, random_state=42) 
random_dataset.shape


# In[ ]:





# In[ ]:




