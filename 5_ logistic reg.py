#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df=pd.read_csv("Social_Network_Ads.csv")

df.head(5)

df.tail(6)

df.info()

df.describe()

x = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

x

y

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Feature Scaling standard minmax scaller clustering classification
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Compute Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Extract TP, FP, TN, FN
TP = conf_matrix[1, 1]
FP = conf_matrix[0, 1]
TN = conf_matrix[0, 0]
FN = conf_matrix[1, 0]

# Compute Evaluation Metrics (performance evaluation took)
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print Evaluation Metrics
print("\nAccuracy:", accuracy)  #(TP + TN) / Total
print("Error Rate:", error_rate)  
print("Precision:", precision) #TP / (TP + FP)
print("Recall:", recall)  #TP / (TP + FN)
print("F1 Score:", f1)  # 2 × (Precision × Recall) / (Precision + Recall)


# In[ ]:




