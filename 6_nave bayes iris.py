#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Step 1: Load the dataset using pandas
data = pd.read_csv('iris.csv')
print('Original Dataset')
print(data.head())

# Step 2: Preprocessing (features and target)
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Naïve Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = nb_classifier.predict(X_test)

# Step 6: Confusion matrix and evaluation metrics
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Step 7: Output
print("Confusion Matrix:") # performance evaluator matrix how well the model’s predicted classifications match the actual labels
print(conf_matrix)
print("Accuracy:", accuracy) #(TP + TN) / Total
print("Error Rate:", error_rate)
print("Precision:", precision) # TP / (TP + FP)
print("Recall:", recall)# TP / (TP + FN)


# Bayes' Theorem:
# -------------------
# In simple words, Bayes' Theorem helps us calculate the probability of a class (A),
# given the evidence or feature (B), using the known probabilities.

# Formula:
#     P(A|B) = (P(B|A) * P(A)) / P(B)

# Where:
# - P(A|B): Posterior probability - probability of class A given feature B
# - P(B|A): Likelihood - probability of feature B given class A
# - P(A): Prior probability of class A (how likely A is in general)
# - P(B): Prior probability of feature B (how likely B is in general)


# In[ ]:




