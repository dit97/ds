#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris_df=pd.read_csv("Iris.csv")
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']


print("Features and their types:")
print(iris_df.dtypes)

#Create a histogram for each feature
plt.figure(figsize=(10,8))
for i, feature in enumerate(iris_df.columns[:-1]):
    plt.subplot(2,2,i+1)
    sns.histplot(iris_df[feature],kde=True, color='skyblue')
    plt.title(f'Histogram of {feature}')
plt.tight_layout()

#boxplot
plt.figure(figsize=(10,8))
for i, feature in enumerate(iris_df.columns[:-1]):
    plt.subplot(2,2, i+1)
    sns.boxplot(y=iris_df[feature],color="skyblue")
    plt.title(f"Boxplot of {feature}")
plt.tight_layout()

#outlier
print("Outliers:")
for feature in iris_df.columns[:-1]:
    Q1 = iris_df[feature].quantile(0.25)
    Q3 = iris_df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = iris_df[(iris_df[feature] < lower_bound) | (iris_df[feature] > upper_bound)]
    if len(outliers) > 0:
        print(f"{feature}: {len(outliers)} outliers")


