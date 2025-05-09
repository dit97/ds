#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

df = pd.read_csv('iris.csv')

# Initial Exploration
print("Column Names:", df.columns)
print("Unique species:", df['species'].unique())
print("Species value counts:\n", df['species'].value_counts())


# Missing value check
print("Missing values:\n", df.isnull().sum())

# Fill missing values (if any)
df['sepal_length'] = df['sepal_length'].fillna(df['sepal_length'].mean())
df['sepal_width'] = df['sepal_width'].fillna(df['sepal_width'].mean())
df['petal_length'] = df['petal_length'].fillna(df['petal_length'].mean())
df['petal_width'] = df['petal_width'].fillna(df['petal_width'].mean())

#Summary Statistics
print(df.describe(include='all'))
print("Data Types:\n", df.dtypes)
print("Shape of DataFrame:", df.shape)

# Data type summary
print("\nVariable Types:")
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"{col}: Character (String)")
    elif df[col].dtype == 'int64':
        print(f"{col}: Integer")
    elif df[col].dtype == 'float64':
        print(f"{col}: Numeric")
    elif df[col].dtype == 'bool':
        print(f"{col}: Logical (Boolean)")
    else:
        print(f"{col}: Unknown")

# Convert categorical to numerical
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])


# Normalize numeric columns
scaler = MinMaxScaler()
df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] = scaler.fit_transform(
    df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
)

print(df.sample(5))


