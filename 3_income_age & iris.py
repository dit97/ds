#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("adult.csv")

data

print(data.columns)

data.describe()   #instead also use data.min() max() mean() median() nunique()

data.groupby('income')['age'].describe()

data["age"].min()

# Group by 'income' and compute summary stats for 'age' and 'hours-per-week'
grouped_stats = data.groupby('income').agg({
    'age': ['mean', 'median', 'min', 'max', 'std'],
    
})

print("Summary statistics grouped by income:\n", grouped_stats)


# List of 'age' values for each 'income' group
income_age_list = data.groupby('income')['age'].apply(list)
print("\nList of ages per income group:\n", income_age_list)


import seaborn as sns
import pandas as pd
import numpy as np

data = sns.load_dataset('iris')
print('Original Dataset')
data.head()

data.describe()

data["sepal_length"].quantile(0.25)

data["sepal_length"].quantile(0.5)

data["sepal_length"].quantile(0.75)

data["sepal_length"].std()

