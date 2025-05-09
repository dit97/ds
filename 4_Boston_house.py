#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('train.csv')
data.head()

nan_values = data.isna().sum()  #data.dropna(inplace=True) if contains null values
print(nan_values)

#Calcuates corelation matrix (shows strength between two variables)
corr_matrix = data.corr()
# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap - Boston Housing Data')
plt.show()

features = data.drop('medv', axis=1)
target = data['medv']

#splitting data for training and testing
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)
model = LinearRegression()
#Train model using training data
model.fit(x_train, y_train)
#Predict target value using testing data
y_pred = model.predict(x_test)

print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))

plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # ideal line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.grid(True)
plt.show()


# In[ ]:




