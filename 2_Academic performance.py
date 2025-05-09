#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np

# Creating a dataset with missing values, outliers, and skewness
np.random.seed(42)

data = {
    "Student_ID": range(1, 21),
    "Name": ["Student_" + str(i) for i in range(1, 21)],
    "Math_Score": [np.nan, 40, 55, 78, 1000, 62, 59, 95, 89, 45, 33, 102, 65, np.nan, 50, 72, 110, 98, 39, 101],
    "Science_Score": [80, 70, 60, 55, 100, 85, np.nan, 105, 75, 65, 50, 45, np.nan, 90, 120, 200, 30, 20, 10, 5],
    "English_Score": [55, 56, 57, 55, 54, 53, 52, 90, 92, 95, 97, 96, 93, 94, 95, 120, 130, 140, 150, np.nan],
    "Attendance_Percentage": [np.nan, 85, 90, 87, 88, 95, 75, 72, 80, 81, 30, 40, 35, 85, 82, 70, 73, 78, 79, 84]
}

df = pd.DataFrame(data)
print("Original Dataset:\n", df)

print(df.isnull().sum())

df["Math_Score"] = df["Math_Score"].fillna(df["Math_Score"].median())
df["Science_Score"] = df["Science_Score"].fillna(df["Science_Score"].median())
df["English_Score"] = df["English_Score"].fillna(df["English_Score"].median())
df["Attendance_Percentage"] = df["Attendance_Percentage"].fillna(df["Attendance_Percentage"].median())

print(df.isnull().sum())

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Cap the outliers
    df[column] = np.where(df[column] > upper_bound, upper_bound,
                          np.where(df[column] < lower_bound, lower_bound, df[column]))
    return df

numeric_cols = ["Math_Score", "Science_Score", "English_Score", "Attendance_Percentage"]
for col in numeric_cols:
    df = remove_outliers_iqr(df, col)
    
print("\n Dataset after handling missing values and outliers:\n")
print(df)

#To reduce skewness or normalize the distribution
df["Attendence_Power_03"] = df["Attendance_Percentage"] ** 0.3
print(df["Attendence_Power_03"])

print(df.columns)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(df["Attendance_Percentage"], df["Attendence_Power_03"], color='purple', alpha=0.7)
plt.title('Scatter Plot\n(Original vs Transformed Attendance)')
plt.xlabel('Original Attendance Percentage')
plt.ylabel('Transformed Attendance (Power 0.3)')
plt.show()


# In[ ]:





# In[ ]:




