#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt

df=sns.load_dataset('titanic')
df.head()

df.tail()

df.describe()

sns.set_style("whitegrid") 
sns.boxplot(x='sex',y='age',hue='survived',data=df)


#aged females were more likely to survive
#among male there i no such major difference between age and survival
#outliers are seen in males rather than females
#overall age has strong relationship with survival for fimales than males


# In[ ]:




