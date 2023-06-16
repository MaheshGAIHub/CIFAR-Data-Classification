#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[10]:


import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt


# # Self Correlation

# In[ ]:


folder = "D:\downloads folder\PAF_DATA\EXCEL"
for name in os.listdir(folder):
    
    df = pd.read_csv(os.path.join(folder, name))
    df.dropna(axis='rows')
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',ax=ax)
    print(name)


# # Split Data into 3 parts

# In[76]:


df = pd.read_csv("F:\ME20S033\PAF_DATA\EXCEL\PAF_9.csv")
df1 = df.iloc[1:480,10:16]
df2 = df.iloc[481:960,10:16]
df3 = df.iloc[961:1440,10:16]
#df.describe()
df1.rename(columns = {'PAF B MTR BRG X VIB':'PAF B MTR BRG X VIB_1','PAF B MTR BRG Y VIB':'PAF B MTR BRG Y VIB_1','PAF B DRV SIDE BRG X VIB':'PAF B DRV SIDE BRG X VIB_1','PAF B DRV SIDE BRG Y VIB':'PAF B DRV SIDE BRG Y VIB_1','PAF B NON DRV SIDE BRG X VIB':'PAF B NON DRV SIDE BRG X VIB_1','PAF B NON DRV SIDE BRG Y VIB':'PAF B NON DRV SIDE BRG Y VIB_1'},inplace='True')
df4 = df1.append(df2)
 


df4.info()
#df2.describe()
#df2.describe()


# In[77]:


corr_matrix = df4.corr()
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',ax=ax)


# 
# from sklearn.model_selection import train_test_split
# 
# #Assuming 'data' is your dataset
# 
# #Split the dataset into train and test sets
# train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
# 
# #Split the test set into validation and test sets
# validation_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

# # Cross Correlation

# In[19]:


import numpy as np

# Assuming 'dataset1', 'dataset2', and 'dataset3' are your datasets

# Compute the cross-correlation matrix between dataset1 and dataset2
cross_corr_1_2 = np.correlate(df1.flatten(), df2.flatten(), mode='same')

# Compute the cross-correlation matrix between dataset1 and dataset3
cross_corr_1_3 = np.correlate(df1.flatten(), df3.flatten(), mode='same')

# Compute the cross-correlation matrix between dataset2 and dataset3
cross_corr_2_3 = np.correlate(df2.flatten(), df3.flatten(), mode='same')

# Print the cross-correlation matrices
print("Cross-correlation between df1 and df22:\n", cross_corr_1_2)
print("Cross-correlation between df1 and df3:\n", cross_corr_1_3)
print("Cross-correlation between df2 and df3:\n", cross_corr_2_3)

