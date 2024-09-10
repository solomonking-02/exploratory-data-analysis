#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[2]:


df = pd.read_csv(r'C:\Users\shamh\Downloads\retail_sales_dataset.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


# checking duplicated values
df.duplicated().sum()


# In[6]:


# checking not available values or null values
df.isna().any()


# In[7]:


# descriptive statistics
df.describe()


# In[8]:


df.shape


# In[9]:


df['Date'] = df['Date'].astype(str)


# In[10]:


df['Day'] = df['Date'].str.split('-').str[2]


# In[11]:


df['Year'] = df['Date'].str.split('-').str[0]


# In[12]:


df['Date'] = pd.to_datetime(df['Date'])


# In[13]:


# seperating the month and date and year for the time series analysis
df['Month']  = df['Date'].dt.month_name()


# In[14]:


df.head()


# In[15]:


df.info()


# In[16]:


#Time series sales over trend

df['Date'] = pd.to_datetime(df['Date'])
monthly_sales = df.groupby(df['Date'].dt.to_period('M'))['Total Amount'].sum() #Monthly convertion
plt.figure(figsize=(10,8))
monthly_sales.plot(linestyle = '-', marker = 'o', color = 'b', kind = 'line')
plt.grid(True)
plt.xlabel("Months")
plt.ylabel("Total_sales")
plt.title("Monthly sales Trend")
plt.show()


# In[17]:


# date convertion into day names
df['DayName'] = df['Date'].dt.day_name() 


# In[18]:


daily_sales = df.groupby(df['DayName'])['Total Amount'].sum()


# In[19]:


week_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_sales = daily_sales.reindex(week_days)

# daily sales trend

plt.figure(figsize=(11,6))
daily_sales.plot(kind='line', marker = 'o', linestyle = '-', color = 'purple')
plt.grid(True)
plt.xlabel('Week days')
plt.ylabel('Sales Amount')
plt.title('Daily sales Trend')
plt.show()


# In[20]:


# Quaterly convertion
df['Date'] = pd.to_datetime(df['Date'])
monthly_sales = df.groupby(df['Date'].dt.to_period('Q'))['Total Amount'].sum() 
plt.figure(figsize=(11,6))
monthly_sales.plot(linestyle = '-', marker = 'o', color = 'b', kind = 'line')
plt.grid(True)
plt.xlabel("Quaters")
plt.ylabel("Total_sales")
plt.title("Quater sales Trend")
plt.show()


# In[21]:


# Customer and Product Analysis
# Male
male_data =df[df['Gender'] == 'Male']


# In[22]:


grp_data = male_data.groupby(['Product Category'])['Quantity'].count()


# In[23]:


grp_data.plot(kind = 'bar', color = 'g')
plt.ylabel('count')
plt.title('Male customer with product category (Count)')
plt.xticks(rotation=0)
plt.show()


# In[24]:


grp_data = male_data.groupby(['Product Category'])['Total Amount'].sum()


# In[25]:


grp_data.plot(kind = 'bar', color = 'orange')
plt.ylabel('Total Amount')
plt.title('Male customer with product category (Total sales)')
plt.xticks(rotation=0)
plt.show()


# In[26]:


# Female data
female_data =df[df['Gender'] == 'Female']


# In[27]:


grp_data = female_data.groupby(['Product Category'])['Quantity'].count()


# In[28]:


grp_data.plot(kind = 'bar', color = 'purple')
plt.ylabel('count')
plt.title('Female customer with product category (Count)')
plt.xticks(rotation=0)
plt.show()


# In[29]:


grp_data = female_data.groupby(['Product Category'])['Total Amount'].sum()


# In[30]:


grp_data.plot(kind = 'bar', color = 'orange')
plt.ylabel('Total Amount')
plt.title('Female customer with product category (Total sales)')
plt.xticks(rotation=0)
plt.show()


# In[31]:


category_counts = df['Product Category'].value_counts()


# In[32]:


category_counts.plot(kind = 'pie', autopct='%1.1f%%', title='Product Category Distribution')
plt.ylabel(None)
plt.show()


# In[33]:


grp_data = df.groupby(['Product Category'])['Price per Unit'].mean()


# In[34]:


plt.figure(figsize=(11,6))
grp_data.plot(kind = 'bar', color='Purple')
plt.title('Average Price per Unit by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Average Price')
plt.xticks(rotation=0)
plt.show()


# In[ ]:





# In[35]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))  # 1 row, 2 columns

sns.boxplot(x='Gender', y='Total Amount', data=df, ax=axes[0])
axes[0].set_title('Relationship between Gender and Total Amount')

sns.boxplot(x='Product Category', y='Price per Unit', data=df, ax=axes[1])
axes[1].set_title('Relationship between Product Categories and their Unit Price')

plt.tight_layout()  # Adjust subplots to give some spacing between plots
plt.show()


# In[36]:


# Gender counts
df['Gender'].value_counts()


# In[37]:


# Age description
df['Age'].describe()


# In[38]:


# counts of product category
df['Product Category'].value_counts()


# In[39]:


# Bivariate analysis
#Numerical (vs) numerical


# In[40]:


# corr score
#price per unit (vs) total amount
df['Price per Unit'].corr(df['Total Amount'])


# In[41]:


sns.scatterplot(df, x='Price per Unit', y='Total Amount')


# In[42]:


# corr score
# quantity (vs) total amount
df['Quantity'].corr(df['Total Amount'])


# In[43]:


sns.scatterplot(df, x='Quantity', y='Total Amount')


# In[44]:


# categorical vs categorical
# gender and product category
pd.crosstab(df['Gender'], df['Product Category'])


# In[46]:


# gender and quantity
pd.crosstab(df['Gender'], df['Quantity'])


# In[47]:


# product_category and quantity
pd.crosstab(df['Product Category'], df['Quantity'])


# In[48]:


#Multi variate analysis
# product category (vs) total amount (vs) gender
sns.boxplot(df, x='Product Category', y='Total Amount', hue='Gender')


# In[49]:


# price per unit (vs) total amount (vs) gender
sns.boxplot(df, x='Price per Unit', y='Total Amount', hue='Gender')


# In[ ]:




