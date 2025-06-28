#!/usr/bin/env python
# coding: utf-8

# Introduction
# Let's look at the 2024 hotel sales data. We'll explore how many rooms were full, what customers did, and 
# how prices changed. We might even joke a bit about 
# busy hotel lobbies and sudden cancellations by using Customer Segmentation Using K-Means Clustering on Transactional Data

# In[1]:


# Suppress warnings for a clean notebook output
import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries
import numpy as np
import pandas as pd


# In[3]:


# Read the Excel file 
data_path = 'transactions.xlsx'
df = pd.read_excel(data_path)


# In[6]:


## Display the first five rows
df.head()


# In[9]:


# Display DataFrame rows and columns size
df.shape


# In[11]:


# check data types of all columns
print(df.dtypes)


# In[12]:


# rename the column

df = df.rename(columns={'reg':'region','cus_name':'customer_name','emel':'email','resv_status':'status'})
df.head()


# # EDA
# 
# - Convert date columns (`arrival_date` and `depature_date`) from strings to datetime objects.
# - Check for and handle missing and duplicate values if needed.
# - Ensure that numeric columns are correctly assigned.
# 

# In[14]:


# Convert date columns to datetime, inferring the date type

date_columns = ['arrival_date', 'depature_date']
for col in date_columns:
                    # Errors in conversion are handled to avoid crashes if the data is not in the expected format
                    #invalid parsing will be set as NaN
    df[col] = pd.to_datetime(df[col], errors='coerce')


# In[17]:


# Check for missing values 
missing_values = df[date_columns].isnull().sum()
print('Missing values in date columns after conversion:')
print(missing_values)


# In[18]:


# Check for duplicates
print("\nDuplicate Records:", df.duplicated().sum())


# In[19]:


# Additional cleaning steps 
df.rename(columns={
    '(child)': 'child'
}, inplace=True)


# In[22]:


# Print summary statistics of numeric columns 
print(df.describe())


# In[23]:


# Example categorical variables
print("\nReservation Status Counts:")
print(df['status'].value_counts())

print("\nRoom Type Counts:")
print(df['room-type'].value_counts())

print("\nPayment Methods:")
print(df['payment_method'].value_counts())


# In[24]:


# plot
import seaborn as sns
import matplotlib.pyplot as plt

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[29]:


# Select numerical columns
numerical_cols = df.select_dtypes(include='number').columns
df_num = df[numerical_cols]

print('Select numerical columns:')
#print(df_num)
df_num


# # Normalize Data

# In[30]:


#Standardization   DOUbt
#Transforms data to have mean = 0 and std = 1.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df_num), columns=numerical_cols)

print('Standardization (Z-score normalization):')
df_standardized.head()


# In[32]:


# Min-Max Normalization
# Scales data to a range of 0 to 1.

from sklearn.preprocessing import MinMaxScaler

minmax_scaler = MinMaxScaler()
df_normalized = pd.DataFrame(minmax_scaler.fit_transform(df_num), columns=numerical_cols)

print('Min-Max Normalization:')
df_normalized.head()


# # **Apply K-Means clustering(using k=3)**

# In[33]:


# prepare scale data

from sklearn.preprocessing import StandardScaler

# Select numeric columns and scale them
numerical_cols = df.select_dtypes(include='number').columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numerical_cols])


# In[34]:


# Find Optimal Number of Clusters (Elbow Method)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Elbow method
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Choose the "elbow" point where the decrease in inertia slows down.


# In[56]:


# Apply K-Means with Chosen k(clustering)
# Assuming k = 3

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# analyze the cluster
# Cluster sizes
print(df['cluster'].value_counts())

# Compare cluster means
df.groupby('cluster')[numerical_cols].mean()


# In[57]:


# visualized cluster using Elbow Method
from sklearn.decomposition import PCA
import seaborn as sns

# Reduce to 2D
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

# Plot clusters
df['pca1'] = components[:, 0]
df['pca2'] = components[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='Set2')
plt.title('K-Means Clustering Visualization')
plt.show()


# # **Determine optimal clusters using Elbow Method and Silhouette Score**

# In[39]:


# clusters using Elbow Method and Silhouette Score

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Select numeric columns
numerical_cols = df.select_dtypes(include='number').columns
X = df[numerical_cols]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# random sample 1000 rows
np.random.seed(42)
sample_idx = np.random.choice(X_scaled.shape[0], 1000, replace=False)
X_sample = X_scaled[sample_idx]

# Elbow and Silhouette
inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_sample)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_sample, labels))

# Plot
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, marker='o', color='orange')
plt.title('Silhouette Score (Sample of 1000)')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()


# In[49]:


#PCA & K-Means Cluster Visualization
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming X_sample is your scaled sample (1000 rows)
# Reduce dimensions to 2 with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_sample)

# Apply K-Means of clusters (e.g., 3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(X_sample)

# Create a DataFrame for plotting
pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = labels

# Scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=60)
plt.title(f'K-Means Clusters (k={optimal_k}) Visualized with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()


# # **Additional Cluster Visualizations**

# In[50]:


#Pairplot (Pairwise Feature Plots)
#Great for small numbers of numerical features.
# Create a DataFrame with cluster labels
sample_df = pd.DataFrame(X_sample, columns=numerical_cols)
sample_df['Cluster'] = labels

# Pairplot
sns.pairplot(sample_df, hue='Cluster', palette='Set2', diag_kind='kde', corner=True)
plt.suptitle('Pairwise Feature Relationships by Cluster', y=1.02)
plt.show()


# In[ ]:


#Boxplots of Features by Cluster
#To compare distributions of features across clusters.

# Melt for easier boxplotting
melted_df = sample_df.melt(id_vars='Cluster', var_name='Feature', value_name='Value')

# Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=melted_df, x='Feature', y='Value', hue='Cluster')
plt.xticks(rotation=45)
plt.title('Feature Distribution by Cluster')
plt.tight_layout()
plt.show()


# In[44]:


#Cluster Centroids Heatmap
#Displays feature averages per cluster.

# Cluster centroids
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=numerical_cols)

plt.figure(figsize=(10, 6))
sns.heatmap(centroids, annot=True, cmap='coolwarm')
plt.title('Cluster Centroids Heatmap')
plt.xlabel('Features')
plt.ylabel('Cluster')
plt.show()


# Summary / Abstract :
# 
# This project applies K-Means Clustering to segment customers using their transactional data. After cleaning, scaling, and analyzing the dataset, customers were grouped based on their behavior across several features such as purchase amounts and frequency.
# 
# The resulting clusters highlight meaningful patterns in customer behavior. These behavioral patterns form the basis of customer segmentation, enabling businesses to design more targeted marketing strategies, enhance customer engagement, and optimize services.

# In[ ]:




