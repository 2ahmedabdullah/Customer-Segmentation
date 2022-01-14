#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 19:53:34 2022

@author: abdul
"""


import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle

#IMPORT DATA
df_segmentation = pd.read_csv('segmentation data.csv', index_col = 0)

df_segmentation.head()

df_segmentation.describe()


#CORRELATION ESTIMATION
df_segmentation.corr()

plt.figure(figsize = (12, 9))
s = sns.heatmap(df_segmentation.corr(),
               annot = True, 
               cmap = 'RdBu',
               vmin = -1, 
               vmax = 1)
s.set_yticklabels(s.get_yticklabels(), rotation = 0, fontsize = 12)
s.set_xticklabels(s.get_xticklabels(), rotation = 90, fontsize = 12)
plt.title('Correlation Heatmap')
plt.savefig('Correlation Heatmap.png')
plt.show()


#VISUALIZATION OF RAW DATA
plt.figure(figsize = (12, 9))
plt.scatter(df_segmentation.iloc[:, 2], df_segmentation.iloc[:, 4])
plt.xlabel('Age')
plt.ylabel('Income')
plt.savefig('Visualization of raw data.png')
plt.title('Visualization of raw data')

#STANDARDIZATION
scaler = StandardScaler()
segmentation_std = scaler.fit_transform(df_segmentation)

#HIERARCHICAL CLUSTERING
hier_clust = linkage(segmentation_std, method = 'ward')

plt.figure(figsize = (12,9))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Observations')
plt.ylabel('Distance')
dendrogram(hier_clust,
           truncate_mode = 'level',
           p = 5,
           show_leaf_counts = False,
           no_labels = True)
plt.savefig('hierarchical clustering.png')
plt.show()


#KMEANS CLUSTERING
wcss = []
for i in range(1,11):
    print(i)
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(segmentation_std)
    wcss.append(kmeans.inertia_)


plt.figure(figsize = (12,9))
plt.plot(range(1, 11), wcss, marker = 'o', linestyle = '--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('K-means Clustering')
plt.savefig('K-means Clustering.png')
plt.show()


kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)

kmeans.fit(segmentation_std)


#RESULT ANALYSIS
df_segm_kmeans = df_segmentation.copy()
df_segm_kmeans['Segment K-means'] = kmeans.labels_


df_segm_analysis = df_segm_kmeans.groupby(['Segment K-means']).mean()
df_segm_analysis


df_segm_analysis['N Obs'] = df_segm_kmeans[['Segment K-means','Sex']].groupby(['Segment K-means']).count()
df_segm_analysis['Probs'] = df_segm_analysis['N Obs'] / df_segm_analysis['N Obs'].sum()

print(df_segm_analysis)

df_segm_analysis.rename({0:'well-off',
                         1:'fewer-opportunities',
                         2:'standard',
                         3:'career focused'})


df_segm_kmeans['Labels'] = df_segm_kmeans['Segment K-means'].map({0:'well-off', 
                                                                  1:'fewer opportunities',
                                                                  2:'standard', 
                                                                  3:'career focused'})


x_axis = df_segm_kmeans['Age']
y_axis = df_segm_kmeans['Income']
plt.figure(figsize = (12, 9))
sns.scatterplot(x_axis, y_axis, hue = df_segm_kmeans['Labels'], palette = ['g', 'r', 'c', 'm'])
plt.title('Segmentation K-means')
plt.savefig('Segmentation K-means.png')
plt.show()


#PCA

pca = PCA()

pca.fit(segmentation_std)

pca.explained_variance_ratio_    
    
plt.figure(figsize = (12,9))
plt.plot(range(1,8), pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')
plt.title('PCA Explained Variance by Components')
plt.xlabel('Number of Components')
plt.savefig('PCA Variance.png')
plt.ylabel('Cumulative Explained Variance')
    
pca = PCA(n_components = 3)
    
pca.fit(segmentation_std)    
    
    
#PCA COMPONENTS
pca.components_

df_pca_comp = pd.DataFrame(data = pca.components_,
                           columns = df_segmentation.columns.values,
                           index = ['Component 1', 'Component 2', 'Component 3'])
df_pca_comp
    
    
sns.heatmap(df_pca_comp,
            vmin = -1, 
            vmax = 1,
            cmap = 'RdBu',
            annot = True)
plt.yticks([0.5, 1.5, 2.5], 
           ['Component 1', 'Component 2', 'Component 3'],
           rotation = 0,
           fontsize = 9)
plt.savefig('PCA Correlation Heatmap.png')
plt.show()
    

pca.transform(segmentation_std)

scores_pca = pca.transform(segmentation_std)


#K-MEANS WITH PCA
wcss = []
for i in range(1,11):
    print(i)
    kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans_pca.fit(scores_pca)
    wcss.append(kmeans_pca.inertia_)


plt.figure(figsize = (12,9))
plt.plot(range(1, 11), wcss, marker = 'o', linestyle = '--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('K-means with PCA Clustering')
plt.savefig('K-means with PCA Clustering.png')
plt.show()


kmeans_pca = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)

kmeans_pca.fit(scores_pca)


#K-MEANS WITH PCA RESULTS

df_segm_pca_kmeans = pd.concat([df_segmentation.reset_index(drop = True), pd.DataFrame(scores_pca)], axis = 1)
df_segm_pca_kmeans.columns.values[-3: ] = ['Component 1', 'Component 2', 'Component 3']
df_segm_pca_kmeans['Segment K-means PCA'] = kmeans_pca.labels_

df_segm_pca_kmeans_freq = df_segm_pca_kmeans.groupby(['Segment K-means PCA']).mean()
df_segm_pca_kmeans_freq


df_segm_pca_kmeans_freq['N Obs'] = df_segm_pca_kmeans[['Segment K-means PCA','Sex']].groupby(['Segment K-means PCA']).count()
df_segm_pca_kmeans_freq['Probs'] = df_segm_pca_kmeans_freq['N Obs'] / df_segm_pca_kmeans_freq['N Obs'].sum()
df_segm_pca_kmeans_freq = df_segm_pca_kmeans_freq.rename({0:'standard', 
                                                          1:'career focused',
                                                          2:'fewer opportunities', 
                                                          3:'well-off'})
df_segm_pca_kmeans_freq


df_segm_pca_kmeans['Label'] = df_segm_pca_kmeans['Segment K-means PCA'].map({0:'standard', 
                                                          1:'career focused',
                                                          2:'fewer opportunities', 
                                                          3:'well-off'})

x_axis = df_segm_pca_kmeans['Component 2']
y_axis = df_segm_pca_kmeans['Component 1']
plt.figure(figsize = (12, 9))
sns.scatterplot(x_axis, y_axis, hue = df_segm_pca_kmeans['Label'], palette = ['g', 'r', 'c', 'm'])
plt.title('Clusters by PCA Components')
plt.savefig('Segmentation K-means with PCA Clustering.png')
plt.show()


#SAVE PICKLE
pickle.dump(scaler, open('scaler.pkl', 'wb'))

pickle.dump(pca, open('pca.pkl', 'wb'))

pickle.dump(kmeans_pca, open('kmeans_pca.pkl', 'wb'))






