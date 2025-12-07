# stage2_model_optimization/clustering_analysis.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def kmeans_clustering(df, n_clusters=3):
    """
    Apply KMeans clustering to student features.
    
    Returns:
        cluster_labels (pd.Series)
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(df)
    return pd.Series(cluster_labels, name='Cluster')

def find_optimal_k(df, max_k=10):
    """
    Elbow method to find optimal number of clusters.
    """
    inertia = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)
    
    plt.figure(figsize=(8,5))
    plt.plot(range(2, max_k+1), inertia, marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.show()
