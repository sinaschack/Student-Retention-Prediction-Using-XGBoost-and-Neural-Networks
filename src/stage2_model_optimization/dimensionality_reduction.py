import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def apply_pca(df, n_components=2):
    """
    Apply PCA to reduce dimensionality for visualization or modeling.
    """
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(df)
    return pd.DataFrame(reduced, columns=[f'PC{i+1}' for i in range(n_components)])

def apply_tsne(df, n_components=2, random_state=42):
    """
    Apply t-SNE for visualization of clusters in high-dimensional data.
    """
    tsne = TSNE(n_components=n_components, random_state=random_state)
    reduced = tsne.fit_transform(df)
    return pd.DataFrame(reduced, columns=[f'tSNE{i+1}' for i in range(n_components)])

def plot_2d(df, target=None, title='2D Projection'):
    """
    Simple 2D scatter plot.
    """
    plt.figure(figsize=(8,6))
    if target is not None:
        plt.scatter(df.iloc[:,0], df.iloc[:,1], c=target, cmap='coolwarm', alpha=0.7)
    else:
        plt.scatter(df.iloc[:,0], df.iloc[:,1], alpha=0.7)
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.title(title)
    plt.colorbar()
    plt.show()
