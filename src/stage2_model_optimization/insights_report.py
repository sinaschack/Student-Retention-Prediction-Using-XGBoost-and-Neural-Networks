import pandas as pd

def generate_student_risk_report(df, cluster_labels, target='CompletedCourse'):
    """
    Create a simple report highlighting high-risk clusters or student groups.
    
    Parameters:
        df (pd.DataFrame): Feature dataframe
        cluster_labels (pd.Series): Cluster assignments
    """
    df = df.copy()
    df['Cluster'] = cluster_labels
    summary = df.groupby('Cluster')[target].value_counts(normalize=True).unstack(fill_value=0)
    print("Cluster-wise dropout risk distribution:\n", summary)
    return summary
