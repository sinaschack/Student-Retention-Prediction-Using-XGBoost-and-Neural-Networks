import pandas as pd
import numpy as np

def add_engagement_features(df):
    """
    Create features based on student engagement, e.g., attendance rate,
    module completion ratios, or average grades per semester.
    
    Parameters:
        df (pd.DataFrame): Input dataframe from Stage 1.
        
    Returns:
        pd.DataFrame: DataFrame with new features.
    """
    # Example: ratio of passed modules to total modules
    if 'ModulesPassed' in df.columns and 'TotalModules' in df.columns:
        df['PassRatio'] = df['ModulesPassed'] / df['TotalModules']
    
    # Example: cumulative average of grades if column exists
    if 'Grades' in df.columns:
        df['AvgGrades'] = df['Grades'].expanding().mean()
    
    return df

def feature_selection(df, target='CompletedCourse', top_n=20):
    """
    Select top_n features based on correlation with target or domain knowledge.
    
    Parameters:
        df (pd.DataFrame)
        target (str)
        top_n (int)
    
    Returns:
        pd.DataFrame with selected features
    """
    corr = df.corr()[target].abs().sort_values(ascending=False)
    selected_features = corr.index[1:top_n+1]  # skip target itself
    return df[selected_features.tolist() + [target]]
