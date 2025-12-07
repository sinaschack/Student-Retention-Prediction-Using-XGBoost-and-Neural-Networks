import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    df = pd.read_csv(file_path)
    return df

def drop_unhelpful_columns(df: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
    """Drop columns not useful for modeling."""
    return df.drop(columns=cols_to_drop, errors='ignore')

def handle_missing_values(df: pd.DataFrame, numeric_strategy='mean') -> pd.DataFrame:
    """Impute missing numeric values."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        imputer = SimpleImputer(strategy=numeric_strategy)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df

def encode_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Convert target variable to binary encoding."""
    df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})
    return df
