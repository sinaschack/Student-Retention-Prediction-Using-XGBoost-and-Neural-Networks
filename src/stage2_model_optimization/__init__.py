# stage2_model_optimization/__init__.py

# Feature engineering
from .feature_engineering import add_engagement_features, feature_selection

# Dimensionality reduction
from .dimensionality_reduction import apply_pca, apply_tsne, plot_2d

# Clustering analysis
from .clustering_analysis import kmeans_clustering, find_optimal_k

# Model hyperparameter tuning
from .model_hyperparameter_tuning import xgb_hyperparameter_tuning

# Model evaluation
from .model_evaluation import evaluate_model

# Insights report
from .insights_report import generate_student_risk_report

# Optional: define __all__ for explicit export
__all__ = [
    'add_engagement_features', 'feature_selection',
    'apply_pca', 'apply_tsne', 'plot_2d',
    'kmeans_clustering', 'find_optimal_k',
    'xgb_hyperparameter_tuning',
    'evaluate_model',
    'generate_student_risk_report'
]
