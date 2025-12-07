import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Load Stage 3 data from Google Drive
file_url_3 = "https://drive.google.com/uc?id=18oyu-RQotQN6jaibsLBoPdqQJbj_cV2-"
stage3_data = pd.read_csv(f"https://drive.google.com/uc?export=download&id={file_url_3.split('=')[-1]}")

# Drop unnecessary columns
stage3_data.drop(columns=['LearnerCode'], inplace=True)
stage3_data['DateofBirth'] = pd.to_datetime(stage3_data['DateofBirth'], errors='coerce', dayfirst=True)
stage3_data['Age'] = 2016 - stage3_data['DateofBirth'].dt.year
stage3_data.drop(columns=['DateofBirth'], inplace=True)

# Drop high cardinality columns (>200 unique values)
high_card_cols = [col for col in stage3_data.columns if stage3_data[col].nunique() > 200]
stage3_data.drop(columns=high_card_cols, inplace=True)

# Drop columns with >50% missing values
threshold = len(stage3_data) * 0.5
stage3_data.dropna(thresh=threshold, axis=1, inplace=True)

# Drop rows with <2% missing in a column
for col in stage3_data.columns:
    missing_ratio = stage3_data[col].isnull().mean()
    if 0 < missing_ratio < 0.02:
        stage3_data = stage3_data[~stage3_data[col].isnull()]

# Impute numeric columns
numeric_cols = stage3_data.select_dtypes(include=np.number).columns
if len(numeric_cols) > 0:
    imputer = SimpleImputer(strategy='mean')
    stage3_data[numeric_cols] = imputer.fit_transform(stage3_data[numeric_cols])

# Encode target and categorical variables
stage3_data['CompletedCourse'] = stage3_data['CompletedCourse'].map({'Yes': 1, 'No': 0})
categorical_cols = stage3_data.select_dtypes(include=['object', 'category']).columns
categorical_cols = categorical_cols.drop('CompletedCourse', errors='ignore')
stage3_data = pd.get_dummies(stage3_data, columns=categorical_cols, drop_first=True)

# Split features and target
X = stage3_data.drop(columns=['CompletedCourse'])
y = stage3_data['CompletedCourse']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
