import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from jsonpath_nz import log, jprint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def convert_mostly_numeric(series: pd.Series, threshold: float = 0.75) -> pd.Series:
    """Convert series to numeric if majority of values are numeric"""
    def remove_outliers_zscore(series: pd.Series, threshold_outliers: float = 3) -> pd.Series:
        """Remove outliers using Z-score method"""
        z_scores = (series - series.mean()) / series.std()
        return series[abs(z_scores) < threshold_outliers]
    try:
        if series.dtype == 'object':
            # Pattern for numeric values (including negatives and decimals)
            pattern = r'^-?\d*\.?\d+$'
            
            # Get unique values and their counts
            unique_values = series.unique()
            numeric_values = [v for v in unique_values if pd.notna(v) and 
                            bool(re.match(pattern, str(v).strip().replace(',','')))]
            
            # Calculate percentage of numeric unique values
            numeric_percentage = len(numeric_values) / len(unique_values)
            
            if numeric_percentage >= threshold:
                # Convert numeric strings to float
                def is_numeric(x):
                    if pd.isna(x):
                        return False
                    return bool(re.match(pattern, str(x).strip().replace(',','')))
                
                numeric_mask = series.apply(is_numeric)
                numeric_series = pd.to_numeric(series[numeric_mask].str.replace(',',''), 
                                            errors='coerce')
                
                # Calculate mean of numeric values truncate to 0 decimal places
                mean_value = numeric_series.mean().round(0)
                
                # Replace non-numeric values with mean
                series = pd.to_numeric(series.astype(str).str.strip().str.replace(',',''), 
                                     errors='coerce').fillna(mean_value)
                
                series = remove_outliers_zscore(series)
                
                log.info(f"Converted mostly numeric column: {numeric_percentage:.2%} numeric values")
                log.info(f"Replaced non-numeric values with mean: {mean_value:.0f}")
                
                # Determine if we need float or int
                if series.dropna().apply(float.is_integer).all():
                    return series.astype('Int64')
                return series.astype('float64')
                
        return series
        
    except Exception as e:
        log.error(f"Error converting mostly numeric column: {str(e)}")
        return series
    
def clean_and_transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame by removing empty columns and applying get_dummies"""
    try:
        # Copy DataFrame to avoid modifying original
        df_clean = df.copy()
        
        # Calculate percentage of missing values in each column
        missing_percentages = (df_clean.isna().sum() / len(df_clean)) * 100
        
        # Remove columns with more than 50% missing values
        columns_to_drop = missing_percentages[missing_percentages > 50].index
        df_clean = df_clean.drop(columns=columns_to_drop)
        # log.info(f"Dropped columns with >50% missing values: {columns_to_drop.tolist()} -- {len(columns_to_drop.tolist())}")
        
        # Fill remaining missing values
        # For numeric columns: fill with mean
        numeric_columns = df_clean.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean().round(2))
            # log.info(f"Filled missing values for numeric columns: {col} --  value: {df_clean[col].mean().round(2)}")
        
        # For categorical columns: fill with mode
        
        categorical_columns = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = convert_mostly_numeric(df_clean[col])
        
        log.info(f"Original shape: {df.shape}")
        log.info(f"Cleaned shape: {df_clean.shape}")
        log.info(f"Remaining missing values: {df_clean.isna().sum().sum()}")
        
        return df_clean
        
    except Exception as e:
        log.error(f"Error cleaning and transforming data: {str(e)}")
        return df
    
np.random.seed(0)

# Artificially add noise to make task harder
#Load data from db_file
field_list = []
filter_list = []
exclude_list = []
db_file = "C:\\ymohammad\\se350\\DataAnt\\mockDashboard\\learning\\heart.csv"
df = pd.read_csv(db_file, low_memory=False)
log.info(f"Loaded data from {db_file} -- len(df): {len(df)}")
#Clean data
df = df.drop_duplicates()
df = df.reset_index(drop=True)
log.info(f"Cleaned data -- len(df): {len(df)}")
df = clean_and_transform_data(df)

colList = df.columns.tolist()        
log.info(f"Length of colList (Fields): {len(colList)}")

if len(field_list) > 0:
    for field in field_list:
        for col in colList:
            if re.search(field, col, re.IGNORECASE):
                filter_list.append(col)
else:
    filter_list = colList

if len(exclude_list) > 0:
    for exclude in exclude_list:
        filter_list = [
            filter_item for filter_item in filter_list 
            if not re.search(exclude, filter_item, re.IGNORECASE)
        ]

df = df[filter_list]

# Prepare the data
X = df.drop(columns=['output'])
y = df['output']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model with better parameters
model = LogisticRegression(
    max_iter=2000,          # Increased iterations
    solver='saga',          # Better for large datasets
    n_jobs=-1,             # Parallel processing
    random_state=42,
    tol=1e-4,              # Relaxed tolerance
    C=1.0,                 # Regularization strength
    class_weight='balanced' # Handle imbalanced classes
)

# Fit the model
model.fit(X_train_scaled, y_train)

# Get predictions
y_scores_train = model.predict_proba(X_train_scaled)
y_scores_test = model.predict_proba(X_test_scaled)

# One hot encode the labels
y_train_onehot = pd.get_dummies(y_train, columns=model.classes_)
y_test_onehot = pd.get_dummies(y_test, columns=model.classes_)

# Create ROC curve plot
fig = go.Figure()
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

# Plot ROC curves for both training and test sets
for i in range(y_scores_train.shape[1]):
    # Training set ROC
    fpr_train, tpr_train, _ = roc_curve(y_train_onehot.iloc[:, i], y_scores_train[:, i])
    auc_train = roc_auc_score(y_train_onehot.iloc[:, i], y_scores_train[:, i])
    
    # Test set ROC
    fpr_test, tpr_test, _ = roc_curve(y_test_onehot.iloc[:, i], y_scores_test[:, i])
    auc_test = roc_auc_score(y_test_onehot.iloc[:, i], y_scores_test[:, i])
    
    name_train = f"Train {y_train_onehot.columns[i]} (AUC={auc_train:.2f})"
    name_test = f"Test {y_test_onehot.columns[i]} (AUC={auc_test:.2f})"
    
    fig.add_trace(go.Scatter(x=fpr_train, y=tpr_train, name=name_train, mode='lines'))
    fig.add_trace(go.Scatter(x=fpr_test, y=tpr_test, name=name_test, mode='lines', line=dict(dash='dash')))

fig.update_layout(
    xaxis=dict(title='False Positive Rate'),
    yaxis=dict(
        title='True Positive Rate',
        scaleanchor='x',
        scaleratio=1
    ),
    width=700,
    height=500,
    title='ROC Curves (Training vs Test)',
    showlegend=True
)

# Log model performance
log.info(f"Model training completed successfully")
log.info(f"Training set shape: {X_train.shape}")
log.info(f"Test set shape: {X_test.shape}")
log.info(f"Number of features: {X_train.shape[1]}")
log.info(f"Number of classes: {len(model.classes_)}")

fig.show()