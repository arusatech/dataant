import plotly.graph_objects as go
import plotly.express as px
#classification models
import numpy as np
from sklearn import metrics, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    AdaBoostClassifier,
    RandomForestClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
#Clustering models
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

#Generic models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression  # or any other model
from jsonpath_nz import log, jprint
import pandas as pd
import time
from collections import deque
from datetime import datetime

class ModelTrainer:
    _instance = None 
    def __new__(cls, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelTrainer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, random_state=42, model_name='logistic', test_size=0.2, X=None, y=None):
        if self._initialized:
            return
            
        # Validate inputs
        if X is None or y is None:
            raise ValueError("X and y must not be None")
            
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
            
        if len(X) == 0 or len(y) == 0:
            raise ValueError("X and y must not be empty")
            
        self.random_state = random_state    
        self.model_name = model_name
        self.test_size = test_size
        self.X = X
        self.y = y
        self.scaler = StandardScaler()
        self._initialized = True
        self.is_trained = False
        self.feature_columns = None
        self.last_training_shape = None
        self.training_history = deque(maxlen=100)
        self.cached_results = None
        self.model = {
            #classification models
            'logistic': LogisticRegression(
                max_iter=2000,          # Increased iterations
                solver='saga',          # Better for large datasets
                n_jobs=-1,             # Parallel processing
                random_state=self.random_state,
                tol=1e-4,              # Relaxed tolerance
                C=1.0,                 # Regularization strength
                class_weight='balanced' # Handle imbalanced classes
            ),
            'svm': SVC(
                kernel='rbf',
                random_state=self.random_state
            ),
            'decision_tree': DecisionTreeClassifier(
                max_depth=10,   
                random_state=self.random_state
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=5
            ),
            #regression models
            'linear': LinearRegression(
                n_jobs=-1
            ),
            'ridge': Ridge(),
            'lasso': Lasso(),
            'random_forest': RandomForestRegressor(),
            'svr': SVR(
                kernel='rbf'
            ),
            #clustering models
            'kmeans': KMeans(),
            'dbscan': DBSCAN(),
            'hierarchical': AgglomerativeClustering(),
            #ensemble models
            'gradient_boosting': GradientBoostingClassifier(),
            'adaboost': AdaBoostClassifier(),
            'xgboost': XGBClassifier(),
            'lightgbm': LGBMClassifier()
        }
    
    def train_model(self):
        """Train the model if needed"""
        try:
            current_shape = self.X.shape
            
            # Check if model needs retraining
            if (self.is_trained and 
                self.last_training_shape == current_shape and 
                self.feature_columns == list(self.X.columns) and 
                self.cached_results is not None):
                log.info("Using cached model results - no retraining needed")
                return self.cached_results
            
            log.info(f"Training model with shape {current_shape}")
            start_time = time.time()
            
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=self.test_size, random_state=self.random_state
            )
            
            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            model = self.model[self.model_name]
            model.fit(X_train_scaled, y_train)
            
            # Get predictions
            y_scores_train = model.predict_proba(X_train_scaled)
            y_scores_test = model.predict_proba(X_test_scaled)
            
            # Store state for cache validation
            self.is_trained = True
            self.last_training_shape = current_shape
            self.feature_columns = list(self.X.columns)
            
            
            
            # Log training time
            training_time = (time.time() - start_time) * 1000
            self.training_history.append({
                'timestamp': datetime.now(),
                'training_time': training_time,
                'data_size': len(self.X),
                'n_features': self.X.shape[1]
            })
            log.info(f"Model training completed in {training_time:.2f}ms")
            # Cache the results
            self.cached_results = {
                'y_train': y_train,
                'y_test': y_test,
                'y_scores_train': y_scores_train,
                'y_scores_test': y_scores_test,
                'classes': model.classes_,
                'training_history': self.training_history
            }
            
            return self.cached_results
            
        except Exception as e:
            self.is_trained = False
            self.cached_results = None
            log.error(f"Error in train_model: {str(e)}")
            log.traceback(e)
            raise
        
    def calculate_metrics(self, y_true, y_pred_proba):
        """Calculate ROC and Precision-Recall metrics"""
        try:
            # ROC Curve metrics
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Precision-Recall Curve metrics
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = average_precision_score(y_true, y_pred_proba)
            
            return {
                'fpr': fpr,
                'tpr': tpr,
                'auc_roc': roc_auc,
                'precision': precision,
                'recall': recall,
                'auc_pr': pr_auc
            }
            
        except Exception as e:
            log.error(f"Error calculating metrics: {str(e)}")
            raise
    
    def predict(self, X):
        """Make predictions on new data"""
        try:
            # Validate input
            if not isinstance(X, pd.DataFrame):
                raise ValueError("X must be a pandas DataFrame")
                
            if not self.is_trained:
                raise ValueError("Model not trained yet. Call train_model() first")
                
            if list(X.columns) != self.feature_columns:
                raise ValueError(f"Feature columns don't match training data. Expected {self.feature_columns}, got {list(X.columns)}")
            
            # Scale the features using the same scaler from training
            X_scaled = self.scaler.transform(X)
            
            # Get predictions
            y_scores = self.model[self.model_name].predict_proba(X_scaled)
            
            # Return prediction data in consistent format
            prediction_data = {
                'scores': y_scores,
                'classes': self.model[self.model_name].classes_,
                'feature_importance': None,  # Could add feature importance if needed
                'metadata': {
                    'prediction_time': time.time(),
                    'model_name': self.model_name,
                    'n_samples': len(X),
                    'n_features': X.shape[1]
                }
            }
            
            # Log prediction info
            log.info(f"Made predictions for {len(X)} samples using {self.model_name}")
            
            return prediction_data
            
        except Exception as e:
            log.error(f"Error in predict: {str(e)}")
            log.traceback(e)
            raise
        